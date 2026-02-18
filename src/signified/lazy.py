from collections import defaultdict, deque
from collections.abc import Generator, Callable, Iterator
from typing import Any

from .core import Observer, Variable
from .types import _OrderedWeakrefSet as OWRS, HasValue


type DepMap = defaultdict[int, OWRS[Variable[Any]]]


def changed(was: Any, now: Any) -> bool:
    if was is now:
        return False
    
    match was, now:
        case Variable(), Variable():
            return was is not now

        case Generator() | Callable(), Generator() | Callable():
            return True

    not_equal = was != now
    try:
        return bool(not_equal)
    except ValueError:
        # numpy array.any()
        return bool(getattr(not_equal, 'any', lambda: False)())
    return False

class VariableStore:
    """Container for tracking state of the current namespace
    
    Usage:
        ```py
        >>> from sigified import _STORE, Signal
        >>> 
        >>> s = Signal([1, 2, 3])
        >>> y = s.rx.map(lambda i: i**2)
        >>> print(_STORE)
        Tracking: 2 Variables
        ```
    """
    def __init__(self, *, max_deps: int| None = None) -> None:
        self.max_stack = max_deps
        self.tracked = OWRS[Variable[Any]]()
        self.dep_map: DepMap = defaultdict(OWRS[Variable[Any]])
        self.stack = deque[Variable[Any]](maxlen=max_deps)

    def __repr__(self) -> str:
        return f'Tracking: {len(self.tracked)} Variables'

    def track(self, variable: Variable[Any]) -> None:
        self.tracked.add(variable)
        # NOTE: variable._observers is only referenced
        # If the variable mutates this set, changes will be reflected here.
        # This is likely needs to be managed with a special threadsafe implementation
        # if at any point we want to support async/paralell updates 
        self.dep_map[id(variable)] = variable._observers # type: ignore (observers should all support Variable interface)

    def _get_deps(self, variable: Variable[Any]) -> Iterator[Variable[Any]]:
        self._prevent_circular(variable)
        for dep in self.dep_map.get(id(variable), []):
            yield dep

    def _prevent_circular(self, variable: Variable[Any]) -> None:
        # prevent infinite looping
        # e.g.
        # >>> x = Signal(1)
        # >>> y = Signal(x)
        # >>> x.value = y
        # RecursionError ...
        for dep in self.dep_map.get(id(variable), []):
            if id(dep) == id(variable):
                raise RecursionError(f'{dep}@{id(dep)} is dependant on self!')
    
    def refresh(self, variable: Variable[Any]) -> None:
        """Refresh the dependency tree for the provided Variable"""
        try:
            # get deps
            self.stack.extend(self._get_deps(variable))
            # recursively refresh deps
            while self.stack:
                self.refresh(self.stack.pop())
            # detect change
            if changed(variable._value, variable.value):
                # notify subscribers
                variable.notify()
        finally:
            # Ensure that the stack is cleared if the 
            # refresh fails
            self.stack.clear()