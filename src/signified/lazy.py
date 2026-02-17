from collections import defaultdict, deque
from collections.abc import Generator, Callable
from typing import Any

from .core import Observer, Variable
from .types import _OrderedWeakrefSet as OWRS, HasValue


type DepMap = defaultdict[int, OWRS[Observer]]

def changed(was: Any, now: Any) -> bool:
    # Same object
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
        return bool(getattr(not_equal, 'any', lambda: False)())
    return False

class ComputeContext:
    def __init__(self, *, max_stack: int| None = None) -> None:
        self.max_stack = max_stack
        self.tracked = OWRS[Variable[Any]]()
        self.dep_map: DepMap = defaultdict(OWRS[Observer])

    def track(self, variable: Variable[Any]) -> None:
        self.tracked.add(variable)
        self.dep_map[id(variable)] = variable._observers

    def sync_context(self, variable: Variable[Any]) -> None:
        for dep in self.dep_map.get(id(variable), []):
            dep.update()

    def refresh(self, variable: Variable[Any]) -> None:
        _stack = deque[Variable[Any]](maxlen=self.max_stack)
        _stack.append(variable)
        _deps = self.dep_map.get(id(variable), [])
        if not _deps:
            return
        