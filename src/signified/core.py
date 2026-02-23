"""Core reactive programming functionality."""

from __future__ import annotations

import importlib
import importlib.util
import math
import operator
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Concatenate, Literal, Protocol, Self, SupportsIndex, TypeGuard, Union, cast, overload

from .plugins import pm
from .types import HasValue, ReactiveValue, _OrderedWeakrefSet
from .rx import ReactiveMixIn

if importlib.util.find_spec("numpy") is not None:
    import numpy as np  # pyright: ignore[reportMissingImports]
else:
    np = None  # User doesn't have numpy installed


__all__ = [
    "Observer",
    "Variable",
    "Signal",
    "Computed",
    "Effect",
    "computed",
    "unref",
    "has_value",
    "deep_unref",
    "reactive_method",
    "as_signal",
]


class _SupportsAdd[OtherT, ResultT](Protocol):
    def __add__(self, other: OtherT, /) -> ResultT: ...


class _SupportsGetItem[KeyT, ValueT](Protocol):
    def __getitem__(self, key: KeyT, /) -> ValueT: ...


def computed[R](func: Callable[..., R]) -> Callable[..., Computed[R]]:
    """Wrap a function so calls produce a reactive ``Computed`` result.

    The returned wrapper accepts plain values, reactive values, or nested
    containers that include reactive values. On each recomputation, arguments
    are normalized with :func:`deep_unref`, so ``func`` receives plain Python
    values.

    The created :class:`Computed` tracks dependencies dynamically while the
    wrapped function runs. Any reactive value read during evaluation becomes a
    dependency for subsequent updates.

    Args:
        func: Function that computes a derived value from its inputs.

    Returns:
        A wrapper that returns a :class:`Computed` when called.

    Example:
        ```py
        >>> @computed
        ... def total(price, quantity):
        ...     return price * quantity
        >>> price = Signal(10)
        >>> quantity = Signal(2)
        >>> subtotal = total(price, quantity)
        >>> subtotal.value
        20
        >>> quantity.value = 3
        >>> subtotal.value
        30

        ```
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Computed[R]:
        def compute_func() -> R:
            resolved_args = tuple(deep_unref(arg) for arg in args)
            resolved_kwargs = {key: deep_unref(value) for key, value in kwargs.items()}
            return func(*resolved_args, **resolved_kwargs)

        return Computed(compute_func)

    return wrapper


def _warn_deprecated_alias(method: str, replacement: str) -> None:
    """Warn that a legacy `ReactiveMixIn` helper method is deprecated."""
    warnings.warn(
        f"`ReactiveMixIn.{method}()` is deprecated; use `{replacement}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _RxOps[T]:
    """Helper methods available under ``signal.rx``."""

    __slots__ = ("_source",)

    def __init__(self, source: "ReactiveMixIn[T]") -> None:
        self._source = source

    def map[R](self, fn: Callable[[T], R]) -> Computed[R]:
        """Return a reactive value by applying ``fn`` to ``self._source``.

        Args:
            fn: Function used to transform the current source value.

        Returns:
            A reactive value for ``fn(source.value)``.

        Example:
            ```py
            >>> s = Signal(4)
            >>> doubled = s.rx.map(lambda x: x * 2)
            >>> doubled.value
            8
            >>> s.value = 5
            >>> doubled.value
            10

            ```
        """
        return computed(fn)(self._source)

    def effect(self, fn: Callable[[T], None]) -> "Effect":
        """Eagerly run ``fn`` for side effects whenever the source changes.

        ``fn`` is called immediately on creation and again on every subsequent
        change to the source — without requiring the caller to read ``.value``.

        The effect is active as long as the caller holds the returned
        :class:`Effect` instance. Letting it be garbage-collected will silently
        stop the effect; call :meth:`Effect.dispose` to stop it explicitly.

        Args:
            fn: Callback that receives the current source value on each change.

        Returns:
            An :class:`Effect` instance whose lifetime controls the subscription.

        Example:
            ```py
            >>> seen = []
            >>> s = Signal(1)
            >>> e = s.rx.effect(seen.append)
            >>> seen
            [1]
            >>> s.value = 2
            >>> s.value = 3
            >>> seen
            [1, 2, 3]
            >>> e.dispose()
            >>> s.value = 99
            >>> seen
            [1, 2, 3]

            ```
        """
        assert isinstance(self._source, Variable)
        return Effect(self._source, fn)

    def peek(self, fn: Callable[[T], Any]) -> Computed[T]:
        """Run ``fn`` for side effects and pass through the original value.

        This is a lazy pipeline operator. ``fn`` only executes when the
        returned :class:`Computed` is read, not on every upstream change.
        Intermediate values are skipped if the source changes multiple times
        between reads.

        .. warning::
            The returned :class:`Computed` must be kept alive by the caller.
            Observers are held as weak references, so if nothing holds a strong
            reference to the returned value, it will be garbage-collected and
            ``fn`` will silently stop running.

            This is **not** an eagerly-evaluated effect. For code like::

                s.rx.peek(print)  # returned Computed immediately GC'd

            ``fn`` will never fire after the initial read. Assign the result
            to a variable that outlives the reactive computation.

        Args:
            fn: Side-effect callback that receives the current source value.

        Returns:
            A reactive value that always equals ``source.value``.

        Example:
            ```py
            >>> seen = []
            >>> s = Signal(1)
            >>> passthrough = s.rx.peek(lambda x: seen.append(x))
            >>> passthrough.value
            1
            >>> s.value = 3
            >>> passthrough.value
            3
            >>> seen
            [1, 3]

            ```
        """

        @computed
        def _peek(value: T) -> T:
            fn(value)
            return value

        return _peek(self._source)

    def len(self) -> Computed[int]:
        """Return a reactive value for ``len(source.value)``.

        Returns:
            A reactive value for ``len(source.value)``.

        Example:
            ```py
            >>> s = Signal([1, 2, 3])
            >>> length = s.rx.len()
            >>> length.value
            3
            >>> s.value = [10]
            >>> length.value
            1

            ```
        """
        return computed(len)(self._source)

    def is_(self, other: Any) -> Computed[bool]:
        """Return a reactive value for identity check ``source.value is other``.

        Args:
            other: Value to compare against with identity semantics.

        Returns:
            A reactive value for ``source.value is other``.

        Example:
            ```py
            >>> marker = object()
            >>> s = Signal(marker)
            >>> result = s.rx.is_(marker)
            >>> result.value
            True
            >>> s.value = object()
            >>> result.value
            False

            ```
        """
        return computed(operator.is_)(self._source, other)

    def is_not(self, other: Any) -> Computed[bool]:
        """Return a reactive value for identity check ``source.value is not other``.

        Args:
            other: Value to compare against with identity semantics.

        Returns:
            A reactive value for ``source.value is not other``.

        Example:
            ```py
            >>> marker = object()
            >>> s = Signal(marker)
            >>> result = s.rx.is_not(marker)
            >>> result.value
            False
            >>> s.value = object()
            >>> result.value
            True

            ```
        """
        return computed(operator.is_not)(self._source, other)

    def in_(self, container: Any) -> Computed[bool]:
        """Return a reactive value for containment check ``source.value in container``.

        Args:
            container: Value checked for membership, e.g. list/string/set.

        Returns:
            A reactive value for ``source.value in container``.

        Example:
            ```py
            >>> needle = Signal("a")
            >>> haystack = Signal("cat")
            >>> result = needle.rx.in_(haystack)
            >>> result.value
            True
            >>> needle.value = "z"
            >>> result.value
            False

            ```
        """
        return computed(operator.contains)(container, self._source)

    def contains(self, other: Any) -> Computed[bool]:
        """Return a reactive value for whether `other` is in `self._source`.

        Args:
            other: The value to check for containment.

        Returns:
            A reactive value for ``other in source.value``.

        Example:
            ```py
            >>> s = Signal([1, 2, 3, 4])
            >>> result = s.rx.contains(3)
            >>> result.value
            True
            >>> s.value = [5, 6, 7, 8]
            >>> result.value
            False

            ```
        """
        return computed(operator.contains)(self._source, other)

    def eq(self, other: Any) -> Computed[bool]:
        """Return a reactive value for whether ``source.value == other``.

        Args:
            other: Value to compare against.

        Returns:
            A reactive value for ``source.value == other``.

        Example:
            ```py
            >>> s = Signal(10)
            >>> result = s.rx.eq(10)
            >>> result.value
            True
            >>> s.value = 25
            >>> result.value
            False

            ```
        """
        return computed(operator.eq)(self._source, other)

    def where[A, B](self, a: HasValue[A], b: HasValue[B]) -> Computed[A | B]:
        """Return a reactive value for ``a`` if ``source`` is truthy, else ``b``.

        Args:
            a: The value to return if source is truthy.
            b: The value to return if source is falsy.

        Returns:
            A reactive value for ``a if source.value else b``.

        Example:
            ```py
            >>> condition = Signal(True)
            >>> result = condition.rx.where("Yes", "No")
            >>> result.value
            'Yes'
            >>> condition.value = False
            >>> result.value
            'No'

            ```
        """

        @computed
        def ternary(a: A, b: B, condition: Any) -> A | B:
            return a if condition else b

        return ternary(a, b, self._source)

    def as_bool(self) -> Computed[bool]:
        """Return a reactive value for the boolean value of ``self._source``.

        Note:
            ``__bool__`` cannot be implemented to return a non-``bool``, so it is provided as a method.

        Returns:
            A reactive value for ``bool(source.value)``.

        Example:
            ```py
            >>> s = Signal(1)
            >>> result = s.rx.as_bool()
            >>> result.value
            True
            >>> s.value = 0
            >>> result.value
            False

            ```
        """
        return computed(bool)(self._source)


def _coerce_to_bool(value: Any) -> bool:
    """Convert a value to bool, including ambiguous array-like values.

    Some array/series-style objects raise ``ValueError`` when coerced with
    ``bool(...)``. For those, fall back to ``value.all()`` semantics so
    partial matches are treated as unequal in comparison contexts.
    """
    try:
        return bool(value)
    except ValueError:
        # Handle numpy arrays, pandas Series, and similar objects.
        return bool(value.all())


class Observer(Protocol):
    def update(self) -> None:
        pass


class Variable[T](ABC, ReactiveMixIn[T]):
    """An abstract base class for reactive values.

    A reactive value is an object that can be observed by observer for changes and
    can notify observers when its value changes. This class implements both the observer
    and observable patterns.

    This class implements both the observer and observable pattern.

    Subclasses should implement the `update` method.

    Attributes:
        _observers (list[Observer]): List of observers subscribed to this variable.
    """

    __slots__ = ["_observers", "__name", "_version", "__weakref__"]

    def __init__(self):
        """Initialize the variable."""
        self._observers = _OrderedWeakrefSet[Observer]()
        self.__name = ""
        self._version = 0

    @staticmethod
    def _iter_variables(item: Any) -> Generator[Variable[Any], None, None]:
        """Yield `Variable` instances found in arbitrarily nested containers."""
        if isinstance(item, Variable):
            yield item
            return
        if isinstance(item, str):
            return
        if isinstance(item, Iterable):
            for sub_item in item:
                yield from Variable._iter_variables(sub_item)

    def subscribe(self, observer: Observer) -> None:
        """Subscribe an observer to this variable.

        Args:
            observer: The observer to subscribe.
        """
        self._observers.add(observer)

    def unsubscribe(self, observer: Observer) -> None:
        """Unsubscribe an observer from this variable.

        Args:
            observer: The observer to unsubscribe.
        """
        self._observers.discard(observer)

    def observe(self, items: Any) -> Self:
        """Subscribe the observer (`self`) to all items that are Observable.

        This method handles arbitrarily nested iterables.

        Args:
            items: A single item, an iterable, or a nested structure of items to potentially subscribe to.

        Returns:
            self
        """

        for item in self._iter_variables(items):
            if item is not self:
                item.subscribe(self)
        return self

    def unobserve(self, items: Any) -> Self:
        """Unsubscribe the observer (`self`) from all items that are Observable.

        Args:
            items: A single item or an iterable of items to potentially unsubscribe from.

        Returns:
            self
        """

        for item in self._iter_variables(items):
            if item is not self:
                item.unsubscribe(self)
        return self

    def notify(self) -> None:
        """Notify all observers by calling their update method."""
        for observer in tuple(self._observers):
            observer.update()

    def __repr__(self) -> str:
        """Represent the object in a way that shows the inner value."""
        return f"<{self.value!r}>"

    @abstractmethod
    def update(self) -> None:
        """Update method to be overridden by subclasses.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError("Update method should be overridden by subclasses")

    def _ipython_display_(self) -> None:
        from .display import _HAS_IPYTHON, IPythonObserver

        if not _HAS_IPYTHON:
            return

        try:
            display = importlib.import_module("IPython.display").display
        except ImportError:
            return

        handle = display(self.value, display_id=True)
        assert handle is not None
        IPythonObserver(self, handle)

    def add_name(self, name: str) -> Self:
        self.__name = name
        pm.hook.named(value=self)
        return self

    def __format__(self, format_spec: str) -> str:
        """Format the variable with custom display options.

        Format options:
        :n  - just the name (or type+id if unnamed)
        :d  - full debug info
        empty - just the value in brackets (default)
        """
        if not format_spec:  # Default - just show value in brackets
            return f"<{self.value}>"
        if format_spec == "n":  # Name only
            return self.__name if self.__name else f"{type(self).__name__}(id={id(self)})"
        if format_spec == "d":  # Debug
            name_part = f"name='{self.__name}', " if self.__name else ""
            return f"{type(self).__name__}({name_part}value={self.value!r}, id={id(self)})"
        return super().__format__(format_spec)  # Handles other format specs


_COMPUTE_STACK: list[Any] = []
"""Internal state that supports inferring reactive dependencies.

When a reactive value is read, we attach that read to the Computed at the
top of this stack so dependency subscriptions can be reconciled on refresh.
"""


def _track_read(variable: Variable[Any]) -> None:
    """Register `variable` as a dependency of the currently computing Computed."""
    if not _COMPUTE_STACK:
        # Reads outside Computed evaluation do not participate in dependency tracking.
        return
    owner = _COMPUTE_STACK[-1]
    if owner is variable:
        # Ignore self-reads to avoid self-dependency loops.
        return
    owner_impl = getattr(owner, "_impl", None)
    if owner_impl is not None:
        # Add this read for the current refresh run.
        owner_impl.register_dependency(variable)


def unref[T](value: HasValue[T]) -> T:
    """Resolve a value by unwrapping reactive containers until plain data remains.

    This utility repeatedly unwraps :class:`Variable` objects by following
    their internal ``_value`` references. It intentionally bypasses dependency
    tracking, which keeps this helper side-effect free inside reactive
    computations.

    Args:
        value: Plain value, reactive value, or nested reactive value.

    Returns:
        The fully unwrapped value.

    Example:
        ```py
        >>> nested = Signal(Signal(5))
        >>> unref(nested)
        5

        ```
    """
    current: Any = value
    while isinstance(current, Variable):
        if isinstance(current, Computed):
            current._impl.ensure_uptodate()
        current = current._value
    return current


def _has_changed(previous: Any, current: Any) -> bool:
    """Best-effort change detection for assignments into reactive values.

    This function is intentionally fail-open: if comparison is ambiguous or
    raises, we treat the value as changed to avoid missing invalidations.
    """
    # Compare callables by identity to avoid invoking custom `__eq__` logic and
    # to preserve stable references as unchanged.
    if callable(previous) or callable(current):
        return previous is not current
    # Reactive wrappers compare by identity rather than value equality.
    # Distinct wrapper objects should invalidate even if they currently resolve
    # to equal values.
    if isinstance(previous, Variable) or isinstance(current, Variable):
        return previous is not current

    # Keep NaN stable: treat NaN -> NaN as unchanged.
    if isinstance(previous, float) and isinstance(current, float):
        if math.isnan(previous) and math.isnan(current):
            return False

    try:
        # `==` may return non-scalar array-like values; coerce those with
        # all-elements semantics before negating.
        return not _coerce_to_bool(current == previous)
    except Exception:
        # Fail-open for exotic/buggy equality implementations.
        return True


def has_value[T](obj: Any, type_: type[T]) -> TypeGuard[HasValue[T]]:
    """Check whether an object's resolved value is an instance of ``type_``.

    This helper is a typed guard around :func:`unref`. It is useful when code
    accepts either plain values or reactive values and needs a narrowed type
    before continuing.

    Args:
        obj: Value to inspect. May be plain or reactive.
        type_: Expected resolved value type.

    Returns:
        ``True`` if ``unref(obj)`` is an instance of ``type_``; otherwise
        ``False``.

    Example:
        ```py
        >>> candidate = Signal(42)
        >>> has_value(candidate, int)
        True
        >>> has_value(candidate, str)
        False

        ```
    """
    return isinstance(unref(obj), type_)


class Signal[T](Variable[T]):
    """Mutable source-of-truth reactive value.

    ``Signal`` stores a value and notifies subscribers when that value changes.
    It is typically used for application state that should be observed by
    derived :class:`Computed` values.

    The ``value`` property is read/write:
    - reading ``value`` returns the resolved plain value
    - assigning ``value`` updates dependencies and notifies observers when the
      value changed

    Signals can also proxy mutation operations (for example ``__setattr__`` and
    ``__setitem__``) so in-place updates on wrapped objects can still trigger
    reactivity.

    Args:
        value: Initial value to wrap. May be plain or reactive.

    Example:
        ```py
        >>> count = Signal(1)
        >>> doubled = count * 2
        >>> doubled.value
        2
        >>> count.value = 3
        >>> doubled.value
        6

        ```
    """

    __slots__ = ["_value"]

    @overload
    def __init__(self, value: ReactiveValue[T]) -> None: ...

    @overload
    def __init__(self, value: T) -> None: ...

    def __init__(self, value: HasValue[T]) -> None:
        super().__init__()
        self._value: HasValue[T] = value
        self.observe(value)
        pm.hook.created(value=self)

    @property
    def value(self) -> T:
        """Get or set the current value."""
        pm.hook.read(value=self)
        _track_read(self)
        return unref(self._value)

    @value.setter
    def value(self, new_value: HasValue[T]) -> None:
        old_value = self._value
        if _has_changed(old_value, new_value):
            self._value = new_value
            self._version += 1
            pm.hook.updated(value=self)
            self.unobserve(old_value)
            self.observe(new_value)
            self.notify()

    @contextmanager
    def at(self, value: T) -> Generator[None, None, None]:
        """Temporarily set the signal to a given value within a context."""
        before = self.value
        try:
            self.value = value
            yield
        finally:
            self.value = before

    def update(self) -> None:
        """Update the signal and notify subscribers."""
        self._version += 1
        self.notify()


class _ComputedImpl:
    """Internal state and dependency tracking for :class:`Computed`."""

    __slots__ = ["_owner", "_deps", "_next_deps", "_dirty", "_has_value", "_is_computing", "_dep_versions"]

    def __init__(self, owner: "Computed[Any]") -> None:
        self._owner = owner
        self._deps = _OrderedWeakrefSet[Variable[Any]]()
        self._next_deps: _OrderedWeakrefSet[Variable[Any]] | None = None
        self._dirty = True
        self._has_value = False
        self._is_computing = False
        self._dep_versions: dict[int, int] = {}

    def register_dependency(self, dependency: Variable[Any]) -> None:
        if self._next_deps is not None and dependency is not self._owner:
            self._next_deps.add(dependency)

    def refresh(self) -> None:
        owner = self._owner
        if self._is_computing:
            raise RuntimeError("Cycle detected while evaluating Computed")

        previous_value = owner._value
        had_value = self._has_value

        # 1) Evaluate with dependency tracking enabled.
        self._is_computing = True
        self._next_deps = _OrderedWeakrefSet[Variable[Any]]()
        _COMPUTE_STACK.append(owner)
        try:
            next_value = owner.f()
        finally:
            popped = _COMPUTE_STACK.pop()
            assert popped is owner
            next_deps = self._next_deps
            self._next_deps = None
            self._is_computing = False

        # 2) Reconcile subscriptions against the dependency set from this run.
        assert next_deps is not None
        for dep in tuple(self._deps):
            if dep not in next_deps:
                dep.unsubscribe(owner)
        for dep in tuple(next_deps):
            if dep not in self._deps:
                dep.subscribe(owner)
        self._deps = next_deps
        self._dep_versions = {id(dep): dep._version for dep in tuple(next_deps)}

        # 3) Commit value/version if the computed result actually changed.
        self._dirty = False
        self._has_value = True
        if not had_value or _has_changed(previous_value, next_value):
            owner._value = next_value
            owner._version += 1
            pm.hook.updated(value=owner)

    def dependencies_changed(self) -> bool:
        """Return True when any dependency has a newer observed version."""
        for dep in tuple(self._deps):
            if isinstance(dep, Computed):
                dep._impl.ensure_uptodate()
            if self._dep_versions.get(id(dep), -1) != dep._version:
                return True
        return False

    def ensure_uptodate(self) -> None:
        # Fast path 1: already fresh.
        if not self._dirty and self._has_value:
            return

        # Fast path 2: dirty marker is stale, but dependency versions unchanged.
        if self._has_value and not self.dependencies_changed():
            self._dirty = False
            return

        # Slow path: recompute and reconcile dependencies.
        self.refresh()

    def invalidate(self) -> bool:
        """Mark stale and return True when this call changed the marker."""
        if self._dirty:
            return False
        self._dirty = True
        return True


class Computed[T](Variable[T]):
    """Read-only reactive value derived from a computation.

    ``Computed`` tracks dependencies as it executes and lazily recalculates the
    value when it is read after dependencies change. In most usage, instances
    are created implicitly via :func:`computed`, operator overloads, or helper
    APIs such as :func:`reactive_method`.

    Unlike :class:`Signal`, ``Computed.value`` is read-only and updated by
    re-running the stored function.

    Args:
        f: Zero-argument function used to compute the current value.
        dependencies: Deprecated compatibility argument. Still accepted for
            backwards compatibility but ignored. Runtime reads determine the
            true dependency set.

    Example:
        ```py
        >>> count = Signal(2)
        >>> squared = Computed(lambda: count.value ** 2)
        >>> squared.value
        4
        >>> count.value = 5
        >>> squared.value
        25

        ```
    """

    __slots__ = ["f", "_value", "_impl"]

    def __init__(self, f: Callable[[], T], dependencies: Any = None) -> None:
        super().__init__()
        self.f = f
        self._value: Any = None
        self._impl = _ComputedImpl(self)

        if dependencies is not None:
            warnings.warn(
                "`Computed(..., dependencies=...)` is deprecated and ignored; "
                "dependencies are tracked automatically during evaluation.",
                DeprecationWarning,
                stacklevel=2,
            )

        pm.hook.created(value=self)

    def update(self) -> None:
        """Mark this computed stale and propagate invalidation."""
        if not self._impl.invalidate():
            return
        self.notify()

    @property
    def value(self) -> T:
        """Get the current value, recomputing lazily when stale."""
        pm.hook.read(value=self)
        _track_read(self)
        self._impl.ensure_uptodate()
        return unref(self._value)


class Effect:
    """Eagerly run a side-effect whenever a reactive source changes.

    Unlike :meth:`_RxOps.peek`, ``Effect`` subscribes directly to the source
    and calls ``fn`` immediately on creation and on every subsequent change,
    without requiring the caller to read ``.value``.

    The effect is active for as long as the caller holds a reference to this
    object. Because observers are stored as weak references, letting the
    ``Effect`` instance be garbage-collected will silently stop the effect.
    Call :meth:`dispose` to stop it explicitly before the instance is released.

    Args:
        source: The reactive value to observe.
        fn: Callback that receives the current unwrapped value on each change.

    Example:
        ```py
        >>> seen = []
        >>> s = Signal(1)
        >>> e = Effect(s, seen.append)
        >>> seen
        [1]
        >>> s.value = 2
        >>> s.value = 3
        >>> seen
        [1, 2, 3]
        >>> e.dispose()
        >>> s.value = 99
        >>> seen
        [1, 2, 3]

        ```
    """

    __slots__ = ("_fn", "_source", "__weakref__")

    def __init__(self, source: Variable[Any], fn: Callable[[Any], None]) -> None:
        self._fn = fn
        self._source = source
        source.subscribe(self)
        fn(source.value)

    def update(self) -> None:
        """Called by the source when its value changes."""
        self._fn(self._source.value)

    def dispose(self) -> None:
        """Unsubscribe from the source and stop the effect."""
        self._source.unsubscribe(self)


# ---------------------------------------------------------------------------
# Utility functions that depend on the core types above
# ---------------------------------------------------------------------------

_SCALAR_TYPES = {int, float, str, bool, type(None)}


def deep_unref(value: Any) -> Any:
    """Recursively resolve reactive values within nested containers.

    ``deep_unref`` is the structural counterpart to :func:`unref`. It unwraps
    reactive values that appear inside supported containers while preserving the
    container type where practical.

    Supported behavior:
    - scalar primitives are returned unchanged
    - reactive values are unwrapped recursively
    - ``dict``, ``list``, and ``tuple`` contents are recursively unwrapped
    - generic iterables are reconstructed when possible; otherwise returned as-is
    - ``numpy.ndarray`` values with ``dtype=object`` are recursively unwrapped
      element-wise

    Args:
        value: Any value, possibly containing reactive values.

    Returns:
        Value with reactive nodes recursively replaced by plain values.

    Example:
        ```py
        >>> payload = {"a": Signal(1), "b": [Signal(2), 3]}
        >>> deep_unref(payload)
        {'a': 1, 'b': [2, 3]}

        ```
    """
    # Fast path for common scalar types (faster than isinstance check)
    if type(value) in _SCALAR_TYPES:
        return value

    # Base case - if it's a reactive value, resolve through `.value` so reads
    # are tracked while inside computed evaluations.
    if isinstance(value, Variable):
        return deep_unref(value.value)

    # For containers, recursively unref their elements
    if np is not None and isinstance(value, np.ndarray):
        assert np is not None
        return np.array([deep_unref(item) for item in value]).reshape(value.shape) if value.dtype == object else value
    if isinstance(value, dict):
        return {deep_unref(k): deep_unref(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(deep_unref(item) for item in value)
    if isinstance(value, Iterable) and not isinstance(value, str):
        constructor: Any = type(value)
        try:
            return constructor(deep_unref(item) for item in value)
        except TypeError:
            return value

    return value


def reactive_method[**P, T](
    *dep_names: str,
) -> Callable[[Callable[Concatenate[Any, P], T]], Callable[Concatenate[Any, P], Computed[T]]]:
    """Deprecated helper for method-style computed values.

    This decorator now delegates to :func:`computed`. It is retained only for
    backwards compatibility and will be removed in a future release.

    Args:
        *dep_names: Deprecated compatibility argument. Ignored.

    Returns:
        A decorator that transforms an instance method into one that returns
        :class:`Computed`.
    """

    warnings.warn(
        "`reactive_method(...)` is deprecated and will be removed in a future "
        "release; use `@computed` instead. Any dependency-name arguments are "
        "ignored.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(func: Callable[Concatenate[Any, P], T]) -> Callable[Concatenate[Any, P], Computed[T]]:
        return cast(Callable[Concatenate[Any, P], Computed[T]], computed(func))

    return decorator


def as_signal[T](val: HasValue[T]) -> Signal[T]:
    """Normalize a value to a signal-compatible reactive object.

    If ``val`` is already reactive, it is returned unchanged to avoid wrapping
    an existing reactive node. Otherwise a new :class:`Signal` is created.

    Args:
        val: Plain value or reactive value.

    Returns:
        A reactive value suitable for APIs expecting ``Signal``-like behavior.

    Note:
        Existing reactive values are returned as-is at runtime, including
        ``Computed`` instances.

    Example:
        ```py
        >>> from signified import Signal, as_signal
        >>> as_signal(3).value
        3
        >>> s = Signal(4)
        >>> as_signal(s) is s
        True

        ```
    """
    return cast(Signal[T], val) if isinstance(val, Variable) else Signal(val)
