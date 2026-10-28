"""What every message shares: the `Msg` base, read-only arrays, value equality, dict round-trips.

A message is an immutable value. Its numbers live in read-only numpy arrays, so a message can
hand out views of its data (slices, rows, whole columns) without copying and without anyone
being able to change it underneath you.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
import weakref
from collections.abc import Mapping
from typing import Any, ClassVar, TypeVar

import numpy as np

Array = np.ndarray
"""Annotation alias for array fields. Arrays inside messages are always read-only."""

M = TypeVar("M", bound="Msg")

_REGISTRY: dict[str, type[Msg]] = {}

# Arrays this library allocated and froze itself. Only their views may skip the defensive copy:
# a caller's own read-only array could be flipped back to writeable by its owner.
_OWNED: weakref.WeakValueDictionary[int, np.ndarray] = weakref.WeakValueDictionary()


def frozen_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Return `value` as a read-only array nobody else can mutate.

    Copies, unless `value` is already a view onto memory this library froze (a slice or row
    of another message's array). WHY: slicing a trajectory must stay O(1), never a copy.
    `dtype=None` keeps the input's dtype (labels, ints, bools in trajectory channels).
    """
    if isinstance(value, np.ndarray) and (dtype is None or value.dtype == np.dtype(dtype)):
        if not value.flags.writeable and _is_owned(value):
            return value
    array = np.array(value, dtype=dtype, copy=True)
    return adopt(array)


def adopt(array: np.ndarray) -> np.ndarray:
    """Freeze an array this library just allocated, without copying it. Internal use only."""
    array.setflags(write=False)
    _OWNED[id(array)] = array
    return array


def _is_owned(array: np.ndarray) -> bool:
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return _OWNED.get(id(root)) is root


def optional_array(value: Any, dtype: Any = np.float64) -> np.ndarray | None:
    return None if value is None else frozen_array(value, dtype)


def set_field(msg: Msg, name: str, value: Any) -> None:
    """Normalise a field inside `__post_init__` of a frozen dataclass."""
    object.__setattr__(msg, name, value)


class Msg:
    """Base of every message. Subclasses are `@dataclass(frozen=True, eq=False, repr=False)`.

    The base supplies array-aware `==` and `hash`, a compact `repr` that never dumps a large
    array, `replace()`, and a lossless `to_dict()` / `from_dict()`.
    """

    __slots__ = ()

    type_name: ClassVar[str]
    """Stable name like ``"trajectory/JointTrajectory"``, stored in serialised dicts."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        module = cls.__module__.split(".")
        domain = module[1] if module[0] == "robot_msgs" and len(module) > 2 else cls.__module__
        cls.type_name = f"{domain}/{cls.__name__}"
        _REGISTRY[cls.type_name] = cls

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            values_equal(getattr(self, f.name), getattr(other, f.name))
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
        )

    def __hash__(self) -> int:
        fields = dataclasses.fields(self)  # type: ignore[arg-type]
        return hash((type(self), *(_hashable(getattr(self, f.name)) for f in fields)))

    def __repr__(self) -> str:
        parts = []
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if _is_unset(value, f):
                continue
            parts.append(f"{f.name}={short_repr(value)}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def replace(self: M, **changes: Any) -> M:
        """A copy with some fields changed. Validation runs again."""
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict with a ``"_type"`` key; arrays keep their dtype and shape."""
        return to_dict(self)

    @classmethod
    def from_dict(cls: type[M], data: Mapping[str, Any]) -> M:
        return from_dict(data, cls)


# ── equality, hashing, repr ──────────────────────────────────────────────────


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            value = value + 0.0  # -0.0 → 0.0 so equal arrays hash equally
            value = np.where(np.isnan(value), np.nan, value)
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _is_unset(value: Any, f: dataclasses.Field) -> bool:
    if value is None:
        return True
    if isinstance(value, (Mapping, tuple)) and not value:
        return True
    if isinstance(value, Msg) and f.default is not dataclasses.MISSING:
        return value == f.default
    if (
        isinstance(value, Msg)
        and f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    ):
        return value == f.default_factory()  # type: ignore[misc]
    return False


def short_repr(value: Any) -> str:
    """repr that summarises big arrays as ``float64[500×6]`` instead of printing them."""
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return np.array2string(value, precision=4, separator=", ", suppress_small=True)
        return f"{value.dtype}[{'×'.join(str(n) for n in value.shape)}]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k!r}: {short_repr(v)}" for k, v in value.items()) + "}"
    if isinstance(value, tuple) and len(value) > 6:
        head = ", ".join(short_repr(v) for v in value[:3])
        return f"({head}, … {len(value) - 3} more)"
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def allclose(a: Any, b: Any, *, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """Structural approximate equality for messages (same types, numbers within tolerance)."""
    if isinstance(a, Msg) or isinstance(b, Msg):
        if type(a) is not type(b):
            return False
        return all(
            allclose(getattr(a, f.name), getattr(b, f.name), rtol=rtol, atol=atol)
            for f in dataclasses.fields(a)  # type: ignore[arg-type]
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fciu" and b.dtype.kind in "fciu":
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol) or (math.isnan(a) and math.isnan(b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(allclose(a[k], b[k], rtol=rtol, atol=atol) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(
            allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True)
        )
    return values_equal(a, b)


# ── dict / JSON / npz round-trips ────────────────────────────────────────────


def to_dict(msg: Msg, *, arrays: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Encode a message as plain dicts and lists.

    With `arrays`, array payloads go into that dict (for `save_npz`) and the encoded tree only
    keeps a reference to them.
    """
    out: dict[str, Any] = {"_type": msg.type_name}
    for f in dataclasses.fields(msg):  # type: ignore[arg-type]
        out[f.name] = _encode(getattr(msg, f.name), arrays)
    return out


def _encode(value: Any, arrays: dict[str, np.ndarray] | None) -> Any:
    if isinstance(value, Msg):
        return to_dict(value, arrays=arrays)
    if isinstance(value, np.ndarray):
        if arrays is not None:
            key = f"a{len(arrays)}"
            arrays[key] = value
            return {"npz": key}
        return {"dtype": value.dtype.str, "shape": list(value.shape), "data": value.ravel().tolist()}
    if isinstance(value, Mapping):
        return {str(k): _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v, arrays) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def from_dict(
    data: Mapping[str, Any],
    cls: type[M] | None = None,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> M:
    """Decode what `to_dict` produced. `cls` defaults to the class named by ``"_type"``."""
    target = _REGISTRY[data["_type"]] if "_type" in data else cls
    if target is None:
        raise ValueError("from_dict needs a class or a '_type' key")
    if cls is not None and not issubclass(target, cls):
        raise TypeError(f"{data.get('_type')} is not a {cls.__name__}")
    hints = typing.get_type_hints(target)
    kwargs = {
        f.name: _decode(data[f.name], hints[f.name], arrays)
        for f in dataclasses.fields(target)  # type: ignore[arg-type]
        if f.init and f.name in data
    }
    return target(**kwargs)  # type: ignore[return-value]


def _decode(value: Any, hint: Any, arrays: Mapping[str, np.ndarray] | None) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _decode(value, options[0], arrays) if len(options) == 1 else value
    if hint is np.ndarray or origin is np.ndarray:
        return _decode_array(value, arrays)
    if isinstance(hint, type) and issubclass(hint, Msg):
        return from_dict(value, hint, arrays=arrays)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(v, args[0], arrays) for v in value)
        return tuple(_decode(v, a, arrays) for v, a in zip(value, args, strict=True))
    if origin in (dict, Mapping) or origin is getattr(typing, "Mapping", None):
        value_hint = args[1] if len(args) == 2 else Any
        return {k: _decode(v, value_hint, arrays) for k, v in value.items()}
    if hint is Any and isinstance(value, Mapping) and ("npz" in value or "dtype" in value):
        return _decode_array(value, arrays)
    return value


def _decode_array(value: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None) -> np.ndarray:
    if "npz" in value:
        if arrays is None:
            raise ValueError("array reference without an npz archive")
        return frozen_array(arrays[value["npz"]], None)
    array = np.array(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    return adopt(array)


def save_npz(msg: Msg, path: Any) -> None:
    """Save a message compactly: arrays as raw npz entries, everything else as JSON."""
    arrays: dict[str, np.ndarray] = {}
    tree = to_dict(msg, arrays=arrays)
    np.savez(path, __msg__=np.array(json.dumps(tree)), **arrays)


def load_npz(path: Any) -> Msg:
    with np.load(path, allow_pickle=False) as archive:
        tree = json.loads(str(archive["__msg__"]))
        arrays = {k: archive[k] for k in archive.files if k != "__msg__"}
    return from_dict(tree, arrays=arrays)


def message_types() -> dict[str, type[Msg]]:
    """Every registered message type by its `type_name`."""
    return dict(_REGISTRY)
