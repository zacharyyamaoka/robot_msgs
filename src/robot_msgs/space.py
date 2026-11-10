"""Spaces: a message holds a value OR a range, and you can sample it or check against it.

    from robot_msgs.space import Value, Space
    pick = Pose(position=Point(x=Value(0.5, "m", low=0.4, high=0.6), y=0.0, z=0.3))
    Space(pick).sample(rng)          # -> a Pose, x drawn in [0.4, 0.6], marked basis=SAMPLED
    Space(pick).contains(measured)   # -> the acceptable range in a test
    Space(traj, positions=(low, high)).sample(rng)   # array fields: gym-Box bounds

WHY the same class and a derived space (Zach, Oct 5 2026, "do not re-litigate"): a second
`PoseSpace` / `CheckedBelt` class "drifts" from the message it shadows. So a field carries its
own range as a `Value`, which IS the builtin float/int/str/tuple ("I definitely don't want to
have to do width_m.value"), and `Space(msg)` is a view over the instance whose `sample()`
returns the ordinary class. `plain()` drops the notes at a runtime boundary (ROS, JSON).
WHY here, at the lowest level (Zach, Nov 10 2026): every package can then state an acceptable
range or sample a message without a vocabulary of its own. Moved from
py_dataclass_viewer/values.py (the reference implementation); stdlib only.

Value: a number (or string) that remembers where it came from, and still is the number.
`Value(1.2, "m")` returns a FloatValue, so `isinstance(x, Value)` and `isinstance(x, float)`
are both true. Arithmetic returns a plain number: a derived number is a new claim.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np


class Basis(StrEnum):
    """Where a number comes from."""

    CHOSEN = "chosen"
    DATASHEET = "datasheet"
    MARKET = "market"
    ASSUMED = "assumed"
    MEASURED = "measured"
    SAMPLED = "sampled"  # drawn by Space.sample() from this value's range, choices or distribution


# --- distributions: per-value, first class ---
@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def draw(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class Normal:
    mean: float
    std: float

    def draw(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)


@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    def draw(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))


@dataclass(frozen=True)
class Choice:
    """A discrete distribution: options with optional weights."""

    options: tuple
    weights: tuple | None = None

    def draw(self, rng: random.Random):
        return rng.choices(self.options, weights=self.weights)[0]


Distribution = Uniform | Normal | LogUniform | Choice

META = (
    "unit",
    "low",
    "high",
    "step",
    "choices",
    "dist",
    "basis",
    "source",
    "measured_by",
    "notes",
    "date_accessed",
)


def _within(raw, low, high) -> bool:
    """low <= raw <= high; a tuple is checked elementwise (a scalar bound applies to each)."""
    if isinstance(raw, tuple):
        lows = low if isinstance(low, tuple) else (low,) * len(raw)
        highs = high if isinstance(high, tuple) else (high,) * len(raw)
        if len(lows) != len(raw) or len(highs) != len(raw):
            raise ValueError(f"{raw!r} has {len(raw)} elements but low={low!r}, high={high!r}")
        return all(lo <= x <= hi for lo, x, hi in zip(lows, raw, highs, strict=True))
    return low <= raw <= high


def _rebuild(cls, raw, meta):
    meta = dict(meta)
    unit = meta.pop("unit")  # unit is positional-only in the constructor
    return cls(raw, unit, **meta)


class Value:
    """`Value(raw, unit=None, *, low, high, step, choices, dist, basis, source, …)`.

    Calling Value(...) picks the carrier for the raw type (FloatValue / IntValue / StrValue /
    TupleValue), each subclassing both Value and the builtin. bool and None cannot carry
    evidence: Python forbids subclassing them.
    """

    _base: type = object

    def __new__(
        cls,
        raw=None,
        unit=None,
        /,
        *,
        low=None,
        high=None,
        step=None,
        choices=None,
        dist=None,
        basis=None,
        source="",
        measured_by="",
        notes="",
        date_accessed="",
    ):
        if cls is Value:
            if isinstance(raw, bool) or raw is None:
                raise TypeError(
                    "bool and None cannot carry evidence (Python forbids subclassing them)"
                )
            for base, carrier in (
                (float, FloatValue),
                (int, IntValue),
                (str, StrValue),
                (tuple, TupleValue),
            ):
                if isinstance(raw, base):
                    cls = carrier
                    break
            else:
                raise TypeError(f"Value wraps float, int, str or tuple, not {type(raw).__name__}")
        obj = cls._base.__new__(cls, raw)
        meta = dict(
            unit=unit,
            low=low,
            high=high,
            step=step,
            choices=tuple(choices) if choices is not None else None,
            dist=dist,
            basis=basis,
            source=source,
            measured_by=measured_by,
            notes=notes,
            date_accessed=date_accessed,
        )
        for k, val in meta.items():
            object.__setattr__(obj, k, val)
        if low is not None and high is not None and not _within(raw, low, high):
            raise ValueError(
                f"{raw!r} outside [{low}, {high}] ({source or unit or ''})".rstrip(" ()")
            )
        if meta["choices"] is not None and raw not in meta["choices"]:
            raise ValueError(f"{raw!r} not one of {meta['choices']}")
        return obj

    def __init__(self, *args, **kwargs):  # all work happens in __new__
        pass

    def __setattr__(self, key, val):
        raise AttributeError(
            f"{type(self).__name__} is immutable; write a new Value(...) in the source"
        )

    def __reduce__(self):  # copy, deepcopy, asdict and pickle rebuild through __new__
        return (_rebuild, (type(self), self._base(self), self.meta()))

    @property
    def value(self):
        """The plain builtin. New code just uses the value itself."""
        return self._base(self)

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in META}


class FloatValue(Value, float):
    _base = float


class IntValue(Value, int):
    _base = int


class StrValue(Value, str):
    _base = str


class TupleValue(Value, tuple):
    """A vector or pair with a source: `Value((0.0, 0.0, 0.3), "m", low=(-1, -1, 0))`."""

    _base = tuple


def meta(x) -> dict | None:
    """The evidence on a value, or None for a plain one."""
    return x.meta() if isinstance(x, Value) else None


def plain(obj):
    """The same object with every Value replaced by its builtin: for ROS, JSON and the like."""
    if isinstance(obj, TupleValue):
        return tuple(plain(x) for x in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **{f.name: plain(getattr(obj, f.name)) for f in fields(obj) if f.init})
    if isinstance(obj, (list, tuple)):
        return type(obj)(plain(x) for x in obj)
    return obj.value if isinstance(obj, Value) else obj


class Space:
    """A message seen as a space (gymnasium-style), derived from the instance itself.

    Scalars: every `Value` with a dist, choices or [low, high] can be drawn and checked.
    Arrays: `Space(traj, positions=(low, high))` bounds a top-level array field the gym `Box`
    way; low/high broadcast against the field (per joint, per sample, or one scalar).
    `sample()` returns the same class as the instance; nothing else is discarded.
    """

    def __init__(self, instance: Any, **array_bounds: tuple[Any, Any]) -> None:
        self.instance = instance
        self.array_bounds: dict[str, tuple[Any, Any]] = {}
        for name, (low, high) in array_bounds.items():
            current = getattr(instance, name, None)
            if not isinstance(current, np.ndarray):
                raise ValueError(f"{type(instance).__name__} has no array field {name!r}")
            low_b, high_b = (
                np.broadcast_to(low, current.shape),
                np.broadcast_to(high, current.shape),
            )
            if np.any(low_b > high_b):
                raise ValueError(f"{name}: low > high")
            self.array_bounds[name] = (low_b, high_b)

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample(self, rng: random.Random | None = None):
        rng = rng or random.Random()
        drawn = self._draw(self.instance, rng)
        if self.array_bounds:
            gen = np.random.default_rng(rng.getrandbits(64))
            drawn = replace(
                drawn,
                **{name: gen.uniform(low, high) for name, (low, high) in self.array_bounds.items()},
            )
        return drawn

    def _draw(self, x, rng):
        if is_dataclass(x) and not isinstance(x, type):
            return replace(
                x, **{f.name: self._draw(getattr(x, f.name), rng) for f in fields(x) if f.init}
            )
        if isinstance(x, TupleValue) and x.dist is None and x.choices is None:
            return x  # a vector is one claim: drawn only through its own choices or dist
        if isinstance(x, (list, tuple)) and not isinstance(x, TupleValue):
            return type(x)(self._draw(i, rng) for i in x)
        m = meta(x)
        if not m or (
            m["dist"] is None and m["choices"] is None and (m["low"] is None or m["high"] is None)
        ):
            return x
        if m["dist"] is not None:
            drawn = m["dist"].draw(rng)
        elif m["choices"] is not None:
            drawn = rng.choice(m["choices"])
        else:
            drawn = rng.uniform(m["low"], m["high"])
        if isinstance(drawn, (int, float)) and not isinstance(drawn, bool):
            if m["low"] is not None:
                drawn = max(m["low"], drawn)
            if m["high"] is not None:
                drawn = min(m["high"], drawn)
            if isinstance(x, int):
                drawn = round(drawn)
            if m["step"]:
                base = m["low"] if m["low"] is not None else 0
                drawn = base + round((drawn - base) / m["step"]) * m["step"]
                drawn = round(drawn) if isinstance(x, int) else drawn
        return Value(
            drawn,
            m["unit"],
            **{
                **{k: m[k] for k in META if k != "unit"},
                "basis": Basis.SAMPLED,
                "notes": f"sampled from {x!r}" + (f"; {m['notes']}" if m["notes"] else ""),
            },
        )

    # ── checking ──────────────────────────────────────────────────────────────

    def contains(self, other: Any) -> bool:
        """True when every ranged field of `other` lies inside this space."""
        return not self.violations(other)

    def violations(self, other: Any) -> list[str]:
        """One line per field of `other` outside its range: ``position.x=0.7 outside [0.4, 0.6]``"""
        if type(other) is not type(self.instance):
            raise TypeError(
                f"a {type(self.instance).__name__} space cannot hold a {type(other).__name__}"
            )
        found: list[str] = []
        self._check(self.instance, other, "", found)
        for name, (low, high) in self.array_bounds.items():
            values = np.asarray(getattr(other, name))
            if values.shape != low.shape:
                found.append(f"{name} has shape {values.shape}, the space {low.shape}")
                continue
            outside = np.argwhere((values < low) | (values > high))
            if len(outside):
                index = tuple(int(i) for i in outside[0])
                where = ", ".join(str(i) for i in index)
                found.append(
                    f"{name}[{where}]={float(values[index])!r} outside "
                    f"[{float(low[index])!r}, {float(high[index])!r}]"
                    f" ({len(outside)} samples outside)"
                )
        return found

    def _check(self, ranged, other, path, found):
        if is_dataclass(ranged) and not isinstance(ranged, type):
            for f in fields(ranged):
                self._check(
                    getattr(ranged, f.name), getattr(other, f.name), f"{path}{f.name}.", found
                )
            return
        m = meta(ranged)
        if not m:
            return
        name = path.rstrip(".")
        if (
            m["low"] is not None
            and m["high"] is not None
            and not _within(other, m["low"], m["high"])
        ):
            found.append(f"{name}={plain(other)!r} outside [{m['low']}, {m['high']}]")
        elif m["choices"] is not None and other not in m["choices"]:
            found.append(f"{name}={plain(other)!r} not one of {m['choices']}")
