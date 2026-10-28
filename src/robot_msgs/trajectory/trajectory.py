"""Trajectory — samples at strictly increasing times. One entity, many views.

The base of every trajectory, in any domain: a time axis `t`, typed per-sample fields that a
subclass declares with `sample()`, and free-form `channels`. Storage is struct-of-arrays — one
read-only array per field, time on axis 0 — and every other shape is a view of it:

    traj.positions      the whole (N, J) array               batch view, no copy
    traj["kp"]          any field or channel, by name
    traj[i]             one point (row views)                per-step view
    for p in traj       every point
    traj[a:b]           a trajectory over the same memory    O(1)
    traj.at(1.25)       an interpolated point
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np

from robot_msgs._core import Array, FrozenDict, Msg, adopt, frozen_array, set_field, short_repr
from robot_msgs.geometry.rotations import quat_slerp
from robot_msgs.std import Header

Interpolation = Literal["linear", "slerp", "hold"]
Extrapolation = Literal["raise", "clamp"]
TrajectoryT = TypeVar("TrajectoryT", bound="Trajectory")


def sample(
    *,
    interp: Interpolation = "linear",
    default: Any = dataclasses.MISSING,
    point_name: str | None = None,
    dtype: Any = np.float64,
) -> Any:
    """Declare a per-sample field: an array whose first axis is time.

    `interp` says how `at()` and `resample()` fill between samples: ``"linear"``, ``"slerp"``
    (xyzw quaternions on the last axis) or ``"hold"`` (previous sample — modes, labels, ints).
    `point_name` renames the field on the point type, e.g. ``positions (N, 3)`` → ``position``.
    `dtype=None` keeps whatever dtype the caller passes.
    """
    metadata = {"sample": interp, "point_name": point_name, "dtype": dtype}
    if default is dataclasses.MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


@functools.cache
def sample_fields(cls: type[Trajectory]) -> tuple[dataclasses.Field, ...]:
    """The fields of a trajectory class whose first axis is time."""
    return tuple(f for f in dataclasses.fields(cls) if "sample" in f.metadata)


def _point_name(f: dataclasses.Field) -> str:
    return f.metadata.get("point_name") or f.name


def _point_value(value: Any, dtype: Any) -> Any:
    if value is None or isinstance(value, (Msg, str)):
        return value
    array = frozen_array(value, dtype)
    return array.item() if array.ndim == 0 else array


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class TrajectoryPoint(Msg):
    """One sample: its time and the value of every per-sample field at that time."""

    t: float
    channels: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_field(self, "t", float(self.t))
        channels = {str(k): _point_value(v, None) for k, v in self.channels.items()}
        set_field(self, "channels", FrozenDict(channels))
        for f in dataclasses.fields(self):
            if f.name not in ("t", "channels"):
                dtype = f.metadata.get("dtype", np.float64)
                set_field(self, f.name, _point_value(getattr(self, f.name), dtype))

    @property
    def time_from_start(self) -> float:
        """ROS's name for `t`."""
        return self.t

    def __getitem__(self, name: str) -> Any:
        if name in self.channels:
            return self.channels[name]
        if name in {f.name for f in dataclasses.fields(self)}:
            return getattr(self, name)
        raise KeyError(name)


@functools.cache
def point_class(cls: type[Trajectory]) -> type[TrajectoryPoint]:
    """The point type a trajectory class yields; generated when the class doesn't declare one.

    WHY generated: a new domain should only have to declare its `sample()` fields to get
    indexing and iteration, not also hand-write a matching point class.
    """
    declared = cls.point_type
    declared_names = {f.name for f in dataclasses.fields(declared)}
    missing = [f for f in sample_fields(cls) if _point_name(f) not in declared_names]
    if not missing:
        return declared
    spec = [
        (_point_name(f), Any, field(default=None, metadata={"dtype": f.metadata["dtype"]}))
        for f in missing
    ]
    return dataclasses.make_dataclass(
        f"{cls.__name__}Point",
        spec,
        bases=(declared,),
        frozen=True,
        eq=False,
        repr=False,
        kw_only=True,
        namespace={"__module__": cls.__module__},
    )


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Trajectory(Msg):
    """Samples at strictly increasing times `t` (seconds after `header.stamp`).

    A bare `Trajectory` is just `t` plus `channels` — enough for any time series. Domain
    trajectories (`JointTrajectory`, `PoseTrajectory`, your own) subclass it and declare typed
    per-sample fields with `sample()`; they inherit everything below.
    """

    t: Array
    channels: Mapping[str, Array] = field(default_factory=dict)
    header: Header = field(default_factory=Header)

    point_type: ClassVar[type[TrajectoryPoint]] = TrajectoryPoint

    def __post_init__(self) -> None:
        t = frozen_array(self.t)
        if t.ndim != 1:
            raise ValueError(f"{type(self).__name__}.t must be 1-D, got shape {t.shape}")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"{type(self).__name__}.t must be finite")
        if t.size > 1 and np.any(np.diff(t) <= 0.0):
            first_bad = int(np.argmax(np.diff(t) <= 0.0))
            raise ValueError(
                f"{type(self).__name__}.t must be strictly increasing; "
                f"t[{first_bad}]={t[first_bad]!r} then t[{first_bad + 1}]={t[first_bad + 1]!r}"
            )
        set_field(self, "t", t)
        count = t.shape[0]
        names = {f.name for f in dataclasses.fields(self)}
        for f in sample_fields(type(self)):
            value = getattr(self, f.name)
            if value is None:
                continue
            array = frozen_array(value, f.metadata["dtype"])
            self._check_length(f.name, array, count)
            set_field(self, f.name, array)
        channels = {}
        for name, value in self.channels.items():
            if name in names:
                raise ValueError(f"channel {name!r} would shadow the field of the same name")
            array = frozen_array(value, None)
            self._check_length(f"channels[{name!r}]", array, count)
            channels[str(name)] = array
        set_field(self, "channels", FrozenDict(channels))

    def _check_length(self, name: str, array: np.ndarray, count: int) -> None:
        if array.ndim == 0 or array.shape[0] != count:
            got = "a scalar" if array.ndim == 0 else f"{array.shape[0]} samples"
            raise ValueError(f"{type(self).__name__}.{name} has {got}, t has {count}")

    # ── views ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        for index in range(len(self)):
            yield self._point(index)

    def __getitem__(self, key: Any) -> Any:
        """``str`` → that field's array · ``int`` → a point · slice, mask or indices → a trajectory."""
        if isinstance(key, str):
            return self._array(key)
        if isinstance(key, (int, np.integer)):
            index = operator.index(key)
            if not -len(self) <= index < len(self):
                raise IndexError(f"sample {index} out of range for {len(self)} samples")
            return self._point(index % len(self))
        return self._take(key)

    @property
    def sample_names(self) -> tuple[str, ...]:
        """Names of every per-sample array that is set: typed fields, then channels."""
        typed = tuple(f.name for f in sample_fields(type(self)) if getattr(self, f.name) is not None)
        return typed + tuple(self.channels)

    def _array(self, name: str) -> np.ndarray:
        if name == "t":
            return self.t
        if name in self.channels:
            return self.channels[name]
        for f in sample_fields(type(self)):
            if f.name == name:
                value = getattr(self, name)
                if value is None:
                    raise KeyError(f"{name!r} is not set on this {type(self).__name__}")
                return value
        raise KeyError(f"{name!r} is not a field or channel; have {('t',) + self.sample_names}")

    def _point(self, index: int) -> TrajectoryPoint:
        values = {}
        for f in sample_fields(type(self)):
            array = getattr(self, f.name)
            values[_point_name(f)] = None if array is None else array[index]
        return point_class(type(self))(
            t=self.t[index],
            channels={name: array[index] for name, array in self.channels.items()},
            **values,
        )

    def _take(self, key: Any) -> Any:
        if isinstance(key, slice):
            return self._map_samples(lambda array: array[key])
        index = np.asarray(key)
        if index.dtype == bool and index.shape != (len(self),):
            raise IndexError(f"mask has shape {index.shape}, trajectory has {len(self)} samples")
        if index.dtype != bool and index.dtype.kind not in "iu":
            raise TypeError(f"cannot index a trajectory with {index.dtype} values")
        return self._map_samples(lambda array: adopt(array[index]))

    def _map_samples(self: TrajectoryT, function: Any) -> TrajectoryT:
        """The same trajectory with `function` applied to every per-sample array (and `t`)."""
        changes = {
            f.name: function(getattr(self, f.name))
            for f in sample_fields(type(self))
            if getattr(self, f.name) is not None
        }
        channels = {name: function(array) for name, array in self.channels.items()}
        return dataclasses.replace(self, t=function(self.t), channels=channels, **changes)

    def to_points(self) -> tuple[TrajectoryPoint, ...]:
        """Array-of-structs view (what ROS puts on the wire)."""
        return tuple(self)

    @classmethod
    def from_points(cls: type[TrajectoryT], points: Sequence[TrajectoryPoint], **static: Any) -> TrajectoryT:
        """Stack points back into one trajectory; `static` sets the per-trajectory fields."""
        if not points:
            raise ValueError("from_points needs at least one point")
        fields = {}
        for f in sample_fields(cls):
            values = [getattr(p, _point_name(f), None) for p in points]
            present = [v is not None for v in values]
            if any(present) and not all(present):
                raise ValueError(f"{_point_name(f)} is set on some points but not all")
            if all(present):
                fields[f.name] = np.stack([np.asarray(v) for v in values])
        keys = set(points[0].channels)
        if any(set(p.channels) != keys for p in points):
            raise ValueError("every point must carry the same channels")
        channels = {k: np.stack([np.asarray(p.channels[k]) for p in points]) for k in keys}
        times = [p.t for p in points]
        return cls(t=times, channels=channels, **fields, **static)

    # ── time ─────────────────────────────────────────────────────────────────

    @property
    def start_time(self) -> float:
        self._require_samples()
        return float(self.t[0])

    @property
    def end_time(self) -> float:
        self._require_samples()
        return float(self.t[-1])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) > 1 else 0.0

    @property
    def absolute_times(self) -> np.ndarray:
        """`t` on the header's clock: ``header.stamp + t``."""
        return self.header.stamp + self.t

    def is_uniform(self, rtol: float = 1e-6) -> bool:
        """True when every sample step equals the mean step within `rtol`."""
        if len(self) < 3:
            return True
        steps = np.diff(self.t)
        return bool(np.allclose(steps, steps.mean(), rtol=rtol, atol=0.0))

    @property
    def dt(self) -> float:
        """The fixed sample step. Raises on a non-uniform trajectory rather than guess."""
        if len(self) < 2:
            raise ValueError("dt needs at least two samples")
        if not self.is_uniform():
            raise ValueError("samples are not uniformly spaced; use resample(dt=...) first")
        return float(np.diff(self.t).mean())

    def window(self: TrajectoryT, start: float, end: float) -> TrajectoryT:
        """Samples with ``start <= t <= end``, as a view."""
        lower = int(np.searchsorted(self.t, start, side="left"))
        upper = int(np.searchsorted(self.t, end, side="right"))
        return self[lower:upper]

    def time_shifted(self: TrajectoryT, seconds: float) -> TrajectoryT:
        """Every sample `seconds` later (earlier if negative)."""
        return self.replace(t=self.t + seconds)

    def rebased(self: TrajectoryT) -> TrajectoryT:
        """Same trajectory with ``t[0] == 0``; the offset moves into `header.stamp`."""
        if len(self) == 0:
            return self
        offset = float(self.t[0])
        return self.replace(t=self.t - offset, header=self.header.time_shifted(offset))

    def with_channel(self: TrajectoryT, name: str, values: Any) -> TrajectoryT:
        return self.replace(channels={**self.channels, name: values})

    def without_channel(self: TrajectoryT, name: str) -> TrajectoryT:
        if name not in self.channels:
            raise KeyError(name)
        return self.replace(channels={k: v for k, v in self.channels.items() if k != name})

    # ── interpolation ────────────────────────────────────────────────────────

    def at(self, time: float, *, extrapolate: Extrapolation = "raise") -> TrajectoryPoint:
        """The point at `time`, interpolated per field (`sample(interp=...)`)."""
        return self.resample([time], extrapolate=extrapolate)[0]

    def resample(
        self: TrajectoryT,
        times: Any = None,
        *,
        dt: float | None = None,
        extrapolate: Extrapolation = "raise",
    ) -> TrajectoryT:
        """The trajectory at new `times`, or on a fixed `dt` grid from `start_time`.

        A grid of `dt` ends at the last grid time that does not pass `end_time`.
        """
        if (times is None) == (dt is None):
            raise TypeError("resample takes exactly one of times or dt")
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("dt must be positive")
            count = int(math.floor((self.end_time - self.start_time) / dt + 1e-9)) + 1
            times = self.start_time + dt * np.arange(count)
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        lower, upper, alpha = self._bracket(times, extrapolate)
        changes = {
            f.name: _interpolate(getattr(self, f.name), f.metadata["sample"], lower, upper, alpha)
            for f in sample_fields(type(self))
            if getattr(self, f.name) is not None
        }
        channels = {
            name: _interpolate(array, "linear" if array.dtype.kind in "fc" else "hold",
                               lower, upper, alpha)
            for name, array in self.channels.items()
        }
        return dataclasses.replace(self, t=times, channels=channels, **changes)

    def _bracket(
        self, times: np.ndarray, extrapolate: Extrapolation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_samples()
        start, end = self.t[0], self.t[-1]
        if extrapolate == "raise":
            tolerance = 1e-9 * max(1.0, abs(float(end)))
            if times.min() < start - tolerance or times.max() > end + tolerance:
                raise ValueError(
                    f"time outside [{start:.6g}, {end:.6g}] s; "
                    "pass extrapolate='clamp' to hold the end values"
                )
        elif extrapolate != "clamp":
            raise ValueError(f"extrapolate must be 'raise' or 'clamp', got {extrapolate!r}")
        clamped = np.clip(times, start, end)
        if len(self) == 1:
            zeros = np.zeros(times.shape, dtype=np.intp)
            return zeros, zeros, np.zeros(times.shape)
        lower = np.clip(np.searchsorted(self.t, clamped, side="right") - 1, 0, len(self) - 2)
        upper = lower + 1
        alpha = np.clip((clamped - self.t[lower]) / (self.t[upper] - self.t[lower]), 0.0, 1.0)
        return lower, upper, alpha

    def _require_samples(self) -> None:
        if len(self) == 0:
            raise ValueError(f"empty {type(self).__name__}")

    def __repr__(self) -> str:
        if len(self) == 0:
            span = "empty"
        else:
            span = f"N={len(self)}, t=[{self.t[0]:.4g} … {self.t[-1]:.4g}] s"
        parts = [span]
        for f in dataclasses.fields(self):
            if f.name == "t" or not f.repr:
                continue
            value = getattr(self, f.name)
            if value is None or (isinstance(value, (Mapping, tuple)) and not value):
                continue
            if f.name == "header" and value == Header():
                continue
            parts.append(f"{f.name}={short_repr(value)}")
        return f"{type(self).__name__}({', '.join(parts)})"


def _interpolate(
    array: np.ndarray,
    kind: Interpolation,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    if kind == "hold":
        return adopt(array[np.where(alpha >= 1.0, upper, lower)])
    weight = alpha.reshape(-1, *([1] * (array.ndim - 1)))
    if kind == "linear":
        # (1 - w)·a + w·b, not a + w·(b - a): both endpoints come back bit-exact.
        return adopt((1.0 - weight) * array[lower] + weight * array[upper])
    if kind == "slerp":
        return adopt(quat_slerp(array[lower], array[upper], weight[..., 0]))
    raise ValueError(f"unknown interpolation {kind!r}")


def concatenate(trajectories: Sequence[TrajectoryT]) -> TrajectoryT:
    """Join trajectories of one type end to end, on the first one's clock.

    Times are aligned through each header's stamp, so pieces recorded against different start
    stamps join correctly; frames, channels and per-trajectory fields must match.
    """
    if not trajectories:
        raise ValueError("concatenate needs at least one trajectory")
    first = trajectories[0]
    kind = type(first)
    static = [
        f.name for f in dataclasses.fields(first)
        if f.name not in ("t", "channels", "header") and "sample" not in f.metadata
    ]
    for other in trajectories[1:]:
        if type(other) is not kind:
            raise TypeError(f"cannot concatenate {kind.__name__} with {type(other).__name__}")
        if other.header.frame_id != first.header.frame_id:
            raise ValueError("trajectories are in different frames")
        if set(other.channels) != set(first.channels):
            raise ValueError("trajectories carry different channels")
        for name in static:
            if getattr(other, name) != getattr(first, name):
                raise ValueError(f"trajectories differ in {name}")
    times = np.concatenate([tr.t + (tr.header.stamp - first.header.stamp) for tr in trajectories])
    changes = {}
    for f in sample_fields(kind):
        present = [getattr(tr, f.name) is not None for tr in trajectories]
        if any(present) and not all(present):
            raise ValueError(f"{f.name} is set on some trajectories but not all")
        if all(present):
            changes[f.name] = np.concatenate([getattr(tr, f.name) for tr in trajectories])
    channels = {
        name: np.concatenate([tr.channels[name] for tr in trajectories]) for name in first.channels
    }
    return dataclasses.replace(first, t=times, channels=channels, **changes)
