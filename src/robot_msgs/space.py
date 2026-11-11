"""Spaces: a message holds a value OR a range, and you can sample it or check against it.

    from robot_msgs.space import Value, Space
    pick = Pose(position=Point(x=Value(0.5, "m", low=0.4, high=0.6), y=0.0, z=0.3))
    Space(pick).sample(rng)          # -> a Pose, x drawn in [0.4, 0.6], marked basis=SAMPLED
    Space(pick).contains(measured)   # -> the acceptable range in a test
    Space(traj, positions=(low, high)).sample(rng)   # array fields: gym-Box bounds

WHY a re-export (Zach, Nov 10 2026): "the pydataclass syntax can technically apply to any
domain! robot_msgs is just a particular domain." `Value`, `Basis`, the distributions, `meta`,
`plain` and the scalar `Space` live in the domain-free, stdlib-only `dataclass_values` package,
which this vocabulary and py_dataclass_viewer both build on; every name that was importable from
`robot_msgs.space` still is. Only the numpy array bounds below are robot_msgs's own, as a subclass
of `dataclass_values.Space` through its two hooks.

WHY the same class and a derived space (Zach, Oct 5 2026, "do not re-litigate"): a second
`PoseSpace` class "drifts" from the message it shadows. So a field carries its own range as a
`Value`, which IS the builtin float/int/str/tuple ("I definitely don't want to have to do
width_m.value"), and `Space(msg)` is a view over the instance whose `sample()` returns the ordinary
class. `plain()` drops the notes at a runtime boundary (ROS, JSON).
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import numpy as np
from dataclass_values import (
    Basis,
    Choice,
    FloatValue,
    IntValue,
    LogUniform,
    Normal,
    StrValue,
    TupleValue,
    Uniform,
    Value,
    meta,
    plain,
)
from dataclass_values import Space as ScalarSpace
from dataclass_values.values import META, Distribution

__all__ = [
    "META",
    "Basis",
    "Choice",
    "Distribution",
    "FloatValue",
    "IntValue",
    "LogUniform",
    "Normal",
    "Space",
    "StrValue",
    "TupleValue",
    "Uniform",
    "Value",
    "meta",
    "plain",
]


class Space(ScalarSpace):
    """A message seen as a space (gymnasium-style), derived from the instance itself.

    Scalars: every `Value` with a dist, choices or [low, high] can be drawn and checked (the
    `dataclass_values.Space` core). Arrays: `Space(traj, positions=(low, high))` bounds a top-level
    array field the gym `Box` way; low/high broadcast against the field (per joint, per sample, or
    one scalar). `sample()` returns the same class as the instance; nothing else is discarded.
    """

    def __init__(self, instance: Any, **array_bounds: tuple[Any, Any]) -> None:
        super().__init__(instance)
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

    def extra_draw(self, drawn, rng: random.Random):
        if self.array_bounds:
            gen = np.random.default_rng(rng.getrandbits(64))
            drawn = replace(
                drawn,
                **{name: gen.uniform(low, high) for name, (low, high) in self.array_bounds.items()},
            )
        return drawn

    def extra_violations(self, other: Any) -> list[str]:
        found: list[str] = []
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
