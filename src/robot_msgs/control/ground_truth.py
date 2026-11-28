"""GroundTruth — what a simulator knows about its joints and a real arm never can.

A simulated `ServoState` carries one in `ServoState.truth`; a real one carries `None`. WHY on the state and not
beside it (Zach, Daily Note 2026-11-28): "I think in metadata here if you're in simulator you should include the
ground truth as well... That way if you want to switch onto real it's kinda like the 1 line switch". The same loop
reads `state` on sim and real; only a test or a metric that needs the truth looks at `state.truth`, and on real it
finds `None` instead of a different API.

Always joint-named and in the joint frame (rad, rad/s, N·m; m, m/s, N for a prismatic joint), whatever names the
measured sample uses around it: the plant's truth has no servo frame, no encoder offset and no homed zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from collections.abc import Mapping
from typing import Any

from robot_msgs._core import Array, FrozenDict, Msg, frozen_array, optional_array, set_field
from robot_msgs.std import Header

_OPTIONAL = ("velocity", "effort", "stop_effort")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class GroundTruth(Msg):
    """The simulator's truth for joints `name[i]` at `header.stamp` (s), the instant its sample was taken.

    - `position`, `velocity`: the true joint angle and rate (no encoder offset, no sensor model).
    - `effort`: the torque the actuator applied to the joint (after its own cap).
    - `stop_effort`: the torque a hard stop pushes back with (0 = not touching one); `None` when the scene has none.
    - `source`: which oracle said so (``"mujoco"``): the sensor that produces truth, present on some backends only.
    - `meta`: everything else that oracle knows and not every oracle can give (contacts, object poses, solver stats),
      a flat ``str -> value`` mapping, untyped on purpose.

    WHY the state's own field names plus `meta` (Zach, Daily Note 2026-11-28): "I think it would be nice if the
    structure of truth is similar to that of the state... truth could almost just match state exact but just with
    the real values... if we still want more flexibility you can have like state.truth.meta". The parts of the truth
    every oracle can give are typed (`name`/`position`/`velocity`/`effort` read exactly like `ServoState`'s); the
    parts only some can give go in `meta`. It stays its own type so a function that asks for the truth cannot be
    handed a measurement.
    """

    name: tuple[str, ...]
    position: Array
    velocity: Array | None = None
    effort: Array | None = None
    stop_effort: Array | None = None
    source: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.name)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"GroundTruth.name has duplicate names: {duplicates}")
        set_field(self, "name", names)
        set_field(self, "position", frozen_array(self.position))
        for field_name in _OPTIONAL:
            set_field(self, field_name, optional_array(getattr(self, field_name)))
        set_field(self, "source", str(self.source))
        set_field(self, "meta", FrozenDict((str(key), value) for key, value in dict(self.meta).items()))
        for field_name in ("position", *_OPTIONAL):
            values = getattr(self, field_name)
            if values is None:
                continue
            if values.ndim != 1:
                raise ValueError(f"GroundTruth.{field_name} must be 1-D, got shape {values.shape}")
            if len(values) != len(names):
                raise ValueError(
                    f"GroundTruth.{field_name} has {len(values)} values, name has {len(names)}"
                )

    def __len__(self) -> int:
        return len(self.name)

    def of(self, joint: str) -> float:
        """The true position of `joint`."""
        return float(self.position[self.name.index(joint)])
