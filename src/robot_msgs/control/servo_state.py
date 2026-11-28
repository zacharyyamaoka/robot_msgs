"""ServoState — one sample of a set of named servos, stamped when it was taken.

A servo here is a motor with its own inner position/velocity loop on its board (moteus, Damiao,
RobStride, CubeMars, Dynamixel). WHY "Servo" and not "Actuator" (Zach, 2026-11-11): at a glance the
name says what the type is; a pneumatic cylinder or a suction cup is an actuator too, but it takes
a different command, so it gets its own type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.control.ground_truth import GroundTruth
from robot_msgs.sensor import JointState
from robot_msgs.std import Header

_OPTIONAL = ("velocity", "effort", "temperature", "voltage", "fault")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoState(Msg):
    """`position[i]` (rad or m), `velocity[i]`, `effort[i]` (N·m or N) belong to servo `name[i]`.

    `header.stamp` is when the sample was taken (s): every timing metric (sensor-to-command latency,
    period jitter, stalls) is measured from it, so a driver must set it. `seq` counts samples from
    one driver, so a dropped or repeated sample shows as a gap or a repeat.
    `temperature` is °C (as ROS sensor_msgs/Temperature), `voltage` V, `fault` the board's integer
    fault code per servo with 0 = no fault. Unmeasured fields are `None`.

    `truth` (additive, 2026-11-28): a simulator's `GroundTruth` for the same instant, joint-named and in the joint
    frame; `None` on real hardware. Controllers never read it (they see what a real arm gives them); a test or
    a metric does, and on real it finds `None` rather than a different API (Zach, 2026-11-28: "if you're in
    simulator you should include the ground truth as well... switch onto real it's kinda like the 1 line switch").
    """

    name: tuple[str, ...]
    position: Array
    velocity: Array | None = None
    effort: Array | None = None
    temperature: Array | None = None
    voltage: Array | None = None
    fault: Array | None = None
    seq: int = 0
    header: Header = field(default_factory=Header)
    truth: GroundTruth | None = None

    def __post_init__(self) -> None:
        if self.truth is not None and not isinstance(self.truth, GroundTruth):
            raise TypeError(f"ServoState.truth must be a GroundTruth or None, got {type(self.truth).__name__}")
        names = tuple(str(n) for n in self.name)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"ServoState.name has duplicate names: {duplicates}")
        set_field(self, "name", names)
        set_field(self, "position", frozen_array(self.position))
        for field_name in _OPTIONAL:
            dtype = np.int64 if field_name == "fault" else np.float64
            set_field(self, field_name, optional_array(getattr(self, field_name), dtype=dtype))
        set_field(self, "seq", int(self.seq))
        for field_name in ("position", *_OPTIONAL):
            values = getattr(self, field_name)
            if values is None:
                continue
            if values.ndim != 1:
                raise ValueError(f"ServoState.{field_name} must be 1-D, got shape {values.shape}")
            if len(values) != len(names):
                raise ValueError(
                    f"ServoState.{field_name} has {len(values)} values, name has {len(names)}"
                )

    def __len__(self) -> int:
        return len(self.name)

    @property
    def faulted(self) -> bool:
        """True when any servo reports a non-zero fault code."""
        return self.fault is not None and bool(np.any(self.fault != 0))

    def joint_state(self) -> JointState:
        """The kinematic part, as the rest of the stack reads joints."""
        return JointState(
            name=self.name,
            position=self.position,
            velocity=self.velocity,
            effort=self.effort,
            header=self.header,
        )

    def compare(self, measured: ServoState) -> Any:
        """`measured` against this one as the reference → a `robot_msgs.compare.Comparison`.

        Values only (stamps ignored), as one-sample streams; see `robot_msgs.compare.compare`.
        """
        from robot_msgs.compare import compare

        return compare(self, measured)
