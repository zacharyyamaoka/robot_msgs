"""JointLimits — per-joint bounds a trajectory must respect, named like Ruckig's inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, optional_array, set_field

BOUNDS = ("min_position", "max_position", "max_velocity", "max_acceleration", "max_jerk", "max_effort")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointLimits(Msg):
    """One value per joint for each bound that is known; `None` means unbounded.

    WHY Ruckig's names (`max_velocity`, `max_acceleration`, `max_jerk`): the limits feed
    straight into the trajectory generator, and RobotDescription can carry the same value.
    `max_*` bounds are magnitudes (|velocity| <= max_velocity); `inf` is allowed.
    """

    joint_names: tuple[str, ...]
    min_position: Array | None = None
    max_position: Array | None = None
    max_velocity: Array | None = None
    max_acceleration: Array | None = None
    max_jerk: Array | None = None
    max_effort: Array | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError(f"JointLimits.joint_names must be unique, got {names}")
        set_field(self, "joint_names", names)
        for bound in BOUNDS:
            values = optional_array(getattr(self, bound))
            if values is None:
                continue
            if values.shape != (len(names),):
                raise ValueError(
                    f"JointLimits.{bound} has shape {values.shape}, expected ({len(names)},)"
                )
            if np.any(np.isnan(values)):
                raise ValueError(f"JointLimits.{bound} contains NaN; use inf for unbounded")
            if bound.startswith("max_") and bound != "max_position" and np.any(values < 0):
                raise ValueError(f"JointLimits.{bound} is a magnitude and must be >= 0")
            set_field(self, bound, values)
        if self.min_position is not None and self.max_position is not None:
            if np.any(self.min_position > self.max_position):
                raise ValueError("JointLimits.min_position exceeds max_position")

    def select(self, names: Sequence[str]) -> JointLimits:
        """These limits for `names`, in that order."""
        index = [self.joint_names.index(name) for name in names]
        bounds = {b: None if getattr(self, b) is None else getattr(self, b)[index] for b in BOUNDS}
        return JointLimits(joint_names=tuple(names), **bounds)
