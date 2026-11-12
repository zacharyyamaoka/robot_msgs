"""CartesianLimits — how fast the tool may move in Cartesian space."""

from __future__ import annotations

from dataclasses import dataclass

from robot_msgs._core import Msg, set_field
from robot_msgs.manipulation._common import positive

FIELDS = (
    "max_linear_velocity",
    "max_linear_acceleration",
    "max_linear_jerk",
    "max_angular_velocity",
    "max_angular_acceleration",
    "max_angular_jerk",
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CartesianLimits(Msg):
    """Magnitude bounds on the tool frame's motion: linear in m/s, m/s², m/s³; angular in
    rad/s, rad/s², rad/s³. `inf` means unbounded.

    WHY these defaults: bam_traj_gen's `CartesianLimits` (2026-11-11), so moving it onto this
    type changes no plan.
    """

    max_linear_velocity: float = 1.0
    max_linear_acceleration: float = 1.0
    max_linear_jerk: float = 10.0
    max_angular_velocity: float = 2.0
    max_angular_acceleration: float = 2.0
    max_angular_jerk: float = 20.0

    def __post_init__(self) -> None:
        for name in FIELDS:
            set_field(
                self, name, positive("CartesianLimits", name, getattr(self, name), allow_inf=True)
            )
