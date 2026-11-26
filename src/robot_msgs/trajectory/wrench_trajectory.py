"""WrenchTrajectory — force and torque over time (an F/T sensor's log, or a commanded wrench)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array
from robot_msgs.geometry import Vector3, Wrench
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class WrenchTrajectoryPoint(TrajectoryPoint):
    """One sample of a `WrenchTrajectory`: `force` (N) and `torque` (N·m)."""

    force: Array | None = None
    torque: Array | None = None

    @property
    def wrench(self) -> Wrench:
        return Wrench(Vector3.from_array(self.force), Vector3.from_array(self.torque))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class WrenchTrajectory(Trajectory):
    """`forces (N, 3)` in N and `torques (N, 3)` in N·m, in `header.frame_id`.

    geometry_msgs/WrenchStamped is one sample of it. WHY `from_wrenches(t, wrenches)` and no
    `from_wrench_stampeds`: robot_msgs has no WrenchStamped; a `Wrench` carries no time, so the
    times come alongside, as `PoseTrajectory.from_poses` takes them.
    """

    forces: Array = sample(point_name="force")
    torques: Array = sample(point_name="torque")

    point_type: ClassVar[type[TrajectoryPoint]] = WrenchTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("forces", "torques"):
            values = getattr(self, name)
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(
                    f"WrenchTrajectory.{name} has shape {values.shape}, expected ({len(self)}, 3)"
                )

    def wrenches(self) -> tuple[Wrench, ...]:
        return tuple(point.wrench for point in self)

    def wrench_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> Wrench:
        return self.at(time, extrapolate=extrapolate).wrench

    @classmethod
    def from_wrenches(
        cls, t: Sequence[float], wrenches: Sequence[Wrench], **static: object
    ) -> WrenchTrajectory:
        """From sample times (s after `header.stamp`) and one `Wrench` per time."""
        arrays = np.array([wrench.as_array() for wrench in wrenches]).reshape(-1, 6)
        return cls(t=t, forces=arrays[:, :3], torques=arrays[:, 3:], **static)  # type: ignore[arg-type]
