"""ImuTrajectory — a log of `Imu` samples: orientation, angular rate and linear acceleration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array, freeze_new, set_field
from robot_msgs.geometry import Quaternion, Vector3
from robot_msgs.geometry.rotations import quat_normalize
from robot_msgs.sensor import Imu
from robot_msgs.std import Header
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ImuTrajectoryPoint(TrajectoryPoint):
    """One sample of an `ImuTrajectory`."""

    orientation: Array | None = None
    angular_velocity: Array | None = None
    linear_acceleration: Array | None = None

    def imu(self) -> Imu:
        """This sample as a sensor `Imu` (no covariances), stamped ``header.stamp + t``."""
        return Imu(
            orientation=Quaternion.from_array(self.orientation),
            angular_velocity=Vector3.from_array(self.angular_velocity),
            linear_acceleration=Vector3.from_array(self.linear_acceleration),
            header=Header(stamp=self.header.stamp + self.t, frame_id=self.header.frame_id),
        )


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ImuTrajectory(Trajectory):
    """`orientations (N, 4)` xyzw, `angular_velocities (N, 3)` rad/s, `linear_accelerations (N, 3)`
    m/s², all in `header.frame_id` (the IMU's frame), as sensor_msgs/Imu.

    Orientations are slerped between samples and normalised at construction. WHY no covariances:
    they describe the sensor, not the motion; a covariance that changes per sample can ride as a
    channel (``(N, 3, 3)``).
    """

    orientations: Array = sample(interp="slerp", point_name="orientation")
    angular_velocities: Array = sample(point_name="angular_velocity")
    linear_accelerations: Array = sample(point_name="linear_acceleration")

    point_type: ClassVar[type[TrajectoryPoint]] = ImuTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        widths = {"orientations": 4, "angular_velocities": 3, "linear_accelerations": 3}
        for name, width in widths.items():
            values = getattr(self, name)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(
                    f"ImuTrajectory.{name} has shape {values.shape}, "
                    f"expected ({len(self)}, {width})"
                )
        if len(self):
            norms = np.linalg.norm(self.orientations, axis=1)
            if np.any(norms < 1e-9):
                raise ValueError("ImuTrajectory.orientations contains a zero quaternion")
            if not np.allclose(norms, 1.0, atol=1e-12):
                set_field(self, "orientations", freeze_new(quat_normalize(self.orientations)))

    def imu_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> Imu:
        """The sample at `time` (orientation slerped, rates linear) as a sensor `Imu`."""
        return self.at(time, extrapolate=extrapolate).imu()

    @classmethod
    def from_imus(cls, imus: Sequence[Imu], **static: object) -> ImuTrajectory:
        """A recorded stream from a log of Imu messages: stamps become `t`, the first one the
        header's stamp. Covariances are dropped (see the class)."""
        if not imus:
            raise ValueError("from_imus needs at least one Imu")
        stamps = np.array([imu.header.stamp for imu in imus], dtype=np.float64)
        header = Header(stamp=float(stamps[0]), frame_id=imus[0].header.frame_id)
        return cls(
            t=stamps - stamps[0],
            orientations=np.array([imu.orientation.as_array() for imu in imus]),
            angular_velocities=np.array([imu.angular_velocity.as_array() for imu in imus]),
            linear_accelerations=np.array([imu.linear_acceleration.as_array() for imu in imus]),
            header=static.pop("header", header),  # type: ignore[arg-type]
            **static,  # type: ignore[arg-type]
        )
