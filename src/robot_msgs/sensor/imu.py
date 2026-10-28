"""Imu — orientation, angular rate and linear acceleration. Mirrors sensor_msgs/Imu."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Array, Msg, optional_array, set_field
from robot_msgs.geometry import Quaternion, Vector3
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Imu(Msg):
    """Angular velocity in rad/s, linear acceleration in m/s², both in `header.frame_id`.

    Each covariance is a (3, 3) matrix (ROS flattens to 9), or `None` when unknown.
    """

    orientation: Quaternion
    angular_velocity: Vector3
    linear_acceleration: Vector3
    orientation_covariance: Array | None = None
    angular_velocity_covariance: Array | None = None
    linear_acceleration_covariance: Array | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        for name in (
            "orientation_covariance",
            "angular_velocity_covariance",
            "linear_acceleration_covariance",
        ):
            covariance = optional_array(getattr(self, name))
            if covariance is not None and covariance.shape != (3, 3):
                raise ValueError(f"Imu.{name} must have shape (3, 3), got {covariance.shape}")
            set_field(self, name, covariance)
