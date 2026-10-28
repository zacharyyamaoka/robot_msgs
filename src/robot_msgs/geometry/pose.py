"""Pose — a position and orientation in a frame. Mirrors geometry_msgs/Pose."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry.point import Point
from robot_msgs.geometry.quaternion import Quaternion, V
from robot_msgs.geometry.transform import Transform
from robot_msgs.geometry.vector3 import Vector3, expect_type


@dataclass(frozen=True, eq=False, repr=False)
class Pose(Msg):
    """Same maths as `Transform` (every method converts and delegates), different intent: a
    Pose says where something *is*, a Transform says how to move between frames."""

    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)

    def __post_init__(self) -> None:
        expect_type("Pose.position", self.position, Point)
        expect_type("Pose.orientation", self.orientation, Quaternion)

    @classmethod
    def identity(cls) -> Pose:
        return cls()

    @classmethod
    def from_xyz_rpy(
        cls, x: float, y: float, z: float, roll: float, pitch: float, yaw: float
    ) -> Pose:
        return Transform.from_xyz_rpy(x, y, z, roll, pitch, yaw).as_pose()

    @classmethod
    def from_matrix(cls, matrix: Any) -> Pose:
        return Transform.from_matrix(matrix).as_pose()

    def to_matrix(self) -> np.ndarray:
        return self.as_transform().to_matrix()

    def as_array(self) -> np.ndarray:
        """``(7,)`` = x y z qx qy qz qw."""
        return np.concatenate([self.position.as_array(), self.orientation.as_array()])

    @classmethod
    def from_array(cls, array: Any) -> Pose:
        return Transform.from_array(array).as_pose()

    def compose(self, other: Pose) -> Pose:
        """``T_world_tool = T_world_base.compose(T_base_tool)``: apply `other` in our frame."""
        expect_type("Pose.compose(other)", other, Pose)
        return (self.as_transform() @ other.as_transform()).as_pose()

    def __matmul__(self, other: object) -> Pose:
        if not isinstance(other, Pose):
            return NotImplemented
        return self.compose(other)

    def inverse(self) -> Pose:
        return self.as_transform().inverse().as_pose()

    def transform_point(self, point: V) -> V:
        """Child-frame Point, Vector3 or ``(..., 3)`` array to the parent frame (same kind)."""
        return self.as_transform().transform_point(point)

    def interpolate(self, other: Pose, alpha: float) -> Pose:
        """Lerp the position, slerp the orientation; 0 gives `self`, 1 gives `other`."""
        expect_type("Pose.interpolate(other)", other, Pose)
        return self.as_transform().interpolate(other.as_transform(), alpha).as_pose()

    def shifted(
        self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, *, local: bool = False
    ) -> Pose:
        """Moved by ``(dx, dy, dz)`` along the parent axes, or the pose's own with ``local=True``:
        ``pregrasp = grasp.shifted(dz=-0.10, local=True)``."""
        return self.as_transform().shifted(dx, dy, dz, local=local).as_pose()

    def rotated(
        self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, *, local: bool = False
    ) -> Pose:
        """Turned in place by ROS RPY about the parent axes, or its own with ``local=True``."""
        return self.as_transform().rotated(roll, pitch, yaw, local=local).as_pose()

    def as_transform(self) -> Transform:
        return Transform(Vector3(*self.position), self.orientation)
