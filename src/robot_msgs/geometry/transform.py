"""Transform — where a child frame sits in its parent frame. Mirrors geometry_msgs/Transform.

`Transform` carries the rigid-transform maths; `Pose` is the same maths on a Point + Quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry import rotations as rot
from robot_msgs.geometry.point import Point
from robot_msgs.geometry.quaternion import Quaternion, V, map_vectors
from robot_msgs.geometry.vector3 import Vector3, array_of, expect_type

if TYPE_CHECKING:
    from robot_msgs.geometry.pose import Pose


@dataclass(frozen=True, eq=False, repr=False)
class Transform(Msg):
    """``T_parent_child``: maps points expressed in the child frame into the parent frame."""

    translation: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    def __post_init__(self) -> None:
        expect_type("Transform.translation", self.translation, Vector3)
        expect_type("Transform.rotation", self.rotation, Quaternion)

    @classmethod
    def identity(cls) -> Transform:
        return cls()

    @classmethod
    def from_xyz_rpy(
        cls, x: float, y: float, z: float, roll: float, pitch: float, yaw: float
    ) -> Transform:
        return cls(Vector3(x, y, z), Quaternion.from_euler(roll, pitch, yaw))

    @classmethod
    def from_matrix(cls, matrix: Any) -> Transform:
        """From a ``(4, 4)`` homogeneous matrix whose last row is ``[0, 0, 0, 1]``."""
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (4, 4):
            raise ValueError(f"Transform.from_matrix needs shape (4, 4), got {array.shape}")
        if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"Transform.from_matrix last row must be [0, 0, 0, 1], got {array[3]}")
        translation, quaternion = rot.matrix_to_pose(array)
        return cls(Vector3.from_array(translation), Quaternion.from_array(quaternion))

    def to_matrix(self) -> np.ndarray:
        return rot.pose_to_matrix(self.translation.as_array(), self.rotation.as_array())

    def as_array(self) -> np.ndarray:
        """``(7,)`` = x y z qx qy qz qw."""
        return np.concatenate([self.translation.as_array(), self.rotation.as_array()])

    @classmethod
    def from_array(cls, array: Any) -> Transform:
        values = array_of("Transform.from_array", array, 7)
        return cls(Vector3.from_array(values[:3]), Quaternion.from_array(values[3:]))

    def compose(self, other: Transform) -> Transform:
        """``T_world_tool = T_world_base.compose(T_base_tool)``: apply `other` in our frame."""
        expect_type("Transform.compose(other)", other, Transform)
        return Transform(
            self.translation + self.rotation.rotate(other.translation),
            self.rotation * other.rotation,
        )

    def __matmul__(self, other: object) -> Transform:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.compose(other)

    def inverse(self) -> Transform:
        rotation = self.rotation.inverse()
        return Transform(-rotation.rotate(self.translation), rotation)

    def transform_point(self, point: V) -> V:
        """Child-frame Point, Vector3 or ``(..., 3)`` array to the parent frame (same kind).

        A Vector3 is treated as a position here, so it is translated too.
        """
        translation = self.translation.as_array()
        unit = self.rotation._unit_array()
        return map_vectors(point, lambda array: rot.quat_rotate(unit, array) + translation)

    def interpolate(self, other: Transform, alpha: float) -> Transform:
        """Lerp the translation, slerp the rotation; 0 gives `self`, 1 gives `other`."""
        expect_type("Transform.interpolate(other)", other, Transform)
        return Transform(
            self.translation + (other.translation - self.translation) * alpha,
            self.rotation.slerp(other.rotation, alpha),
        )

    def shifted(
        self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, *, local: bool = False
    ) -> Transform:
        """Moved by ``(dx, dy, dz)`` along the parent frame's axes, or along its own with
        ``local=True``. Back a grasp off along the tool: ``grasp.shifted(dz=-0.1, local=True)``.
        """
        offset = np.array([dx, dy, dz], dtype=np.float64)
        if local:
            offset = rot.quat_rotate(self.rotation._unit_array(), offset)
        return Transform(Vector3.from_array(self.translation.as_array() + offset), self.rotation)

    def rotated(
        self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, *, local: bool = False
    ) -> Transform:
        """Turned in place (the origin stays put) by ROS RPY about the parent frame's axes, or
        about its own with ``local=True``."""
        turn = Quaternion.from_euler(roll, pitch, yaw)
        rotation = self.rotation.multiply(turn) if local else turn.multiply(self.rotation)
        return Transform(self.translation, rotation)

    def as_pose(self) -> Pose:
        # WHY local import: pose.py imports Transform at module level (Pose delegates its maths).
        from robot_msgs.geometry.pose import Pose

        return Pose(Point(*self.translation), self.rotation)
