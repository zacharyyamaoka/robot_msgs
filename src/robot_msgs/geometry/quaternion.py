"""Quaternion — an orientation as ``(x, y, z, w)``. Mirrors geometry_msgs/Quaternion.

All maths goes through the batch kernel in `rotations`; this file only adapts it to one value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import rotations as rot
from robot_msgs.geometry.point import Point
from robot_msgs.geometry.vector3 import (
    MIN_NORM,
    Vector3,
    array_of,
    expect_type,
    finite_float,
)

V = TypeVar("V", Vector3, Point, np.ndarray)


def map_vectors(value: V, func: Callable[[np.ndarray], np.ndarray]) -> V:
    """Apply `func` to a Vector3, Point or ``(..., 3)`` array and give back the same kind."""
    if isinstance(value, Vector3):
        return Vector3.from_array(func(value.as_array()))
    if isinstance(value, Point):
        return Point.from_array(func(value.as_array()))
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"expected a Vector3, Point or (..., 3) array, got shape {array.shape}")
    return func(array)


def _axis_array(label: str, value: Vector3 | Point | Any) -> np.ndarray:
    if isinstance(value, (Vector3, Point)):
        return value.as_array()
    return array_of(label, value, 3)


@dataclass(frozen=True, eq=False, repr=False)
class Quaternion(Msg):
    """Unit quaternions are rotations; the fields are stored exactly as given.

    WHY not normalised at construction: a stored property must read back as authored. Every
    operation that needs a rotation (`rotate`, `to_matrix`, `multiply`, ...) normalises first;
    `normalized()` and `is_unit()` are there when you want the unit value yourself.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "w"):
            set_field(self, name, finite_float(type(self).__name__, name, getattr(self, name)))
        if math.hypot(self.x, self.y, self.z, self.w) < MIN_NORM:
            raise ValueError("Quaternion must be non-zero, got a zero-length (x, y, z, w)")

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
        yield self.w

    # ── construction and conversion ──────────────────────────────────────────

    @classmethod
    def identity(cls) -> Quaternion:
        return cls()

    def as_array(self) -> np.ndarray:
        """``(4,)`` in x, y, z, w order."""
        return np.array([self.x, self.y, self.z, self.w])

    @classmethod
    def from_array(cls, array: Any) -> Quaternion:
        x, y, z, w = array_of("Quaternion.from_array", array, 4).tolist()
        return cls(x, y, z, w)

    @classmethod
    def from_wxyz(cls, array: Any) -> Quaternion:
        """From MuJoCo / viser order."""
        return cls.from_array(rot.wxyz_to_xyzw(array_of("Quaternion.from_wxyz", array, 4)))

    def to_wxyz(self) -> np.ndarray:
        return rot.xyzw_to_wxyz(self.as_array())

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """ROS RPY: fixed axes X, then Y, then Z (``R = Rz·Ry·Rx``)."""
        return cls.from_array(rot.quat_from_euler(roll, pitch, yaw))

    def to_euler(self) -> tuple[float, float, float]:
        """``(roll, pitch, yaw)``; pitch in [-π/2, π/2]."""
        roll, pitch, yaw = rot.quat_to_euler(self.as_array()).tolist()
        return roll, pitch, yaw

    @classmethod
    def from_matrix(cls, matrix: Any) -> Quaternion:
        """From a ``(3, 3)`` rotation matrix."""
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (3, 3):
            raise ValueError(f"Quaternion.from_matrix needs shape (3, 3), got {array.shape}")
        return cls.from_array(rot.quat_from_matrix(array))

    def to_matrix(self) -> np.ndarray:
        return rot.quat_to_matrix(self.as_array())

    @classmethod
    def from_axis_angle(cls, axis: Any, angle: float) -> Quaternion:
        """`axis` need not be unit length, but must be non-zero. Radians."""
        axis_array = _axis_array("Quaternion.from_axis_angle axis", axis)
        return cls.from_array(rot.quat_from_axis_angle(axis_array, np.asarray(angle)))

    @classmethod
    def from_rotvec(cls, rotvec: Any) -> Quaternion:
        """From axis × angle (radians)."""
        return cls.from_array(rot.quat_from_rotvec(_axis_array("Quaternion.from_rotvec", rotvec)))

    def to_rotvec(self) -> np.ndarray:
        """Axis × angle for the shorter rotation, angle in [0, π]."""
        return rot.quat_to_rotvec(self.as_array())

    # ── algebra ──────────────────────────────────────────────────────────────

    def _unit_array(self) -> np.ndarray:
        return rot.quat_normalize(self.as_array())

    def normalized(self) -> Quaternion:
        return Quaternion.from_array(self._unit_array())

    def is_unit(self, tol: float = 1e-9) -> bool:
        return abs(math.hypot(self.x, self.y, self.z, self.w) - 1.0) <= tol

    def multiply(self, other: Quaternion) -> Quaternion:
        """Hamilton product of the unit rotations: apply `other` first, then `self`."""
        expect_type("Quaternion.multiply(other)", other, Quaternion)
        return Quaternion.from_array(rot.quat_multiply(self._unit_array(), other._unit_array()))

    def __mul__(self, other: object) -> Quaternion:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.multiply(other)

    def conjugate(self) -> Quaternion:
        """Negated vector part, fields otherwise as stored."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> Quaternion:
        """The unit quaternion of the opposite rotation."""
        return Quaternion.from_array(rot.quat_conjugate(self._unit_array()))

    def rotate(self, vector: V) -> V:
        """Rotate a Vector3, Point or ``(..., 3)`` array; returns the same kind."""
        unit = self._unit_array()
        return map_vectors(vector, lambda array: rot.quat_rotate(unit, array))

    def slerp(self, other: Quaternion, alpha: float) -> Quaternion:
        """Along the shorter arc; 0 gives `self`, 1 gives `other`, outside [0, 1] extrapolates."""
        expect_type("Quaternion.slerp(other)", other, Quaternion)
        alpha = finite_float("Quaternion.slerp", "alpha", alpha)
        return Quaternion.from_array(
            rot.quat_slerp(self.as_array(), other.as_array(), np.asarray(alpha))
        )

    def angle_to(self, other: Quaternion) -> float:
        """Radians in [0, π] of the rotation that takes `self` to `other`."""
        expect_type("Quaternion.angle_to(other)", other, Quaternion)
        return float(rot.quat_angle(self.as_array(), other.as_array()))
