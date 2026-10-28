"""Accel — linear (m/s²) and angular (rad/s²) acceleration. Mirrors geometry_msgs/Accel."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry.vector3 import Vector3, array_of, expect_type


@dataclass(frozen=True, eq=False, repr=False)
class Accel(Msg):
    """Linear and angular acceleration of a body, both in the frame the header names."""

    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)

    __array_ufunc__ = None  # see Vector3

    def __post_init__(self) -> None:
        expect_type("Accel.linear", self.linear, Vector3)
        expect_type("Accel.angular", self.angular, Vector3)

    def as_array(self) -> np.ndarray:
        """``(6,)`` = linear xyz then angular xyz."""
        return np.concatenate([self.linear.as_array(), self.angular.as_array()])

    @classmethod
    def from_array(cls, array: Any) -> Accel:
        values = array_of("Accel.from_array", array, 6)
        return cls(Vector3.from_array(values[:3]), Vector3.from_array(values[3:]))

    def __add__(self, other: object) -> Accel:
        if not isinstance(other, Accel):
            return NotImplemented
        return Accel(self.linear + other.linear, self.angular + other.angular)

    def __sub__(self, other: object) -> Accel:
        if not isinstance(other, Accel):
            return NotImplemented
        return Accel(self.linear - other.linear, self.angular - other.angular)

    def __mul__(self, scalar: object) -> Accel:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Accel(self.linear * scalar, self.angular * scalar)

    __rmul__ = __mul__
