"""Twist — linear (m/s) and angular (rad/s) velocity. Mirrors geometry_msgs/Twist."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry.vector3 import Vector3, array_of, expect_type


@dataclass(frozen=True, eq=False, repr=False)
class Twist(Msg):
    """Linear and angular velocity of a body, both in the frame the header names."""

    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)

    __array_ufunc__ = None  # see Vector3

    def __post_init__(self) -> None:
        expect_type("Twist.linear", self.linear, Vector3)
        expect_type("Twist.angular", self.angular, Vector3)

    def as_array(self) -> np.ndarray:
        """``(6,)`` = linear xyz then angular xyz."""
        return np.concatenate([self.linear.as_array(), self.angular.as_array()])

    @classmethod
    def from_array(cls, array: Any) -> Twist:
        values = array_of("Twist.from_array", array, 6)
        return cls(Vector3.from_array(values[:3]), Vector3.from_array(values[3:]))

    def __add__(self, other: object) -> Twist:
        if not isinstance(other, Twist):
            return NotImplemented
        return Twist(self.linear + other.linear, self.angular + other.angular)

    def __sub__(self, other: object) -> Twist:
        if not isinstance(other, Twist):
            return NotImplemented
        return Twist(self.linear - other.linear, self.angular - other.angular)

    def __mul__(self, scalar: object) -> Twist:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Twist(self.linear * scalar, self.angular * scalar)

    __rmul__ = __mul__
