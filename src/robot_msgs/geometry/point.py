"""Point — a position in a frame. Mirrors geometry_msgs/Point."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.vector3 import array_of, expect_type, finite_float


@dataclass(frozen=True, eq=False, repr=False)
class Point(Msg):
    """Metres. Arithmetic returns Point, as the ROS type has no separate displacement type."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    __array_ufunc__ = None  # see Vector3

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            set_field(self, name, finite_float(type(self).__name__, name, getattr(self, name)))

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, array: Any) -> Point:
        x, y, z = array_of("Point.from_array", array, 3).tolist()
        return cls(x, y, z)

    def __add__(self, other: object) -> Point:
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: object) -> Point:
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: object) -> Point:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Point(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: object) -> Point:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Point(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Point:
        return Point(-self.x, -self.y, -self.z)

    def norm(self) -> float:
        """Distance from the frame origin."""
        return math.hypot(self.x, self.y, self.z)

    def dot(self, other: Point) -> float:
        expect_type("Point.dot(other)", other, Point)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def distance_to(self, other: Point) -> float:
        expect_type("Point.distance_to(other)", other, Point)
        return math.hypot(self.x - other.x, self.y - other.y, self.z - other.z)
