"""Vector3 — a free 3-D vector (direction, velocity, force). Mirrors geometry_msgs/Vector3.

Also home of the small validators every geometry type shares.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field

MIN_NORM = 1e-12


def finite_float(owner: str, name: str, value: Any) -> float:
    """`value` as a Python float, or a ValueError naming `owner.name` if it is NaN or infinite."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{owner}.{name} must be finite, got {number}")
    return number


def expect_type(label: str, value: Any, expected: type) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}, got {type(value).__name__}")


def array_of(label: str, value: Any, length: int) -> np.ndarray:
    """`value` as a float64 array of exactly shape ``(length,)``."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{label} needs shape ({length},), got {array.shape}")
    return array


@dataclass(frozen=True, eq=False, repr=False)
class Vector3(Msg):
    """Metres, m/s, newtons... whatever the owning field says. Arithmetic returns Vector3."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # WHY: make numpy defer to our operators, so ``np.float64(2) * vector`` is a Vector3 and
    # not an object array (we are iterable, which numpy would otherwise try to unpack).
    __array_ufunc__ = None

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
    def from_array(cls, array: Any) -> Vector3:
        x, y, z = array_of("Vector3.from_array", array, 3).tolist()
        return cls(x, y, z)

    def __add__(self, other: object) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: object) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: object) -> Vector3:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: object) -> Vector3:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.hypot(self.x, self.y, self.z)

    def dot(self, other: Vector3) -> float:
        expect_type("Vector3.dot(other)", other, Vector3)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        expect_type("Vector3.cross(other)", other, Vector3)
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def normalized(self) -> Vector3:
        length = self.norm()
        if length < MIN_NORM:
            raise ValueError("cannot normalise a zero-length Vector3")
        return self / length
