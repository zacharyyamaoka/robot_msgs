"""Wrench — force and torque on a body. Mirrors geometry_msgs/Wrench."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry.vector3 import Vector3, array_of, expect_type


@dataclass(frozen=True, eq=False, repr=False)
class Wrench(Msg):
    """Force (N) and torque (N·m) applied to a body, both in the frame the header names."""

    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)

    __array_ufunc__ = None  # see Vector3

    def __post_init__(self) -> None:
        expect_type("Wrench.force", self.force, Vector3)
        expect_type("Wrench.torque", self.torque, Vector3)

    def as_array(self) -> np.ndarray:
        """``(6,)`` = force xyz then torque xyz."""
        return np.concatenate([self.force.as_array(), self.torque.as_array()])

    @classmethod
    def from_array(cls, array: Any) -> Wrench:
        values = array_of("Wrench.from_array", array, 6)
        return cls(Vector3.from_array(values[:3]), Vector3.from_array(values[3:]))

    def __add__(self, other: object) -> Wrench:
        if not isinstance(other, Wrench):
            return NotImplemented
        return Wrench(self.force + other.force, self.torque + other.torque)

    def __sub__(self, other: object) -> Wrench:
        if not isinstance(other, Wrench):
            return NotImplemented
        return Wrench(self.force - other.force, self.torque - other.torque)

    def __mul__(self, scalar: object) -> Wrench:
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return Wrench(self.force * scalar, self.torque * scalar)

    __rmul__ = __mul__
