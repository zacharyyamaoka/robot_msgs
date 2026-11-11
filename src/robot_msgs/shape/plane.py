"""Plane — ax + by + cz + d = 0. Mirrors shape_msgs/Plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field
from robot_msgs.geometry import Vector3
from robot_msgs.shape._common import points_of, vector_of


@dataclass(frozen=True, eq=False, repr=False)
class Plane(Msg):
    """`coef` = ``(a, b, c, d)``: the points with ``a·x + b·y + c·z + d = 0``, in the frame the
    owner names. ``(a, b, c)`` is the normal and need not be a unit vector, as in ROS."""

    coef: Array

    def __post_init__(self) -> None:
        coef = frozen_array(self.coef)
        if coef.shape != (4,):
            raise ValueError(f"Plane.coef needs shape (4,) = (a, b, c, d), got {coef.shape}")
        if not np.all(np.isfinite(coef)):
            raise ValueError(f"Plane.coef must be finite, got {coef.tolist()}")
        if not np.any(coef[:3]):
            raise ValueError(f"Plane.coef needs a non-zero normal (a, b, c), got {coef.tolist()}")
        set_field(self, "coef", coef)

    @property
    def a(self) -> float:
        return float(self.coef[0])

    @property
    def b(self) -> float:
        return float(self.coef[1])

    @property
    def c(self) -> float:
        return float(self.coef[2])

    @property
    def d(self) -> float:
        return float(self.coef[3])

    @property
    def normal(self) -> Vector3:
        """The unit normal ``(a, b, c) / |(a, b, c)|``."""
        return Vector3.from_array(self.coef[:3] / np.linalg.norm(self.coef[:3]))

    @classmethod
    def from_point_normal(cls, point: Any, normal: Any) -> Plane:
        """The plane through `point` with the given `normal` (any length); `coef` comes out with a
        unit normal, so ``d`` is minus the distance from the origin along it."""
        origin = vector_of("Plane.from_point_normal(point)", point)
        direction = vector_of("Plane.from_point_normal(normal)", normal)
        length = float(np.linalg.norm(direction))
        if length == 0.0 or not np.isfinite(length):
            raise ValueError("Plane.from_point_normal needs a non-zero finite normal")
        unit = direction / length
        return cls([*unit, -float(unit @ origin)])

    def normalized(self) -> Plane:
        """The same plane with a unit normal."""
        return Plane(self.coef / np.linalg.norm(self.coef[:3]))

    def distance(self, points: Any) -> Any:
        """Signed distance (m), positive on the side the normal points to.

        A `Point`, `Vector3` or ``(3,)`` sequence gives a float; ``(N, 3)`` gives ``(N,)``.
        """
        array, single = points_of("Plane.distance", points)
        distances = (array @ self.coef[:3] + self.coef[3]) / np.linalg.norm(self.coef[:3])
        return float(distances[0]) if single else distances
