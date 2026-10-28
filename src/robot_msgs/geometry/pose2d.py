"""Pose2D — a planar position and heading. Mirrors geometry_msgs/Pose2D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.vector3 import array_of, finite_float


@dataclass(frozen=True, eq=False, repr=False)
class Pose2D(Msg):
    """Metres and radians; `theta` is the heading about +Z, stored as given (not wrapped)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "theta"):
            set_field(self, name, finite_float("Pose2D", name, getattr(self, name)))

    def as_array(self) -> np.ndarray:
        """``(3,)`` = x y theta."""
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, array: Any) -> Pose2D:
        x, y, theta = array_of("Pose2D.from_array", array, 3).tolist()
        return cls(x, y, theta)
