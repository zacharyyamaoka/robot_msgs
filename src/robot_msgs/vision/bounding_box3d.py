"""BoundingBox3D — an oriented box in space. Mirrors vision_msgs/BoundingBox3D."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg
from robot_msgs.geometry import Pose, Vector3

_CORNER_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


@dataclass(frozen=True, eq=False, repr=False)
class BoundingBox3D(Msg):
    """`center` is the box pose in the parent frame; `size` is the extent along its own x, y, z."""

    center: Pose
    size: Vector3

    def __post_init__(self) -> None:
        extent = self.size.as_array()
        if not np.all(np.isfinite(extent)) or np.any(extent < 0.0):
            raise ValueError(f"BoundingBox3D.size must be finite and >= 0, got {tuple(extent)}")

    def corners(self) -> Array:
        """(8, 3) corners in the parent frame.

        Row i has signs (±x, ±y, ±z) of half the size in binary counting order, with z the
        fastest axis: row 0 is (-,-,-), row 1 is (-,-,+), row 2 is (-,+,-), row 7 is (+,+,+).
        """
        rotation = self.center.orientation.to_matrix()
        local = _CORNER_SIGNS * (self.size.as_array() / 2.0)
        return local @ rotation.T + self.center.position.as_array()

    @property
    def volume(self) -> float:
        return float(math.prod(self.size.as_array()))

    def contains(self, points: Array) -> Array:
        """(N, 3) points in the parent frame → (N,) bool; box edges count as inside."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        rotation = self.center.orientation.to_matrix()
        local = (points - self.center.position.as_array()) @ rotation
        return np.all(np.abs(local) <= self.size.as_array() / 2.0, axis=1)
