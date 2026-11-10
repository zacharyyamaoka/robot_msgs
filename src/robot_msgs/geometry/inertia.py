"""Inertia — a rigid body's mass, centre of mass and inertia tensor. Mirrors geometry_msgs/Inertia.

The tensor is about the centre of mass, in the frame the owner names (ROS's convention). A URDF
``<inertial>`` is this plus an ``origin`` rotation for the tensor's axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.quaternion import Quaternion
from robot_msgs.geometry.vector3 import Vector3, array_of, expect_type, finite_float

_MOMENTS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


@dataclass(frozen=True, eq=False, repr=False)
class Inertia(Msg):
    """Mass `m` (kg), centre of mass `com` (m) and the six independent tensor entries (kg·m²)."""

    m: float = 0.0
    com: Vector3 = field(default_factory=Vector3)
    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0

    def __post_init__(self) -> None:
        mass = finite_float("Inertia", "m", self.m)
        if mass < 0.0:
            raise ValueError(f"Inertia.m must not be negative, got {mass}")
        set_field(self, "m", mass)
        expect_type("Inertia.com", self.com, Vector3)
        for name in _MOMENTS:
            set_field(self, name, finite_float("Inertia", name, getattr(self, name)))

    def to_matrix(self) -> np.ndarray:
        """The symmetric ``(3, 3)`` tensor."""
        return np.array(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ]
        )

    @classmethod
    def from_matrix(cls, m: float, com: Vector3, matrix: Any) -> Inertia:
        """From a symmetric ``(3, 3)`` tensor; an asymmetric one is refused, never averaged."""
        tensor = np.asarray(matrix, dtype=np.float64)
        if tensor.shape != (3, 3):
            raise ValueError(f"Inertia.from_matrix needs shape (3, 3), got {tensor.shape}")
        if not np.array_equal(tensor, tensor.T):
            raise ValueError("Inertia.from_matrix needs a symmetric tensor")
        return cls(
            m,
            com,
            tensor[0, 0],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 1],
            tensor[1, 2],
            tensor[2, 2],
        )

    def rotated(self, rotation: Quaternion) -> Inertia:
        """Re-expressed in a frame turned by `rotation`: ``I' = R·I·Rᵀ``, ``com' = R·com``."""
        expect_type("Inertia.rotated(rotation)", rotation, Quaternion)
        r = rotation.to_matrix()
        tensor = r @ self.to_matrix() @ r.T
        tensor = (tensor + tensor.T) / 2.0  # WHY: R I Rᵀ is symmetric up to rounding only
        return Inertia.from_matrix(self.m, rotation.rotate(self.com), tensor)

    def is_physical(self) -> bool:
        """Positive mass, a positive semi-definite tensor and the triangle inequality on its
        principal moments: what a simulator needs to accept the body."""
        if self.m <= 0.0:
            return False
        principal = np.linalg.eigvalsh(self.to_matrix())
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(principal))))
        if np.any(principal < -tolerance):
            return False
        a, b, c = principal
        return bool(a + b >= c - tolerance and a + c >= b - tolerance and b + c >= a - tolerance)

    def as_array(self) -> np.ndarray:
        """``(10,)`` = m, com xyz, ixx ixy ixz iyy iyz izz."""
        return np.array([self.m, *self.com, *(getattr(self, name) for name in _MOMENTS)])

    @classmethod
    def from_array(cls, array: Any) -> Inertia:
        values = array_of("Inertia.from_array", array, 10).tolist()
        return cls(values[0], Vector3(*values[1:4]), *values[4:])
