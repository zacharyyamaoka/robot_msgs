"""Mesh — a triangle mesh in a frame. Mirrors shape_msgs/Mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field
from robot_msgs.geometry import Inertia, Point, Pose, Vector3
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.shape._common import positive_mass


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Mesh(Msg):
    """`vertices` is ``(N, 3)`` float64 metres; `triangles` is ``(M, 3)`` int32 vertex indices
    (ROS's ``MeshTriangle.vertex_indices``), counter-clockwise seen from outside.

    The volume, centroid and inertia methods read the mesh as a solid, so they are exact for a
    closed mesh whose triangles face outward and meaningless for an open one.
    """

    vertices: Array
    triangles: Array

    def __post_init__(self) -> None:
        vertices = frozen_array(self.vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Mesh.vertices must have shape (N, 3), got {vertices.shape}")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Mesh.vertices must be finite")
        triangles = frozen_array(self.triangles, dtype=np.int32)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"Mesh.triangles must have shape (M, 3), got {triangles.shape}")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            bad = int(triangles.min() if triangles.min() < 0 else triangles.max())
            raise ValueError(
                f"Mesh.triangles has vertex index {bad}, out of range for {len(vertices)} vertices"
            )
        set_field(self, "vertices", vertices)
        set_field(self, "triangles", triangles)

    def __len__(self) -> int:
        return len(self.triangles)

    def bounding_box(self) -> tuple[Point, Point]:
        """``(min, max)`` corners of the axis-aligned box around the vertices."""
        if len(self.vertices) == 0:
            raise ValueError("Mesh.bounding_box of an empty mesh")
        return Point.from_array(self.vertices.min(axis=0)), Point.from_array(
            self.vertices.max(axis=0)
        )

    def transformed(self, pose: Pose) -> Mesh:
        """The mesh moved rigidly: every vertex rotated by `pose.orientation`, then shifted."""
        expect_type("Mesh.transformed(pose)", pose, Pose)
        rotation = pose.orientation.to_matrix()
        moved = self.vertices @ rotation.T + pose.position.as_array()
        return self.replace(vertices=moved)

    def volume(self) -> float:
        """Signed volume (m³) by the divergence theorem: positive when the triangles face
        outward, negative for an inside-out mesh, 0 for an empty one."""
        return self._moments()[0]

    def centroid(self) -> Point:
        """Centre of volume of the solid (see `volume` for what the mesh must be)."""
        volume, centroid, _ = self._moments()
        if volume == 0.0:
            raise ValueError("Mesh.centroid needs a mesh with non-zero volume")
        return Point.from_array(centroid)

    def inertia(self, mass: float) -> Inertia:
        """Inertia of the solid at uniform density, about its centroid (which is `com`).

        Exact for a closed outward-facing mesh: each triangle and the origin make a tetrahedron
        whose second moment has a closed form, and the signed tetrahedra sum to the solid.
        """
        mass = positive_mass("Mesh.inertia", mass)
        volume, centroid, second = self._moments()
        if volume <= 0.0:
            raise ValueError(
                f"Mesh.inertia needs a closed mesh with outward-facing triangles, "
                f"but its volume is {volume}"
            )
        # second is ∫ r rᵀ dV about the centroid; I = ρ (tr(C) 1 − C) with ρ = m / V
        tensor = mass / volume * (np.trace(second) * np.eye(3) - second)
        return Inertia.from_matrix(mass, Vector3.from_array(centroid), (tensor + tensor.T) / 2.0)

    def _moments(self) -> tuple[float, np.ndarray, np.ndarray]:
        """(signed volume, centroid, ∫ r rᵀ dV about the centroid), summed over the tetrahedra
        (o, a, b, c) of every triangle, with `o` the box centre so large offsets lose no digits."""
        if len(self.triangles) == 0:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        origin = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) / 2.0
        a, b, c = np.moveaxis(self.vertices[self.triangles] - origin, 1, 0)
        volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        total = float(volumes.sum())
        if total == 0.0:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        s = a + b + c
        centroid = (volumes[:, None] * s).sum(axis=0) / 4.0 / total
        # ∫ x xᵀ dV over a tetrahedron (0, a, b, c) is V/20 (a aᵀ + b bᵀ + c cᵀ + s sᵀ)
        raw = sum(np.einsum("i,ij,ik->jk", volumes / 20.0, v, v) for v in (a, b, c, s))
        second = raw - total * np.outer(centroid, centroid)
        return total, centroid + origin, second
