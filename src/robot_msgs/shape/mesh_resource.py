"""MeshResource — a mesh kept in a file, by name and scale. Not a ROS message.

URDF's ``<mesh filename=... scale=...>`` and ``visualization_msgs/Marker`` MESH_RESOURCE both
point at a file instead of carrying triangles. shape_msgs has no such type (its `Mesh` holds the
triangles), so there is no ROS conversion here: read the file into a `Mesh` to cross a wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import Vector3


@dataclass(frozen=True, eq=False, repr=False)
class MeshResource(Msg):
    """`filename` as the owner writes it (``package://…``, ``file://…`` or a path), unresolved;
    `scale` multiplies the file's x, y, z (URDF's default is 1, 1, 1)."""

    filename: str
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))

    def __post_init__(self) -> None:
        filename = str(self.filename)
        if not filename:
            raise ValueError("MeshResource.filename must not be empty")
        set_field(self, "filename", filename)
        scale: Any = self.scale
        if not isinstance(scale, Vector3):
            values = tuple(scale)
            if len(values) != 3:
                raise ValueError(f"MeshResource.scale needs 3 values (x, y, z), got {len(values)}")
            scale = Vector3(*values)
        if 0.0 in (scale.x, scale.y, scale.z):
            raise ValueError(f"MeshResource.scale must be non-zero, got {tuple(scale)}")
        set_field(self, "scale", scale)
