"""SolidPrimitive — a box, sphere, cylinder, cone or prism. Mirrors shape_msgs/SolidPrimitive.

Describes a solid in any 3-D scene (a robot's collision geometry, an obstacle, a bin), so it lives
here and not with the robot description. ROS defines every primitive with its bounding box centred
on the origin; so does this: a box and a sphere about their centre, a cylinder and a cone along z,
a prism extruded along z. The pose that places it is the owner's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.geometry import Inertia, Point, Vector3
from robot_msgs.shape._common import points_of, positive_mass
from robot_msgs.shape.mesh import Mesh

_NAMES = {1: "BOX", 2: "SPHERE", 3: "CYLINDER", 4: "CONE", 5: "PRISM"}
_DIMENSIONS = {1: 3, 2: 1, 3: 2, 4: 2, 5: 1}

# Vertices are indexed 4·ix + 2·iy + iz with i = 0 for the − side and 1 for the + side.
_BOX_FACES = np.array(
    [
        [0, 1, 2], [1, 3, 2],  # x = −, normal −x
        [4, 6, 5], [5, 6, 7],  # x = +, normal +x
        [0, 4, 1], [1, 4, 5],  # y = −, normal −y
        [2, 3, 6], [3, 7, 6],  # y = +, normal +y
        [0, 2, 4], [2, 6, 4],  # z = −, normal −z
        [1, 5, 3], [3, 5, 7],  # z = +, normal +z
    ],
    dtype=np.int32,
)  # fmt: skip


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class SolidPrimitive(Msg):
    """`type` is one of the ROS constants below and `dimensions` follows ROS's index layout:

    =========  =============================  ==========================================
    type       dimensions                     origin
    =========  =============================  ==========================================
    BOX        x, y, z extents                centre
    SPHERE     radius                         centre
    CYLINDER   height, radius                 centre, axis along z
    CONE       height, radius                 bounding-box centre, base down, apex up
    PRISM      height                         `polygon` extruded along z, centred on z=0
    =========  =============================  ==========================================

    WHY the cone's origin is the middle of its height, not its base or centroid: ROS defines every
    primitive with its bounding box centred on the origin. The base is at ``z = -h/2``, the apex
    at ``+h/2`` and the centroid at ``-h/4``; `inertia` reports that offset as `com`.

    `polygon` is ``(N, 2)`` metres in the xy plane, for PRISM only: a simple polygon (no crossing
    edges), either winding. ROS's Polygon holds Point32s whose z is not used here.

    Build one with `box`, `sphere`, `cylinder`, `cone` or `prism`; read it with `size`, `radius`
    and `height`.
    """

    BOX: ClassVar[int] = 1
    SPHERE: ClassVar[int] = 2
    CYLINDER: ClassVar[int] = 3
    CONE: ClassVar[int] = 4
    PRISM: ClassVar[int] = 5

    BOX_X: ClassVar[int] = 0
    BOX_Y: ClassVar[int] = 1
    BOX_Z: ClassVar[int] = 2
    SPHERE_RADIUS: ClassVar[int] = 0
    CYLINDER_HEIGHT: ClassVar[int] = 0
    CYLINDER_RADIUS: ClassVar[int] = 1
    CONE_HEIGHT: ClassVar[int] = 0
    CONE_RADIUS: ClassVar[int] = 1
    PRISM_HEIGHT: ClassVar[int] = 0

    type: int
    dimensions: Array
    polygon: Array | None = None

    def __post_init__(self) -> None:
        kind = int(self.type)
        if kind not in _NAMES:
            raise ValueError(
                f"SolidPrimitive.type {kind} is unknown (BOX=1, SPHERE=2, CYLINDER=3, CONE=4, "
                "PRISM=5)"
            )
        name, count = _NAMES[kind], _DIMENSIONS[kind]
        dimensions = frozen_array(self.dimensions)
        if dimensions.shape != (count,):
            raise ValueError(
                f"SolidPrimitive {name} needs {count} dimension{'s' if count != 1 else ''}, "
                f"got {dimensions.size}"
            )
        if not np.all(np.isfinite(dimensions)):
            raise ValueError(f"SolidPrimitive.dimensions must be finite, got {dimensions.tolist()}")
        if np.any(dimensions <= 0.0):
            raise ValueError(
                f"SolidPrimitive.dimensions must be positive, got {dimensions.tolist()}"
            )
        polygon = optional_array(self.polygon)
        if kind == self.PRISM:
            if polygon is None:
                raise ValueError("SolidPrimitive PRISM needs a polygon")
            _check_polygon(polygon)
        elif polygon is not None:
            raise ValueError(f"only PRISM has a polygon, not {name}")
        set_field(self, "type", kind)
        set_field(self, "dimensions", dimensions)
        set_field(self, "polygon", polygon)

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def box(cls, x: float, y: float, z: float) -> SolidPrimitive:
        """A box with extents `x`, `y`, `z` (m) along its own axes, centred on the origin."""
        return cls(type=cls.BOX, dimensions=[x, y, z])

    @classmethod
    def sphere(cls, radius: float) -> SolidPrimitive:
        return cls(type=cls.SPHERE, dimensions=[radius])

    @classmethod
    def cylinder(cls, height: float, radius: float) -> SolidPrimitive:
        """Axis along z, centred on the origin (ROS and URDF)."""
        return cls(type=cls.CYLINDER, dimensions=[height, radius])

    @classmethod
    def cone(cls, height: float, radius: float) -> SolidPrimitive:
        """Base (radius `radius`) at ``z = -height/2``, apex at ``z = +height/2``."""
        return cls(type=cls.CONE, dimensions=[height, radius])

    @classmethod
    def prism(cls, height: float, polygon: Any) -> SolidPrimitive:
        """`polygon` ``(N, 2)`` extruded from ``z = -height/2`` to ``+height/2``."""
        return cls(type=cls.PRISM, dimensions=[height], polygon=polygon)

    # ── readable dimensions ──────────────────────────────────────────────────

    @property
    def size(self) -> Vector3:
        """x, y, z extents of a BOX."""
        if self.type != self.BOX:
            raise ValueError(f"SolidPrimitive {_NAMES[self.type]} has no size (only BOX does)")
        return Vector3.from_array(self.dimensions)

    @property
    def radius(self) -> float:
        """Radius of a SPHERE, CYLINDER or CONE."""
        if self.type == self.SPHERE:
            return float(self.dimensions[self.SPHERE_RADIUS])
        if self.type in (self.CYLINDER, self.CONE):
            return float(self.dimensions[self.CYLINDER_RADIUS])
        raise ValueError(f"SolidPrimitive {_NAMES[self.type]} has no radius")

    @property
    def height(self) -> float:
        """Height of a CYLINDER, CONE or PRISM."""
        if self.type in (self.CYLINDER, self.CONE, self.PRISM):
            return float(self.dimensions[0])
        raise ValueError(f"SolidPrimitive {_NAMES[self.type]} has no height")

    # ── standard functions ───────────────────────────────────────────────────

    def volume(self) -> float:
        """Exact volume (m³)."""
        d = self.dimensions
        if self.type == self.BOX:
            return float(d[0] * d[1] * d[2])
        if self.type == self.SPHERE:
            return 4.0 / 3.0 * math.pi * float(d[0]) ** 3
        if self.type == self.CYLINDER:
            return math.pi * float(d[1]) ** 2 * float(d[0])
        if self.type == self.CONE:
            return math.pi * float(d[1]) ** 2 * float(d[0]) / 3.0
        return abs(_signed_area(self.polygon)) * float(d[0])

    def centroid(self) -> Point:
        """Centre of volume, in the primitive's frame: the origin, except a cone's (``z = -h/4``,
        see the class docstring) and a prism's (the polygon's centroid, ``z = 0``)."""
        if self.type == self.CONE:
            return Point(0.0, 0.0, -self.height / 4.0)
        if self.type == self.PRISM:
            x, y = _polygon_centroid(self.polygon)
            return Point(x, y, 0.0)
        return Point(0.0, 0.0, 0.0)

    def inertia(self, mass: float) -> Inertia:
        """Inertia of the uniform solid, about its centroid, axes parallel to the primitive's
        (ROS/URDF conventions); `com` is `centroid()`, so a cone's is not at the origin.

        Closed forms for box, sphere, cylinder (along z) and cone (along z); a prism is exact
        through its own mesh.
        """
        mass = positive_mass("SolidPrimitive.inertia", mass)
        com = Vector3.from_array(self.centroid().as_array())
        d = self.dimensions.tolist()
        if self.type == self.BOX:
            x, y, z = d
            return Inertia(
                mass, com, mass / 12 * (y * y + z * z), 0.0, 0.0,
                mass / 12 * (x * x + z * z), 0.0, mass / 12 * (x * x + y * y),
            )  # fmt: skip
        if self.type == self.SPHERE:
            moment = 2.0 / 5.0 * mass * d[0] ** 2
            return Inertia(mass, com, moment, 0.0, 0.0, moment, 0.0, moment)
        if self.type == self.CYLINDER:
            height, radius = d
            transverse = mass / 12 * (3 * radius**2 + height**2)
            return Inertia(mass, com, transverse, 0.0, 0.0, transverse, 0.0, mass / 2 * radius**2)
        if self.type == self.CONE:
            height, radius = d
            transverse = 3 / 20 * mass * radius**2 + 3 / 80 * mass * height**2
            return Inertia(
                mass, com, transverse, 0.0, 0.0, transverse, 0.0, 3 / 10 * mass * radius**2
            )
        return self.to_mesh().inertia(mass)

    def bounding_box(self) -> tuple[Point, Point]:
        """``(min, max)`` corners of the axis-aligned box around the primitive, in its own frame."""
        d = self.dimensions
        if self.type == self.BOX:
            half = d / 2.0
        elif self.type == self.SPHERE:
            half = np.full(3, d[0])
        elif self.type in (self.CYLINDER, self.CONE):
            half = np.array([d[1], d[1], d[0] / 2.0])
        else:
            low, high = self.polygon.min(axis=0), self.polygon.max(axis=0)
            return (
                Point(low[0], low[1], -d[0] / 2.0),
                Point(high[0], high[1], d[0] / 2.0),
            )
        return Point.from_array(-half), Point.from_array(half)

    def contains(self, points: Any) -> Any:
        """Whether each point (in the primitive's frame) is inside; the surface counts as inside.

        A `Point`, `Vector3` or ``(3,)`` sequence gives a bool; ``(N, 3)`` gives ``(N,)`` bools.
        """
        array, single = points_of("SolidPrimitive.contains", points)
        x, y, z = array.T
        d = self.dimensions
        if self.type == self.BOX:
            inside = np.all(np.abs(array) <= d / 2.0, axis=1)
        elif self.type == self.SPHERE:
            inside = x * x + y * y + z * z <= d[0] ** 2
        elif self.type == self.CYLINDER:
            inside = (np.hypot(x, y) <= d[1]) & (np.abs(z) <= d[0] / 2.0)
        elif self.type == self.CONE:
            # the radius at height z shrinks linearly from the base (z = -h/2) to the apex (+h/2)
            inside = (np.hypot(x, y) <= d[1] * (0.5 - z / d[0])) & (np.abs(z) <= d[0] / 2.0)
        else:
            inside = _in_polygon(self.polygon, x, y) & (np.abs(z) <= d[0] / 2.0)
        return bool(inside[0]) if single else inside

    def to_mesh(self, resolution: int = 32) -> Mesh:
        """A closed, outward-facing triangle mesh with vertices on the surface (so it is inscribed
        and its volume converges to `volume()` from below as `resolution` grows).

        `resolution` is the number of segments around the axis of a sphere, cylinder or cone
        (at least 3; a sphere gets half as many rings); box and prism are exact and ignore it.
        """
        if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
            raise ValueError(f"to_mesh resolution must be an integer, got {resolution!r}")
        if resolution < 3:
            raise ValueError(f"to_mesh resolution must be at least 3, got {resolution}")
        d = self.dimensions
        if self.type == self.BOX:
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            return Mesh(vertices=signs * d / 2.0, triangles=_BOX_FACES)
        if self.type == self.SPHERE:
            return _sphere_mesh(float(d[0]), int(resolution))
        if self.type == self.CYLINDER:
            return _cylinder_mesh(float(d[0]), float(d[1]), int(resolution))
        if self.type == self.CONE:
            return _cone_mesh(float(d[0]), float(d[1]), int(resolution))
        return _prism_mesh(float(d[0]), self.polygon)


# ── meshes ───────────────────────────────────────────────────────────────────


def _ring(radius: float, z: float, count: int) -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(count) / count
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), np.full(count, z)])


def _sides(lower: int, upper: int, count: int) -> list[list[int]]:
    """Quads between two rings of `count` vertices starting at indices `lower` and `upper`
    (lower ring first), as outward triangles when the rings run counter-clockwise from +z."""
    triangles = []
    for i in range(count):
        j = (i + 1) % count
        triangles.append([lower + i, lower + j, upper + j])
        triangles.append([lower + i, upper + j, upper + i])
    return triangles


def _cylinder_mesh(height: float, radius: float, count: int) -> Mesh:
    bottom, top = 0, count
    vertices = np.vstack(
        [
            _ring(radius, -height / 2.0, count),
            _ring(radius, height / 2.0, count),
            [[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]],
        ]
    )
    triangles = _sides(bottom, top, count)
    for i in range(count):
        j = (i + 1) % count
        triangles.append([2 * count + 1, top + i, top + j])  # top cap faces +z
        triangles.append([2 * count, bottom + j, bottom + i])  # bottom cap faces -z
    return Mesh(vertices=vertices, triangles=triangles)


def _cone_mesh(height: float, radius: float, count: int) -> Mesh:
    apex, centre = count, count + 1
    vertices = np.vstack(
        [
            _ring(radius, -height / 2.0, count),
            [[0.0, 0.0, height / 2.0], [0.0, 0.0, -height / 2.0]],
        ]
    )
    triangles = []
    for i in range(count):
        j = (i + 1) % count
        triangles.append([i, j, apex])
        triangles.append([centre, j, i])  # base faces -z
    return Mesh(vertices=vertices, triangles=triangles)


def _sphere_mesh(radius: float, count: int) -> Mesh:
    rings = max(2, count // 2)  # latitude rings between the poles: rings - 1
    north, south = 0, 1
    polar = np.pi * np.arange(1, rings) / rings
    vertices = np.vstack(
        [
            [[0.0, 0.0, radius], [0.0, 0.0, -radius]],
            *[_ring(radius * np.sin(t), radius * np.cos(t), count) for t in polar],
        ]
    )
    first = lambda k: 2 + k * count  # noqa: E731  (index of ring k's first vertex, k from the north)
    triangles = []
    for i in range(count):
        j = (i + 1) % count
        triangles.append([north, first(0) + i, first(0) + j])
        triangles.append([south, first(rings - 2) + j, first(rings - 2) + i])
    for k in range(rings - 2):  # quads between ring k (above) and ring k + 1 (below)
        for i in range(count):
            j = (i + 1) % count
            triangles.append([first(k) + i, first(k + 1) + i, first(k + 1) + j])
            triangles.append([first(k) + i, first(k + 1) + j, first(k) + j])
    return Mesh(vertices=vertices, triangles=triangles)


def _prism_mesh(height: float, polygon: np.ndarray) -> Mesh:
    if _signed_area(polygon) < 0.0:
        polygon = polygon[::-1]
    count = len(polygon)
    vertices = np.vstack(
        [
            np.column_stack([polygon, np.full(count, -height / 2.0)]),
            np.column_stack([polygon, np.full(count, height / 2.0)]),
        ]
    )
    triangles = _sides(0, count, count)
    for a, b, c in _triangulate(polygon):
        triangles.append([count + a, count + b, count + c])  # top faces +z
        triangles.append([a, c, b])  # bottom faces -z
    return Mesh(vertices=vertices, triangles=triangles)


# ── polygons ─────────────────────────────────────────────────────────────────


def _signed_area(polygon: np.ndarray) -> float:
    x, y = polygon.T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    x, y = polygon.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(cross.sum())
    return (
        float(np.sum((x + x1) * cross) / (6.0 * area)),
        float(np.sum((y + y1) * cross) / (6.0 * area)),
    )


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _check_polygon(polygon: np.ndarray) -> None:
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"SolidPrimitive.polygon needs shape (N, 2), got {polygon.shape}")
    if len(polygon) < 3:
        raise ValueError(f"SolidPrimitive.polygon needs at least 3 points, got {len(polygon)}")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("SolidPrimitive.polygon must be finite")
    if _signed_area(polygon) == 0.0:
        raise ValueError("SolidPrimitive.polygon has zero area")
    edges = len(polygon)
    for i in range(edges):
        p1, p2 = polygon[i], polygon[(i + 1) % edges]
        for j in range(i + 2, edges):
            if i == 0 and j == edges - 1:
                continue  # the first and last edges share a vertex
            p3, p4 = polygon[j], polygon[(j + 1) % edges]
            if (
                _cross(p3, p4, p1) * _cross(p3, p4, p2) < 0
                and _cross(p1, p2, p3) * _cross(p1, p2, p4) < 0
            ):
                raise ValueError(f"SolidPrimitive.polygon edges {i} and {j} cross")
    _triangulate(polygon if _signed_area(polygon) > 0 else polygon[::-1])


def _triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping of a counter-clockwise simple polygon: index triples, counter-clockwise."""
    remaining = list(range(len(polygon)))
    triangles = []
    while len(remaining) > 3:
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            turn = _cross(a, b, c)
            if turn < 0.0:
                continue  # a reflex corner is never an ear
            if turn > 0.0 and any(
                _in_triangle(polygon[k], a, b, c)
                for k in remaining
                if k not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[position]
            break
        else:
            raise ValueError("SolidPrimitive.polygon is not a simple polygon")
    triangles.append(tuple(remaining))
    return triangles


def _in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return _cross(a, b, p) >= 0.0 and _cross(b, c, p) >= 0.0 and _cross(c, a, p) >= 0.0


def _in_polygon(polygon: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd test of points against a polygon, the boundary counting as inside."""
    inside = np.zeros(px.shape, dtype=bool)
    on_edge = np.zeros(px.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        within = (
            (min(x1, x2) <= px) & (px <= max(x1, x2)) & (min(y1, y2) <= py) & (py <= max(y1, y2))
        )
        on_edge |= (cross == 0.0) & within
        if y1 == y2:
            continue  # a horizontal edge is never crossed by a horizontal ray
        crosses = (y1 > py) != (y2 > py)
        inside ^= crosses & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
    return inside | on_edge
