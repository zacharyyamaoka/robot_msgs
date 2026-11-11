"""robot_msgs.shape (SolidPrimitive, Mesh, Plane, MeshResource) and std.ColorRGBA.

The oracles come from outside the implementation: textbook closed forms typed in as formulas
here, a Monte-Carlo integral over an indicator function written in this file (never the code's
own `contains`), the mesh's own divergence-theorem volume, scipy's rotations and matplotlib's
colour parser.
"""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import robot_msgs
from robot_msgs.geometry import Inertia, Point, Pose, Quaternion, Vector3
from robot_msgs.shape import Mesh, MeshResource, Plane, SolidPrimitive
from robot_msgs.std import ColorRGBA

L_SHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]  # concave, area 3, centroid (5/6, 5/6)


# ── helpers: independent of the code under test ──────────────────────────────


def indicator(solid: SolidPrimitive, points: np.ndarray) -> np.ndarray:
    """Is each point inside? Written from the shape definitions, not from `contains`."""
    x, y, z = points.T
    if solid.type == SolidPrimitive.BOX:
        a, b, c = solid.dimensions
        return (abs(x) <= a / 2) & (abs(y) <= b / 2) & (abs(z) <= c / 2)
    if solid.type == SolidPrimitive.SPHERE:
        return x * x + y * y + z * z <= solid.dimensions[0] ** 2
    if solid.type == SolidPrimitive.PRISM:  # the L shape: two rectangles
        in_l = ((0 <= x) & (x <= 2) & (0 <= y) & (y <= 1)) | (
            (0 <= x) & (x <= 1) & (1 <= y) & (y <= 2)
        )
        return in_l & (abs(z) <= solid.dimensions[0] / 2)
    height, radius = solid.dimensions
    if solid.type == SolidPrimitive.CYLINDER:
        return (np.hypot(x, y) <= radius) & (abs(z) <= height / 2)
    if solid.type == SolidPrimitive.CONE:  # base at z = -h/2, apex at z = +h/2
        return (np.hypot(x, y) <= radius * (height / 2 - z) / height) & (abs(z) <= height / 2)
    raise AssertionError(solid.type)


def monte_carlo_inertia(solid, mass, n=1_500_000, seed=11):
    """(centroid, tensor about the centroid) of a uniform solid by sampling its bounding box."""
    lo, hi = (corner.as_array() for corner in solid.bounding_box())
    points = np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))
    inside = points[indicator(solid, points)]
    centroid = inside.mean(axis=0)
    rel = inside - centroid
    # I = m * mean(|r|² 1 - r rᵀ) over the solid
    tensor = mass * (np.mean(np.sum(rel**2, axis=1)) * np.eye(3) - rel.T @ rel / len(rel))
    return centroid, tensor


def is_watertight(mesh: Mesh) -> bool:
    """Every directed edge appears once and its reverse appears once: a closed, oriented mesh."""
    edges = {}
    for a, b, c in mesh.triangles.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            edges[edge] = edges.get(edge, 0) + 1
    return all(count == 1 and (edge[1], edge[0]) in edges for edge, count in edges.items())


def shoelace(polygon) -> float:
    pts = np.asarray(polygon, dtype=float)
    return 0.5 * float(
        np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    )


SOLIDS = {
    "box": SolidPrimitive.box(0.3, 0.2, 0.5),
    "sphere": SolidPrimitive.sphere(0.25),
    "cylinder": SolidPrimitive.cylinder(0.6, 0.15),
    "cone": SolidPrimitive.cone(0.5, 0.2),
    "prism": SolidPrimitive.prism(0.4, L_SHAPE),
}


# ── SolidPrimitive: ROS shape, constructors, validation ──────────────────────


def test_ros_constants_are_the_ros_values():
    assert (SolidPrimitive.BOX, SolidPrimitive.SPHERE, SolidPrimitive.CYLINDER) == (1, 2, 3)
    assert (SolidPrimitive.CONE, SolidPrimitive.PRISM) == (4, 5)
    assert (SolidPrimitive.BOX_X, SolidPrimitive.BOX_Y, SolidPrimitive.BOX_Z) == (0, 1, 2)
    assert SolidPrimitive.SPHERE_RADIUS == 0
    assert (SolidPrimitive.CYLINDER_HEIGHT, SolidPrimitive.CYLINDER_RADIUS) == (0, 1)
    assert (SolidPrimitive.CONE_HEIGHT, SolidPrimitive.CONE_RADIUS) == (0, 1)
    assert SolidPrimitive.PRISM_HEIGHT == 0


def test_constructors_fill_the_ros_dimension_layout():
    assert SolidPrimitive.box(1, 2, 3).dimensions.tolist() == [1, 2, 3]
    assert SolidPrimitive.sphere(0.5).dimensions.tolist() == [0.5]
    assert SolidPrimitive.cylinder(height=2, radius=0.1).dimensions.tolist() == [2, 0.1]
    assert SolidPrimitive.cone(height=2, radius=0.1).dimensions.tolist() == [2, 0.1]
    prism = SolidPrimitive.prism(0.4, L_SHAPE)
    assert prism.dimensions.tolist() == [0.4] and prism.polygon.shape == (6, 2)
    assert SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1, 2, 3]) == SolidPrimitive.box(
        1, 2, 3
    )
    assert SolidPrimitive.box(1, 2, 3).type == 1 and SolidPrimitive.cone(1, 1).type == 4


def test_readable_properties_per_type():
    assert SolidPrimitive.box(1, 2, 3).size == Vector3(1, 2, 3)
    assert SolidPrimitive.sphere(0.5).radius == 0.5
    cylinder = SolidPrimitive.cylinder(height=2, radius=0.1)
    assert (cylinder.height, cylinder.radius) == (2, 0.1)
    cone = SolidPrimitive.cone(height=3, radius=0.4)
    assert (cone.height, cone.radius) == (3, 0.4)
    assert SolidPrimitive.prism(0.4, L_SHAPE).height == 0.4


def test_a_property_the_type_does_not_have_says_so():
    with pytest.raises(ValueError, match="SPHERE has no size"):
        _ = SolidPrimitive.sphere(1).size
    with pytest.raises(ValueError, match="BOX has no radius"):
        _ = SolidPrimitive.box(1, 1, 1).radius
    with pytest.raises(ValueError, match="SPHERE has no height"):
        _ = SolidPrimitive.sphere(1).height


def test_arrays_are_read_only_and_copied():
    dims = np.array([1.0, 2.0, 3.0])
    box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dims)
    dims[0] = 99.0
    assert box.dimensions[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        box.dimensions[0] = 5.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"type": 9, "dimensions": [1.0]}, "type 9"),
        ({"type": 1, "dimensions": [1.0, 2.0]}, r"BOX.*3 dimensions.*got 2"),
        ({"type": 2, "dimensions": [1.0, 2.0]}, r"SPHERE.*1 dimension.*got 2"),
        ({"type": 3, "dimensions": [1.0]}, r"CYLINDER.*2 dimensions.*got 1"),
        ({"type": 4, "dimensions": [1.0, 1.0, 1.0]}, r"CONE.*2 dimensions.*got 3"),
        ({"type": 1, "dimensions": [1.0, 0.0, 1.0]}, "dimensions must be positive"),
        ({"type": 2, "dimensions": [-0.1]}, "dimensions must be positive"),
        ({"type": 2, "dimensions": [float("nan")]}, "finite"),
        ({"type": 2, "dimensions": [float("inf")]}, "finite"),
        ({"type": 5, "dimensions": [1.0]}, "PRISM needs a polygon"),
        ({"type": 5, "dimensions": [1.0], "polygon": [(0, 0), (1, 0)]}, "at least 3"),
        ({"type": 5, "dimensions": [1.0], "polygon": [(0, 0, 0), (1, 0, 0), (0, 1, 0)]}, "shape"),
        ({"type": 5, "dimensions": [1.0], "polygon": [(0, 0), (1, 1), (2, 2)]}, "zero area"),
        ({"type": 1, "dimensions": [1, 1, 1], "polygon": L_SHAPE}, "only PRISM has a polygon"),
    ],
)
def test_invalid_primitives_fail_at_construction_naming_the_field(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolidPrimitive(**kwargs)


def test_replace_revalidates():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        SolidPrimitive.box(1, 1, 1).replace(dimensions=[1.0, -1.0, 1.0])


# ── volume: textbook closed forms ────────────────────────────────────────────


def test_volume_matches_the_textbook_formulas():
    assert SolidPrimitive.box(0.3, 0.2, 0.5).volume() == pytest.approx(0.3 * 0.2 * 0.5)
    assert SolidPrimitive.sphere(0.25).volume() == pytest.approx(4 / 3 * math.pi * 0.25**3)
    assert SolidPrimitive.cylinder(0.6, 0.15).volume() == pytest.approx(math.pi * 0.15**2 * 0.6)
    assert SolidPrimitive.cone(0.5, 0.2).volume() == pytest.approx(math.pi * 0.2**2 * 0.5 / 3)
    assert SolidPrimitive.prism(0.4, L_SHAPE).volume() == pytest.approx(3.0 * 0.4)


def test_prism_volume_does_not_depend_on_winding():
    assert SolidPrimitive.prism(1.0, L_SHAPE[::-1]).volume() == pytest.approx(3.0)
    assert shoelace(L_SHAPE) == pytest.approx(3.0)


@pytest.mark.parametrize("name", ["sphere", "cylinder", "cone"])
def test_mesh_volume_converges_to_the_exact_volume_from_below(name):
    solid = SOLIDS[name]
    errors = [
        abs(solid.volume() - solid.to_mesh(resolution).volume()) / solid.volume()
        for resolution in (8, 16, 32, 64, 128)
    ]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 2e-3
    # inscribed polyhedra are strictly smaller than the solid
    assert solid.to_mesh(16).volume() < solid.volume()


@pytest.mark.parametrize("name", ["box", "prism"])
def test_flat_sided_meshes_are_exact(name):
    solid = SOLIDS[name]
    assert solid.to_mesh().volume() == pytest.approx(solid.volume(), rel=1e-12)


# ── inertia: closed form, then an independent Monte-Carlo integral ───────────


def test_inertia_matches_the_textbook_tensors():
    m = 2.0
    box = SolidPrimitive.box(0.3, 0.2, 0.5).inertia(m)
    assert (box.ixx, box.iyy, box.izz) == pytest.approx(
        (m / 12 * (0.2**2 + 0.5**2), m / 12 * (0.3**2 + 0.5**2), m / 12 * (0.3**2 + 0.2**2))
    )
    sphere = SolidPrimitive.sphere(0.25).inertia(m)
    assert (sphere.ixx, sphere.iyy, sphere.izz) == pytest.approx((2 / 5 * m * 0.25**2,) * 3)
    h, r = 0.6, 0.15  # cylinder along z, URDF convention
    cylinder = SolidPrimitive.cylinder(h, r).inertia(m)
    assert (cylinder.ixx, cylinder.iyy, cylinder.izz) == pytest.approx(
        (m / 12 * (3 * r**2 + h**2), m / 12 * (3 * r**2 + h**2), m / 2 * r**2)
    )
    h, r = 0.5, 0.2  # cone along z, about its centroid
    cone = SolidPrimitive.cone(h, r).inertia(m)
    assert (cone.ixx, cone.iyy, cone.izz) == pytest.approx(
        (
            3 / 20 * m * r**2 + 3 / 80 * m * h**2,
            3 / 20 * m * r**2 + 3 / 80 * m * h**2,
            3 / 10 * m * r**2,
        )
    )


def test_inertia_is_about_the_centroid_and_carries_the_mass_and_com():
    for name, solid in SOLIDS.items():
        inertia = solid.inertia(1.5)
        assert isinstance(inertia, Inertia) and inertia.m == 1.5, name
        assert inertia.is_physical(), name


def test_symmetric_shapes_have_their_centroid_at_the_origin():
    for name in ("box", "sphere", "cylinder"):
        assert SOLIDS[name].centroid() == Point(0, 0, 0), name
        assert SOLIDS[name].inertia(1.0).com == Vector3(0, 0, 0), name


def test_ros_cone_origin_is_the_bounding_box_centre_so_the_centroid_sits_a_quarter_below():
    """ROS: all shapes have their bounding boxes centred on 0,0,0. A cone of height h spans
    z = -h/2 (base) to +h/2 (apex); its centroid is h/4 above the base: z = -h/4."""
    cone = SolidPrimitive.cone(0.8, 0.3)
    assert cone.centroid().z == pytest.approx(-0.2)
    assert cone.inertia(1.0).com.z == pytest.approx(-0.2)
    assert cone.bounding_box() == (Point(-0.3, -0.3, -0.4), Point(0.3, 0.3, 0.4))


def test_prism_centroid_is_the_polygons_centroid():
    centroid = SOLIDS["prism"].centroid()
    assert (centroid.x, centroid.y, centroid.z) == pytest.approx((5 / 6, 5 / 6, 0.0))
    assert SolidPrimitive.prism(1.0, L_SHAPE[::-1]).centroid() == centroid


@pytest.mark.parametrize("name", list(SOLIDS))
def test_inertia_and_centroid_match_a_monte_carlo_integral(name):
    solid, mass = SOLIDS[name], 1.7
    centroid, tensor = monte_carlo_inertia(solid, mass)
    inertia = solid.inertia(mass)
    scale = float(np.abs(tensor).max())
    assert solid.centroid().as_array() == pytest.approx(centroid, abs=5e-3)
    assert inertia.com.as_array() == pytest.approx(centroid, abs=5e-3)
    assert inertia.to_matrix() == pytest.approx(tensor, abs=0.02 * scale)


@pytest.mark.parametrize("name", list(SOLIDS))
def test_inertia_matches_the_meshs_own_divergence_theorem_inertia(name):
    solid, mass = SOLIDS[name], 1.7
    mesh = solid.to_mesh(256)
    expected = mesh.inertia(mass)
    inertia = solid.inertia(mass)
    scale = float(np.abs(expected.to_matrix()).max())
    assert inertia.to_matrix() == pytest.approx(expected.to_matrix(), abs=0.01 * scale)
    assert inertia.com.as_array() == pytest.approx(expected.com.as_array(), abs=2e-3)


def test_inertia_needs_a_positive_mass():
    for mass in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="mass"):
            SolidPrimitive.box(1, 1, 1).inertia(mass)


# ── bounding_box ─────────────────────────────────────────────────────────────


def test_bounding_boxes_are_centred_on_the_origin_except_where_the_polygon_is_not():
    assert SOLIDS["box"].bounding_box() == (Point(-0.15, -0.1, -0.25), Point(0.15, 0.1, 0.25))
    assert SOLIDS["sphere"].bounding_box() == (Point(-0.25, -0.25, -0.25), Point(0.25, 0.25, 0.25))
    assert SOLIDS["cylinder"].bounding_box() == (
        Point(-0.15, -0.15, -0.3),
        Point(0.15, 0.15, 0.3),
    )
    assert SOLIDS["cone"].bounding_box() == (Point(-0.2, -0.2, -0.25), Point(0.2, 0.2, 0.25))
    assert SOLIDS["prism"].bounding_box() == (Point(0, 0, -0.2), Point(2, 2, 0.2))


@pytest.mark.parametrize("name", list(SOLIDS))
def test_the_mesh_fits_in_the_bounding_box_and_touches_it(name):
    solid = SOLIDS[name]
    low, high = (corner.as_array() for corner in solid.bounding_box())
    mesh_low, mesh_high = (corner.as_array() for corner in solid.to_mesh(64).bounding_box())
    assert np.all(mesh_low >= low - 1e-12) and np.all(mesh_high <= high + 1e-12)
    # every face of the box is touched (the sphere's round equator needs res 64 to get close)
    tolerance = 0.01 * float(np.max(high - low))
    assert mesh_low == pytest.approx(low, abs=tolerance)
    assert mesh_high == pytest.approx(high, abs=tolerance)


# ── contains ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(SOLIDS))
def test_contains_agrees_with_the_independent_indicator_on_random_points(name):
    solid = SOLIDS[name]
    low, high = (corner.as_array() for corner in solid.bounding_box())
    points = np.random.default_rng(5).uniform(low - 0.05, high + 0.05, size=(20_000, 3))
    mine = solid.contains(points)
    expected = indicator(solid, points)
    assert mine.dtype == bool and mine.shape == (20_000,)
    assert np.array_equal(mine, expected)
    assert 0.05 < expected.mean() < 0.95  # the test sees both inside and outside


def test_contains_takes_a_point_a_vector_or_a_row_and_returns_a_bool():
    box = SOLIDS["box"]
    assert box.contains(Point(0, 0, 0)) is True
    assert box.contains(Point(1, 0, 0)) is False
    assert box.contains([0.1, 0.0, 0.0]) is True
    assert box.contains(Vector3(0.0, 0.0, 0.3)) is False
    assert box.contains(np.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        box.contains([1.0, 2.0])


def test_the_boundary_counts_as_inside():
    assert SOLIDS["box"].contains(Point(0.15, 0.1, 0.25))
    assert SOLIDS["sphere"].contains(Point(0.25, 0, 0))
    assert SOLIDS["cylinder"].contains(Point(0.15, 0, 0.3))
    assert SOLIDS["cone"].contains(Point(0.2, 0, -0.25))  # base rim
    assert SOLIDS["cone"].contains(Point(0, 0, 0.25))  # apex
    assert not SOLIDS["cone"].contains(Point(0.2, 0, -0.2))  # outside the slanted side
    assert SOLIDS["prism"].contains(Point(2, 0.5, 0))  # on an edge of the L
    assert SOLIDS["prism"].contains(Point(1, 1, 0.2))  # on the reflex corner, on the top face
    assert not SOLIDS["prism"].contains(Point(1.5, 1.5, 0))  # the notch of the L


# ── to_mesh ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(SOLIDS))
@pytest.mark.parametrize("resolution", [3, 8, 32])
def test_to_mesh_is_closed_outward_oriented_and_on_the_surface(name, resolution):
    solid = SOLIDS[name]
    mesh = solid.to_mesh(resolution)
    assert isinstance(mesh, Mesh) and mesh.vertices.shape[1] == 3
    assert is_watertight(mesh)
    assert mesh.volume() > 0  # a flipped winding would be negative
    # every vertex is on or inside the solid, inflated by rounding
    inflated = solid.replace(dimensions=solid.dimensions * (1 + 1e-9)) if name != "prism" else solid
    assert np.all(inflated.contains(mesh.vertices))


def test_resolution_counts_segments_around_the_axis():
    cylinder = SolidPrimitive.cylinder(1.0, 0.5)
    ring = lambda mesh, z: int(np.sum(np.isclose(mesh.vertices[:, 2], z)))  # noqa: E731
    assert ring(cylinder.to_mesh(12), 0.5) == 12 + 1  # ring + cap centre
    assert ring(cylinder.to_mesh(20), -0.5) == 20 + 1
    assert len(SOLIDS["box"].to_mesh(64).triangles) == 12  # flat solids ignore it


@pytest.mark.parametrize("resolution", [0, 2, -4, 1.5, True])
def test_a_bad_resolution_is_refused(resolution):
    with pytest.raises((ValueError, TypeError), match="resolution"):
        SOLIDS["sphere"].to_mesh(resolution)


# ── Mesh ─────────────────────────────────────────────────────────────────────

CUBE_FACES = np.array(
    [
        [0, 1, 2], [1, 3, 2],  # x = 0, normal -x
        [4, 6, 5], [5, 6, 7],  # x = 1, normal +x
        [0, 4, 1], [1, 4, 5],  # y = 0, normal -y
        [2, 3, 6], [3, 7, 6],  # y = 1, normal +y
        [0, 2, 4], [2, 6, 4],  # z = 0, normal -z
        [1, 5, 3], [3, 5, 7],  # z = 1, normal +z
    ]
)  # fmt: skip
CUBE_VERTICES = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
)  # index = 4x + 2y + z
TETRA = Mesh(
    vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    triangles=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
)


def test_mesh_volume_of_known_solids():
    cube = Mesh(vertices=CUBE_VERTICES, triangles=CUBE_FACES)
    assert is_watertight(cube)
    assert cube.volume() == pytest.approx(1.0)
    assert TETRA.volume() == pytest.approx(1 / 6)  # corner tetrahedron: base 1/2 · height 1 / 3


def test_mesh_volume_is_signed_by_winding():
    cube = Mesh(vertices=CUBE_VERTICES, triangles=CUBE_FACES)
    assert cube.replace(triangles=CUBE_FACES[:, ::-1]).volume() == pytest.approx(-1.0)


def test_mesh_volume_does_not_depend_on_where_the_origin_is():
    cube = Mesh(vertices=CUBE_VERTICES + [10.0, -4.0, 3.0], triangles=CUBE_FACES)
    assert cube.volume() == pytest.approx(1.0)


def test_mesh_centroid_and_inertia_of_the_unit_cube():
    cube = Mesh(vertices=CUBE_VERTICES, triangles=CUBE_FACES)
    inertia = cube.inertia(6.0)
    assert inertia.com.as_array() == pytest.approx([0.5, 0.5, 0.5])
    assert inertia.to_matrix() == pytest.approx(np.eye(3) * (6.0 / 12 * 2))  # m/12 (a² + b²)


def test_mesh_inertia_needs_a_closed_outward_mesh():
    flipped = Mesh(vertices=CUBE_VERTICES, triangles=CUBE_FACES[:, ::-1])
    with pytest.raises(ValueError, match="volume"):
        flipped.inertia(1.0)


def test_mesh_bounding_box():
    assert TETRA.bounding_box() == (Point(0, 0, 0), Point(1, 1, 1))
    low, high = Mesh(vertices=CUBE_VERTICES - 2.0, triangles=CUBE_FACES).bounding_box()
    assert (low, high) == (Point(-2, -2, -2), Point(-1, -1, -1))


def test_mesh_transformed_is_the_rigid_motion_of_every_vertex():
    rotation = Rotation.from_euler("xyz", [0.3, -0.7, 1.1])  # extrinsic xyz: ROS RPY
    pose = Pose.from_xyz_rpy(0.5, -1.0, 2.0, 0.3, -0.7, 1.1)
    mesh = SOLIDS["cone"].to_mesh(24)
    moved = mesh.transformed(pose)
    # a writeable copy: scipy on Python 3.11 refuses read-only arrays in Rotation.apply
    expected = rotation.apply(np.array(mesh.vertices)) + [0.5, -1.0, 2.0]
    assert moved.vertices == pytest.approx(expected, abs=1e-12)
    assert np.array_equal(moved.triangles, mesh.triangles)
    assert moved.volume() == pytest.approx(mesh.volume())
    assert moved.inertia(1.0).to_matrix() == pytest.approx(
        rotation.as_matrix() @ mesh.inertia(1.0).to_matrix() @ rotation.as_matrix().T, abs=1e-12
    )
    low, high = (corner.as_array() for corner in moved.bounding_box())
    assert low == pytest.approx(expected.min(axis=0)) and high == pytest.approx(
        expected.max(axis=0)
    )


def test_mesh_transformed_by_the_identity_is_equal_and_the_original_is_untouched():
    mesh = SOLIDS["box"].to_mesh()
    before = mesh.vertices.copy()
    assert mesh.transformed(Pose()) == mesh
    mesh.transformed(Pose.from_xyz_rpy(1, 2, 3, 0, 0, 0))
    assert np.array_equal(mesh.vertices, before)


def test_mesh_arrays_are_read_only_ros_shaped():
    mesh = SOLIDS["box"].to_mesh()
    assert mesh.vertices.dtype == np.float64 and mesh.vertices.shape == (8, 3)
    assert mesh.triangles.dtype.kind in "iu" and mesh.triangles.shape == (12, 3)
    assert not mesh.vertices.flags.writeable and not mesh.triangles.flags.writeable
    assert len(mesh) == 12


@pytest.mark.parametrize(
    ("vertices", "triangles", "match"),
    [
        ([[0, 0], [1, 0]], [[0, 1, 0]], r"Mesh.vertices.*\(N, 3\)"),
        ([[0, 0, 0]] * 3, [[0, 1]], r"Mesh.triangles.*\(M, 3\)"),
        ([[0, 0, 0]] * 3, [[0, 1, 3]], "index 3.*3 vertices"),
        ([[0, 0, 0]] * 3, [[0, 1, -1]], "index -1"),
        ([[0, 0, 0], [1, 0, 0], [0, float("nan"), 0]], [[0, 1, 2]], "finite"),
    ],
)
def test_invalid_meshes_fail_at_construction_naming_the_field(vertices, triangles, match):
    with pytest.raises(ValueError, match=match):
        Mesh(vertices=vertices, triangles=triangles)


def test_an_empty_mesh_is_allowed_and_has_no_volume():
    empty = Mesh(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=int))
    assert len(empty) == 0 and empty.volume() == 0.0
    with pytest.raises(ValueError, match="empty"):
        empty.bounding_box()


# ── Plane ────────────────────────────────────────────────────────────────────


def test_plane_coefficients_are_ros_ax_by_cz_d():
    plane = Plane([0.0, 0.0, 1.0, -0.5])  # z = 0.5
    assert plane.coef.tolist() == [0.0, 0.0, 1.0, -0.5]
    assert (plane.a, plane.b, plane.c, plane.d) == (0.0, 0.0, 1.0, -0.5)
    assert plane.normal == Vector3(0, 0, 1)


def test_plane_distance_is_signed_positive_on_the_normal_side():
    plane = Plane([0.0, 0.0, 1.0, -0.5])
    assert plane.distance(Point(7, -3, 2.0)) == pytest.approx(1.5)
    assert plane.distance(Point(0, 0, 0)) == pytest.approx(-0.5)
    assert plane.distance(Point(1, 1, 0.5)) == pytest.approx(0.0)
    # an unnormalised plane measures the same distance
    assert Plane([0.0, 0.0, 4.0, -2.0]).distance(Point(7, -3, 2.0)) == pytest.approx(1.5)


def test_plane_distance_of_random_planes_matches_the_point_to_plane_formula():
    rng = np.random.default_rng(2)
    for _ in range(20):
        coef = rng.normal(size=4)
        points = rng.normal(size=(50, 3))
        expected = (points @ coef[:3] + coef[3]) / np.linalg.norm(coef[:3])
        assert Plane(coef).distance(points) == pytest.approx(expected)
        # moving a point by the signed distance along the unit normal lands on the plane
        unit = coef[:3] / np.linalg.norm(coef[:3])
        landed = points - np.outer(Plane(coef).distance(points), unit)
        assert Plane(coef).distance(landed) == pytest.approx(0.0, abs=1e-12)


def test_plane_distance_returns_a_float_for_a_point_and_an_array_for_many():
    plane = Plane([1.0, 0.0, 0.0, 0.0])
    assert isinstance(plane.distance(Point(2, 0, 0)), float)
    assert plane.distance(np.array([[1.0, 0, 0], [-2.0, 0, 0]])).tolist() == [1.0, -2.0]


def test_plane_from_point_normal_normalises_and_passes_through_the_point():
    plane = Plane.from_point_normal(Point(1.0, 2.0, 3.0), Vector3(0.0, 0.0, 5.0))
    assert plane.coef.tolist() == [0.0, 0.0, 1.0, -3.0]
    tilted = Plane.from_point_normal([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])
    assert tilted.coef[:3] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    assert tilted.distance(Point(1.0, 2.0, 3.0)) == pytest.approx(0.0, abs=1e-15)
    assert tilted.distance(Point(1.0, 2.0, 3.0) + Point(1 / 3, 2 / 3, 2 / 3)) == pytest.approx(1.0)


def test_plane_normalized_has_a_unit_normal_and_the_same_distances():
    plane = Plane([0.0, 3.0, 4.0, -10.0])
    unit = plane.normalized()
    assert unit.coef.tolist() == pytest.approx([0.0, 0.6, 0.8, -2.0])
    assert unit.distance(Point(1, 2, 3)) == pytest.approx(plane.distance(Point(1, 2, 3)))


@pytest.mark.parametrize("coef", [[0, 0, 0, 1.0], [0, 0, 0, 0.0]])
def test_a_plane_needs_a_normal(coef):
    with pytest.raises(ValueError, match="Plane.coef.*normal"):
        Plane(coef)


@pytest.mark.parametrize("coef", [[1, 0, 0], [1, 0, 0, 0, 0], [float("nan"), 0, 1, 0]])
def test_a_plane_needs_four_finite_coefficients(coef):
    with pytest.raises(ValueError, match="Plane.coef"):
        Plane(coef)


def test_from_point_normal_refuses_a_zero_normal():
    with pytest.raises(ValueError, match="normal"):
        Plane.from_point_normal(Point(), Vector3())


# ── MeshResource ─────────────────────────────────────────────────────────────


def test_mesh_resource_is_a_urdf_mesh_a_filename_and_a_scale():
    resource = MeshResource("package://bam_arm/meshes/link1.stl", scale=(0.001, 0.001, 0.001))
    assert resource.filename == "package://bam_arm/meshes/link1.stl"
    assert resource.scale == Vector3(0.001, 0.001, 0.001)
    assert MeshResource("a.dae").scale == Vector3(1.0, 1.0, 1.0)  # URDF's default scale
    assert MeshResource("a.dae", scale=Vector3(1, 2, 3)).scale == Vector3(1, 2, 3)


def test_mesh_resource_scale_keeps_a_value_s_range():
    from robot_msgs.space import FloatValue, Value

    value = Value(0.001, "m/mm", low=0.0005, high=0.002)
    scale = Vector3(value, value, value)
    assert isinstance(MeshResource("a.stl", scale=scale).scale.x, FloatValue)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"filename": ""}, "filename"),
        ({"filename": "a.stl", "scale": (1, 2)}, "scale"),
        ({"filename": "a.stl", "scale": (1, 0, 1)}, "scale must be non-zero"),
        ({"filename": "a.stl", "scale": (1, float("nan"), 1)}, "finite"),
    ],
)
def test_invalid_mesh_resources_fail_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MeshResource(**kwargs)


# ── ColorRGBA ────────────────────────────────────────────────────────────────


def test_color_defaults_to_opaque_black_and_takes_ros_field_order():
    assert ColorRGBA() == ColorRGBA(0.0, 0.0, 0.0, 1.0)
    color = ColorRGBA(0.1, 0.2, 0.3, 0.4)
    assert (color.r, color.g, color.b, color.a) == (0.1, 0.2, 0.3, 0.4)
    assert color.as_array().tolist() == [0.1, 0.2, 0.3, 0.4]
    assert ColorRGBA.from_array([0.1, 0.2, 0.3, 0.4]) == color


@pytest.mark.parametrize("channel", ["r", "g", "b", "a"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_channels_must_be_in_zero_to_one(channel, value):
    with pytest.raises(ValueError, match=f"ColorRGBA.{channel}"):
        ColorRGBA(**{channel: value})


def test_from_hex_agrees_with_matplotlibs_colour_parser():
    from matplotlib.colors import to_hex, to_rgba

    for text in ("#ff8000", "#FF8000", "#00000000", "#12345678", "#ffffff", "#0a0b0c"):
        assert tuple(ColorRGBA.from_hex(text).as_array()) == pytest.approx(to_rgba(text))
    assert ColorRGBA.from_hex("#f80") == ColorRGBA.from_hex("#ff8800")  # CSS shorthand
    assert ColorRGBA.from_hex("ff8000") == ColorRGBA.from_hex("#ff8000")
    assert ColorRGBA(1.0, 128 / 255, 0.0, 1.0).to_hex() == "#ff8000" == to_hex((1, 128 / 255, 0))


def test_to_hex_writes_the_alpha_pair_only_when_not_opaque():
    assert ColorRGBA(1.0, 0.0, 0.0).to_hex() == "#ff0000"
    assert ColorRGBA(1.0, 0.0, 0.0, 0.5).to_hex() == "#ff000080"
    assert ColorRGBA(1.0, 0.0, 0.0, 0.5).to_hex(alpha=False) == "#ff0000"
    assert ColorRGBA(0.0, 0.0, 0.0).to_hex(alpha=True) == "#000000ff"


def test_hex_round_trips_every_8_bit_channel_value():
    for value in range(256):
        text = f"#{value:02x}{255 - value:02x}{value:02x}{value:02x}"
        assert ColorRGBA.from_hex(text).to_hex(alpha=True) == text


@pytest.mark.parametrize("text", ["", "#", "#ff", "#ff80", "#gg0000", "#ff800", "#ff8000ff00"])
def test_a_bad_hex_string_is_refused(text):
    with pytest.raises(ValueError, match="hex"):
        ColorRGBA.from_hex(text)


# ── the package ──────────────────────────────────────────────────────────────


def test_shape_is_a_domain_of_robot_msgs():
    assert robot_msgs.shape.SolidPrimitive is SolidPrimitive
    assert SolidPrimitive.type_name == "shape/SolidPrimitive"
    assert Mesh.type_name == "shape/Mesh" and Plane.type_name == "shape/Plane"
    assert MeshResource.type_name == "shape/MeshResource"
    assert ColorRGBA.type_name == "std/ColorRGBA"
    assert Quaternion  # geometry stays the owner of poses; shape only imports it
