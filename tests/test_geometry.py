"""Geometry messages vs scipy (the independent oracle), plus value semantics and validation."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial import transform as scipy_transform
from scipy.spatial.transform import Rotation, Slerp

from robot_msgs._core import allclose
from robot_msgs.geometry import (
    Accel,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
    Vector3,
    Wrench,
    rotations,
)
from robot_msgs.std import Header

RNG = np.random.default_rng(11)
N = 50
QUATS = Rotation.random(N, random_state=1).as_quat()  # scipy is xyzw too
QUATS2 = Rotation.random(N, random_state=2).as_quat()
VECS = RNG.normal(size=(N, 3))
VECS2 = RNG.normal(size=(N, 3))

has_rigid_transform = hasattr(scipy_transform, "RigidTransform")


def same_rotation(a, b, atol=1e-10):
    """Quaternions q and -q are one rotation."""
    return abs(abs(float(np.dot(a, b))) - 1.0) < atol


def quat(i, pool=QUATS):
    return Quaternion.from_array(pool[i])


def transform(i):
    return Transform(Vector3.from_array(VECS[i]), quat(i))


def pose(i):
    return Pose(Point.from_array(VECS[i]), quat(i))


# ── Vector3 / Point ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [Vector3, Point])
def test_vector_arithmetic_matches_numpy(cls):
    a, b = cls.from_array(VECS[0]), cls.from_array(VECS2[0])
    np.testing.assert_allclose((a + b).as_array(), VECS[0] + VECS2[0])
    np.testing.assert_allclose((a - b).as_array(), VECS[0] - VECS2[0])
    np.testing.assert_allclose((a * 2.5).as_array(), VECS[0] * 2.5)
    np.testing.assert_allclose((2.5 * a).as_array(), VECS[0] * 2.5)
    np.testing.assert_allclose((np.float64(2.5) * a).as_array(), VECS[0] * 2.5)
    np.testing.assert_allclose((a / 4).as_array(), VECS[0] / 4)
    np.testing.assert_allclose((-a).as_array(), -VECS[0])
    assert isinstance(a + b, cls) and isinstance(-a, cls) and isinstance(a * 2, cls)
    assert a.norm() == pytest.approx(np.linalg.norm(VECS[0]))
    assert a.dot(b) == pytest.approx(np.dot(VECS[0], VECS2[0]))


@pytest.mark.parametrize("cls", [Vector3, Point])
def test_vector_unpacks_and_stores_floats(cls):
    x, y, z = cls(1, 2, 3)
    assert (x, y, z) == (1.0, 2.0, 3.0)
    assert all(type(v) is float for v in cls(np.float32(1), 2, np.int64(3)))
    assert cls() == cls(0.0, 0.0, 0.0)


@pytest.mark.parametrize("cls", [Vector3, Point])
def test_vector_rejects_wrong_operands(cls):
    other = Point if cls is Vector3 else Vector3
    with pytest.raises(TypeError):
        cls(1, 2, 3) + other(1, 2, 3)
    with pytest.raises(TypeError):
        cls(1, 2, 3) * cls(1, 2, 3)
    with pytest.raises(TypeError):
        cls(1, 2, 3).dot(other(1, 2, 3))


def test_vector3_cross_and_normalized_match_numpy():
    a, b = Vector3.from_array(VECS[1]), Vector3.from_array(VECS2[1])
    np.testing.assert_allclose(a.cross(b).as_array(), np.cross(VECS[1], VECS2[1]))
    assert a.normalized().norm() == pytest.approx(1.0)
    np.testing.assert_allclose(a.normalized().as_array(), VECS[1] / np.linalg.norm(VECS[1]))
    with pytest.raises(ValueError, match="zero-length"):
        Vector3().normalized()


def test_point_distance_to():
    a, b = Point.from_array(VECS[2]), Point.from_array(VECS2[2])
    assert a.distance_to(b) == pytest.approx(np.linalg.norm(VECS[2] - VECS2[2]))
    assert a.distance_to(a) == 0.0


@pytest.mark.parametrize("cls", [Vector3, Point])
@pytest.mark.parametrize("bad", [(1, 2), (1, 2, 3, 4), ((1, 2, 3),)])
def test_vector_from_array_checks_shape(cls, bad):
    with pytest.raises(ValueError, match="shape"):
        cls.from_array(bad)


# ── Quaternion vs scipy ──────────────────────────────────────────────────────


def test_quaternion_defaults_to_identity():
    assert Quaternion() == Quaternion.identity() == Quaternion(0, 0, 0, 1)
    assert Quaternion().is_unit()


def test_quaternion_is_stored_as_given_and_normalized_on_demand():
    stored = Quaternion(0.0, 0.0, 2.0, 0.0)
    assert tuple(stored) == (0.0, 0.0, 2.0, 0.0)
    assert not stored.is_unit()
    assert stored.normalized() == Quaternion(0.0, 0.0, 1.0, 0.0)
    assert stored.normalized().is_unit()
    assert stored.is_unit(tol=1.5)
    # operations see the rotation, not the length
    np.testing.assert_allclose(stored.rotate([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(stored.to_matrix(), np.diag([-1.0, -1.0, 1.0]), atol=1e-15)


def test_quaternion_multiply_matches_scipy_composition():
    for i in range(N):
        expected = (Rotation.from_quat(QUATS[i]) * Rotation.from_quat(QUATS2[i])).as_quat()
        assert same_rotation((quat(i) * quat(i, QUATS2)).as_array(), expected)
        assert quat(i).multiply(quat(i, QUATS2)) == quat(i) * quat(i, QUATS2)


def test_quaternion_multiply_of_non_unit_inputs_is_a_unit_rotation():
    product = Quaternion(0, 0, 3, 0) * Quaternion(2, 0, 0, 0)
    assert product.is_unit()
    expected = Rotation.from_quat([0, 0, 1, 0]) * Rotation.from_quat([1, 0, 0, 0])
    assert same_rotation(product.as_array(), expected.as_quat())


def test_quaternion_inverse_and_conjugate():
    for i in range(N):
        inverse = quat(i).inverse()
        assert same_rotation(inverse.as_array(), Rotation.from_quat(QUATS[i]).inv().as_quat())
        assert same_rotation((quat(i) * inverse).as_array(), [0, 0, 0, 1])
        assert quat(i).conjugate() == Quaternion(*(-QUATS[i][:3]), QUATS[i][3])
    assert Quaternion(0, 0, 2, 0).conjugate() == Quaternion(0, 0, -2, 0)  # as stored
    assert Quaternion(0, 0, 2, 0).inverse().is_unit()


def test_quaternion_rotate_matches_scipy_for_every_kind():
    for i in range(N):
        expected = Rotation.from_quat(QUATS[i]).apply(VECS[i])
        np.testing.assert_allclose(quat(i).rotate(VECS[i]), expected, atol=1e-12)
        np.testing.assert_allclose(
            quat(i).rotate(Vector3.from_array(VECS[i])).as_array(), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            quat(i).rotate(Point.from_array(VECS[i])).as_array(), expected, atol=1e-12
        )
    batch = quat(0).rotate(VECS)
    np.testing.assert_allclose(batch, Rotation.from_quat(QUATS[0]).apply(VECS), atol=1e-12)
    assert isinstance(quat(0).rotate(Vector3(1, 0, 0)), Vector3)
    assert isinstance(quat(0).rotate(Point(1, 0, 0)), Point)
    assert isinstance(quat(0).rotate([1, 0, 0]), np.ndarray)
    assert quat(0).rotate(VECS).shape == (N, 3)


def test_quaternion_rotate_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        Quaternion().rotate([1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        Quaternion().rotate(np.zeros((4, 2)))


def test_quaternion_matrix_round_trip_matches_scipy():
    for i in range(N):
        matrix = quat(i).to_matrix()
        np.testing.assert_allclose(matrix, Rotation.from_quat(QUATS[i]).as_matrix(), atol=1e-12)
        assert same_rotation(
            Quaternion.from_matrix(Rotation.from_quat(QUATS[i]).as_matrix()).as_array(), QUATS[i]
        )
    with pytest.raises(ValueError, match="shape"):
        Quaternion.from_matrix(np.eye(4))


def test_quaternion_euler_matches_scipy_xyz_extrinsic():
    angles = RNG.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi], size=(N, 3))
    for roll, pitch, yaw in angles:
        built = Quaternion.from_euler(roll, pitch, yaw)
        expected = Rotation.from_euler("xyz", [roll, pitch, yaw])  # lowercase = fixed axes
        assert same_rotation(built.as_array(), expected.as_quat())
        np.testing.assert_allclose(built.to_euler(), expected.as_euler("xyz"), atol=1e-10)
        np.testing.assert_allclose(built.to_euler(), (roll, pitch, yaw), atol=1e-10)
    assert all(type(angle) is float for angle in Quaternion().to_euler())


def test_quaternion_euler_matches_the_ros_definition():
    roll, pitch, yaw = 0.3, -0.4, 1.1
    expected = (
        Rotation.from_euler("z", yaw)
        * Rotation.from_euler("y", pitch)
        * Rotation.from_euler("x", roll)
    )
    np.testing.assert_allclose(
        Quaternion.from_euler(roll, pitch, yaw).to_matrix(), expected.as_matrix(), atol=1e-12
    )


def test_quaternion_axis_angle_and_rotvec_match_scipy():
    for i in range(N):
        rotvec = Rotation.from_quat(QUATS[i]).as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle
        assert same_rotation(Quaternion.from_axis_angle(axis, angle).as_array(), QUATS[i])
        assert same_rotation(Quaternion.from_axis_angle(axis * 7.0, angle).as_array(), QUATS[i])
        assert same_rotation(Quaternion.from_rotvec(rotvec).as_array(), QUATS[i])
        np.testing.assert_allclose(quat(i).to_rotvec(), rotvec, atol=1e-10)
    same = Quaternion.from_axis_angle(Vector3(0, 0, 1), 0.5)
    assert same_rotation(same.as_array(), Rotation.from_rotvec([0, 0, 0.5]).as_quat())
    with pytest.raises(ValueError, match="axis"):
        Quaternion.from_axis_angle([0, 0, 0], 1.0)
    np.testing.assert_allclose(Quaternion().to_rotvec(), 0.0)


def test_quaternion_slerp_matches_scipy():
    for i in range(N):
        pair = Rotation.from_quat([QUATS[i], QUATS2[i]])
        for alpha in (0.0, 0.1, 0.5, 0.9, 1.0):
            expected = Slerp([0.0, 1.0], pair)([alpha]).as_quat()[0]
            assert same_rotation(quat(i).slerp(quat(i, QUATS2), alpha).as_array(), expected)


def test_quaternion_slerp_takes_the_short_way_round():
    start, end = Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, -1)  # same rotation, opposite sign
    assert same_rotation(start.slerp(end, 0.5).as_array(), [0, 0, 0, 1])
    assert start.angle_to(end) == pytest.approx(0.0, abs=1e-7)


def test_quaternion_angle_to_matches_scipy_relative_rotation():
    for i in range(N):
        relative = Rotation.from_quat(QUATS[i]).inv() * Rotation.from_quat(QUATS2[i])
        assert quat(i).angle_to(quat(i, QUATS2)) == pytest.approx(relative.magnitude(), abs=1e-7)


def test_quaternion_wxyz_conversion():
    q = Quaternion(0.1, 0.2, 0.3, 0.9)
    np.testing.assert_array_equal(q.to_wxyz(), [0.9, 0.1, 0.2, 0.3])
    assert Quaternion.from_wxyz(q.to_wxyz()) == q
    np.testing.assert_array_equal(q.to_wxyz(), rotations.xyzw_to_wxyz(q.as_array()))
    assert Quaternion.from_wxyz([1, 0, 0, 0]) == Quaternion.identity()


# ── Transform / Pose vs scipy ────────────────────────────────────────────────


@pytest.mark.parametrize("make", [transform, pose])
def test_compose_matches_matrix_product_and_scipy(make):
    for i in range(N):
        a, b = make(i), make((i + 1) % N)
        composed = a @ b
        np.testing.assert_allclose(composed.to_matrix(), a.to_matrix() @ b.to_matrix(), atol=1e-12)
        assert composed == a.compose(b)
        assert type(composed) is type(a)
        expected = Rotation.from_quat(QUATS[i]) * Rotation.from_quat(QUATS[(i + 1) % N])
        rotation = composed.rotation if make is transform else composed.orientation
        assert same_rotation(rotation.as_array(), expected.as_quat())


@pytest.mark.parametrize("make", [transform, pose])
def test_compose_means_apply_other_in_our_frame(make):
    world_base = Transform.from_xyz_rpy(1, 0, 0, 0, 0, np.pi / 2)  # base is 1 m along x, turned 90°
    base_tool = Transform.from_xyz_rpy(1, 0, 0, 0, 0, 0)  # tool is 1 m along the base's x
    world_tool = world_base @ base_tool
    np.testing.assert_allclose(world_tool.translation.as_array(), [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(
        world_base.transform_point(Point(1, 0, 0)).as_array(), [1, 1, 0], atol=1e-12
    )


@pytest.mark.parametrize("make", [transform, pose])
def test_inverse_matches_matrix_inverse(make):
    for i in range(N):
        value = make(i)
        np.testing.assert_allclose(
            value.inverse().to_matrix(), np.linalg.inv(value.to_matrix()), atol=1e-10
        )
        identity = (value @ value.inverse()).to_matrix()
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-12)
        assert type(value.inverse()) is type(value)


@pytest.mark.parametrize("make", [transform, pose])
def test_transform_point_matches_scipy_and_keeps_kind(make):
    for i in range(N):
        value = make(i)
        expected = Rotation.from_quat(QUATS[i]).apply(VECS2[i]) + VECS[i]
        np.testing.assert_allclose(value.transform_point(VECS2[i]), expected, atol=1e-12)
        np.testing.assert_allclose(
            value.transform_point(Point.from_array(VECS2[i])).as_array(), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            value.transform_point(Vector3.from_array(VECS2[i])).as_array(), expected, atol=1e-12
        )
    batch = make(3).transform_point(VECS2)
    expected = Rotation.from_quat(QUATS[3]).apply(VECS2) + VECS[3]
    np.testing.assert_allclose(batch, expected, atol=1e-12)
    assert isinstance(make(3).transform_point(Point(1, 2, 3)), Point)
    assert isinstance(make(3).transform_point(Vector3(1, 2, 3)), Vector3)
    assert batch.shape == (N, 3)


@pytest.mark.parametrize("make", [transform, pose])
def test_matrix_round_trip_matches_scipy(make):
    cls = type(make(0))
    for i in range(N):
        value = make(i)
        matrix = value.to_matrix()
        np.testing.assert_allclose(
            matrix[:3, :3], Rotation.from_quat(QUATS[i]).as_matrix(), atol=1e-12
        )
        np.testing.assert_allclose(matrix[:3, 3], VECS[i], atol=1e-15)
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1])
        assert same_rotation(cls.from_matrix(matrix).as_array()[3:], QUATS[i])
        np.testing.assert_allclose(cls.from_matrix(matrix).as_array()[:3], VECS[i], atol=1e-15)
        np.testing.assert_allclose(cls.from_matrix(matrix).to_matrix(), matrix, atol=1e-12)


@pytest.mark.parametrize("cls", [Transform, Pose])
def test_from_matrix_validation(cls):
    with pytest.raises(ValueError, match="shape"):
        cls.from_matrix(np.eye(3))
    bad = np.eye(4)
    bad[3, 0] = 1.0
    with pytest.raises(ValueError, match="last row"):
        cls.from_matrix(bad)


@pytest.mark.parametrize("cls", [Transform, Pose])
def test_from_xyz_rpy_matches_scipy(cls):
    for roll, pitch, yaw in RNG.uniform(-1.4, 1.4, size=(N, 3)):
        built = cls.from_xyz_rpy(1.0, -2.0, 3.0, roll, pitch, yaw)
        expected = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        np.testing.assert_allclose(built.to_matrix()[:3, :3], expected, atol=1e-12)
        np.testing.assert_allclose(built.to_matrix()[:3, 3], [1.0, -2.0, 3.0])


@pytest.mark.parametrize("make", [transform, pose])
def test_interpolate_matches_scipy_slerp_and_lerp(make):
    for i in range(N):
        start, end = make(i), make((i + 7) % N)
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([QUATS[i], QUATS[(i + 7) % N]]))
        for alpha in (0.0, 0.25, 0.5, 1.0):
            middle = start.interpolate(end, alpha)
            array = middle.as_array()
            np.testing.assert_allclose(
                array[:3], VECS[i] + alpha * (VECS[(i + 7) % N] - VECS[i]), atol=1e-12
            )
            assert same_rotation(array[3:], slerp([alpha]).as_quat()[0])


def test_pose_transform_conversions_are_lossless():
    p = pose(5)
    t = p.as_transform()
    assert isinstance(t, Transform)
    assert t.translation.as_array().tolist() == p.position.as_array().tolist()
    assert t.rotation == p.orientation
    assert t.as_pose() == p
    assert Transform.identity().as_pose() == Pose.identity()


@pytest.mark.parametrize("cls", [Transform, Pose])
def test_pose_array_layout_and_round_trip(cls):
    value = cls.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
    array = value.as_array()
    assert array.shape == (7,)
    np.testing.assert_array_equal(array[:3], [1, 2, 3])
    np.testing.assert_array_equal(array[3:], value.as_array()[3:])
    assert cls.from_array(array) == value
    with pytest.raises(ValueError, match="shape"):
        cls.from_array(np.zeros(6))


def test_compose_rejects_other_types():
    with pytest.raises(TypeError):
        Transform().compose(Pose())
    with pytest.raises(TypeError):
        Transform() @ Pose()
    with pytest.raises(TypeError):
        Pose() @ Transform()
    with pytest.raises(TypeError):
        Pose().interpolate(Transform(), 0.5)


@pytest.mark.skipif(not has_rigid_transform, reason="scipy RigidTransform needs scipy >= 1.16")
def test_against_scipy_rigid_transform():
    from scipy.spatial.transform import RigidTransform

    for i in range(N):
        mine_a, mine_b = transform(i), transform((i + 3) % N)
        ref_a = RigidTransform.from_components(VECS[i], Rotation.from_quat(QUATS[i]))
        ref_b = RigidTransform.from_components(
            VECS[(i + 3) % N], Rotation.from_quat(QUATS[(i + 3) % N])
        )
        np.testing.assert_allclose(
            (mine_a @ mine_b).to_matrix(), (ref_a * ref_b).as_matrix(), atol=1e-12
        )
        np.testing.assert_allclose(
            mine_a.inverse().to_matrix(), ref_a.inv().as_matrix(), atol=1e-12
        )
        np.testing.assert_allclose(
            mine_a.transform_point(VECS2[i]), ref_a.apply(VECS2[i]), atol=1e-12
        )
        np.testing.assert_allclose(mine_a.to_matrix(), ref_a.as_matrix(), atol=1e-12)
        round_trip = Transform.from_matrix(ref_a.as_matrix())
        np.testing.assert_allclose(round_trip.to_matrix(), mine_a.to_matrix(), atol=1e-12)


# ── Twist / Wrench / Accel ───────────────────────────────────────────────────

SPATIAL = [(Twist, "linear", "angular"), (Wrench, "force", "torque"), (Accel, "linear", "angular")]


@pytest.mark.parametrize("cls,first,second", SPATIAL)
def test_spatial_array_layout_and_round_trip(cls, first, second):
    value = cls(Vector3(1, 2, 3), Vector3(4, 5, 6))
    np.testing.assert_array_equal(value.as_array(), [1, 2, 3, 4, 5, 6])
    assert cls.from_array(value.as_array()) == value
    assert getattr(value, first) == Vector3(1, 2, 3) and getattr(value, second) == Vector3(4, 5, 6)
    assert cls() == cls(Vector3(), Vector3())
    assert cls(**{first: Vector3(1, 0, 0)}).as_array().tolist() == [1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="shape"):
        cls.from_array(np.zeros(3))


@pytest.mark.parametrize("cls,first,second", SPATIAL)
def test_spatial_arithmetic_matches_numpy(cls, first, second):
    a_array, b_array = RNG.normal(size=6), RNG.normal(size=6)
    a, b = cls.from_array(a_array), cls.from_array(b_array)
    np.testing.assert_allclose((a + b).as_array(), a_array + b_array)
    np.testing.assert_allclose((a - b).as_array(), a_array - b_array)
    np.testing.assert_allclose((a * 3.0).as_array(), a_array * 3.0)
    np.testing.assert_allclose((3.0 * a).as_array(), a_array * 3.0)
    np.testing.assert_allclose((np.float64(3.0) * a).as_array(), a_array * 3.0)
    assert isinstance(a + b, cls)
    with pytest.raises(TypeError):
        a + 1.0
    with pytest.raises(TypeError):
        a * b


def test_spatial_types_are_distinct():
    assert Twist() != Accel() and Wrench() != Twist()
    with pytest.raises(TypeError):
        Twist() + Accel()


# ── Pose2D, stamped types ────────────────────────────────────────────────────


def test_pose2d_fields_and_array():
    p = Pose2D(1, 2, 0.5)
    assert (p.x, p.y, p.theta) == (1.0, 2.0, 0.5)
    np.testing.assert_array_equal(p.as_array(), [1, 2, 0.5])
    assert Pose2D.from_array(p.as_array()) == p
    assert Pose2D(0, 0, 7.0).theta == 7.0  # not wrapped: stored as authored
    with pytest.raises(ValueError, match="shape"):
        Pose2D.from_array([1, 2])


def test_pose_stamped_from_xyz_rpy():
    stamped = PoseStamped.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3, frame_id="map", stamp=12.5)
    assert stamped.header == Header(12.5, "map")
    assert stamped.pose == Pose.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
    assert PoseStamped.from_xyz_rpy(0, 0, 0, 0, 0, 0).header == Header()
    assert PoseStamped().pose == Pose() and PoseStamped().header == Header()


def test_transform_stamped_inverse_swaps_frames():
    stamped = TransformStamped(
        Header(3.0, "world"), "base", Transform.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
    )
    inverse = stamped.inverse()
    assert inverse.header == Header(3.0, "base")
    assert inverse.child_frame_id == "world"
    np.testing.assert_allclose(
        inverse.transform.to_matrix(), np.linalg.inv(stamped.transform.to_matrix()), atol=1e-12
    )
    assert allclose(inverse.inverse(), stamped, atol=1e-9)
    assert TransformStamped().child_frame_id == ""
    assert TransformStamped(child_frame_id=5).child_frame_id == "5"


# ── value semantics ──────────────────────────────────────────────────────────

SAMPLES = [
    Vector3(1, 2, 3),
    Point(1, 2, 3),
    Quaternion(0.1, 0.2, 0.3, 0.9),
    Pose2D(1, 2, 0.3),
    Pose(Point(1, 2, 3), Quaternion(0, 0, 0.6, 0.8)),
    Transform(Vector3(1, 2, 3), Quaternion(0, 0, 0.6, 0.8)),
    Twist(Vector3(1, 2, 3), Vector3(4, 5, 6)),
    Wrench(Vector3(1, 2, 3), Vector3(4, 5, 6)),
    Accel(Vector3(1, 2, 3), Vector3(4, 5, 6)),
    PoseStamped(Header(1.5, "map"), Pose(Point(1, 2, 3), Quaternion(0, 0, 0.6, 0.8))),
    TransformStamped(Header(1.5, "map"), "tool", Transform(Vector3(1, 2, 3))),
]
IDS = [type(sample).__name__ for sample in SAMPLES]


@pytest.mark.parametrize("sample", SAMPLES, ids=IDS)
def test_dict_round_trip_is_lossless_and_json_safe(sample):
    import json

    data = sample.to_dict()
    assert data["_type"] == f"geometry/{type(sample).__name__}"
    revived = type(sample).from_dict(json.loads(json.dumps(data)))
    assert revived == sample
    assert hash(revived) == hash(sample)
    assert type(sample).from_dict(data) == sample


@pytest.mark.parametrize("sample", SAMPLES, ids=IDS)
def test_values_are_frozen_and_compare_by_value(sample):
    field_name = dataclasses.fields(sample)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(sample, field_name, getattr(sample, field_name))
    twin = type(sample).from_dict(sample.to_dict())
    assert twin is not sample and twin == sample and hash(twin) == hash(sample)
    assert len({sample, twin}) == 1
    assert sample != Header()  # different type never equal
    assert sample.replace(**{field_name: getattr(sample, field_name)}) == sample


def test_equality_distinguishes_values_and_types():
    assert Vector3(1, 2, 3) == Vector3(1.0, 2.0, 3.0)
    assert Vector3(1, 2, 3) != Vector3(1, 2, 4)
    assert Vector3(1, 2, 3) != Point(1, 2, 3)
    assert Quaternion(0, 0, 0, 1) != Quaternion(0, 0, 0, -1)
    assert hash(Point(0.0, 0, 0)) == hash(Point(-0.0, 0, 0))


def test_replace_revalidates():
    assert Point(1, 2, 3).replace(y=9) == Point(1, 9, 3)
    with pytest.raises(ValueError, match="Point.y"):
        Point(1, 2, 3).replace(y=math.nan)
    with pytest.raises(ValueError, match="zero"):
        Quaternion().replace(w=0.0).replace(x=0.0)


def test_repr_is_compact_and_hides_defaults():
    assert repr(Vector3(1, 2, 3)) == "Vector3(x=1.0, y=2.0, z=3.0)"
    assert repr(Pose()) == "Pose()"
    assert repr(Twist(linear=Vector3(1, 0, 0))) == "Twist(linear=Vector3(x=1.0, y=0.0, z=0.0))"
    assert "child_frame_id='tool'" in repr(TransformStamped(Header(0.0, "map"), "tool"))
    # Floats print exactly, so the repr of a value rebuilds that same value.
    transform = Transform.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
    namespace = {"Transform": Transform, "Vector3": Vector3, "Quaternion": Quaternion}
    assert eval(repr(transform), namespace) == transform


def test_hashes_are_stable_for_dict_keys():
    table = {Point(1, 2, 3): "a", Pose(): "b"}
    assert table[Point(1.0, 2.0, 3.0)] == "a" and table[Pose()] == "b"


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [Vector3, Point])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["x", "y", "z"])
def test_vector_rejects_non_finite_naming_the_field(cls, bad, name):
    with pytest.raises(ValueError, match=f"{cls.__name__}.{name} must be finite"):
        cls(**{name: bad})


@pytest.mark.parametrize("name", ["x", "y", "z", "w"])
def test_quaternion_rejects_non_finite_naming_the_field(name):
    with pytest.raises(ValueError, match=f"Quaternion.{name} must be finite"):
        Quaternion(**{name: math.nan})


@pytest.mark.parametrize("name", ["x", "y", "theta"])
def test_pose2d_rejects_non_finite_naming_the_field(name):
    with pytest.raises(ValueError, match=f"Pose2D.{name} must be finite"):
        Pose2D(**{name: math.inf})


def test_quaternion_rejects_zero():
    with pytest.raises(ValueError, match="non-zero"):
        Quaternion(0, 0, 0, 0)
    with pytest.raises(ValueError, match="non-zero"):
        Quaternion.from_array([0, 0, 0, 0])
    with pytest.raises(ValueError, match="non-zero"):
        Quaternion.from_wxyz([0, 0, 0, 0])


def test_non_finite_arrays_are_rejected_on_the_way_in():
    with pytest.raises(ValueError, match="Point.x"):
        Point.from_array([math.nan, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        Transform.from_array([0, 0, 0, 0, 0, math.nan, 1])
    with pytest.raises(ValueError, match="finite"):
        Quaternion.from_euler(math.nan, 0, 0)


def test_composites_reject_wrong_field_types():
    with pytest.raises(TypeError, match="Pose.position"):
        Pose(position=Vector3())
    with pytest.raises(TypeError, match="Transform.translation"):
        Transform(translation=Point())
    with pytest.raises(TypeError, match="Twist.angular"):
        Twist(angular=np.zeros(3))
    with pytest.raises(TypeError, match="Wrench.force"):
        Wrench(force=(1, 2, 3))
    with pytest.raises(TypeError, match="PoseStamped.pose"):
        PoseStamped(pose=Transform())
    with pytest.raises(TypeError, match="TransformStamped.header"):
        TransformStamped(header="map")
    with pytest.raises(TypeError, match="Accel.linear"):
        Accel(linear=1.0)


def test_scalar_division_by_zero_does_not_produce_a_non_finite_vector():
    with pytest.raises(ZeroDivisionError):
        Vector3(1, 2, 3) / 0
    with pytest.raises(ValueError, match="finite"):
        Vector3(1e308, 0, 0) * 10


def test_package_reexports_every_type_and_the_rotation_module():
    import robot_msgs.geometry as geometry

    expected = {
        "Accel", "Point", "Pose", "Pose2D", "PoseStamped", "Quaternion", "Transform",
        "TransformStamped", "Twist", "Vector3", "Wrench", "rotations",
    }  # fmt: skip
    assert set(geometry.__all__) == expected
    assert all(hasattr(geometry, name) for name in expected)
