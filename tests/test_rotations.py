"""Rotation kernel vs scipy, the independent oracle (scipy is test-only)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from robot_msgs.geometry import rotations as rot

RNG = np.random.default_rng(7)
Q = Rotation.random(200, random_state=7).as_quat()  # scipy is xyzw too
Q2 = Rotation.random(200, random_state=8).as_quat()


def same_rotation(a, b, atol=1e-10):
    return np.allclose(np.abs(np.sum(a * b, axis=-1)), 1.0, atol=atol)


def test_multiply_matches_scipy_composition():
    expected = (Rotation.from_quat(Q) * Rotation.from_quat(Q2)).as_quat()
    assert same_rotation(rot.quat_multiply(Q, Q2), expected)


def test_rotate_matches_scipy_apply():
    v = RNG.normal(size=(200, 3))
    np.testing.assert_allclose(rot.quat_rotate(Q, v), Rotation.from_quat(Q).apply(v), atol=1e-12)


def test_matrix_round_trip_matches_scipy():
    m = rot.quat_to_matrix(Q)
    np.testing.assert_allclose(m, Rotation.from_quat(Q).as_matrix(), atol=1e-12)
    assert same_rotation(rot.quat_from_matrix(m), Q)


def test_from_matrix_handles_every_shepperd_branch():
    # 180° turns about each axis force the x, y and z branches; identity forces w.
    cases = Rotation.from_rotvec(np.pi * np.eye(3)).as_matrix()
    cases = np.concatenate([cases, np.eye(3)[None]])
    assert same_rotation(rot.quat_from_matrix(cases), Rotation.from_matrix(cases).as_quat())


def test_euler_is_ros_rpy_fixed_xyz():
    rpy = RNG.uniform(-np.pi, np.pi, size=(200, 3))
    rpy[:, 1] = RNG.uniform(-1.5, 1.5, size=200)  # stay off gimbal lock
    expected = Rotation.from_euler("xyz", rpy).as_quat()  # lowercase = extrinsic = ROS RPY
    q = rot.quat_from_euler(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    assert same_rotation(q, expected)
    np.testing.assert_allclose(rot.quat_to_euler(q), rpy, atol=1e-9)


def test_rotvec_round_trip_matches_scipy_including_tiny_angles():
    v = np.concatenate([RNG.normal(size=(100, 3)), 1e-10 * RNG.normal(size=(5, 3))])
    q = rot.quat_from_rotvec(v)
    assert same_rotation(q, Rotation.from_rotvec(v).as_quat())
    np.testing.assert_allclose(rot.quat_to_rotvec(q), Rotation.from_quat(q).as_rotvec(), atol=1e-12)


def test_slerp_matches_scipy():
    alpha = RNG.uniform(size=50)
    for i in range(50):
        pair = Rotation.from_quat(np.stack([Q[i], Q2[i]]))
        expected = Slerp([0.0, 1.0], pair)(alpha[i]).as_quat()
        assert same_rotation(rot.quat_slerp(Q[i], Q2[i], alpha[i]), expected, atol=1e-9)


def test_slerp_endpoints_are_exact_rotations():
    assert same_rotation(rot.quat_slerp(Q, Q2, np.zeros(200)), Q)
    assert same_rotation(rot.quat_slerp(Q, Q2, np.ones(200)), Q2)


def test_angle_matches_scipy_magnitude():
    expected = (Rotation.from_quat(Q).inv() * Rotation.from_quat(Q2)).magnitude()
    np.testing.assert_allclose(rot.quat_angle(Q, Q2), expected, atol=1e-7)


def test_wxyz_conversion_is_explicit_and_invertible():
    wxyz = rot.xyzw_to_wxyz(Q)
    np.testing.assert_array_equal(wxyz[:, 0], Q[:, 3])
    np.testing.assert_array_equal(rot.wxyz_to_xyzw(wxyz), Q)


def test_pose_matrix_round_trip():
    p = RNG.normal(size=(20, 3))
    m = rot.pose_to_matrix(p, Q[:20])
    p2, q2 = rot.matrix_to_pose(m)
    np.testing.assert_allclose(p2, p)
    assert same_rotation(q2, Q[:20])


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        rot.quat_normalize(np.zeros(4))


def test_quat_slerp_is_exact_for_small_angles_where_nlerp_was_used():
    """Regression (2026-11-11): under ~1.8 deg (dot > 0.9995) slerp fell back to a normalised lerp,
    up to 8e-4 mrad off true slerp, which failed data-ui's trajectory-contract fixture."""
    from scipy.spatial.transform import Rotation, Slerp

    from robot_msgs.geometry.rotations import quat_slerp

    rng = np.random.default_rng(11)
    for angle in (1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.03, 0.5, 3.0):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        start = Rotation.random(random_state=int(angle * 1e6) % 97)
        end = Rotation.from_rotvec(axis * angle) * start
        alphas = np.linspace(0.0, 1.0, 7)
        expected = Slerp([0.0, 1.0], Rotation.concatenate([start, end]))(alphas)
        ours = quat_slerp(
            np.broadcast_to(start.as_quat(), (7, 4)), np.broadcast_to(end.as_quat(), (7, 4)), alphas
        )
        error = (Rotation.from_quat(ours) * expected.inv()).magnitude()
        assert np.max(error) < 1e-12, (angle, np.max(error))
