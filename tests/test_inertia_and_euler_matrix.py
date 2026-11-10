"""Inertia (geometry_msgs/Inertia) and the direct ROS-RPY <-> rotation-matrix pair.

Gaps found when bam_robot_descriptions adopted robot_msgs (bam_ws, 2026-11-10): a URDF
`<inertial>` is ROS's Inertia plus an origin rotation, and a URDF `rpy` is printed verbatim,
so turning a matrix into RPY must not detour through a quaternion (that changes the last
digits of every decal/scene frame the description writes). scipy is the independent oracle.
"""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_msgs.geometry import Inertia, Quaternion, Vector3
from robot_msgs.geometry import rotations as rot

RNG = np.random.default_rng(11)
RPY = np.column_stack(
    [
        RNG.uniform(-np.pi, np.pi, 300),
        RNG.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 300),
        RNG.uniform(-np.pi, np.pi, 300),
    ]
)


# ── euler_to_matrix / matrix_to_euler ──────────────────────────────────────


def test_euler_to_matrix_is_ros_fixed_xyz():
    expected = Rotation.from_euler("xyz", RPY).as_matrix()  # lower case = extrinsic = ROS RPY
    np.testing.assert_allclose(rot.euler_to_matrix(RPY[:, 0], RPY[:, 1], RPY[:, 2]), expected,
                               atol=1e-14)


def test_euler_to_matrix_agrees_with_the_quaternion_path():
    via_quat = rot.quat_to_matrix(rot.quat_from_euler(RPY[:, 0], RPY[:, 1], RPY[:, 2]))
    np.testing.assert_allclose(rot.euler_to_matrix(*RPY.T), via_quat, atol=1e-14)


def test_matrix_to_euler_inverts_euler_to_matrix_away_from_gimbal():
    np.testing.assert_allclose(rot.matrix_to_euler(rot.euler_to_matrix(*RPY.T)), RPY, atol=1e-12)


def test_matrix_to_euler_matches_scipy():
    matrices = Rotation.from_euler("xyz", RPY).as_matrix()
    np.testing.assert_allclose(rot.matrix_to_euler(matrices),
                               Rotation.from_matrix(matrices).as_euler("xyz"), atol=1e-12)


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2])
def test_matrix_to_euler_at_gimbal_lock_puts_everything_in_roll(pitch):
    """At pitch = ±π/2 only roll ∓ yaw is defined; the convention is yaw = 0 (tf's)."""
    matrix = rot.euler_to_matrix(0.4, pitch, -0.3)
    roll, recovered_pitch, yaw = rot.matrix_to_euler(matrix)
    assert yaw == 0.0
    assert recovered_pitch == pytest.approx(pitch)
    np.testing.assert_allclose(rot.euler_to_matrix(roll, recovered_pitch, yaw), matrix, atol=1e-12)


def test_scalar_inputs_give_unbatched_shapes():
    assert rot.euler_to_matrix(0.1, 0.2, 0.3).shape == (3, 3)
    assert rot.matrix_to_euler(np.eye(3)).shape == (3,)
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
        rot.matrix_to_euler(np.eye(4))


def test_matrix_to_euler_is_exact_on_axis_aligned_frames():
    """A decal on a box face is axis-aligned: its RPY must come out as exact multiples of π/2."""
    flip_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert tuple(rot.matrix_to_euler(flip_x)) == (math.pi, -0.0, 0.0)


def test_quaternion_methods_wrap_them():
    q = Quaternion.from_euler(0.1, -0.2, 0.3)
    np.testing.assert_allclose(Quaternion.from_matrix(rot.euler_to_matrix(0.1, -0.2, 0.3)).as_array(),
                               q.as_array(), atol=1e-14)


# ── Inertia ────────────────────────────────────────────────────────────────


def test_inertia_has_ros_fields_and_defaults():
    inertia = Inertia(2.0, Vector3(0.0, 0.0, 0.1), ixx=0.3, iyy=0.4, izz=0.5)
    assert (inertia.m, inertia.com, inertia.ixy, inertia.ixz, inertia.iyz) == (
        2.0, Vector3(0.0, 0.0, 0.1), 0.0, 0.0, 0.0)
    assert Inertia() == Inertia(0.0, Vector3(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_inertia_matrix_is_the_symmetric_tensor():
    inertia = Inertia(1.0, Vector3(), 0.3, 0.01, -0.02, 0.4, 0.03, 0.5)
    expected = np.array([[0.3, 0.01, -0.02], [0.01, 0.4, 0.03], [-0.02, 0.03, 0.5]])
    np.testing.assert_array_equal(inertia.to_matrix(), expected)
    assert Inertia.from_matrix(1.0, Vector3(), expected) == inertia


def test_inertia_from_matrix_refuses_asymmetric_tensors():
    with pytest.raises(ValueError, match="symmetric"):
        Inertia.from_matrix(1.0, Vector3(), [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])


def test_inertia_validates_at_construction():
    with pytest.raises(ValueError, match="Inertia.m"):
        Inertia(-1.0)
    with pytest.raises(ValueError, match="finite"):
        Inertia(1.0, Vector3(), ixx=float("nan"))
    with pytest.raises(TypeError, match="Inertia.com"):
        Inertia(1.0, (0.0, 0.0, 0.0))


def test_inertia_rotated_matches_r_i_rt():
    """`rotated(q)` re-expresses the tensor in a frame turned by q (I' = R I Rᵀ, com rotated)."""
    inertia = Inertia(1.5, Vector3(0.1, 0.0, 0.0), 0.3, 0.01, -0.02, 0.4, 0.03, 0.5)
    q = Quaternion.from_euler(0.3, -0.5, 1.1)
    R = q.to_matrix()
    turned = inertia.rotated(q)
    np.testing.assert_allclose(turned.to_matrix(), R @ inertia.to_matrix() @ R.T, atol=1e-15)
    np.testing.assert_allclose(turned.com.as_array(), R @ inertia.com.as_array(), atol=1e-15)
    assert turned.m == inertia.m


def test_inertia_is_physical_checks_triangle_inequality_and_definiteness():
    assert Inertia(1.0, Vector3(), 1.0, 0.0, 0.0, 1.0, 0.0, 1.0).is_physical()
    assert not Inertia(1.0, Vector3(), 1.0, 0.0, 0.0, 1.0, 0.0, 3.0).is_physical()  # 1+1 < 3
    assert not Inertia(1.0, Vector3(), 1.0, 2.0, 0.0, 1.0, 0.0, 1.0).is_physical()  # not PSD
    assert not Inertia(0.0, Vector3(), 1.0, 0.0, 0.0, 1.0, 0.0, 1.0).is_physical()  # massless


def test_inertia_as_array_round_trips():
    inertia = Inertia(1.5, Vector3(0.1, 0.2, 0.3), 0.3, 0.01, -0.02, 0.4, 0.03, 0.5)
    np.testing.assert_array_equal(inertia.as_array(),
                                  [1.5, 0.1, 0.2, 0.3, 0.3, 0.01, -0.02, 0.4, 0.03, 0.5])
    assert Inertia.from_array(inertia.as_array()) == inertia
