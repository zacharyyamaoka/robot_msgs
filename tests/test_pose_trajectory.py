"""PoseTrajectory: positions + xyzw orientations over time, slerped, with frame changes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from robot_msgs.geometry import Point, Pose, Quaternion, Transform
from robot_msgs.trajectory import PoseTrajectory


def spin(n=101, rate=0.5, radius=0.2):
    """The tool circles the base at `radius` while yawing at `rate` rad/s."""
    t = np.linspace(0.0, 4.0, n)
    angle = rate * t
    positions = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.full(n, 0.3)], 1)
    orientations = Rotation.from_euler("z", angle[:, None]).as_quat()
    return PoseTrajectory(t=t, positions=positions, orientations=orientations, child_frame_id="tool0")


def test_shapes_and_normalisation():
    with pytest.raises(ValueError, match=r"expected \(2, 4\)"):
        PoseTrajectory(t=[0, 1], positions=np.zeros((2, 3)), orientations=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="zero quaternion"):
        PoseTrajectory(t=[0], positions=np.zeros((1, 3)), orientations=np.zeros((1, 4)))
    traj = PoseTrajectory(t=[0], positions=np.zeros((1, 3)), orientations=[[0, 0, 0, 2.0]])
    assert traj.orientations.tolist() == [[0, 0, 0, 1.0]]


def test_points_are_poses_and_from_poses_round_trips():
    traj = spin()
    pose = traj[7].pose
    assert isinstance(pose, Pose)
    np.testing.assert_allclose(pose.position.as_array(), traj.positions[7])
    rebuilt = PoseTrajectory.from_poses(traj.t, traj.poses(), child_frame_id="tool0")
    assert rebuilt == traj


def test_orientation_is_slerped_like_scipy():
    traj = spin(n=5)
    times = np.linspace(0.0, 4.0, 37)
    expected = Slerp(traj.t, Rotation.from_quat(traj.orientations))(times).as_quat()
    got = traj.resample(times).orientations
    np.testing.assert_allclose(np.abs(np.sum(got * expected, axis=1)), 1.0, atol=1e-10)


def test_transformed_matches_matrix_composition():
    traj = spin()
    offset = Transform.from_xyz_rpy(1.0, -0.5, 0.2, 0.1, 0.2, 0.3)
    moved = traj.transformed(offset, frame_id="world")
    np.testing.assert_allclose(moved.matrices(), offset.to_matrix() @ traj.matrices(), atol=1e-12)
    assert moved.header.frame_id == "world"


def test_derivatives_recover_the_circle_and_the_yaw_rate():
    traj = spin(n=4001).with_derivatives()
    np.testing.assert_allclose(traj.angular_velocities, [[0.0, 0.0, 0.5]] * 4001, atol=1e-9)
    speed = np.linalg.norm(traj.linear_velocities, axis=1)
    np.testing.assert_allclose(speed, 0.5 * 0.2, rtol=1e-4)


def test_path_length_and_matrices_round_trip():
    traj = spin(n=2001)
    assert traj.path_length() == pytest.approx(0.5 * 4.0 * 0.2, rel=1e-6)
    again = PoseTrajectory.from_matrices(traj.t, traj.matrices(), child_frame_id="tool0")
    np.testing.assert_allclose(again.positions, traj.positions, atol=1e-12)


def test_pose_at_uses_geometry_types():
    pose = spin().pose_at(1.0)
    assert isinstance(pose.position, Point) and isinstance(pose.orientation, Quaternion)
    assert pose.orientation.to_euler()[2] == pytest.approx(0.5, abs=1e-6)
