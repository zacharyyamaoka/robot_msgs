"""JointTrajectory: the ROS shape, stored as (N, J) arrays, with the helpers ROS never had."""

import numpy as np
import pytest

from robot_msgs.sensor import JointState
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, JointTrajectoryPoint

NAMES = ("shoulder", "elbow", "wrist")


def sine(n=201, duration=2.0):
    t = np.linspace(0.0, duration, n)
    phase = np.array([0.0, 1.0, 2.0])
    q = np.sin(t[:, None] + phase)
    return JointTrajectory(
        t=t,
        joint_names=NAMES,
        positions=q,
        velocities=np.cos(t[:, None] + phase),
        header=Header(stamp=50.0, frame_id="base_link"),
    )


def test_shapes_are_checked_against_joint_names():
    with pytest.raises(ValueError, match=r"expected \(2, 3\)"):
        JointTrajectory(t=[0, 1], joint_names=NAMES, positions=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unique"):
        JointTrajectory(t=[0], joint_names=("a", "a"), positions=np.zeros((1, 2)))


def test_point_rows_are_views_of_the_batch_arrays():
    traj = sine()
    point = traj[10]
    assert isinstance(point, JointTrajectoryPoint)
    assert np.shares_memory(point.positions, traj.positions)
    np.testing.assert_array_equal(point.positions, traj.positions[10])
    assert point.accelerations is None


def test_joint_and_select_keep_the_type_and_the_names():
    traj = sine()
    elbow = traj.joint("elbow")
    assert type(elbow) is JointTrajectory and elbow.joint_names == ("elbow",)
    np.testing.assert_array_equal(elbow.positions[:, 0], traj.positions[:, 1])
    assert traj.select(["wrist", "shoulder"]).joint_names == ("wrist", "shoulder")
    with pytest.raises(KeyError, match="no joint 'knee'"):
        traj.joint("knee")


def test_finite_difference_derivatives_match_the_analytic_ones():
    traj = sine(n=2001).replace(velocities=None)
    filled = traj.with_derivatives()
    t = traj.t[:, None] + np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(filled.velocities, np.cos(t), atol=1e-5)
    # Differentiating twice loses an order of accuracy at the two ends; the interior is tight.
    np.testing.assert_allclose(filled.accelerations[2:-2], -np.sin(t)[2:-2], atol=1e-6)
    np.testing.assert_allclose(filled.accelerations, -np.sin(t), atol=2e-3)


def test_time_scaling_slows_the_same_path_down():
    traj = sine().replace(effort=np.ones((201, 3)))
    slow = traj.time_scaled(2.0)
    assert slow.duration == pytest.approx(2 * traj.duration)
    np.testing.assert_array_equal(slow.positions, traj.positions)
    np.testing.assert_allclose(slow.velocities, traj.velocities / 2)
    assert slow.effort is None


def test_joint_state_at_puts_the_command_on_the_header_clock():
    traj = sine()
    state = traj.joint_state_at(1.0)
    assert isinstance(state, JointState)
    assert state.name == NAMES and state.header.stamp == 51.0
    np.testing.assert_allclose(state.position, np.sin(1.0 + np.array([0.0, 1.0, 2.0])), atol=1e-4)


def test_a_log_of_joint_states_becomes_a_trajectory():
    traj = sine(n=21)
    log = [traj.joint_state_at(t) for t in traj.t]
    recorded = JointTrajectory.from_joint_states(log)
    np.testing.assert_allclose(recorded.t, traj.t, atol=1e-9)
    np.testing.assert_allclose(recorded.positions, traj.positions)
    assert recorded.header.stamp == 50.0 and recorded.velocities is not None


def test_limits_are_reordered_to_the_joints_and_violations_are_named():
    limits = JointLimits(
        joint_names=("wrist", "elbow", "shoulder"),
        max_velocity=[10.0, 0.5, 10.0],
        max_position=[2.0, 2.0, 2.0],
    )
    traj = sine().replace(limits=limits)
    assert traj.limits.joint_names == NAMES
    problems = traj.limit_violations()
    assert len(problems) == 1 and problems[0] == "elbow: |velocity| reaches 0.99 > 0.5 at t=2 s"
    assert sine().replace(limits=limits.replace(max_velocity=[2.0] * 3)).limit_violations() == ()
    with pytest.raises(ValueError, match="no entry for joints"):
        sine().replace(limits=JointLimits(joint_names=("elbow",), max_velocity=[1.0]))


def test_extra_information_rides_along_as_channels():
    """sim2real's measured-run fields fit without a new type: the rest are channels."""
    n = 5
    traj = JointTrajectory(
        t=np.arange(n) * 0.01,
        joint_names=("j2",),
        positions=np.zeros((n, 1)),
        channels={
            "q_target": np.zeros((n, 1)),
            "i_q": np.linspace(0, 1, n)[:, None],
            "mode": np.array([0, 0, 1, 1, 1]),
            "temp_c": np.full(n, 31.5),
        },
    )
    mid = traj.at(0.015)
    assert mid["mode"] == 0 and mid["i_q"][0] == pytest.approx(0.375)
    assert traj[1:3].channels["mode"].tolist() == [0, 1]


def test_waypoints_are_just_a_sparse_trajectory_with_names():
    """bam_traj_gen's t_waypoint/q_waypoint/waypoint_names, without special fields."""
    waypoints = JointTrajectory(
        t=[0.0, 1.2, 2.0],
        joint_names=NAMES,
        positions=np.zeros((3, 3)),
        channels={"name": np.array(["home", "pregrasp", "grasp"])},
    )
    assert waypoints.at(1.5)["name"] == "pregrasp"
    assert waypoints.window(1.0, 3.0).channels["name"].tolist() == ["pregrasp", "grasp"]


def test_round_trips_with_limits():
    traj = sine().replace(limits=JointLimits(joint_names=NAMES, max_jerk=[100.0] * 3))
    assert JointTrajectory.from_dict(traj.to_dict()) == traj
    assert JointTrajectory.from_points(traj.to_points(), joint_names=NAMES,
                                       header=traj.header, limits=traj.limits) == traj
