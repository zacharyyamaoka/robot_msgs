"""robot_msgs.compare: one trajectory measured against another as its reference.

Written before the code. Expected values are built by hand from each fixture's geometry, and
rotation errors are checked against scipy, so none of them comes from the implementation.

Sign convention (ROS 2 control_msgs/JointTrajectoryControllerState): error = reference - measured.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_msgs.compare import Comparison, ErrorStats, compare, error_stats
from robot_msgs.std import Header
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory

NAMES = ("j1", "j2")


def joint_traj(t, positions, **kwargs) -> JointTrajectory:
    return JointTrajectory(t=t, joint_names=NAMES, positions=positions, **kwargs)


def swing(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(t), 0.5 * np.cos(2.0 * t)])


# ── error statistics ─────────────────────────────────────────────────────────


def test_error_stats_by_hand():
    stats = error_stats(np.array([3.0, np.nan, -4.0]))
    assert stats.n == 2
    assert stats.rms == pytest.approx(np.sqrt(12.5))
    assert stats.mean == pytest.approx(-0.5)
    assert stats.median == pytest.approx(-0.5)
    assert stats.std == pytest.approx(3.5)
    assert (stats.min, stats.max) == (-4.0, 3.0)
    assert stats.max_abs == 4.0
    assert stats.final == -4.0  # last finite sample, sign kept


def test_error_stats_of_nothing_are_nan_not_zero():
    stats = error_stats(np.array([np.nan, np.nan]))
    assert stats.n == 0
    assert np.isnan(stats.rms) and np.isnan(stats.max_abs) and np.isnan(stats.final)


# ── joint space ──────────────────────────────────────────────────────────────


def test_joint_error_is_reference_minus_measured_on_the_measured_clock():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    measured_positions = swing(t)
    measured_positions[:, 1] += 0.01  # j2 reads 0.01 rad high
    result = compare(reference, joint_traj(t, measured_positions))

    assert isinstance(result, Comparison)
    assert isinstance(result.error, JointTrajectory)
    assert result.error.joint_names == NAMES
    np.testing.assert_array_equal(result.error.t, t)
    np.testing.assert_allclose(result.error.positions[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.error.positions[:, 1], -0.01, atol=1e-12)
    j2 = result.metrics["positions/j2"]
    assert j2.rms == pytest.approx(0.01) and j2.final == pytest.approx(-0.01)
    assert j2.std == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["positions/j1"].max_abs == pytest.approx(0.0, abs=1e-12)
    assert result.coverage == pytest.approx(1.0)
    assert result.dropped == 0


def test_joints_are_matched_by_name_not_column():
    t = np.linspace(0.0, 1.0, 11)
    reference = joint_traj(t, swing(t))
    flipped = JointTrajectory(t=t, joint_names=("j2", "j1"), positions=swing(t)[:, ::-1])
    result = compare(reference, flipped)
    np.testing.assert_allclose(result.error.positions, 0.0, atol=1e-12)


def test_missing_joint_is_an_error_naming_it():
    t = np.linspace(0.0, 1.0, 11)
    reference = joint_traj(t, swing(t))
    other = JointTrajectory(t=t, joint_names=("j1", "j9"), positions=swing(t))
    with pytest.raises(ValueError, match="j2"):
        compare(reference, other)


def test_measured_on_another_grid_is_compared_on_its_own_samples():
    t_ref = np.linspace(0.0, 2.0, 21)
    t_meas = np.linspace(0.0, 2.0, 157)  # finer, not a multiple
    line = lambda t: np.column_stack([0.3 * t, -0.2 * t])  # noqa: E731 - linear interp is exact
    result = compare(joint_traj(t_ref, line(t_ref)), joint_traj(t_meas, line(t_meas)))
    np.testing.assert_array_equal(result.error.t, t_meas)
    np.testing.assert_allclose(result.error.positions, 0.0, atol=1e-12)
    assert result.metrics["positions/j1"].n == len(t_meas)


def test_never_extrapolates_past_the_reference():
    t_ref = np.linspace(0.0, 2.0, 201)
    t_meas = np.linspace(-0.5, 2.5, 301)
    result = compare(joint_traj(t_ref, swing(t_ref)), joint_traj(t_meas, swing(t_meas)))
    assert result.error.t[0] >= 0.0 and result.error.t[-1] <= 2.0
    assert result.dropped == int(np.sum((t_meas < 0.0) | (t_meas > 2.0)))
    assert result.coverage == pytest.approx(1.0)


def test_coverage_says_how_much_of_the_reference_was_measured():
    t_ref = np.linspace(0.0, 2.0, 201)
    t_meas = np.linspace(0.0, 0.5, 51)
    result = compare(joint_traj(t_ref, swing(t_ref)), joint_traj(t_meas, swing(t_meas)))
    assert result.coverage == pytest.approx(0.25)


def test_no_overlap_is_an_error():
    t = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="overlap"):
        compare(joint_traj(t, swing(t)), joint_traj(t + 5.0, swing(t)))


def test_header_stamps_put_both_on_one_clock():
    t = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t, swing(t), header=Header(stamp=100.0))
    # Same motion, recorded with its own stamp 0.25 s later and t counted from that stamp.
    measured = joint_traj(t[:76], swing(t[25:]), header=Header(stamp=100.25))
    result = compare(reference, measured)
    np.testing.assert_allclose(result.error.positions, 0.0, atol=1e-12)


def test_time_offset_removes_a_known_latency():
    t = np.linspace(0.0, 2.0, 401)
    reference = joint_traj(t, swing(t))
    late = joint_traj(t, swing(t - 0.1))  # the measured run is 0.1 s late
    assert compare(reference, late).metrics["positions/j1"].rms > 1e-2
    shifted = compare(reference, late, time_offset_s=0.1)
    assert shifted.time_offset_s == 0.1
    np.testing.assert_allclose(shifted.error.positions, 0.0, atol=1e-6)


def test_derivatives_and_effort_are_compared_when_both_sides_have_them():
    t = np.linspace(0.0, 1.0, 11)
    ones = np.ones((11, 2))
    reference = joint_traj(t, ones, velocities=ones, effort=2.0 * ones)
    measured = joint_traj(t, ones, velocities=0.5 * ones)  # no effort recorded
    result = compare(reference, measured)
    np.testing.assert_allclose(result.error.velocities, 0.5)
    assert result.error.effort is None
    assert "velocities/j1" in result.metrics and "effort/j1" not in result.metrics


def test_shared_channels_are_compared_and_one_sided_channels_are_not():
    t = np.linspace(0.0, 1.0, 11)
    reference = joint_traj(
        t, swing(t), channels={"closure_gap": np.zeros(11), "kp": np.full((11, 2), 40.0)}
    )
    measured = joint_traj(t, swing(t), channels={"closure_gap": np.full(11, 2e-4)})
    result = compare(reference, measured)
    np.testing.assert_allclose(result.error.channels["closure_gap"], -2e-4)
    assert "kp" not in result.error.channels
    assert result.metrics["channels/closure_gap"].rms == pytest.approx(2e-4)


# ── cartesian space ──────────────────────────────────────────────────────────

SPEED = 0.2  # m/s along x
LAG_S = 0.05
OFFSET_Y = 0.004
YAW = 0.1  # rad


def x_route(t: np.ndarray) -> np.ndarray:
    """Rest 0.5 s, cruise along +x for 2 s, rest."""
    x = SPEED * np.clip(t - 0.5, 0.0, 2.0)
    return np.column_stack([x, np.zeros_like(t), np.zeros_like(t)])


def pose_traj(t, positions, yaw=0.0, frame="L2 tip") -> PoseTrajectory:
    quats = np.tile(Rotation.from_euler("z", yaw).as_quat(), (len(t), 1))  # xyzw
    return PoseTrajectory(t=t, positions=positions, orientations=quats, child_frame_id=frame)


def lagging_pair():
    t = np.linspace(0.0, 3.0, 3001)
    reference = pose_traj(t, x_route(t))
    measured = pose_traj(t, x_route(t - LAG_S) + [0.0, OFFSET_Y, 0.0], yaw=YAW)
    return t, reference, measured


def test_pose_error_axes_and_magnitudes():
    t, reference, measured = lagging_pair()
    result = compare(reference, measured)
    assert isinstance(result.error, PoseTrajectory)
    assert result.error.child_frame_id == "L2 tip"
    cruise = (t > 0.6) & (t < 2.45)
    np.testing.assert_allclose(result.error.positions[cruise, 0], SPEED * LAG_S, atol=1e-9)
    np.testing.assert_allclose(result.error.positions[:, 1], -OFFSET_Y, atol=1e-12)
    rotvec = result.error.channels["rotation_vector"]
    np.testing.assert_allclose(rotvec, [[0.0, 0.0, -YAW]] * len(t), atol=1e-12)
    assert result.metrics["rotation"].rms == pytest.approx(YAW)
    assert result.metrics["rz"].final == pytest.approx(-YAW)
    assert result.metrics["position"].final == pytest.approx(OFFSET_Y)
    assert result.metrics["position"].max_abs == pytest.approx(np.hypot(SPEED * LAG_S, OFFSET_Y))


def test_lag_and_cross_track_split_timing_from_path():
    t, reference, measured = lagging_pair()
    result = compare(reference, measured)
    lag = result.error.channels["lag"]
    cross = result.error.channels["cross_track"]
    cruise = (t > 0.6) & (t < 2.45)
    np.testing.assert_allclose(lag[cruise], SPEED * LAG_S, atol=1e-9)  # + = behind
    np.testing.assert_allclose(cross[cruise], OFFSET_Y, atol=1e-9)
    resting = (t < 0.45) | (t > 2.55)
    assert np.isnan(lag[resting]).all() and np.isnan(cross[resting]).all()
    # Median, not mean: the 50 ms ramp where the reference has started and the run has not
    # is real (smaller) lag, so only the cruise value is exact.
    assert result.metrics["lag"].median == pytest.approx(SPEED * LAG_S, rel=1e-6)


def test_rotation_error_matches_scipy():
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 1.0, 200)
    ref_rot = Rotation.random(200, random_state=1)
    meas_rot = Rotation.from_rotvec(rng.normal(scale=0.8, size=(200, 3))) * ref_rot
    positions = np.zeros((200, 3))
    reference = PoseTrajectory(t=t, positions=positions, orientations=ref_rot.as_quat())
    measured = PoseTrajectory(t=t, positions=positions, orientations=meas_rot.as_quat())
    result = compare(reference, measured)
    expected = (
        ref_rot * meas_rot.inv()
    ).as_rotvec()  # world-frame rotation taking measured to reference
    np.testing.assert_allclose(result.error.channels["rotation_vector"], expected, atol=1e-9)
    np.testing.assert_allclose(
        result.error.channels["rotation_angle"], np.linalg.norm(expected, axis=1), atol=1e-9
    )


# ── dispatch, method, serialisation ──────────────────────────────────────────


def test_method_is_the_function():
    t = np.linspace(0.0, 1.0, 11)
    reference, measured = joint_traj(t, swing(t)), joint_traj(t, 1.1 * swing(t))
    assert reference.compare(measured) == compare(reference, measured)


def test_mixing_kinds_is_a_type_error():
    t = np.linspace(0.0, 1.0, 11)
    with pytest.raises(TypeError):
        compare(joint_traj(t, swing(t)), pose_traj(t, np.zeros((11, 3))))


def test_plain_trajectories_compare_their_channels():
    t = np.linspace(0.0, 1.0, 11)
    reference = Trajectory(t=t, channels={"temp_c": np.full(11, 30.0)})
    measured = Trajectory(t=t, channels={"temp_c": np.full(11, 31.5)})
    result = compare(reference, measured)
    assert type(result.error) is Trajectory
    assert result.metrics["channels/temp_c"].mean == pytest.approx(-1.5)


def test_comparison_is_a_message_that_round_trips(tmp_path):
    from robot_msgs._core import load_npz, save_npz

    _, reference, measured = lagging_pair()
    result = compare(reference, measured)
    assert isinstance(result.metrics["position"], ErrorStats)
    save_npz(result, tmp_path / "comparison.npz")
    assert load_npz(tmp_path / "comparison.npz") == result
    rows = result.rows()
    assert {"metric": "position", **result.metrics["position"].as_dict()} in rows
