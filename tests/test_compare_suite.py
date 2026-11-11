"""robot_msgs.compare, the second wave: verdicts, N-run statistics, timing, path shape, drift.

Written before the code (2026-11-10), from the prior-art sweep's "start doing" list:
tolerances + verdict (ROS 2 joint_trajectory_controller / control_msgs), accuracy vs
repeatability across runs (ISO 9283), a latency estimate (trajectopy's time shift), exact
contour error (CNC contouring), settling / overshoot / goal error (MATLAB stepinfo), relative
error over windows (evo RPE), discrete Fréchet, IAE / ISE / ITAE, and the time-base channels.

Every expected value is built by hand from the fixture's geometry or a closed-form integral,
never from the implementation. Sign convention stays error = reference - measured.
"""

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_msgs.compare import (
    GOAL_TOLERANCE_VIOLATED,
    PATH_TOLERANCE_VIOLATED,
    SUCCESSFUL,
    Aggregate,
    GoalStats,
    JointTolerance,
    Outcome,
    PoseTolerance,
    TimeOffsetEstimate,
    ToleranceMonitor,
    Tolerances,
    Violation,
    aggregate,
    check,
    compare,
    error_stats,
    estimate_time_offset,
    frechet,
    goal,
    relative_error,
)
from robot_msgs.trajectory import JointTrajectory, JointTrajectoryPoint, PoseTrajectory

NAMES = ("j1", "j2")


def joint_traj(t, positions, velocities=None, **kwargs) -> JointTrajectory:
    return JointTrajectory(
        t=t, joint_names=NAMES, positions=positions, velocities=velocities, **kwargs
    )


def swing(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(t), 0.5 * np.cos(2.0 * t)])


def swing_velocity(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.cos(t), -np.sin(2.0 * t)])


def pose_traj(t, positions, yaw=None, frame="TCP") -> PoseTrajectory:
    yaw = np.zeros(len(t)) if yaw is None else np.broadcast_to(yaw, (len(t),))
    quats = Rotation.from_euler("z", yaw[:, None]).as_quat()  # xyzw; (N, 1) for scipy >= 1.17
    return PoseTrajectory(t=t, positions=positions, orientations=quats, child_frame_id=frame)


def circle(t: np.ndarray, radius: float = 0.2, omega: float = 1.0) -> np.ndarray:
    return np.column_stack(
        [radius * np.cos(omega * t), radius * np.sin(omega * t), np.zeros_like(t)]
    )


# ── 1. tolerances + verdict (ROS 2 JTC vocabulary) ───────────────────────────


def test_ros_result_codes_are_the_follow_joint_trajectory_values():
    # control_msgs/action/FollowJointTrajectory Result
    assert (SUCCESSFUL, PATH_TOLERANCE_VIOLATED, GOAL_TOLERANCE_VIOLATED) == (0, -4, -5)


def test_within_tolerance_is_successful_with_margins():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    measured_positions = swing(t)
    measured_positions[:, 1] += 0.01  # j2 reads 0.01 rad high everywhere
    tolerances = Tolerances(
        path=(JointTolerance(name="j1", position=0.05), JointTolerance(name="j2", position=0.05)),
        goal=(JointTolerance(name="j1", position=0.02), JointTolerance(name="j2", position=0.02)),
    )
    outcome = check(reference, joint_traj(t, measured_positions), tolerances)

    assert isinstance(outcome, Outcome)
    assert outcome.ok and outcome.error_code == SUCCESSFUL
    assert outcome.code_name == "SUCCESSFUL"
    assert outcome.violations == ()
    # margin = limit - worst |error| over the phase
    assert outcome.margins["path/positions/j2"] == pytest.approx(0.04)
    assert outcome.margins["path/positions/j1"] == pytest.approx(0.05)
    assert outcome.margins["goal/positions/j2"] == pytest.approx(0.01)
    assert outcome.goal_time == pytest.approx(0.0)  # within goal tolerance at the end instant


def test_path_violation_names_the_joint_the_time_and_the_margin():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    measured_positions = swing(t)
    measured_positions[120:131, 0] -= 0.08  # j1 dips 0.08 rad low over t = 1.20 .. 1.30 s
    tolerances = Tolerances(path=(JointTolerance(name="j1", position=0.05),))
    outcome = check(reference, joint_traj(t, measured_positions), tolerances)

    assert not outcome.ok
    assert outcome.error_code == PATH_TOLERANCE_VIOLATED
    assert outcome.code_name == "PATH_TOLERANCE_VIOLATED"
    first = outcome.violations[0]
    assert isinstance(first, Violation)
    assert (first.phase, first.metric) == ("path", "positions/j1")
    assert first.t == pytest.approx(1.2)  # the first offending sample, like JTC's abort
    assert first.error == pytest.approx(0.08)  # reference - measured, signed
    assert first.limit == pytest.approx(0.05)
    assert outcome.margins["path/positions/j1"] == pytest.approx(-0.03)
    assert "j1" in outcome.error_string


def test_zero_and_minus_one_mean_not_checked():
    t = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t, swing(t))
    measured = joint_traj(t, swing(t) + 1.0)  # wildly off
    tolerances = Tolerances(
        path=(JointTolerance(name="j1", position=0.0), JointTolerance(name="j2", position=-1.0))
    )
    assert check(reference, measured, tolerances).ok


def test_velocity_tolerance_without_velocities_is_an_error_not_a_pass():
    t = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t, swing(t))
    tolerances = Tolerances(path=(JointTolerance(name="j1", velocity=0.1),))
    with pytest.raises(ValueError, match="velocit"):
        check(reference, joint_traj(t, swing(t)), tolerances)


def lagging_settle(goal_time_tolerance: float, stopped_velocity_tolerance: float = 0.01):
    """Reference ramps j1 0 -> 1 rad over [0, 1] s, then the run settles exponentially after it.

    measured j1(t) = 1 - 0.1 * exp(-(t - 1) / 0.1) for t >= 1, so |error| = 0.1 exp(-(t-1)/0.1):
    it drops below 0.01 rad at t - 1 = 0.1 ln 10 = 0.2303 s.
    """
    t_ref = np.linspace(0.0, 1.0, 101)
    ramp = np.column_stack([t_ref, np.zeros_like(t_ref)])
    reference = joint_traj(t_ref, ramp, velocities=np.column_stack([np.ones(101), np.zeros(101)]))
    t_meas = np.linspace(0.0, 2.0, 2001)
    j1 = np.where(t_meas < 1.0, t_meas - 0.1, 1.0 - 0.1 * np.exp(-(t_meas - 1.0) / 0.1))
    j1_velocity = np.where(t_meas < 1.0, 1.0, np.exp(-(t_meas - 1.0) / 0.1))
    measured = joint_traj(
        t_meas,
        np.column_stack([j1, np.zeros_like(t_meas)]),
        velocities=np.column_stack([j1_velocity, np.zeros_like(t_meas)]),
    )
    tolerances = Tolerances(
        goal=(JointTolerance(name="j1", position=0.01),),
        goal_time_tolerance=goal_time_tolerance,
        stopped_velocity_tolerance=stopped_velocity_tolerance,
    )
    return reference, measured, tolerances


def test_goal_is_judged_after_the_reference_ends_against_its_held_final_point():
    reference, measured, tolerances = lagging_settle(goal_time_tolerance=0.5, stopped_velocity_tolerance=1.0)
    outcome = check(reference, measured, tolerances)
    assert outcome.ok
    # first sample with |error| <= 0.01: 0.1 * ln(10) = 0.2303 s after the end (1 ms grid)
    assert outcome.goal_time == pytest.approx(0.1 * np.log(10.0), abs=1.5e-3)


def test_goal_time_tolerance_too_short_is_goal_tolerance_violated():
    reference, measured, tolerances = lagging_settle(goal_time_tolerance=0.1, stopped_velocity_tolerance=1.0)
    outcome = check(reference, measured, tolerances)
    assert outcome.error_code == GOAL_TOLERANCE_VIOLATED
    assert outcome.violations[0].phase == "goal"
    assert outcome.violations[0].metric == "positions/j1"
    assert np.isnan(outcome.goal_time)


def test_stopped_velocity_tolerance_is_the_default_goal_velocity_limit():
    # JTC: stopped_velocity_tolerance is the default goal velocity tolerance. |qd| = exp(-(t-1)/0.1)
    # falls below 0.01 rad/s only at 0.1 ln 100 = 0.4605 s, later than the position (0.2303 s).
    reference, measured, tolerances = lagging_settle(goal_time_tolerance=0.0, stopped_velocity_tolerance=0.01)
    outcome = check(reference, measured, tolerances)
    assert outcome.ok
    assert outcome.goal_time == pytest.approx(0.1 * np.log(100.0), abs=1.5e-3)


def test_goal_never_observed_is_not_a_pass():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    short = joint_traj(t[:120], swing(t[:120]))  # the run stops recording at 1.19 s
    tolerances = Tolerances(goal=(JointTolerance(name="j1", position=0.01),))
    outcome = check(reference, short, tolerances)
    assert outcome.error_code == GOAL_TOLERANCE_VIOLATED
    assert "end" in outcome.error_string  # says the data ended before the goal was observed


def test_pose_tolerances_check_position_and_rotation_magnitudes():
    t = np.linspace(0.0, 2.0, 201)
    route = np.column_stack([0.1 * t, np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, route)
    measured = pose_traj(t, route + np.array([0.0, 0.002, 0.0]), yaw=0.03)
    loose = Tolerances(path=PoseTolerance(position=0.005, rotation=0.05))
    tight = Tolerances(path=PoseTolerance(position=0.005, rotation=0.02))
    assert check(reference, measured, loose).ok
    outcome = check(reference, measured, tight)
    assert outcome.error_code == PATH_TOLERANCE_VIOLATED
    assert outcome.violations[0].metric == "rotation"
    assert outcome.violations[0].error == pytest.approx(0.03)
    assert outcome.margins["path/position"] == pytest.approx(0.003)


def test_online_monitor_gives_the_same_outcome_as_the_offline_check():
    """The trajectory controller's view: one sample at a time, stop as soon as it decides."""
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t), velocities=swing_velocity(t))
    measured_positions = swing(t)
    measured_positions[150:, 1] += 0.06  # j2 jumps off at t = 1.5 s
    measured = joint_traj(t, measured_positions, velocities=swing_velocity(t))
    tolerances = Tolerances(
        path=(JointTolerance(name="j2", position=0.05),),
        goal=(JointTolerance(name="j2", position=0.01),),
    )
    offline = check(reference, measured, tolerances)

    monitor = ToleranceMonitor(tolerances, joint_names=NAMES, end_time=float(t[-1]))
    online = None
    for index in range(len(t)):
        reference_point = JointTrajectoryPoint(
            positions=reference.positions[index], velocities=reference.velocities[index]
        )
        measured_point = JointTrajectoryPoint(
            positions=measured.positions[index], velocities=measured.velocities[index]
        )
        online = monitor.step(float(t[index]), reference_point, measured_point)
        if online is not None:
            break
    assert online is not None and index == 150  # it stopped at the first bad sample
    assert online.error_code == offline.error_code == PATH_TOLERANCE_VIOLATED
    assert online.violations[0].metric == offline.violations[0].metric == "positions/j2"
    assert online.violations[0].t == pytest.approx(offline.violations[0].t)


def test_online_monitor_reports_success_once_the_goal_is_met():
    t = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t, swing(t))
    tolerances = Tolerances(goal=(JointTolerance(name="j1", position=0.01),))
    monitor = ToleranceMonitor(tolerances, joint_names=NAMES, end_time=1.0)
    results = [
        monitor.step(
            float(time_s),
            JointTrajectoryPoint(positions=reference.positions[i]),
            JointTrajectoryPoint(positions=reference.positions[i]),
        )
        for i, time_s in enumerate(t)
    ]
    assert all(result is None for result in results[:-1])  # nothing to decide mid-path
    assert results[-1] is not None and results[-1].ok


# ── 2. accuracy vs repeatability across N runs (ISO 9283) ────────────────────


def test_aggregate_splits_bias_from_spread_by_hand():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    offsets = (0.01, 0.02, 0.03)  # j1 of each run reads this much high
    runs = []
    for offset in offsets:
        positions = swing(t)
        positions[:, 0] += offset
        runs.append(joint_traj(t, positions))
    result = aggregate(reference, runs)

    assert isinstance(result, Aggregate)
    assert result.runs == 3
    # accuracy: reference - barycentre (the bias a feedforward could remove)
    np.testing.assert_allclose(result.accuracy.error.positions[:, 0], -0.02, atol=1e-12)
    np.testing.assert_allclose(result.accuracy.error.positions[:, 1], 0.0, atol=1e-12)
    assert result.accuracy.metrics["positions/j1"].max_abs == pytest.approx(0.02)
    # repeatability: mean + 3 S of each run's distance to the barycentre (ISO RP = l̄ + 3 S_l)
    distances = np.array([0.01, 0.0, 0.01])
    expected = distances.mean() + 3.0 * distances.std(ddof=1)
    assert result.metrics["positions/j1"].max == pytest.approx(expected)
    assert result.metrics["positions/j2"].max == pytest.approx(0.0, abs=1e-12)


def test_aggregate_poses_use_distance_to_the_barycentre():
    t = np.linspace(0.0, 1.0, 101)
    route = np.column_stack([0.1 * t, np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, route)
    shifts = [np.array([0.0, dy, 0.0]) for dy in (0.001, 0.003, 0.005)]
    runs = [pose_traj(t, route + shift) for shift in shifts]
    result = aggregate(reference, runs)
    assert result.accuracy.metrics["position"].max_abs == pytest.approx(0.003)
    distances = np.array([0.002, 0.0, 0.002])
    assert result.metrics["position"].max == pytest.approx(
        distances.mean() + 3.0 * distances.std(ddof=1)
    )
    assert result.metrics["rotation"].max == pytest.approx(0.0, abs=1e-9)


def test_aggregate_never_extrapolates_a_short_run():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    full = joint_traj(t, swing(t))
    short = joint_traj(t[:101], swing(t[:101]))  # covers half
    result = aggregate(reference, [full, short])
    assert result.accuracy.error.t[-1] <= 1.0 + 1e-9
    assert result.coverage == pytest.approx(0.5)


def test_aggregate_needs_two_runs():
    t = np.linspace(0.0, 1.0, 11)
    reference = joint_traj(t, swing(t))
    with pytest.raises(ValueError, match="2"):
        aggregate(reference, [reference])


# ── 3. latency estimate ──────────────────────────────────────────────────────


@pytest.mark.parametrize("delay_s", [0.037, -0.021])
def test_estimate_time_offset_recovers_a_hidden_delay(delay_s):
    rng = np.random.default_rng(7)
    t = np.arange(0.0, 6.0, 1e-3)
    reference = joint_traj(t, swing(3.0 * t))
    measured_positions = swing(3.0 * (t - delay_s)) + rng.normal(0.0, 1e-4, (len(t), 2))
    measured = joint_traj(t, measured_positions)

    estimate = estimate_time_offset(reference, measured, max_offset_s=0.1)
    assert isinstance(estimate, TimeOffsetEstimate)
    assert estimate.offset_s == pytest.approx(delay_s, abs=5e-4)
    assert estimate.rms_after < estimate.rms_before / 20.0
    shifted = compare(reference, measured, time_offset_s=estimate.offset_s)
    assert shifted.metrics["positions/j1"].rms < 5e-4  # what is left is the noise


def test_estimate_time_offset_on_poses():
    t = np.arange(0.0, 4.0, 2e-3)
    reference = pose_traj(t, circle(t, omega=2.0))
    measured = pose_traj(t, circle(t - 0.012, omega=2.0))
    estimate = estimate_time_offset(reference, measured, max_offset_s=0.05)
    assert estimate.offset_s == pytest.approx(0.012, abs=5e-4)


# ── 4. exact contour error (CNC) beside the first-order cross-track ─────────


def test_contour_is_zero_for_a_late_run_on_a_curved_path_where_cross_track_is_not():
    t = np.arange(0.0, 6.0, 1e-3)
    delay = 0.05
    reference = pose_traj(t, circle(t))
    measured = pose_traj(t, circle(t - delay))  # exactly on the circle, 50 ms behind
    result = compare(reference, measured)
    steady = t > 0.2
    contour = result.error.channels["contour"][steady]
    cross = result.error.channels["cross_track"][steady]
    lag_time = result.error.channels["lag_time"][steady]
    assert np.nanmax(contour) < 1e-6  # on the path: no contour error at all
    # the tangent split leaks: chord sagitta R (1 - cos(ω d)) = 0.2 * (1 - cos 0.05) ≈ 2.5e-4 m
    assert np.nanmedian(cross) == pytest.approx(0.2 * (1.0 - np.cos(delay)), rel=0.05)
    np.testing.assert_allclose(lag_time, delay, atol=1e-3)  # + = behind, in seconds
    # (over the steady part: for t < 50 ms the late run is before the path's start, off it)
    assert result.metrics["contour"].median < 1e-6
    assert result.metrics["lag_time"].median == pytest.approx(delay, abs=1e-3)


def test_contour_measures_a_radial_offset():
    t = np.arange(0.0, 6.0, 1e-3)
    reference = pose_traj(t, circle(t, radius=0.2))
    measured = pose_traj(t, circle(t, radius=0.203))
    result = compare(reference, measured)
    np.testing.assert_allclose(result.error.channels["contour"][100:], 0.003, atol=1e-6)
    np.testing.assert_allclose(result.error.channels["lag_time"][100:], 0.0, atol=1e-3)


def test_contour_search_window_cannot_snap_to_the_other_pass_of_a_retraced_route():
    t = np.arange(0.0, 2.0, 1e-3)
    out_and_back = np.column_stack([1.0 - np.abs(1.0 - t), np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, out_and_back)
    delay = 0.02
    measured = pose_traj(t, np.column_stack([1.0 - np.abs(1.0 - (t - delay)), 0 * t, 0 * t]))
    result = compare(reference, measured, contour_window_s=0.25)
    away = (np.abs(t - 1.0) > 0.3) & (t > 0.1) & (t < 1.9)  # both passes never share a window
    np.testing.assert_allclose(result.error.channels["lag_time"][away], delay, atol=2e-3)


# ── 5. goal / settling / overshoot (stepinfo) ───────────────────────────────


def test_goal_stats_settling_time_overshoot_and_final_error():
    # Reference: j1 ramps 0 -> 1 over [0, 1] s and holds. Measured: a step response that overshoots
    # by 20 % of the move and decays: q = 1 + 0.2 * exp(-(t-1)/0.05) * cos(...) is awkward to
    # invert, so use a piecewise-known signal: +0.2 overshoot at t = 1.1, then linear back to 1.0
    # at t = 1.3, then exactly 1.0.
    t_ref = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t_ref, np.column_stack([t_ref, np.zeros_like(t_ref)]))
    t = np.linspace(0.0, 2.0, 2001)
    j1 = np.interp(t, [0.0, 1.0, 1.1, 1.3, 2.0], [0.0, 0.9, 1.2, 1.0, 1.0])
    measured = joint_traj(t, np.column_stack([j1, np.zeros_like(t)]))
    stats = goal(reference, measured, band=0.02)

    j1_stats = stats["positions/j1"]
    assert isinstance(j1_stats, GoalStats)
    assert j1_stats.overshoot == pytest.approx(0.2, abs=1e-3)  # past the final value, along the move
    assert j1_stats.overshoot_percent == pytest.approx(20.0, abs=0.1)  # of the 1 rad move
    # |error| stays inside 0.02 from where the line 1.2 -> 1.0 reaches 1.02: t = 1.1 + 0.2 * 0.9
    assert j1_stats.settling_time == pytest.approx(0.28, abs=2e-3)  # seconds after the end
    assert j1_stats.final_error == pytest.approx(0.0, abs=1e-12)
    assert stats["positions/j2"].settling_time == pytest.approx(0.0)
    assert j1_stats.samples_after_end > 0


def test_goal_stats_never_settling_is_nan_not_the_end_of_data():
    t_ref = np.linspace(0.0, 1.0, 101)
    reference = joint_traj(t_ref, np.column_stack([t_ref, np.zeros_like(t_ref)]))
    t = np.linspace(0.0, 1.5, 1501)
    j1 = np.where(t < 1.0, t, 1.0 + 0.05 * np.sin(40.0 * (t - 1.0)))  # rings at ±0.05, never inside
    measured = joint_traj(t, np.column_stack([j1, np.zeros_like(t)]))
    assert np.isnan(goal(reference, measured, band=0.02)["positions/j1"].settling_time)


# ── 6. relative error over windows (evo RPE) ────────────────────────────────


def test_relative_error_ignores_a_constant_offset_and_measures_drift():
    t = np.linspace(0.0, 10.0, 1001)
    route = np.column_stack([0.1 * t, np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, route)
    offset = pose_traj(t, route + np.array([0.01, 0.0, 0.0]))
    drifting = pose_traj(t, route + np.column_stack([1e-3 * t, 0 * t, 0 * t]))

    no_drift = relative_error(reference, offset, delta=0.5, unit="s")
    assert no_drift["relative/0.5s/position"].max_abs == pytest.approx(0.0, abs=1e-12)
    assert compare(reference, offset).metrics["position"].rms == pytest.approx(0.01)

    drift = relative_error(reference, drifting, delta=0.5, unit="s")
    assert drift["relative/0.5s/position"].rms == pytest.approx(1e-3 * 0.5, rel=1e-6)
    by_distance = relative_error(reference, drifting, delta=0.05, unit="m")  # 0.05 m = 0.5 s here
    assert by_distance["relative/0.05m/position"].rms == pytest.approx(1e-3 * 0.5, rel=1e-6)


def test_relative_error_rotation_drift():
    t = np.linspace(0.0, 10.0, 1001)
    route = np.column_stack([0.1 * t, np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, route)
    turning = pose_traj(t, route, yaw=0.002 * t)  # 2 mrad/s of yaw drift
    result = relative_error(reference, turning, delta=1.0, unit="s")
    assert result["relative/1s/rotation"].rms == pytest.approx(0.002, rel=1e-6)


# ── 7. discrete Fréchet (time-free shape) ───────────────────────────────────


def test_frechet_by_hand():
    reference = pose_traj(np.arange(3.0), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float))
    measured = pose_traj(np.arange(3.0), np.array([[0, 1, 0], [1, 1.5, 0], [2, 1, 0]], float))
    assert frechet(reference, measured) == pytest.approx(1.5)


def test_frechet_ignores_timing_but_not_shape():
    t = np.linspace(0.0, 1.0, 1001)
    line = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    reference = pose_traj(t, line)
    warped = pose_traj(t, np.column_stack([t**2, 0 * t, 0 * t]))  # same path, other timing
    assert frechet(reference, warped) < 2e-3  # at most about one sample spacing
    assert compare(reference, warped).metrics["position"].max_abs > 0.2
    shifted = pose_traj(t, line + np.array([0.0, 0.003, 0.0]))
    assert frechet(reference, shifted) == pytest.approx(0.003)


def test_frechet_joint_space_and_speed():
    t = np.linspace(0.0, 3.0, 3000)
    reference = joint_traj(t, swing(t))
    measured = joint_traj(t, swing(t) + np.array([0.0, 0.004]))
    started = time.perf_counter()
    assert frechet(reference, measured) == pytest.approx(0.004)
    assert time.perf_counter() - started < 3.0  # 3000 x 3000 must stay interactive


# ── 8. integral criteria on every ErrorStats ────────────────────────────────


def test_error_stats_integrals_by_closed_form():
    t = np.linspace(0.0, 2.0, 2001)
    constant = error_stats(np.ones_like(t), t=t)
    assert (constant.iae, constant.ise, constant.itae) == pytest.approx((2.0, 2.0, 2.0), abs=1e-9)
    ramp = error_stats(-t, t=t)
    assert ramp.iae == pytest.approx(2.0, abs=1e-6)
    assert ramp.ise == pytest.approx(8.0 / 3.0, abs=1e-5)
    assert ramp.itae == pytest.approx(8.0 / 3.0, abs=1e-5)


def test_error_stats_integrals_skip_gaps_and_need_time():
    gappy = error_stats(np.array([1.0, 1.0, np.nan, 1.0, 1.0]), t=np.arange(5.0))
    assert gappy.iae == pytest.approx(2.0)  # only intervals with both ends observed
    assert np.isnan(error_stats(np.array([1.0, 2.0])).iae)  # no time, no integral


def test_compare_metrics_carry_integrals():
    t = np.linspace(0.0, 2.0, 201)
    reference = joint_traj(t, swing(t))
    measured_positions = swing(t)
    measured_positions[:, 1] += 0.01
    result = compare(reference, joint_traj(t, measured_positions))
    assert result.metrics["positions/j2"].iae == pytest.approx(0.02)
    assert result.metrics["positions/j2"].ise == pytest.approx(0.0002)


# ── 9. time-base channels pass through, never differenced ───────────────────


def test_speed_scaling_and_segment_index_ride_along_on_the_error():
    t = np.linspace(0.0, 1.0, 11)
    reference = joint_traj(t, swing(t))
    scaling = np.linspace(1.0, 0.5, 11)
    segment = np.repeat([0.0, 1.0], [6, 5])
    measured = joint_traj(
        t, swing(t), channels={"speed_scaling": scaling, "segment_index": segment}
    )
    result = compare(reference, measured)
    np.testing.assert_array_equal(result.error.channels["time_base/speed_scaling"], scaling)
    np.testing.assert_array_equal(result.error.channels["time_base/segment_index"], segment)
    assert not any("speed_scaling" in name or "segment_index" in name for name in result.metrics)
