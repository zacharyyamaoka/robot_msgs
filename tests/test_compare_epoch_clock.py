"""A clock's origin never changes a comparison's verdict (2026-11-24).

Recorded runs carry real stamps: ``header.stamp`` ~1.76e9 s since 1970. Every time tolerance in
robot_msgs.compare was ``1e-9 * max(1, |t|)``, which is 1.76 s at such a stamp, so `check` started
the goal phase 1.76 s early and passed a 1.07° path breach 0.62 s before the end (found by the
robot_msg_viz trajectory lane). Each test runs the same motion at stamp 0 and at an epoch stamp
and expects the same answer, plus the answer itself, worked out by hand from the fixture.

Written before the fix: on the 1e-9-relative tolerance every epoch case here fails.
"""

import numpy as np
import pytest

from robot_msgs.compare import (
    PATH_TOLERANCE_VIOLATED,
    SUCCESSFUL,
    JointTolerance,
    ToleranceMonitor,
    Tolerances,
    aggregate,
    check,
    goal,
)
from robot_msgs.compare.align import align
from robot_msgs.std import Header
from robot_msgs.trajectory import JointTrajectory, JointTrajectoryPoint

NAMES = ("j1", "j2")
EPOCH = 1_764_000_000.0  # 2025-11-24, a real recording's header.stamp
STAMPS = pytest.mark.parametrize("stamp", [0.0, EPOCH], ids=["stamp-0", "stamp-epoch"])
BREACH_RAD = np.deg2rad(1.07)


def joint_traj(t, positions, stamp, **kwargs) -> JointTrajectory:
    return JointTrajectory(
        t=t, joint_names=NAMES, positions=positions, header=Header(stamp=stamp), **kwargs
    )


def swing(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(t), 0.5 * np.cos(2.0 * t)])


def five_seconds() -> np.ndarray:
    return np.round(np.arange(501) * 0.01, 10)  # 0 .. 5 s at 100 Hz


@STAMPS
def test_a_path_breach_near_the_end_is_a_path_violation(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    positions = swing(t)
    positions[438:446, 0] -= (
        BREACH_RAD  # j1 1.07° low over t = 4.38 .. 4.45 s, 0.55 s before the end
    )
    tolerances = Tolerances(path=(JointTolerance(name="j1", position=np.deg2rad(1.0)),))

    outcome = check(reference, joint_traj(t, positions, stamp), tolerances)

    assert outcome.error_code == PATH_TOLERANCE_VIOLATED, outcome.error_string
    first = outcome.violations[0]
    assert (first.phase, first.metric) == ("path", "positions/j1")
    assert first.t == pytest.approx(4.38, abs=1e-6)  # the error trajectory's t, from its stamp


@STAMPS
def test_the_online_monitor_stops_at_the_same_sample(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    positions = swing(t)
    positions[438:446, 0] -= BREACH_RAD
    tolerances = Tolerances(path=(JointTolerance(name="j1", position=np.deg2rad(1.0)),))
    monitor = ToleranceMonitor(tolerances, joint_names=NAMES, end_time=stamp + float(t[-1]))

    outcome = None
    for index, time in enumerate(t):
        outcome = monitor.step(
            stamp + float(time),
            JointTrajectoryPoint(positions=reference.positions[index]),
            JointTrajectoryPoint(positions=positions[index]),
        )
        if outcome is not None:
            break
    assert outcome is not None and index == 438
    assert outcome.error_code == PATH_TOLERANCE_VIOLATED
    assert outcome.violations[0].phase == "path"


@STAMPS
def test_a_run_inside_its_limits_still_succeeds(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    positions = swing(t)
    positions[:, 0] -= np.deg2rad(0.5)
    tolerances = Tolerances(
        path=(JointTolerance(name="j1", position=np.deg2rad(1.0)),),
        goal=(JointTolerance(name="j1", position=np.deg2rad(1.0)),),
    )
    assert check(reference, joint_traj(t, positions, stamp), tolerances).error_code == SUCCESSFUL


@STAMPS
def test_align_drops_samples_outside_the_reference_span(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    # Recorded from 1 s before the reference starts: those 100 samples have no reference.
    early = np.round(np.arange(601) * 0.01, 10)
    measured = joint_traj(early, swing(early - 1.0), stamp - 1.0)

    _, inside, dropped, _ = align(reference, measured)

    assert dropped == 100
    assert inside.absolute_times[0] - stamp == pytest.approx(0.0, abs=1e-6)


@STAMPS
def test_aggregate_never_counts_an_instant_a_run_did_not_reach(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    short = t[:401]  # this run stops 1 s before the reference ends
    runs = [joint_traj(t, swing(t), stamp), joint_traj(short, swing(short), stamp)]

    result = aggregate(reference, runs)

    assert result.accuracy.error.t[-1] == pytest.approx(4.0, abs=1e-6)


@STAMPS
def test_goal_reads_only_samples_at_or_after_the_end(stamp):
    t = five_seconds()
    reference = joint_traj(t, swing(t), stamp)
    longer = np.round(np.arange(601) * 0.01, 10)  # 1 s past the reference's end
    positions = swing(np.minimum(longer, 5.0))
    measured = joint_traj(longer, positions, stamp)

    stats = goal(reference, measured, band=0.01)

    assert stats["positions/j1"].samples_after_end == 101  # t = 5.00 .. 6.00 s
