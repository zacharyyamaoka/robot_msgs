"""Lining a measured trajectory up with its reference: one clock, no extrapolation."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from robot_msgs.trajectory import JointTrajectory, Trajectory

TrajectoryT = TypeVar("TrajectoryT", bound=Trajectory)


def align(
    reference: TrajectoryT, measured: TrajectoryT, *, time_offset_s: float = 0.0
) -> tuple[TrajectoryT, TrajectoryT, int, float]:
    """The reference sampled at the measured run's own instants, and those measured samples.

    Returns ``(reference_at, measured_inside, dropped, coverage)``.

    - Both clocks are absolute (``header.stamp + t``), so runs recorded with their own stamps line
      up. `time_offset_s` is how late the measured clock runs: measured time ``τ`` is compared
      with reference time ``τ - time_offset_s`` (a camera's latency, a one-tick-stale reply).
    - Measured samples outside the reference's span are dropped, never compared against an
      extrapolated or held reference. WHY: an error measured against a frozen last value is
      invented, not observed (data-ui's ``difference(a, b)`` contract, evo's ``max_diff``).
    - Errors are taken on the measured run's native samples, never on a shared resampled grid:
      a 1 kHz joint's ringing disappears on a 100 Hz grid.
    """
    reference_times = reference.absolute_times
    measured_times = measured.absolute_times - float(time_offset_s)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(reference_times))))
    inside = (measured_times >= reference_times[0] - tolerance) & (
        measured_times <= reference_times[-1] + tolerance
    )
    if not inside.any():
        raise ValueError(
            f"no overlap: measured spans [{measured_times[0]:.6g}, {measured_times[-1]:.6g}] s, "
            f"reference [{reference_times[0]:.6g}, {reference_times[-1]:.6g}] s "
            f"(time_offset_s={time_offset_s})"
        )
    measured_inside = measured[inside]
    query = measured_times[inside] - reference.header.stamp
    reference_at = reference.resample(query, extrapolate="clamp")  # clamp only absorbs `tolerance`
    span = reference_times[-1] - reference_times[0]
    if span > 0.0:
        covered = min(measured_times[inside][-1], reference_times[-1]) - max(
            measured_times[inside][0], reference_times[0]
        )
        coverage = float(np.clip(covered / span, 0.0, 1.0))
    else:
        coverage = 1.0
    return reference_at, measured_inside, int(np.count_nonzero(~inside)), coverage


def match_joints(reference: JointTrajectory, measured: JointTrajectory) -> JointTrajectory:
    """The reference on the measured run's joints, in the measured run's order (matched by name).

    Raises `ValueError` naming the joints that differ, rather than comparing a subset by accident.
    """
    missing = [name for name in reference.joint_names if name not in measured.joint_names]
    extra = [name for name in measured.joint_names if name not in reference.joint_names]
    if missing or extra:
        raise ValueError(
            f"joints differ: measured lacks {missing}, reference lacks {extra}; "
            "use .select(names) on either side to compare a subset"
        )
    return reference.select(measured.joint_names)
