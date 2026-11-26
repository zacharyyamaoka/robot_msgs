"""Lining a measured trajectory up with its reference: one clock, no extrapolation."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from robot_msgs._core import time_tolerance
from robot_msgs.trajectory import Trajectory

TrajectoryT = TypeVar("TrajectoryT", bound=Trajectory)
JointT = TypeVar("JointT", bound=Trajectory)


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
    - A measured instant within `time_tolerance` of a reference sample *is* that sample. WHY: two
      streams stamped alike on an epoch clock land a few ulps apart after the subtraction, and a
      held field (a camera frame, a fault code, a mode) would then take the previous sample.
    """
    reference_times = reference.absolute_times
    measured_times = measured.absolute_times - float(time_offset_s)
    tolerance = time_tolerance(np.max(np.abs(reference_times)))
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
    query = _snapped(measured_times[inside] - reference.header.stamp, reference.t, tolerance)
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


def _snapped(query: np.ndarray, times: np.ndarray, tolerance: float) -> np.ndarray:
    """`query` with every value within `tolerance` of a sample of `times` set to that sample."""
    if not len(times) or not len(query):
        return query
    upper = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    lower = np.clip(upper - 1, 0, len(times) - 1)
    nearer = np.where(np.abs(times[lower] - query) < np.abs(times[upper] - query), lower, upper)
    close = np.abs(times[nearer] - query) <= tolerance
    return np.where(close, times[nearer], query)


def match_joints(reference: JointT, measured: JointT) -> JointT:
    """The reference on the measured run's joints, in the measured run's order (matched by name).

    Any joint-keyed stream (`JointTrajectory`, `ServoStateTrajectory`, `ServoCommandTrajectory`):
    it needs `joint_names` and `select(names)`.

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
