"""Where a measured run is on the reference's clock: path (still moving) or goal (it has ended)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from robot_msgs.trajectory import Trajectory


class Phases(NamedTuple):
    """Masks over the measured samples, and the clock they were judged on."""

    path: np.ndarray  # measured samples inside the reference span, before its end
    goal: np.ndarray  # samples from the reference's end on (and within the goal time, if set)
    tau: np.ndarray  # each measured sample's time on the reference clock (absolute)
    end: float  # the reference's last absolute time


def split_phases(
    reference: Trajectory,
    measured: Trajectory,
    *,
    time_offset_s: float = 0.0,
    goal_time_tolerance: float = 0.0,
) -> Phases:
    """Split the measured samples into the path phase and the goal phase.

    - Path: the samples `align` keeps (inside the reference span), except the one at the end.
    - Goal: ``τ >= end`` (inclusive, so a run sampled on the reference's own grid has its last
      sample in the goal phase), and ``τ <= end + goal_time_tolerance`` when that is positive.
    - `time_offset_s` is `align`'s: how late the measured clock runs.

    WHY a held reference is legitimate here, and only here: after the reference ends the
    controller judges the goal against the reference's FINAL point held (control.ros.org
    joint_trajectory_controller docs). That is the controller's own convention for goal checking;
    it never feeds a path statistic, which stays on `align`'s no-extrapolation rule.
    """
    reference_times = reference.absolute_times
    start, end = float(reference_times[0]), float(reference_times[-1])
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(reference_times))))
    tau = measured.absolute_times - float(time_offset_s)
    path = (tau >= start - tolerance) & (tau < end - tolerance)
    goal = tau >= end - tolerance
    if goal_time_tolerance > 0.0:
        goal &= tau <= end + goal_time_tolerance + tolerance
    return Phases(path=path, goal=goal, tau=tau, end=end)
