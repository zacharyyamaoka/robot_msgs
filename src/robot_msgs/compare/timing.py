"""Latency estimate: how late the measured clock runs, from the two runs themselves.

Prior art: trajectopy's time-shift estimation (a coarse correlation lag, then a fit that
minimises the trajectory error) and ``scipy.signal.correlation_lags`` (the lag axis of a
cross-correlation, the sign convention used below). Not a replacement for knowing the latency:
the result is an `align` ``time_offset_s`` to pass explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.compare.align import align, match_joints
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class TimeOffsetEstimate(Msg):
    """An estimated `time_offset_s` and what it bought.

    `offset_s` has `align`'s sign (how late the measured clock runs), so
    ``compare(reference, measured, time_offset_s=estimate.offset_s)`` is the shifted comparison.
    `rms_before` / `rms_after` are the RMS of the position error at offset 0 and at the estimate
    (over every position-error element), so a small gain says the offset was not the problem.
    `max_offset_s` is the search range the estimate was limited to.
    """

    offset_s: float
    rms_before: float
    rms_after: float
    max_offset_s: float

    def __post_init__(self) -> None:
        for name in ("offset_s", "rms_before", "rms_after", "max_offset_s"):
            set_field(self, name, float(getattr(self, name)))


def _rms_position_error(reference: Any, measured: Any, offset_s: float) -> float:
    """RMS over all position-error elements of ``compare(reference, measured, offset_s)``.

    WHY `align` and not `compare`: the same error, without the metrics a search does not need.
    """
    try:
        reference_at, inside, _, _ = align(reference, measured, time_offset_s=offset_s)
    except ValueError:  # no overlap at this offset
        return math.inf
    return float(np.sqrt(np.mean((reference_at.positions - inside.positions) ** 2)))


def _golden_minimum(function: Any, low: float, high: float, tolerance: float) -> float:
    inner_low = high - _GOLDEN * (high - low)
    inner_high = low + _GOLDEN * (high - low)
    value_low, value_high = function(inner_low), function(inner_high)
    while high - low > tolerance:
        if value_low <= value_high:
            high, inner_high, value_high = inner_high, inner_low, value_low
            inner_low = high - _GOLDEN * (high - low)
            value_low = function(inner_low)
        else:
            low, inner_low, value_low = inner_low, inner_high, value_high
            inner_high = low + _GOLDEN * (high - low)
            value_high = function(inner_high)
    return 0.5 * (low + high)


def estimate_time_offset(
    reference: JointTrajectory | PoseTrajectory,
    measured: JointTrajectory | PoseTrajectory,
    *,
    max_offset_s: float = 0.25,
) -> TimeOffsetEstimate:
    """How late the measured clock runs, from positions only (joints, or a pose's position).

    1. Coarse: normalised cross-correlation of the two runs' velocities (``np.gradient``) on a uniform grid at
       the reference's median step over their overlap, lags limited to ±`max_offset_s`.
    2. Refine: minimise the RMS position error (the error `compare` would report at that offset)
       over ``[coarse - 3 dt, coarse + 3 dt]`` by golden-section search, to 1e-6 s.

    WHY velocities for the coarse step: correlating positions of a slow drift or a ramp has no
    sharp peak; their derivatives do. WHY a refinement at all: the correlation is only as fine as
    the grid step.
    """
    if isinstance(reference, JointTrajectory) and isinstance(measured, JointTrajectory):
        reference = match_joints(reference, measured)
    elif not (isinstance(reference, PoseTrajectory) and isinstance(measured, PoseTrajectory)):
        raise TypeError(
            "estimate_time_offset takes a JointTrajectory or PoseTrajectory pair, got "
            f"{type(reference).__name__} and {type(measured).__name__}"
        )
    if len(reference) < 3 or len(measured) < 3:
        raise ValueError("estimate_time_offset needs at least 3 samples on each side")
    step = float(np.median(np.diff(reference.t)))
    reference_times = reference.absolute_times
    measured_times = measured.absolute_times
    low = max(reference_times[0], measured_times[0])
    high = min(reference_times[-1], measured_times[-1])
    count = int(math.floor((high - low) / step + 1e-9)) + 1
    if count < 3:
        raise ValueError("reference and measured barely overlap in time; nothing to correlate")
    grid = low + step * np.arange(count)

    def velocity(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        columns = [np.interp(grid, times, positions[:, c]) for c in range(positions.shape[1])]
        return np.gradient(np.column_stack(columns), step, axis=0)

    reference_velocity = velocity(reference_times, reference.positions)
    measured_velocity = velocity(measured_times, measured.positions)
    max_lag = min(int(round(max_offset_s / step)), count // 2)
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(len(lags))
    for slot, lag in enumerate(lags):
        # a run delayed by d: measured_velocity[k + lag] == reference_velocity[k] at lag = d/step
        if lag >= 0:
            first, second = reference_velocity[: count - lag], measured_velocity[lag:]
        else:
            first, second = reference_velocity[-lag:], measured_velocity[: count + lag]
        # WHY normalised (Pearson-style over the overlap), not divided by the overlap length: a
        # motion between rests keeps the same sum as the overlap shrinks, so a per-sample mean
        # rewards the largest lag (a 10 ms delay read as 88 ms before 2026-11-10).
        energy = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
        scores[slot] = float(np.sum(first * second)) / energy if energy > 0.0 else -np.inf
    coarse = float(lags[int(np.argmax(scores))]) * step
    offset = _golden_minimum(
        lambda candidate: _rms_position_error(reference, measured, candidate),
        coarse - 3.0 * step,
        coarse + 3.0 * step,
        1e-6,
    )
    return TimeOffsetEstimate(
        offset_s=offset,
        rms_before=_rms_position_error(reference, measured, 0.0),
        rms_after=_rms_position_error(reference, measured, offset),
        max_offset_s=max_offset_s,
    )
