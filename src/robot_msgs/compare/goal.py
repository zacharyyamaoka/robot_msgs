"""Goal error, settling time and overshoot once the reference has ended.

Prior art: MATLAB ``stepinfo`` (settling time to a band, overshoot as a percent of the move) and
ROS 2 ``joint_trajectory_controller``'s goal check against the reference's final point held
(`split_phases`, shared with `check`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.compare._phases import split_phases
from robot_msgs.compare.align import match_joints
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class GoalStats(Msg):
    """How a run arrived at the reference's final point, from its samples at or after the end.

    - `final_error`: ``reference - measured`` at the last measured sample (a pose: the distance);
      NaN when the run has no sample at or after the reference's end.
    - `settling_time`: seconds after the reference's end of the first sample from which
      ``|error| <= band`` for every later sample (0 if inside at the end and staying); NaN if the
      last sample is outside the band or there are no such samples. WHY NaN, not the end of the
      data: a run that never settled must not look like one that settled at its last sample.
    - `overshoot`: how far the run went past the final value along the reference's net move
      (same unit as the signal), 0 if it never did; `overshoot_percent`: of that move's size
      (MATLAB ``stepinfo``'s convention), NaN when the reference does not move.
    - `band` is the settling band used; `samples_after_end` how many samples it rests on.
    """

    final_error: float
    settling_time: float
    overshoot: float
    overshoot_percent: float
    band: float
    samples_after_end: int

    def __post_init__(self) -> None:
        for name in ("final_error", "settling_time", "overshoot", "overshoot_percent", "band"):
            set_field(self, name, float(getattr(self, name)))
        set_field(self, "samples_after_end", int(self.samples_after_end))


def _band(band: float | Mapping[str, float], key: str) -> float:
    if isinstance(band, Mapping):
        if key not in band:
            raise ValueError(f"band has no entry for {key!r}; have {sorted(band)}")
        return float(band[key])
    return float(band)


def _stats(
    error: np.ndarray, overshoot_along: np.ndarray, ages: np.ndarray, move: float, band: float
) -> GoalStats:
    """`error` is the signed (or distance) error and `overshoot_along` the measured excess past
    the final value along the move, both at the post-end samples; `ages` their seconds after the
    end."""
    count = len(error)
    if not count:
        nan = math.nan
        return GoalStats(
            final_error=nan, settling_time=nan, overshoot=nan, overshoot_percent=nan,
            band=band, samples_after_end=0,
        )  # fmt: skip
    outside = np.flatnonzero(~(np.abs(error) <= band))
    first_inside = 0 if not outside.size else int(outside[-1]) + 1
    settling = math.nan if first_inside >= count else max(0.0, float(ages[first_inside]))
    overshoot = max(0.0, float(np.max(overshoot_along)))
    percent = math.nan if move <= 0.0 else 100.0 * overshoot / move
    return GoalStats(
        final_error=float(error[-1]), settling_time=settling, overshoot=overshoot,
        overshoot_percent=percent, band=band, samples_after_end=count,
    )  # fmt: skip


def _direction(reference: PoseTrajectory) -> np.ndarray:
    """Unit vector of the reference's net move: its last non-zero linear velocity, else
    ``final - first`` (zeros when it does not move)."""
    velocities = reference.linear_velocities
    if velocities is None and len(reference) >= 2:
        velocities = reference.with_derivatives().linear_velocities
    if velocities is not None:
        speed = np.linalg.norm(velocities, axis=1)
        moving = np.flatnonzero(speed > 1e-12)
        if moving.size:
            return velocities[moving[-1]] / speed[moving[-1]]
    net = reference.positions[-1] - reference.positions[0]
    length = float(np.linalg.norm(net))
    return net / length if length > 0.0 else np.zeros(3)


def goal(
    reference: JointTrajectory | PoseTrajectory,
    measured: JointTrajectory | PoseTrajectory,
    *,
    band: float | Mapping[str, float],
    time_offset_s: float = 0.0,
) -> dict[str, GoalStats]:
    """`GoalStats` per joint (``positions/<joint>``) or for a pose (``position``).

    `band` is the settling band: one number, or a mapping metric name → band. The error after the
    reference's end is taken against its final point held, the controller's own goal convention
    (see `split_phases`); `time_offset_s` is `align`'s.
    """
    if isinstance(reference, JointTrajectory) and isinstance(measured, JointTrajectory):
        reference = match_joints(reference, measured)
    elif not (isinstance(reference, PoseTrajectory) and isinstance(measured, PoseTrajectory)):
        raise TypeError(
            "goal takes a JointTrajectory or PoseTrajectory pair, got "
            f"{type(reference).__name__} and {type(measured).__name__}"
        )
    phases = split_phases(reference, measured, time_offset_s=time_offset_s)
    after = np.flatnonzero(phases.goal)
    ages = phases.tau[after] - phases.end
    positions = measured.positions[after]
    final = reference.positions[-1]
    if isinstance(reference, JointTrajectory):
        results = {}
        for column, name in enumerate(reference.joint_names):
            key = f"positions/{name}"
            net = final[column] - reference.positions[0, column]
            along = (positions[:, column] - final[column]) * np.sign(net)
            results[key] = _stats(
                final[column] - positions[:, column], along, ages, abs(net), _band(band, key)
            )
        return results
    error = final - positions
    along = (positions - final) @ _direction(reference)
    move = float(np.linalg.norm(final - reference.positions[0]))
    return {
        "position": _stats(
            np.linalg.norm(error, axis=1), along, ages, move, _band(band, "position")
        )
    }
