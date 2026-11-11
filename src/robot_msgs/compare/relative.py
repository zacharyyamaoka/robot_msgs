"""Relative (drift) error over a window: what a constant offset hides and a drift shows.

Prior art: evo's relative pose error (RPE), ``E_ij = (Q_i⁻¹ Q_j)⁻¹ (P_i⁻¹ P_j)`` with ``Q`` the
reference and ``P`` the measured poses (Sturm et al., TUM RGB-D benchmark).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from robot_msgs.compare.align import align
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.geometry.rotations import quat_conjugate, quat_multiply, quat_rotate, quat_to_rotvec
from robot_msgs.trajectory import PoseTrajectory


def _partners(axis: np.ndarray, delta: float) -> np.ndarray:
    """For each index i the first j with ``axis[j] - axis[i] >= delta`` (relative tolerance
    1e-9, so 0.5 s on a 0.01 s grid is exactly 50 steps); ``len(axis)`` when there is none."""
    return np.searchsorted(axis, axis + delta * (1.0 - 1e-9), side="left")


def relative_error(
    reference: PoseTrajectory,
    measured: PoseTrajectory,
    *,
    delta: float,
    unit: Literal["s", "m"] = "s",
    time_offset_s: float = 0.0,
) -> dict[str, ErrorStats]:
    """Drift of `measured` against `reference` over windows of `delta`, on the aligned samples.

    - Every sample i is paired with the first later sample j at least `delta` away: seconds
      (``unit="s"``), or metres of the reference's cumulative path length (``unit="m"``).
    - ``E_ij = (Q_i⁻¹ Q_j)⁻¹ (P_i⁻¹ P_j)``: the pose the run gained over the window beyond the
      reference's own motion. A constant offset cancels; a drift does not. Keys
      ``relative/<delta>s/position`` (‖translation of E‖, m) and ``.../rotation`` (angle of E,
      rad), with `delta` printed as ``{delta:g}`` and its unit.
    """
    if not (isinstance(reference, PoseTrajectory) and isinstance(measured, PoseTrajectory)):
        raise TypeError("relative_error takes a PoseTrajectory pair")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if unit not in ("s", "m"):
        raise ValueError(f"unit must be 's' or 'm', got {unit!r}")
    reference_at, inside, _, _ = align(reference, measured, time_offset_s=time_offset_s)
    if unit == "s":
        axis = inside.t
    else:
        steps = np.linalg.norm(np.diff(reference_at.positions, axis=0), axis=1)
        axis = np.concatenate([[0.0], np.cumsum(steps)])
    later = _partners(axis, delta)
    count = int(np.searchsorted(later, len(axis), side="left"))  # the i that have a partner
    start = np.arange(count)
    end = later[:count]

    def window(trajectory: PoseTrajectory) -> tuple[np.ndarray, np.ndarray]:
        """Q_i⁻¹ Q_j as (rotation, translation)."""
        inverse = quat_conjugate(trajectory.orientations[start])
        translation = quat_rotate(inverse, trajectory.positions[end] - trajectory.positions[start])
        return quat_multiply(inverse, trajectory.orientations[end]), translation

    reference_rotation, reference_translation = window(reference_at)
    measured_rotation, measured_translation = window(inside)
    gained = quat_multiply(quat_conjugate(reference_rotation), measured_rotation)
    drift = quat_rotate(
        quat_conjugate(reference_rotation), measured_translation - reference_translation
    )
    times = inside.t[:count]
    prefix = f"relative/{delta:g}{unit}"
    return {
        f"{prefix}/position": error_stats(np.linalg.norm(drift, axis=1), times),
        f"{prefix}/rotation": error_stats(np.linalg.norm(quat_to_rotvec(gained), axis=1), times),
    }
