"""Discrete Fréchet distance: how far two paths are apart as shapes, whatever their timing.

Prior art: Eiter & Mannila, "Computing discrete Fréchet distance" (1994).
"""

from __future__ import annotations

import numpy as np

from robot_msgs.compare.align import match_joints
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory


def _curve(
    reference: JointTrajectory | PoseTrajectory, measured: JointTrajectory | PoseTrajectory
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(reference, JointTrajectory) and isinstance(measured, JointTrajectory):
        return match_joints(reference, measured).positions, measured.positions
    if isinstance(reference, PoseTrajectory) and isinstance(measured, PoseTrajectory):
        return reference.positions, measured.positions
    raise TypeError(
        "frechet takes a JointTrajectory or PoseTrajectory pair, got "
        f"{type(reference).__name__} and {type(measured).__name__}"
    )


def frechet(
    reference: JointTrajectory | PoseTrajectory, measured: JointTrajectory | PoseTrajectory
) -> float:
    """Discrete Fréchet distance between the two curves, in metres (poses) or the joints' unit.

    - Time-free: no alignment, no offset; only the order of the points matters. A run that takes
      the right path at the wrong speed scores ~0 here and badly in `compare`.
    - A `PoseTrajectory` is its position curve; a `JointTrajectory` its joint-position vectors
      (joints matched by name, as `compare` does). Orientation is not part of the distance.
    - Eiter–Mannila's recurrence, evaluated along anti-diagonals: every cell of diagonal ``k``
      depends only on diagonals ``k - 1`` and ``k - 2``, so each diagonal is one numpy
      operation and memory stays O(N + M).
    """
    first, second = _curve(reference, measured)
    rows, columns = len(first), len(second)
    if not rows or not columns:
        raise ValueError("frechet needs at least one sample on each side")
    # squared distances: min / max commute with the square root, so it is taken once at the end
    previous = np.full(rows + 1, np.inf)  # diagonal k-1, indexed by row + 1 (slot 0 is padding)
    before = np.full(rows + 1, np.inf)  # diagonal k-2
    for diagonal in range(rows + columns - 1):
        low = max(0, diagonal - columns + 1)
        high = min(diagonal, rows - 1)
        offset = first[low : high + 1] - second[diagonal - high : diagonal - low + 1][::-1]
        cost = np.einsum("ij,ij->i", offset, offset)
        current = np.full(rows + 1, np.inf)
        if diagonal == 0:
            current[1] = cost[0]
        else:
            slots = slice(low + 1, high + 2)
            reach = np.minimum(
                np.minimum(previous[slots], previous[low : high + 1]), before[low : high + 1]
            )
            current[slots] = np.maximum(reach, cost)
        before, previous = previous, current
    return float(np.sqrt(previous[rows]))
