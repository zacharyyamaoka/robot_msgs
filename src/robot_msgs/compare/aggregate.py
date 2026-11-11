"""Accuracy vs repeatability over N runs of one reference.

Prior art: ISO 9283 (pose accuracy AP / AT is the barycentre's bias from the commanded pose; pose
repeatability RP = l̄ + 3·S_l is the spread of the runs around their own barycentre).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.compare._dispatch import compare
from robot_msgs.compare.align import match_joints
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.geometry.rotations import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotvec,
)
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Aggregate(Msg):
    """What `aggregate(reference, runs)` returns: the bias and the spread of N runs.

    - `accuracy`: ``compare(reference, barycentre)`` on the grid: what a feed-forward could
      remove (ISO AP / AT).
    - `repeatability`: a `Trajectory` over the grid; per instant ISO's RP = l̄ + 3·S_l, where
      ``l_i`` is run i's distance to the barycentre and ``S_l`` its sample standard deviation
      (ddof=1). Joints: channel ``positions`` ``(N, J)``, ``l = |x - b|`` per joint. Poses:
      channels ``position`` ``(N,)`` (``l = ‖p - b‖``) and ``rotation`` ``(N,)``
      (``l`` = angle of ``q_i ⊗ b⁻¹``).
    - `metrics`: `ErrorStats` of the repeatability channels (``positions/<joint>``, ``position``,
      ``rotation``).
    - `runs` and `coverage` (the fraction of the reference's duration that every run covers) say
      what the numbers are made of.
    """

    accuracy: Comparison
    repeatability: Trajectory
    metrics: Mapping[str, ErrorStats] = field(default_factory=dict)
    runs: int = 0
    coverage: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.accuracy, Comparison):
            raise TypeError(f"Aggregate.accuracy must be a Comparison, got {self.accuracy!r}")
        if not isinstance(self.repeatability, Trajectory):
            raise TypeError("Aggregate.repeatability must be a Trajectory")
        for name, stats in self.metrics.items():
            if not isinstance(stats, ErrorStats):
                raise TypeError(f"Aggregate.metrics[{name!r}] must be ErrorStats")
        set_field(self, "metrics", FrozenDict({str(k): v for k, v in self.metrics.items()}))
        set_field(self, "runs", int(self.runs))
        set_field(self, "coverage", float(self.coverage))


def _barycentre_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Mean orientation per instant of ``(R, N, 4)`` quaternions: signs aligned to the first run
    (q and -q are one rotation), averaged, normalised."""
    sign = np.where(np.sum(quaternions * quaternions[:1], axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return quat_normalize((sign * quaternions).mean(axis=0))


def _spread(distances: np.ndarray) -> np.ndarray:
    """ISO 9283 RP over the runs (axis 0): mean + 3 × sample standard deviation."""
    return distances.mean(axis=0) + 3.0 * distances.std(axis=0, ddof=1)


def aggregate(
    reference: JointTrajectory | PoseTrajectory,
    runs: Sequence[JointTrajectory | PoseTrajectory],
    *,
    time_offset_s: float | Sequence[float] = 0.0,
) -> Aggregate:
    """Split N runs of one reference into bias (`accuracy`) and spread (`repeatability`).

    - Needs at least 2 runs. `time_offset_s` is `align`'s (how late a run's clock runs): one
      float for all, or one per run.
    - The grid is the reference's own sample instants that every run covers; each run is
      resampled there and never extrapolated, so the barycentre is the mean of runs that all
      observed that instant.
    - WHY a barycentre and not an average of per-sample error arrays across runs (evo's
      ``merge_results``): per-sample errors only mean something when the runs share a timeline,
      and the mean of signed errors hides the spread. ISO 9283's barycentre separates the two:
      the bias of the mean from the reference, and each run's distance to that mean.
    """
    runs = list(runs)
    if len(runs) < 2:
        raise ValueError(f"aggregate needs at least 2 runs, got {len(runs)}")
    kind = type(reference)
    if kind not in (JointTrajectory, PoseTrajectory):
        raise TypeError(f"aggregate takes JointTrajectory or PoseTrajectory, got {kind.__name__}")
    for run in runs:
        if type(run) is not kind:
            raise TypeError(
                f"cannot aggregate a {type(run).__name__} run against a {kind.__name__} reference"
            )
    offsets = np.atleast_1d(np.asarray(time_offset_s, dtype=np.float64))
    if offsets.size == 1:
        offsets = np.repeat(offsets, len(runs))
    elif offsets.size != len(runs):
        raise ValueError(f"time_offset_s has {offsets.size} values for {len(runs)} runs")
    joint = kind is JointTrajectory
    if joint:
        for run in runs:
            match_joints(reference, run)  # raises, naming the joints that differ
        runs = [run.select(reference.joint_names) for run in runs]

    reference_times = reference.absolute_times
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(reference_times))))
    covered = np.ones(len(reference), dtype=bool)
    for run, offset in zip(runs, offsets, strict=True):
        run_times = run.absolute_times - offset
        covered &= (reference_times >= run_times[0] - tolerance) & (
            reference_times <= run_times[-1] + tolerance
        )
    if not covered.any():
        raise ValueError("no reference instant is covered by every run")
    on_grid = reference[covered]
    grid = reference_times[covered]
    resampled = [
        run.resample(grid - run.header.stamp + offset, extrapolate="clamp")
        for run, offset in zip(runs, offsets, strict=True)
    ]
    span = reference_times[-1] - reference_times[0]
    coverage = 1.0 if span <= 0.0 else float((grid[-1] - grid[0]) / span)

    if joint:
        positions = np.stack([run.positions for run in resampled])  # (R, N, J)
        centre = positions.mean(axis=0)
        barycentre = JointTrajectory(
            t=on_grid.t,
            joint_names=reference.joint_names,
            positions=centre,
            header=reference.header,
        )
        repeatability = {"positions": _spread(np.abs(positions - centre))}
        stats = {
            f"positions/{name}": error_stats(repeatability["positions"][:, column], on_grid.t)
            for column, name in enumerate(reference.joint_names)
        }
    else:
        positions = np.stack([run.positions for run in resampled])  # (R, N, 3)
        quaternions = np.stack([run.orientations for run in resampled])  # (R, N, 4)
        centre = positions.mean(axis=0)
        orientation = _barycentre_quaternions(quaternions)
        barycentre = PoseTrajectory(
            t=on_grid.t,
            positions=centre,
            orientations=orientation,
            child_frame_id=reference.child_frame_id,
            header=reference.header,
        )
        turns = quat_multiply(quaternions, quat_conjugate(orientation))
        repeatability = {
            "position": _spread(np.linalg.norm(positions - centre, axis=-1)),
            "rotation": _spread(np.linalg.norm(quat_to_rotvec(turns), axis=-1)),
        }
        stats = {name: error_stats(values, on_grid.t) for name, values in repeatability.items()}
    return Aggregate(
        accuracy=compare(on_grid, barycentre),
        repeatability=Trajectory(t=on_grid.t, channels=repeatability, header=reference.header),
        metrics=stats,
        runs=len(runs),
        coverage=coverage,
    )
