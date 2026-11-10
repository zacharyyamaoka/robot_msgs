"""`compare`: one generic function, one implementation per trajectory type (like `viz.plot`)."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

from robot_msgs.compare.align import align
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.difference import joint_error, pose_error, trajectory_error
from robot_msgs.compare.metrics import channel_metrics, joint_metrics, pose_metrics
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory


def _require(measured: Any, kind: type[Trajectory], reference: Trajectory) -> None:
    if not isinstance(measured, kind):
        raise TypeError(
            f"cannot compare a {type(measured).__name__} against a {type(reference).__name__} "
            f"reference; both must be {kind.__name__}s"
        )


@singledispatch
def compare(reference: Any, measured: Any, *, time_offset_s: float = 0.0) -> Comparison:
    """`measured` against `reference`: the error trajectory and its metrics.

    - ``error = reference - measured`` (ROS 2 ``JointTrajectoryControllerState``), on the
      measured run's own samples inside the reference's span; see `align` for the clock rules.
    - The type picks the analysis: `JointTrajectory` → per-joint errors; `PoseTrajectory` →
      position / rotation / per-axis / lag / cross-track; plain `Trajectory` → shared channels.
      Joint ↔ cartesian is not this function's job: build a `PoseTrajectory` per point you care
      about (``PoseTrajectory.from_matrices(t, fk(q), child_frame_id="L2 tip")``) and compare
      those.
    - Pairwise on purpose: N runs, or sim vs real with no target involved, is a loop.
    """
    raise TypeError(f"robot_msgs.compare has no comparison for {type(reference).__name__}")


@compare.register
def _compare_trajectory(
    reference: Trajectory, measured: Any, *, time_offset_s: float = 0.0
) -> Comparison:
    _require(measured, Trajectory, reference)
    reference_at, inside, dropped, coverage = align(
        reference, measured, time_offset_s=time_offset_s
    )
    error = trajectory_error(reference_at, inside)
    return Comparison(
        error=error, metrics=channel_metrics(error), time_offset_s=time_offset_s,
        dropped=dropped, coverage=coverage,
    )  # fmt: skip


@compare.register
def _compare_joints(
    reference: JointTrajectory, measured: Any, *, time_offset_s: float = 0.0
) -> Comparison:
    _require(measured, JointTrajectory, reference)
    missing = [name for name in reference.joint_names if name not in measured.joint_names]
    extra = [name for name in measured.joint_names if name not in reference.joint_names]
    if missing or extra:
        raise ValueError(
            f"joints differ: measured lacks {missing}, reference lacks {extra}; "
            "use .select(names) on either side to compare a subset"
        )
    reference = reference.select(measured.joint_names)  # match by name, in measured's order
    reference_at, inside, dropped, coverage = align(
        reference, measured, time_offset_s=time_offset_s
    )
    error = joint_error(reference_at, inside)
    return Comparison(
        error=error, metrics=joint_metrics(error), time_offset_s=time_offset_s,
        dropped=dropped, coverage=coverage,
    )  # fmt: skip


@compare.register
def _compare_poses(
    reference: PoseTrajectory, measured: Any, *, time_offset_s: float = 0.0
) -> Comparison:
    _require(measured, PoseTrajectory, reference)
    if reference.linear_velocities is None and len(reference) >= 2:
        # Lag / cross-track need the reference's direction of travel, taken on its own samples.
        reference = reference.with_derivatives()
    reference_at, inside, dropped, coverage = align(
        reference, measured, time_offset_s=time_offset_s
    )
    error = pose_error(reference_at, inside)
    return Comparison(
        error=error, metrics=pose_metrics(error), time_offset_s=time_offset_s,
        dropped=dropped, coverage=coverage,
    )  # fmt: skip
