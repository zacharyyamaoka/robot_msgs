"""Stable metric names over an error trajectory, each summarised as `ErrorStats`.

Every metric passes the error trajectory's `t`, so each carries its IAE / ISE / ITAE integrals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from robot_msgs.compare.difference import TIME_BASE_PREFIX
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory

_JOINT_FIELDS = ("positions", "velocities", "accelerations", "effort")
_POSE_DERIVED = {"rotation_vector", "rotation_angle", "lag", "cross_track", "contour", "lag_time"}


def _columns(
    prefix: str, values: np.ndarray, names: Sequence[str] | None, t: np.ndarray
) -> dict[str, ErrorStats]:
    if values.ndim == 1:
        return {prefix: error_stats(values, t)}
    flat = values.reshape(len(values), -1)
    labels = names if names is not None and len(names) == flat.shape[1] else range(flat.shape[1])
    return {f"{prefix}/{label}": error_stats(flat[:, col], t) for col, label in enumerate(labels)}


def channel_metrics(
    error: Trajectory, *, skip: set[str] = frozenset(), names: Sequence[str] | None = None
) -> dict[str, ErrorStats]:
    """``channels/<name>`` (one column) or ``channels/<name>/<joint or index>`` per column.

    The ``time_base/`` channels are the run's clock, not an error: no metric is made of them.
    """
    metrics: dict[str, ErrorStats] = {}
    for name, values in error.channels.items():
        if name not in skip and not name.startswith(TIME_BASE_PREFIX):
            metrics.update(_columns(f"channels/{name}", values, names, error.t))
    return metrics


def joint_metrics(error: JointTrajectory) -> dict[str, ErrorStats]:
    """``<field>/<joint>`` for positions, velocities, accelerations and effort, then channels."""
    metrics: dict[str, ErrorStats] = {}
    for field_name in _JOINT_FIELDS:
        values = getattr(error, field_name)
        if values is not None:
            metrics.update(_columns(field_name, values, error.joint_names, error.t))
    metrics.update(channel_metrics(error, names=error.joint_names))
    return metrics


def pose_metrics(error: PoseTrajectory) -> dict[str, ErrorStats]:
    """``position`` / ``rotation`` (magnitudes), ``x y z`` / ``rx ry rz`` (signed, world axes),
    ``lag`` / ``cross_track`` (first-order), ``contour`` / ``lag_time`` (exact), velocity-error
    magnitudes when present, then channels."""
    rotation_vector = error.channels["rotation_vector"]
    metrics = {
        "position": error_stats(np.linalg.norm(error.positions, axis=1), error.t),
        "rotation": error_stats(error.channels["rotation_angle"], error.t),
    }
    for axis, column in (("x", 0), ("y", 1), ("z", 2)):
        metrics[axis] = error_stats(error.positions[:, column], error.t)
    for axis, column in (("rx", 0), ("ry", 1), ("rz", 2)):
        metrics[axis] = error_stats(rotation_vector[:, column], error.t)
    for name in ("lag", "cross_track", "contour", "lag_time"):
        if name in error.channels:
            metrics[name] = error_stats(error.channels[name], error.t)
    if error.linear_velocities is not None:
        metrics["linear_velocity"] = error_stats(
            np.linalg.norm(error.linear_velocities, axis=1), error.t
        )
    if error.angular_velocities is not None:
        metrics["angular_velocity"] = error_stats(
            np.linalg.norm(error.angular_velocities, axis=1), error.t
        )
    metrics.update(channel_metrics(error, skip=_POSE_DERIVED))
    return metrics
