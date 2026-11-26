"""Stable metric names over an error trajectory, each summarised as `ErrorStats`.

Every metric passes the error trajectory's `t`, so each carries its IAE / ISE / ITAE integrals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from robot_msgs.compare.difference import (
    FAULT_MISMATCH,
    IMAGE_ERROR_CHANNELS,
    ROTATION_ERROR_CHANNELS,
    TIME_BASE_PREFIX,
)
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    JointTrajectory,
    PoseTrajectory,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    Trajectory,
    WrenchTrajectory,
)

_JOINT_FIELDS = ("positions", "velocities", "accelerations", "effort")
_SERVO_STATE_FIELDS = ("positions", "velocities", "effort", "temperature", "voltage")
_SERVO_COMMAND_FIELDS = (
    "positions", "velocities", "feedforward_effort", "stiffness", "damping", "max_effort",
)  # fmt: skip
_XYZ = ("x", "y", "z")
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


def _rotation_metrics(error: Trajectory) -> dict[str, ErrorStats]:
    """``rotation`` (the angle) and ``rx ry rz`` (the rotation vector, signed, about the error's
    fixed axes), from the ``rotation_vector`` / ``rotation_angle`` channels of `rotation_error`."""
    rotation_vector = error.channels["rotation_vector"]
    metrics = {"rotation": error_stats(error.channels["rotation_angle"], error.t)}
    for axis, column in (("rx", 0), ("ry", 1), ("rz", 2)):
        metrics[axis] = error_stats(rotation_vector[:, column], error.t)
    return metrics


def _vector_metrics(name: str, values: np.ndarray, t: np.ndarray) -> dict[str, ErrorStats]:
    """``<name>/x|y|z`` (signed) from an ``(N, 3)`` error."""
    return {f"{name}/{axis}": error_stats(values[:, column], t) for column, axis in enumerate(_XYZ)}


def pose_metrics(error: PoseTrajectory) -> dict[str, ErrorStats]:
    """``position`` / ``rotation`` (magnitudes), ``x y z`` / ``rx ry rz`` (signed, world axes),
    ``lag`` / ``cross_track`` (first-order), ``contour`` / ``lag_time`` (exact), velocity-error
    magnitudes when present, then channels."""
    rotation = _rotation_metrics(error)
    metrics = {
        "position": error_stats(np.linalg.norm(error.positions, axis=1), error.t),
        "rotation": rotation.pop("rotation"),
    }
    for axis, column in (("x", 0), ("y", 1), ("z", 2)):
        metrics[axis] = error_stats(error.positions[:, column], error.t)
    metrics.update(rotation)
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


def servo_state_metrics(error: ServoStateTrajectory) -> dict[str, ErrorStats]:
    """``<field>/<servo>`` for positions, velocities, effort, temperature and voltage (each when
    both sides carry it), ``fault_mismatch`` and ``fault_mismatch/<servo>`` when both report
    faults, then channels.

    ``fault_mismatch`` is the `ErrorStats` of a 0/1 indicator per sample (1 where any servo's code
    differs): its ``mean`` is the fraction of samples whose codes disagree, ``max`` 1 if they ever
    do, ``n`` the samples compared. WHY ErrorStats and not a bare float: every metric keeps one
    shape, so tables, `rows()` and the dataset roll-ups need no special case.
    """
    metrics: dict[str, ErrorStats] = {}
    for field_name in _SERVO_STATE_FIELDS:
        values = getattr(error, field_name)
        if values is not None:
            metrics.update(_columns(field_name, values, error.joint_names, error.t))
    mismatch = error.channels.get(FAULT_MISMATCH)
    if mismatch is not None:
        metrics[FAULT_MISMATCH] = error_stats(mismatch.max(axis=1), error.t)
        metrics.update(_columns(FAULT_MISMATCH, mismatch, error.joint_names, error.t))
    metrics.update(channel_metrics(error, skip={FAULT_MISMATCH}, names=error.joint_names))
    return metrics


def servo_command_metrics(error: ServoCommandTrajectory) -> dict[str, ErrorStats]:
    """``<field>/<servo>`` for positions, velocities, feedforward_effort, stiffness, damping and
    max_effort (each when both sides carry it), then channels (``channels/latency``)."""
    metrics: dict[str, ErrorStats] = {}
    for field_name in _SERVO_COMMAND_FIELDS:
        values = getattr(error, field_name)
        if values is not None:
            metrics.update(_columns(field_name, values, error.joint_names, error.t))
    metrics.update(channel_metrics(error, names=error.joint_names))
    return metrics


def imu_metrics(error: ImuTrajectory) -> dict[str, ErrorStats]:
    """``rotation`` / ``rx ry rz`` (as a pose's), ``angular_velocity`` / ``linear_acceleration``
    (magnitudes), ``angular_velocities/x|y|z`` / ``linear_accelerations/x|y|z`` (signed), then
    channels."""
    metrics = _rotation_metrics(error)
    metrics["angular_velocity"] = error_stats(
        np.linalg.norm(error.angular_velocities, axis=1), error.t
    )
    metrics["linear_acceleration"] = error_stats(
        np.linalg.norm(error.linear_accelerations, axis=1), error.t
    )
    metrics.update(_vector_metrics("angular_velocities", error.angular_velocities, error.t))
    metrics.update(_vector_metrics("linear_accelerations", error.linear_accelerations, error.t))
    metrics.update(channel_metrics(error, skip=set(ROTATION_ERROR_CHANNELS)))
    return metrics


def wrench_metrics(error: WrenchTrajectory) -> dict[str, ErrorStats]:
    """``force`` / ``torque`` (magnitudes), ``forces/x|y|z`` / ``torques/x|y|z`` (signed), then
    channels."""
    metrics = {
        "force": error_stats(np.linalg.norm(error.forces, axis=1), error.t),
        "torque": error_stats(np.linalg.norm(error.torques, axis=1), error.t),
    }
    metrics.update(_vector_metrics("forces", error.forces, error.t))
    metrics.update(_vector_metrics("torques", error.torques, error.t))
    metrics.update(channel_metrics(error))
    return metrics


def image_metrics(error: ImageTrajectory) -> dict[str, ErrorStats]:
    """``pixels``: every pixel (and channel) of every error frame as one signal, so its ``rms`` is
    the whole stream's RMSE and its ``mean`` the bias (no time axis: a pixel is not a sample in
    time, so its integrals are NaN). Then the per-frame series over the error's `t`: ``mae``,
    ``rmse`` and, for integer inputs, ``psnr`` (dB; a frame equal to its reference has infinite
    PSNR and is not counted, so ``n`` says how many frames differ). Then channels."""
    metrics = {"pixels": error_stats(error.frames.reshape(-1))}
    for name in IMAGE_ERROR_CHANNELS:
        if name in error.channels:
            metrics[name] = error_stats(error.channels[name], error.t)
    metrics.update(channel_metrics(error, skip=set(IMAGE_ERROR_CHANNELS)))
    return metrics


def cloud_metrics(forward: np.ndarray, reverse: np.ndarray) -> dict[str, ErrorStats]:
    """``distance`` (each measured point to the nearest reference point, m), ``distance_reverse``
    (each reference point to the nearest measured point) and ``chamfer``: the two sets of
    distances together as one signal, so its ``mean`` is the symmetric mean distance weighted by
    point count (the mean of the two means when the clouds are the same size) and its ``max`` the
    Hausdorff distance. None has a time axis (integrals NaN)."""
    return {
        "distance": error_stats(forward),
        "distance_reverse": error_stats(reverse),
        "chamfer": error_stats(np.concatenate([forward, reverse])),
    }
