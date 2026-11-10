"""The error trajectory, ``reference - measured``, of the same type as its inputs."""

from __future__ import annotations

import numpy as np

from robot_msgs.geometry.rotations import quat_conjugate, quat_multiply, quat_to_rotvec
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory

# Channels the pose error adds; a measured channel of the same name would be ambiguous.
POSE_ERROR_CHANNELS = ("rotation_vector", "rotation_angle", "lag", "cross_track")


def _minus(reference: np.ndarray | None, measured: np.ndarray | None) -> np.ndarray | None:
    if reference is None or measured is None:
        return None
    return reference - measured


def channel_errors(reference_at: Trajectory, measured: Trajectory) -> dict[str, np.ndarray]:
    """``reference - measured`` for every numeric channel both carry with the same shape.

    A channel only one side has (a gain, a mode) is not an error and is left out.
    """
    errors = {}
    for name, values in measured.channels.items():
        other = reference_at.channels.get(name)
        if other is None or other.shape != values.shape:
            continue
        if values.dtype.kind not in "fc" or other.dtype.kind not in "fc":
            continue
        errors[name] = other - values
    return errors


def trajectory_error(reference_at: Trajectory, measured: Trajectory) -> Trajectory:
    """Plain `Trajectory`: only channels can be compared."""
    return Trajectory(
        t=measured.t, channels=channel_errors(reference_at, measured), header=measured.header
    )


def joint_error(reference_at: JointTrajectory, measured: JointTrajectory) -> JointTrajectory:
    """Per-joint ``reference - measured`` for positions and whichever derivatives/effort both
    sides carry. `reference_at` must already be on `measured`'s joints and samples."""
    return JointTrajectory(
        t=measured.t,
        joint_names=measured.joint_names,
        positions=reference_at.positions - measured.positions,
        velocities=_minus(reference_at.velocities, measured.velocities),
        accelerations=_minus(reference_at.accelerations, measured.accelerations),
        effort=_minus(reference_at.effort, measured.effort),
        channels=channel_errors(reference_at, measured),
        header=measured.header,
    )


def track_errors(
    position_error: np.ndarray, reference_velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``reference - measured`` into lag (along the reference's direction of travel,
    ``+`` = the run is behind) and cross-track (distance off that direction).

    WHY a tangent split, not nearest-point-on-path: it is the lag / contour-error linearization
    of contouring control, needs no search and cannot snap to the wrong pass of a route that
    retraces itself. It is exact on straight segments and first-order on curves. Where the
    reference is at rest there is no direction of travel, so both are NaN there.
    """
    speed = np.linalg.norm(reference_velocity, axis=1)
    finite = speed[np.isfinite(speed)]
    peak = float(np.max(finite)) if finite.size else 0.0
    moving = np.isfinite(speed) & (speed > max(1e-9, 1e-3 * peak))
    lag = np.full(len(position_error), np.nan)
    cross = np.full(len(position_error), np.nan)
    if moving.any():
        direction = reference_velocity[moving] / speed[moving, None]
        lag[moving] = np.einsum("ij,ij->i", position_error[moving], direction)
        cross[moving] = np.linalg.norm(
            position_error[moving] - lag[moving, None] * direction, axis=1
        )
    return lag, cross


def pose_error(reference_at: PoseTrajectory, measured: PoseTrajectory) -> PoseTrajectory:
    """Pose ``reference - measured`` in the reference frame (`header.frame_id`).

    - `positions`: ``p_ref - p_meas``; `orientations`: ``q_ref · q_meas⁻¹``, the world-frame
      rotation taking the measured orientation to the reference one.
    - channels ``rotation_vector`` (that rotation as axis × angle, so rx / ry / rz errors about
      the fixed axes — not a difference of Euler angles, which couple and jump at gimbal lock),
      ``rotation_angle``, ``lag`` and ``cross_track`` (`track_errors`, needs
      `reference_at.linear_velocities`), plus every shared numeric channel.
    """
    clash = sorted(set(POSE_ERROR_CHANNELS) & set(measured.channels))
    if clash:
        raise ValueError(f"measured channels {clash} would shadow the pose error channels")
    position_error = reference_at.positions - measured.positions
    rotation = quat_multiply(reference_at.orientations, quat_conjugate(measured.orientations))
    rotation_vector = quat_to_rotvec(rotation)
    channels = channel_errors(reference_at, measured)
    channels["rotation_vector"] = rotation_vector
    channels["rotation_angle"] = np.linalg.norm(rotation_vector, axis=1)
    if reference_at.linear_velocities is not None:
        channels["lag"], channels["cross_track"] = track_errors(
            position_error, reference_at.linear_velocities
        )
    return PoseTrajectory(
        t=measured.t,
        positions=position_error,
        orientations=rotation,
        linear_velocities=_minus(reference_at.linear_velocities, measured.linear_velocities),
        angular_velocities=_minus(reference_at.angular_velocities, measured.angular_velocities),
        child_frame_id=measured.child_frame_id,
        channels=channels,
        header=measured.header,
    )
