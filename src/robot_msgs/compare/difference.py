"""The error trajectory, ``reference - measured``, of the same type as its inputs.

Prior art: ROS 2 ``JointTrajectoryControllerState`` (the sign), evo / trajectopy (pose error),
CNC contouring control and ISO 9283 normal planes (the exact contour error).
"""

from __future__ import annotations

import numpy as np

from robot_msgs.geometry.rotations import quat_conjugate, quat_multiply, quat_to_rotvec
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory

# Channels the pose error adds; a measured channel of the same name would be ambiguous.
POSE_ERROR_CHANNELS = (
    "rotation_vector", "rotation_angle", "lag", "cross_track", "contour", "lag_time",
)  # fmt: skip

# Measured channels that describe the run's clock, not its tracking. WHY copied, not differenced:
# a joint_trajectory_controller state records the speed-scaling factor; a scaled run's timing
# error is expected, so the time base has to travel with the error to be read against it.
TIME_BASE_CHANNELS = ("speed_scaling", "segment_index")
TIME_BASE_PREFIX = "time_base/"


def _minus(reference: np.ndarray | None, measured: np.ndarray | None) -> np.ndarray | None:
    if reference is None or measured is None:
        return None
    return reference - measured


def channel_errors(reference_at: Trajectory, measured: Trajectory) -> dict[str, np.ndarray]:
    """``reference - measured`` for every numeric channel both carry with the same shape.

    A channel only one side has (a gain, a mode) is not an error and is left out. The measured
    `TIME_BASE_CHANNELS` come through verbatim as ``time_base/<name>``.
    """
    errors = {}
    for name, values in measured.channels.items():
        if name in TIME_BASE_CHANNELS:
            errors[TIME_BASE_PREFIX + name] = values
            continue
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

    This is the first-order (tangent) split: the lag / contour-error linearization of contouring
    control. It needs no search, is exact on straight segments and leaks a little lag into
    cross-track on curves (a chord's sagitta, R(1 - cos ωd)). Where the reference is at rest there
    is no direction of travel, so both are NaN there. The exact split is `contour_errors`
    (``contour`` / ``lag_time``: CNC contouring, ISO 9283 normal planes), which finds the nearest
    point on the reference polyline within a time window.
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


def contour_errors(
    reference: PoseTrajectory,
    measured: PoseTrajectory,
    *,
    time_offset_s: float = 0.0,
    window_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact contour error and lag time of each measured position against the reference path.

    - ``contour`` (m): distance from the measured position to the nearest point of the reference
      polyline (piecewise linear between its samples).
    - ``lag_time`` (s): ``τ - t_nearest``, where ``τ`` is the measured sample's reference time and
      ``t_nearest`` the reference time interpolated at that nearest point; ``+`` = behind. NaN
      where the nearest segment has no length (the reference is at rest there: it names no
      direction of travel and no moment).
    - WHY a time window: only segments whose times overlap ``[τ - window_s, τ + window_s]`` are
      searched, so a route that retraces itself cannot snap to its other pass. Ties (distances
      within 1e-9 m of the minimum) go to the candidate with the smallest ``|lag_time|``.

    `measured` must be the aligned samples (`align`); `reference` the full reference. Vectorised in
    chunks of measured samples: a 6000 sample run against 500 segments takes well under a second.
    """
    times = reference.t
    points = reference.positions
    if len(reference) < 2:  # a single pose is a polyline of zero length
        times = np.array([times[0], times[0] + 1.0])
        points = np.vstack([points[0], points[0]])
    query = measured.absolute_times - time_offset_s - reference.header.stamp
    start = points[:-1]
    delta = points[1:] - start
    length2 = np.einsum("ij,ij->i", delta, delta)
    degenerate = length2 <= 1e-24
    safe_length2 = np.where(degenerate, 1.0, length2)
    span = np.diff(times)
    segments = len(start)
    first = np.maximum(np.searchsorted(times, query - window_s, side="left") - 1, 0)
    last = np.minimum(np.searchsorted(times, query + window_s, side="right") - 1, segments - 1)
    last = np.maximum(last, first)
    width = int((last - first).max()) + 1
    contour = np.empty(len(query))
    lag_time = np.empty(len(query))
    chunk = max(1, 400_000 // width)
    for begin in range(0, len(query), chunk):
        rows = slice(begin, begin + chunk)
        # indices past a sample's last segment repeat it: a duplicate candidate, never a new one
        index = np.minimum(first[rows, None] + np.arange(width), last[rows, None])
        offset = measured.positions[rows, None, :] - start[index]
        step = delta[index]
        along = np.einsum("cks,cks->ck", offset, step) / safe_length2[index]
        along = np.where(degenerate[index], 0.0, np.clip(along, 0.0, 1.0))
        foot = offset - along[..., None] * step
        distance = np.sqrt(np.einsum("cks,cks->ck", foot, foot))
        nearest = distance.min(axis=1)
        lag = query[rows, None] - (times[index] + along * span[index])
        ranked = np.where(
            (distance <= nearest[:, None] + 1e-9) & ~degenerate[index], np.abs(lag), np.inf
        )
        pick = ranked.argmin(axis=1)
        row = np.arange(len(pick))
        contour[rows] = nearest
        lag_time[rows] = np.where(np.isfinite(ranked[row, pick]), lag[row, pick], np.nan)
    return contour, lag_time


def pose_error(
    reference_at: PoseTrajectory,
    measured: PoseTrajectory,
    *,
    contour: tuple[np.ndarray, np.ndarray] | None = None,
) -> PoseTrajectory:
    """Pose ``reference - measured`` in the reference frame (`header.frame_id`).

    - `positions`: ``p_ref - p_meas``; `orientations`: ``q_ref · q_meas⁻¹``, the world-frame
      rotation taking the measured orientation to the reference one.
    - channels ``rotation_vector`` (that rotation as axis × angle, so rx / ry / rz errors about
      the fixed axes — not a difference of Euler angles, which couple and jump at gimbal lock),
      ``rotation_angle``, ``lag`` and ``cross_track`` (`track_errors`, needs
      `reference_at.linear_velocities`), ``contour`` and ``lag_time`` (`contour_errors`, passed in
      as `contour` because it needs the whole reference path, not just its values at the measured
      instants), plus every shared numeric channel.
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
    if contour is not None:
        channels["contour"], channels["lag_time"] = contour
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
