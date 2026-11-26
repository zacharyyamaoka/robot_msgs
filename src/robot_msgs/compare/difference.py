"""The error trajectory, ``reference - measured``, of the same type as its inputs.

Prior art: ROS 2 ``JointTrajectoryControllerState`` (the sign), evo / trajectopy (pose error),
CNC contouring control and ISO 9283 normal planes (the exact contour error).
"""

from __future__ import annotations

import numpy as np

from robot_msgs._core import freeze_new
from robot_msgs.geometry.rotations import quat_conjugate, quat_multiply, quat_to_rotvec
from robot_msgs.sensor import PointCloud
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

# Channels the pose error adds; a measured channel of the same name would be ambiguous.
POSE_ERROR_CHANNELS = (
    "rotation_vector", "rotation_angle", "lag", "cross_track", "contour", "lag_time",
)  # fmt: skip

# Channels the IMU error adds (the orientation error's rotation vector, as the pose error's).
ROTATION_ERROR_CHANNELS = ("rotation_vector", "rotation_angle")
# Channels the servo-state error adds: 1.0 where the two fault codes differ, per servo.
FAULT_MISMATCH = "fault_mismatch"
# Per-frame channels the image error adds.
IMAGE_ERROR_CHANNELS = ("mae", "rmse", "psnr")
# The channel the point-cloud error adds to the measured cloud.
CLOUD_DISTANCE = "distance"

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


def rotation_error(
    reference_orientations: np.ndarray, measured_orientations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``q_ref · q_meas⁻¹`` (the world-frame rotation taking the measured orientation to the
    reference one) and that rotation as a rotation vector (axis × angle, angle in [0, π]).

    Shared by every orientation error (`pose_error`, `imu_error`), so they cannot disagree.
    """
    rotation = quat_multiply(reference_orientations, quat_conjugate(measured_orientations))
    return rotation, quat_to_rotvec(rotation)


def _refuse_clash(measured: Trajectory, names: tuple[str, ...], what: str) -> None:
    clash = sorted(set(names) & set(measured.channels))
    if clash:
        raise ValueError(f"measured channels {clash} would shadow the {what} error channels")


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
    _refuse_clash(measured, POSE_ERROR_CHANNELS, "pose")
    position_error = reference_at.positions - measured.positions
    rotation, rotation_vector = rotation_error(reference_at.orientations, measured.orientations)
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


# ── recorded streams: servos, IMU, F/T, camera ───────────────────────────────


def servo_state_error(
    reference_at: ServoStateTrajectory, measured: ServoStateTrajectory
) -> ServoStateTrajectory:
    """Per-servo ``reference - measured`` for positions and whichever of velocities, effort,
    temperature and voltage both sides carry; `reference_at` on `measured`'s servos and samples.

    `fault` is a code, not a quantity: the measured run's codes are copied, never differenced, and
    when both sides report faults the channel ``fault_mismatch`` ``(N, J)`` is 1.0 where the two
    codes differ (else 0.0), so a plot shows where they disagree.
    """
    _refuse_clash(measured, (FAULT_MISMATCH,), "servo state")
    channels = channel_errors(reference_at, measured)
    if reference_at.fault is not None and measured.fault is not None:
        channels[FAULT_MISMATCH] = (reference_at.fault != measured.fault).astype(np.float64)
    return ServoStateTrajectory(
        t=measured.t,
        joint_names=measured.joint_names,
        positions=reference_at.positions - measured.positions,
        velocities=_minus(reference_at.velocities, measured.velocities),
        effort=_minus(reference_at.effort, measured.effort),
        temperature=_minus(reference_at.temperature, measured.temperature),
        voltage=_minus(reference_at.voltage, measured.voltage),
        fault=measured.fault,
        channels=channels,
        header=measured.header,
    )


def servo_command_error(
    reference_at: ServoCommandTrajectory, measured: ServoCommandTrajectory
) -> ServoCommandTrajectory:
    """Per-servo ``reference - measured`` for positions and every optional target both carry
    (a NaN "no target" stays NaN and is skipped by the statistics). The ``latency`` channel, when
    both have it, is differenced like any shared float channel."""
    return ServoCommandTrajectory(
        t=measured.t,
        joint_names=measured.joint_names,
        positions=reference_at.positions - measured.positions,
        velocities=_minus(reference_at.velocities, measured.velocities),
        feedforward_effort=_minus(reference_at.feedforward_effort, measured.feedforward_effort),
        stiffness=_minus(reference_at.stiffness, measured.stiffness),
        damping=_minus(reference_at.damping, measured.damping),
        max_effort=_minus(reference_at.max_effort, measured.max_effort),
        channels=channel_errors(reference_at, measured),
        header=measured.header,
    )


def imu_error(reference_at: ImuTrajectory, measured: ImuTrajectory) -> ImuTrajectory:
    """IMU ``reference - measured`` in the IMU frame: `orientations` ``q_ref · q_meas⁻¹``
    (`rotation_error`, the pose error's rule), rates and accelerations subtracted, and channels
    ``rotation_vector`` / ``rotation_angle`` plus every shared numeric channel."""
    _refuse_clash(measured, ROTATION_ERROR_CHANNELS, "IMU")
    rotation, rotation_vector = rotation_error(reference_at.orientations, measured.orientations)
    channels = channel_errors(reference_at, measured)
    channels["rotation_vector"] = rotation_vector
    channels["rotation_angle"] = np.linalg.norm(rotation_vector, axis=1)
    return ImuTrajectory(
        t=measured.t,
        orientations=rotation,
        angular_velocities=reference_at.angular_velocities - measured.angular_velocities,
        linear_accelerations=reference_at.linear_accelerations - measured.linear_accelerations,
        channels=channels,
        header=measured.header,
    )


def wrench_error(reference_at: WrenchTrajectory, measured: WrenchTrajectory) -> WrenchTrajectory:
    """Force and torque ``reference - measured``, in the measured frame, plus shared channels."""
    return WrenchTrajectory(
        t=measured.t,
        forces=reference_at.forces - measured.forces,
        torques=reference_at.torques - measured.torques,
        channels=channel_errors(reference_at, measured),
        header=measured.header,
    )


_ERROR_ENCODINGS = {1: "32FC1", 3: "32FC3", 4: "32FC4"}


def image_error(reference_at: ImageTrajectory, measured: ImageTrajectory) -> ImageTrajectory:
    """Frame ``reference - measured`` as float32, pixel by pixel; `reference_at` is the reference
    frame held (the latest one at or before each measured frame, `align` with ``"hold"``).

    - Frames must have the same encoding and shape; a mismatch raises rather than compare a
      resized or channel-swapped picture.
    - The error is in the input's pixel units (0–255 for 8-bit), except 16UC1 depth, which is
      millimetres by ROS's convention: its error is in metres, as its 32FC1 encoding says (SI).
      Its encoding is 32FC1 / 32FC3 / 32FC4 by channel count.
    - Channels per frame: ``mae`` (mean |e|), ``rmse`` (√ mean e²) and, for integer inputs,
      ``psnr`` (dB, ``10·log10(MAX² / MSE)``, MAX the dtype's largest value; +inf for a frame
      equal to its reference), plus every shared numeric channel.
    """
    if reference_at.encoding != measured.encoding:
        raise ValueError(
            f"frames differ in encoding: reference {reference_at.encoding}, measured "
            f"{measured.encoding}; convert one side first (rgb8 vs bgr8 is a channel swap)"
        )
    if reference_at.frame_shape != measured.frame_shape:
        raise ValueError(
            f"frames differ in shape: reference {reference_at.frame_shape}, "
            f"measured {measured.frame_shape}; resize or crop one side first"
        )
    channels_in = 1 if measured.frames.ndim == 3 else measured.frames.shape[3]
    _refuse_clash(measured, IMAGE_ERROR_CHANNELS, "image")
    difference = reference_at.frames.astype(np.float32) - measured.frames.astype(np.float32)
    if measured.encoding == "16UC1":
        difference = difference / np.float32(1000.0)
    per_frame = difference.reshape(len(difference), -1).astype(np.float64)
    mse = np.mean(per_frame**2, axis=1)
    channels = channel_errors(reference_at, measured)
    channels["mae"] = np.mean(np.abs(per_frame), axis=1)
    channels["rmse"] = np.sqrt(mse)
    if measured.frames.dtype.kind in "iu" and measured.encoding != "16UC1":
        peak = float(np.iinfo(measured.frames.dtype).max)
        with np.errstate(divide="ignore"):
            channels["psnr"] = 10.0 * np.log10(peak**2 / mse)
    return ImageTrajectory(
        t=measured.t,
        encoding=_ERROR_ENCODINGS[channels_in],
        frames=freeze_new(np.ascontiguousarray(difference, dtype=np.float32)),
        channels=channels,
        header=measured.header,
    )


# ── point clouds: no clock, nearest neighbours ───────────────────────────────

_PAIRS_PER_CHUNK = 4_000_000


def nearest_distances(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance (m) from each `query` point ``(N, 3)`` to its nearest `target` point ``(M, 3)``.

    Exact brute force in float64, chunked so no step holds more than 4 M point pairs (~100 MB).
    Cost is O(N·M): 10 k × 10 k points takes under a second on one core (measured 2026-11-25),
    50 k × 50 k some 20 s; past ~20 k points, downsample both clouds first
    (`PointCloud.downsampled(voxel_size)`). WHY not a k-d tree: numpy is the only runtime
    dependency (scipy's cKDTree is the test oracle).
    """
    query = np.asarray(query, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_norm = np.einsum("ij,ij->i", target, target)
    out = np.empty(len(query))
    chunk = max(1, _PAIRS_PER_CHUNK // max(len(target), 1))
    for begin in range(0, len(query), chunk):
        block = query[begin : begin + chunk]
        # |q - p|² = |q|² - 2 q·p + |p|²; clipped at 0 (round-off), then refined exactly below
        squared = np.einsum("ij,ij->i", block, block)[:, None] - 2.0 * block @ target.T
        squared += target_norm[None, :]
        nearest = np.argmin(squared, axis=1)
        out[begin : begin + chunk] = np.linalg.norm(block - target[nearest], axis=1)
    return out


def cloud_error(reference: PointCloud, measured: PointCloud) -> tuple[PointCloud, np.ndarray]:
    """The measured cloud with channel ``distance`` (m, measured → nearest reference point), and
    the reverse distances (reference → nearest measured point) beside it.

    Clouds have no clock, so nothing is aligned in time; both must be in one frame (a set
    `header.frame_id` on both must match; transform one first).
    """
    if len(reference) == 0 or len(measured) == 0:
        raise ValueError(
            f"cannot compare an empty point cloud ({len(reference)} reference points, "
            f"{len(measured)} measured)"
        )
    frames = (reference.header.frame_id, measured.header.frame_id)
    if all(frames) and frames[0] != frames[1]:
        raise ValueError(f"clouds are in different frames {frames}; transform one first")
    if CLOUD_DISTANCE in measured.channels:
        raise ValueError(f"measured channel {CLOUD_DISTANCE!r} would shadow the cloud error")
    forward = nearest_distances(measured.points, reference.points)
    reverse = nearest_distances(reference.points, measured.points)
    error = measured.replace(channels={**measured.channels, CLOUD_DISTANCE: forward})
    return error, reverse
