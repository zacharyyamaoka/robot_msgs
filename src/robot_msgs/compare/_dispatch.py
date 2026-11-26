"""`compare`: one generic function, one implementation per message type (like `viz.plot`)."""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatch
from typing import Any

from robot_msgs.compare.align import align, match_joints
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.difference import (
    cloud_error,
    contour_errors,
    image_error,
    imu_error,
    joint_error,
    pose_error,
    servo_command_error,
    servo_state_error,
    trajectory_error,
    wrench_error,
)
from robot_msgs.compare.metrics import (
    channel_metrics,
    cloud_metrics,
    image_metrics,
    imu_metrics,
    joint_metrics,
    pose_metrics,
    servo_command_metrics,
    servo_state_metrics,
    wrench_metrics,
)
from robot_msgs.control import ServoCommand, ServoState
from robot_msgs.geometry import Wrench
from robot_msgs.sensor import Image, Imu, JointState, PointCloud
from robot_msgs.std import Header
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


def _require(measured: Any, kind: type, reference: Any) -> None:
    if not isinstance(measured, kind):
        raise TypeError(
            f"cannot compare a {type(measured).__name__} against a {type(reference).__name__} "
            f"reference; both must be {kind.__name__}s"
        )


@singledispatch
def compare(
    reference: Any, measured: Any, *, time_offset_s: float = 0.0, contour_window_s: float = 0.25
) -> Comparison:
    """`measured` against `reference`: the error and its metrics.

    - ``error = reference - measured`` (ROS 2 ``JointTrajectoryControllerState``), on the
      measured run's own samples inside the reference's span; see `align` for the clock rules.
    - The type picks the analysis: `JointTrajectory` → per-joint errors; `PoseTrajectory` →
      position / rotation / per-axis / lag / cross-track / contour; the recorded streams
      (`ServoStateTrajectory`, `ServoCommandTrajectory`, `ImuTrajectory`, `WrenchTrajectory`,
      `ImageTrajectory`) → their own fields; plain `Trajectory` → shared channels. Joint ↔
      cartesian is not this function's job: build a `PoseTrajectory` per point you care about
      (``PoseTrajectory.from_matrices(t, fk(q), child_frame_id="L2 tip")``) and compare those.
    - Two single messages (`JointState`, `ServoState`, `ServoCommand`, `Imu`, `Wrench`, `Image`)
      compare their values only: their stamps are ignored (so is `time_offset_s`), and each is
      wrapped as a one-sample stream, so the error is that stream type at ``t = 0``.
    - Two `PointCloud` messages compare by nearest neighbour (no clock); the error is the measured
      cloud with a ``distance`` channel, see `cloud_error`.
    - `contour_window_s` (poses only, ignored elsewhere) is how far in reference time either side
      of a measured sample the exact contour search looks; see `contour_errors`.
    - Channels in `TIME_BASE_CHANNELS` are copied from the measured run as ``time_base/<name>``,
      not differenced.
    - Pairwise on purpose: N runs, or sim vs real with no target involved, is a loop (`aggregate`
      for N runs of one reference, `compare_run` / `compare_dataset` for recorded runs).
    """
    raise TypeError(f"robot_msgs.compare has no comparison for {type(reference).__name__}")


def _aligned(
    reference: Trajectory,
    measured: Trajectory,
    time_offset_s: float,
    error_of: Callable[[Any, Any], Trajectory],
    metrics_of: Callable[[Any], dict],
) -> Comparison:
    """The shared body of every stream compare: align on one clock, difference, summarise."""
    reference_at, inside, dropped, coverage = align(
        reference, measured, time_offset_s=time_offset_s
    )
    error = error_of(reference_at, inside)
    return Comparison(
        error=error, metrics=metrics_of(error), time_offset_s=time_offset_s,
        dropped=dropped, coverage=coverage,
    )  # fmt: skip


@compare.register
def _compare_trajectory(
    reference: Trajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, Trajectory, reference)
    return _aligned(reference, measured, time_offset_s, trajectory_error, channel_metrics)


@compare.register
def _compare_joints(
    reference: JointTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, JointTrajectory, reference)
    reference = match_joints(reference, measured)
    return _aligned(reference, measured, time_offset_s, joint_error, joint_metrics)


@compare.register
def _compare_poses(
    reference: PoseTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, PoseTrajectory, reference)
    if reference.linear_velocities is None and len(reference) >= 2:
        # Lag / cross-track need the reference's direction of travel, taken on its own samples.
        reference = reference.with_derivatives()
    reference_at, inside, dropped, coverage = align(
        reference, measured, time_offset_s=time_offset_s
    )
    contour = contour_errors(
        reference, inside, time_offset_s=time_offset_s, window_s=contour_window_s
    )
    error = pose_error(reference_at, inside, contour=contour)
    return Comparison(
        error=error, metrics=pose_metrics(error), time_offset_s=time_offset_s,
        dropped=dropped, coverage=coverage,
    )  # fmt: skip


@compare.register
def _compare_servo_states(
    reference: ServoStateTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, ServoStateTrajectory, reference)
    reference = match_joints(reference, measured)
    return _aligned(reference, measured, time_offset_s, servo_state_error, servo_state_metrics)


@compare.register
def _compare_servo_commands(
    reference: ServoCommandTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, ServoCommandTrajectory, reference)
    reference = match_joints(reference, measured)
    return _aligned(reference, measured, time_offset_s, servo_command_error, servo_command_metrics)


@compare.register
def _compare_imus(
    reference: ImuTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, ImuTrajectory, reference)
    return _aligned(reference, measured, time_offset_s, imu_error, imu_metrics)


@compare.register
def _compare_wrenches(
    reference: WrenchTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, WrenchTrajectory, reference)
    return _aligned(reference, measured, time_offset_s, wrench_error, wrench_metrics)


@compare.register
def _compare_images(
    reference: ImageTrajectory,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, ImageTrajectory, reference)
    if reference.frame_shape != measured.frame_shape:  # before align resamples every frame
        raise ValueError(
            f"frames differ in shape: reference {reference.frame_shape}, "
            f"measured {measured.frame_shape}; resize or crop one side first"
        )
    return _aligned(reference, measured, time_offset_s, image_error, image_metrics)


# ── single messages: values only, as one-sample streams ──────────────────────


def _at_zero(header: Header) -> Header:
    # WHY drop the stamp: two single messages are compared by value; a sim and a real sample
    # taken at different times must still line up (one sample has no span to align in).
    return header.replace(stamp=0.0)


def _single(reference: Any, measured: Any, kind: type, stream: Callable[[Any], Trajectory]) -> Any:
    _require(measured, kind, reference)
    return compare(stream(reference), stream(measured))


@compare.register
def _compare_joint_states(
    reference: JointState,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    return _single(
        reference, measured, JointState,
        lambda s: JointTrajectory.from_joint_states([s.replace(header=_at_zero(s.header))]),
    )  # fmt: skip


@compare.register
def _compare_servo_state(
    reference: ServoState,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    return _single(
        reference, measured, ServoState,
        lambda s: ServoStateTrajectory.from_servo_states([s.replace(header=_at_zero(s.header))]),
    )  # fmt: skip


@compare.register
def _compare_servo_command(
    reference: ServoCommand,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    def stream(command: ServoCommand) -> ServoCommandTrajectory:
        # the latency survives the restamp: it is the command's own value, not its time
        state_stamp = None if command.latency is None else -command.latency
        moved = command.replace(header=_at_zero(command.header), state_stamp=state_stamp)
        return ServoCommandTrajectory.from_servo_commands([moved])

    return _single(reference, measured, ServoCommand, stream)


@compare.register
def _compare_imu(
    reference: Imu, measured: Any, *, time_offset_s: float = 0.0, contour_window_s: float = 0.25
) -> Comparison:
    return _single(
        reference, measured, Imu,
        lambda s: ImuTrajectory.from_imus([s.replace(header=_at_zero(s.header))]),
    )  # fmt: skip


@compare.register
def _compare_wrench(
    reference: Wrench, measured: Any, *, time_offset_s: float = 0.0, contour_window_s: float = 0.25
) -> Comparison:
    return _single(
        reference, measured, Wrench, lambda w: WrenchTrajectory.from_wrenches([0.0], [w])
    )


@compare.register
def _compare_image(
    reference: Image, measured: Any, *, time_offset_s: float = 0.0, contour_window_s: float = 0.25
) -> Comparison:
    return _single(
        reference, measured, Image,
        lambda s: ImageTrajectory.from_images([s.replace(header=_at_zero(s.header))]),
    )  # fmt: skip


@compare.register
def _compare_clouds(
    reference: PointCloud,
    measured: Any,
    *,
    time_offset_s: float = 0.0,
    contour_window_s: float = 0.25,
) -> Comparison:
    _require(measured, PointCloud, reference)
    error, reverse = cloud_error(reference, measured)
    return Comparison(error=error, metrics=cloud_metrics(error.channels["distance"], reverse))
