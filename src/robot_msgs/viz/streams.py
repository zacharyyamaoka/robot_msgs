"""Plots for the recorded streams: servo states and commands, IMU, F/T wrench, camera frames.

Same look as the joint and pose plots beside them: one row per quantity against t, one colour per
servo (the joint palette) or per axis (x red, y green, z blue), SI units as stored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from robot_msgs.geometry.rotations import quat_to_euler
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    Trajectory,
    WrenchTrajectory,
)
from robot_msgs.viz._common import AXIS_COLORS, color, new_figure, style, tidy, title_of
from robot_msgs.viz._dispatch import plot
from robot_msgs.viz.trajectory import _frame_part, _plottable, _size_label, draw_series

# (field, label, unit) per joint-keyed stream, in row order
SERVO_STATE_ROWS = (
    ("positions", "position", "rad"),
    ("velocities", "velocity", "rad/s"),
    ("effort", "effort", "N·m"),
    ("temperature", "temperature", "°C"),
    ("voltage", "voltage", "V"),
    ("fault", "fault", "code"),
)
SERVO_COMMAND_ROWS = (
    ("positions", "position", "rad"),
    ("velocities", "velocity", "rad/s"),
    ("feedforward_effort", "feedforward", "N·m"),
    ("stiffness", "stiffness", "N·m/rad"),
    ("damping", "damping", "N·m·s/rad"),
    ("max_effort", "max effort", "N·m"),
)
_XYZ_COLORS = [AXIS_COLORS[name] for name in "xyz"]
_CHANNEL_UNITS = {"latency": "s"}  # channels the stream types write themselves


def _channel_rows(trajectory: Trajectory) -> list[tuple[str, np.ndarray]]:
    return [
        (name, np.asarray(values))
        for name, values in trajectory.channels.items()
        if _plottable(np.asarray(values)) and np.asarray(values).ndim <= 2
    ]


def _heading(kind: str, trajectory: Trajectory, *extra: str | None) -> str:
    return title_of(
        kind, _frame_part(trajectory), *extra, _size_label(len(trajectory), trajectory.duration)
    )


def draw_joint_stream(
    fig: Any,
    trajectory: ServoStateTrajectory | ServoCommandTrajectory,
    rows: Sequence[tuple[str, str, str]],
    *,
    title: bool = True,
) -> dict[str, Any]:
    """One row per set field (one line per servo, joint colours), then channels; a figure legend
    of servo names titled with the stream."""
    names = list(trajectory.joint_names)
    colors = [color(i) for i in range(len(names))]
    present: list[tuple[str, str, str | None, np.ndarray]] = [
        (field_name, label, unit, getattr(trajectory, field_name))
        for field_name, label, unit in rows
        if getattr(trajectory, field_name) is not None
    ]
    present += [
        (name, name, _CHANNEL_UNITS.get(name), values) for name, values in _channel_rows(trajectory)
    ]
    axes = fig.subplots(len(present), 1, sharex=True, squeeze=False)[:, 0]
    panels: dict[str, Any] = {}
    joint_lines: list[Any] = []
    for ax, (row, label, unit, values) in zip(axes, present, strict=True):
        tidy(ax)
        panels[row] = ax
        flat = values.reshape(len(trajectory), -1)
        if flat.shape[1] == len(names):
            lines = draw_series(ax, trajectory.t, flat, colors=colors, labels=names, gid=row)
            joint_lines = joint_lines or lines
        else:
            labels = [str(i) for i in range(flat.shape[1])]
            lines = draw_series(
                ax, trajectory.t, flat, colors=[color(i) for i in range(flat.shape[1])],
                labels=labels, gid=row,
            )  # fmt: skip
            if 1 < flat.shape[1] <= 12:
                ax.legend(handles=lines, loc="upper right", ncols=min(len(lines), 6))
        ax.set_ylabel(f"{label} ({unit})" if unit else label)
    axes[-1].set_xlabel("t (s)")
    heading = _heading(type(trajectory).__name__, trajectory)
    if joint_lines:
        legend = fig.legend(
            handles=joint_lines, labels=names, loc="outside upper center",
            ncols=min(len(names), 6), handlelength=1.4, title=heading if title else None,
        )  # fmt: skip
        if title:
            legend.get_title().set_fontsize(11)
    elif title:
        fig.suptitle(heading, fontsize=11)
    return panels


def _rows_count(trajectory: Trajectory, rows: Sequence[tuple[str, str, str]]) -> int:
    typed = sum(getattr(trajectory, field_name) is not None for field_name, _, _ in rows)
    return typed + len(_channel_rows(trajectory))


@plot.register(ServoStateTrajectory)
def _plot_servo_state_trajectory(trajectory: ServoStateTrajectory, *, ax: Any = None) -> Figure:
    """Position, velocity, effort, temperature, voltage and fault rows (those set) vs t, one
    colour per servo; fault codes as steps."""
    with style():
        fig = new_figure((9, 1.7 * _rows_count(trajectory, SERVO_STATE_ROWS) + 1.4))
        draw_joint_stream(fig, trajectory, SERVO_STATE_ROWS)
    return fig


@plot.register(ServoCommandTrajectory)
def _plot_servo_command_trajectory(trajectory: ServoCommandTrajectory, *, ax: Any = None) -> Figure:
    """Position, velocity, feed-forward, stiffness, damping and max-effort rows (those set) vs t,
    one colour per servo; then channels (``latency``)."""
    with style():
        fig = new_figure((9, 1.7 * _rows_count(trajectory, SERVO_COMMAND_ROWS) + 1.4))
        draw_joint_stream(fig, trajectory, SERVO_COMMAND_ROWS)
    return fig


def draw_vector_rows(
    fig: Any,
    trajectory: Trajectory,
    rows: Sequence[tuple[str, str, np.ndarray, Sequence[str]]],
    heading: str,
) -> dict[str, Any]:
    """One row per ``(gid, ylabel, (N, 3) values, axis labels)`` in x/y/z colours, then channels."""
    channel_rows = _channel_rows(trajectory)
    axes = fig.subplots(len(rows) + len(channel_rows), 1, sharex=True, squeeze=False)[:, 0]
    panels: dict[str, Any] = {}
    for ax, (gid, ylabel, values, labels) in zip(axes, rows, strict=False):
        tidy(ax)
        lines = draw_series(ax, trajectory.t, values, colors=_XYZ_COLORS, labels=labels, gid=gid)
        ax.legend(handles=lines, loc="upper right", ncols=3, handlelength=1.2)
        ax.set_ylabel(ylabel)
        panels[gid] = ax
    for ax, (name, values) in zip(axes[len(rows) :], channel_rows, strict=True):
        tidy(ax)
        flat = values.reshape(len(trajectory), -1)
        draw_series(
            ax, trajectory.t, flat, colors=[color(i) for i in range(flat.shape[1])],
            labels=[str(i) for i in range(flat.shape[1])], gid=name,
        )  # fmt: skip
        ax.set_ylabel(name)
        panels[name] = ax
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(heading, fontsize=11)
    return panels


@plot.register(ImuTrajectory)
def _plot_imu_trajectory(trajectory: ImuTrajectory, *, ax: Any = None) -> Figure:
    """Orientation (roll / pitch / yaw, unwrapped, as the pose plot), angular velocity and linear
    acceleration vs t, x/y/z in red/green/blue; then channels."""
    rpy = (
        np.unwrap(quat_to_euler(trajectory.orientations), axis=0)
        if len(trajectory)
        else np.zeros((0, 3))
    )
    rows = [
        ("rpy", "orientation (rad)", rpy, ("roll", "pitch", "yaw")),
        ("angular_velocity", "angular velocity (rad/s)", trajectory.angular_velocities, "xyz"),
        ("linear_acceleration", "linear acceleration (m/s²)", trajectory.linear_accelerations,
         "xyz"),
    ]  # fmt: skip
    with style():
        fig = new_figure((9, 1.8 * (3 + len(_channel_rows(trajectory))) + 1.1))
        draw_vector_rows(fig, trajectory, rows, _heading("ImuTrajectory", trajectory))
    return fig


@plot.register(WrenchTrajectory)
def _plot_wrench_trajectory(trajectory: WrenchTrajectory, *, ax: Any = None) -> Figure:
    """Force and torque vs t, x/y/z in red/green/blue; then channels."""
    rows = [
        ("force", "force (N)", trajectory.forces, "xyz"),
        ("torque", "torque (N·m)", trajectory.torques, "xyz"),
    ]
    with style():
        fig = new_figure((9, 1.8 * (2 + len(_channel_rows(trajectory))) + 1.1))
        draw_vector_rows(fig, trajectory, rows, _heading("WrenchTrajectory", trajectory))
    return fig


@plot.register(ImageTrajectory)
def _plot_image_trajectory(
    trajectory: ImageTrajectory, *, ax: Any = None, frames: int = 6
) -> Figure:
    """A filmstrip of up to `frames` frames evenly spread over the stream, each titled with its
    t; float frames (depth, or a compare error) are drawn on one shared colour scale (a signed
    error: diverging, 0 = white), so frames can be read against each other. Then one row per
    plottable channel (``mae``, ``rmse``, ``psnr`` on an error)."""
    count = min(max(int(frames), 1), max(len(trajectory), 1))
    picks = np.unique(np.linspace(0, len(trajectory) - 1, count).round().astype(int))
    channel_rows = _channel_rows(trajectory)
    data = trajectory.frames
    is_float = data.dtype.kind == "f"
    if is_float:
        grey = data.astype(np.float64)
        grey = grey.mean(axis=3) if grey.ndim == 4 else grey
        finite = grey[np.isfinite(grey)]
        signed = finite.size and finite.min() < 0.0
        bound = float(np.abs(finite).max()) if finite.size else 1.0
        bound = bound or 1.0
        scale = {"cmap": "RdBu_r", "vmin": -bound, "vmax": bound} if signed else {
            "cmap": "viridis", "vmin": float(finite.min()) if finite.size else 0.0,
            "vmax": float(finite.max()) if finite.size else 1.0,
        }  # fmt: skip
    with style():
        fig = new_figure((2.2 * len(picks) + 1.2, 2.6 + 1.5 * len(channel_rows)))
        grid = fig.add_gridspec(
            1 + len(channel_rows), len(picks), height_ratios=(2.6, *[1.5] * len(channel_rows))
        )
        shown = None
        for column, index in enumerate(picks):
            frame_ax = fig.add_subplot(grid[0, column])
            if is_float:
                shown = frame_ax.imshow(grey[index], interpolation="nearest", **scale)
            else:
                frame_ax.imshow(trajectory[int(index)].image.to_rgb(), interpolation="nearest")
            frame_ax.set_xticks([])
            frame_ax.set_yticks([])
            frame_ax.set_title(f"t = {trajectory.t[index]:.3g} s", fontsize=8)
            frame_ax.set_gid(f"frame:{int(index)}")
        if shown is not None:
            fig.colorbar(shown, ax=fig.axes[: len(picks)], shrink=0.8, aspect=18)
        for row, (name, values) in enumerate(channel_rows, start=1):
            row_ax = tidy(fig.add_subplot(grid[row, :]))
            flat = values.reshape(len(trajectory), -1)
            draw_series(
                row_ax, trajectory.t, flat, colors=[color(i) for i in range(flat.shape[1])],
                labels=[str(i) for i in range(flat.shape[1])], gid=name,
            )  # fmt: skip
            row_ax.set_ylabel(name)
            if row == len(channel_rows):
                row_ax.set_xlabel("t (s)")
        fig.suptitle(
            _heading(
                "ImageTrajectory",
                trajectory,
                f"{trajectory.encoding} {trajectory.width}×{trajectory.height}",
            ),
            fontsize=11,
        )
    return fig
