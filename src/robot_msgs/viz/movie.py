"""`animate`: a trajectory (or two) as an .mp4 or .gif with a moving cursor and triad."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from robot_msgs.geometry.rotations import quat_to_matrix
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory
from robot_msgs.viz._common import AXIS_COLORS, draw_triad, style, title_of
from robot_msgs.viz.trajectory import (
    JOINT_ROWS,
    JointPanels,
    PosePanels,
    draw_joint_trajectory,
    draw_path_3d,
    draw_pose_trajectory,
)

CURSOR = "#333333"


@dataclass
class _Scene:
    """A figure plus the function that moves its cursor, markers and triad to time `t`."""

    figure: Figure
    start: float
    end: float
    update: Callable[[float], None]


def _new_figure(size: tuple[float, float], dpi: int) -> Figure:
    figure = Figure(figsize=size, dpi=dpi, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _cursors(axes: list[Any], start: float) -> list[Any]:
    return [ax.axvline(start, color=CURSOR, linewidth=1.1, zorder=5, gid="cursor") for ax in axes]


def _marker(ax: Any, x: float, y: float, line_color: str, gid: str) -> Any:
    (marker,) = ax.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markersize=6.5,
        color=line_color,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=6,
        gid=gid,
    )
    return marker


def _joint_overlay(trajectory: JointTrajectory, panels: JointPanels) -> Callable[[float], None]:
    """A time cursor on every row and a marker per joint at `trajectory.at(t)` on the typed rows."""
    start = float(trajectory.t[0])
    cursors = _cursors(list(panels.axes.values()), start)
    markers: list[tuple[str, int, Any]] = []
    for row, _, _ in JOINT_ROWS:
        if row not in panels.axes:
            continue
        values = getattr(trajectory, row)
        for column, name, line_color in zip(
            panels.columns, panels.names, panels.colors, strict=True
        ):
            markers.append(
                (
                    row,
                    column,
                    _marker(
                        panels.axes[row],
                        start,
                        values[0, column],
                        line_color,
                        f"marker:{row}:{name}",
                    ),
                )
            )

    def update(t: float) -> None:
        point = trajectory.at(t, extrapolate="clamp")
        for cursor in cursors:
            cursor.set_xdata([t, t])
        for row, column, marker in markers:
            marker.set_data([t], [getattr(point, row)[column]])

    return update


def _triad_updater(ax: Any, trajectory: PoseTrajectory, length: float) -> Callable[[float], None]:
    """A bold triad and dot that follow `trajectory.at(t)` along the 3-D path."""
    position = trajectory.positions[0]
    rotation = quat_to_matrix(trajectory.orientations[0])
    lines = draw_triad(ax, position, rotation, length * 1.15, linewidth=3.4, gid="moving")
    (dot,) = ax.plot(
        *position, marker="o", color=CURSOR, markersize=5, linestyle="none", gid="tool"
    )
    clock = ax.text2D(
        0.03, 0.94, "", transform=ax.transAxes, fontsize=10, color=CURSOR, gid="clock"
    )

    def update(t: float) -> None:
        point = trajectory.at(t, extrapolate="clamp")
        origin, axes = np.asarray(point.position), quat_to_matrix(np.asarray(point.orientation))
        for index, line in enumerate(lines):
            end = origin + axes[:, index] * length * 1.15
            line.set_data_3d(*zip(origin, end, strict=True))
        dot.set_data_3d([origin[0]], [origin[1]], [origin[2]])
        clock.set_text(f"t = {t:5.2f} s")

    return update


def _pose_overlay(trajectory: PoseTrajectory, panels: PosePanels) -> Callable[[float], None]:
    """Cursor and one marker per xyz / roll-pitch-yaw line on the two time-series rows."""
    start = float(trajectory.t[0])
    cursors = _cursors([panels.position_ax, panels.orientation_ax], start)
    series = [
        (panels.position_ax, trajectory.positions[:, axis], AXIS_COLORS["xyz"[axis]], f"xyz:{axis}")
        for axis in range(3)
    ] + [
        (panels.orientation_ax, panels.rpy[:, axis], AXIS_COLORS["xyz"[axis]], f"rpy:{axis}")
        for axis in range(3)
    ]
    markers = [
        _marker(ax, start, values[0], line_color, f"marker:{gid}")
        for ax, values, line_color, gid in series
    ]

    def update(t: float) -> None:
        for cursor in cursors:
            cursor.set_xdata([t, t])
        for marker, (_, values, _, _) in zip(markers, series, strict=True):
            marker.set_data([t], [np.interp(t, trajectory.t, values)])

    return update


def _frame_label(trajectory: PoseTrajectory) -> str | None:
    frame = trajectory.header.frame_id
    return (
        f"{trajectory.child_frame_id or 'pose'} in {frame}"
        if frame
        else trajectory.child_frame_id or None
    )


def _build_scene(joints: JointTrajectory | None, pose: PoseTrajectory | None, dpi: int) -> _Scene:
    reference = joints if joints is not None else pose
    assert reference is not None
    if joints is not None and pose is not None:
        figure = _new_figure((14.0, 6.4), dpi)
        with style():
            grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.1))
            left, right = figure.add_subfigure(grid[0]), figure.add_subfigure(grid[1])
            ax3d = left.add_subplot(projection="3d")
            length = draw_path_3d(left, ax3d, pose, triad_alpha=0.3)
            ax3d.set_title(title_of("PoseTrajectory", _frame_label(pose)), loc="left")
            panels = draw_joint_trajectory(
                right, joints, fields=("positions", "velocities"), channels=False, title=False
            )
            moving, overlay = _triad_updater(ax3d, pose, length), _joint_overlay(joints, panels)
        update = lambda t: (moving(t), overlay(t))  # noqa: E731
    elif joints is not None:
        rows = sum(getattr(joints, row) is not None for row, _, _ in JOINT_ROWS)
        figure = _new_figure((10.0, 1.9 * rows + 1.4), dpi)
        with style():
            panels = draw_joint_trajectory(figure, joints, channels=False)
            update = _joint_overlay(joints, panels)
    else:
        figure = _new_figure((11.0, 5.2), dpi)
        with style():
            pose_panels = draw_pose_trajectory(figure, pose, triad_alpha=0.3)
            moving = _triad_updater(pose_panels.ax3d, pose, pose_panels.triad_length)
            overlay = _pose_overlay(pose, pose_panels)
        update = lambda t: (moving(t), overlay(t))  # noqa: E731
    start, end = float(reference.t[0]), float(reference.t[-1])
    update(start)
    figure.canvas.draw()
    figure.set_layout_engine(
        "none"
    )  # WHY: freeze the layout so no frame re-flows when text changes
    return _Scene(figure, start, end, update)


def _split(trajectories: tuple[Any, ...]) -> tuple[JointTrajectory | None, PoseTrajectory | None]:
    joints = [tr for tr in trajectories if isinstance(tr, JointTrajectory)]
    poses = [tr for tr in trajectories if isinstance(tr, PoseTrajectory)]
    if (
        not trajectories
        or len(joints) + len(poses) != len(trajectories)
        or len(joints) > 1
        or len(poses) > 1
    ):
        raise TypeError(
            "animate takes a JointTrajectory, a PoseTrajectory, or one of each; got "
            + (", ".join(type(tr).__name__ for tr in trajectories) or "nothing")
        )
    return (joints[0] if joints else None), (poses[0] if poses else None)


def animate(
    *trajectories: JointTrajectory | PoseTrajectory,
    path: str | Path,
    fps: int = 30,
    duration: float | None = None,
    dpi: int = 100,
) -> Path:
    """Write `trajectories` as a video: ``.mp4`` (needs ffmpeg) or ``.gif`` by extension.

    A JointTrajectory sweeps a time cursor across its plot with a marker per joint at
    ``traj.at(t)``; a PoseTrajectory moves an orientation triad along its 3-D path; one of each
    (same time span) share a figure: tool path left, joint positions and velocities right.

    `duration` is the length of the video in seconds (default: the motion's own length, i.e.
    real time); the motion is stretched or squeezed to fit. The video has ``round(duration *
    fps)`` frames (at least 2), frame ``i`` showing motion time ``start + span · i / (frames − 1)``.
    """
    joints, pose = _split(trajectories)
    for trajectory in (joints, pose):
        if trajectory is not None and len(trajectory) < 2:
            raise ValueError("animate needs trajectories with at least two samples")
    if (
        joints is not None
        and pose is not None
        and not (np.isclose(joints.t[0], pose.t[0]) and np.isclose(joints.t[-1], pose.t[-1]))
    ):
        raise ValueError(
            f"the JointTrajectory spans [{joints.t[0]:.6g}, {joints.t[-1]:.6g}] s but the "
            f"PoseTrajectory spans [{pose.t[0]:.6g}, {pose.t[-1]:.6g}] s; "
            "animate needs the same span"
        )
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("animate(.mp4) needs ffmpeg on PATH; install it or write a .gif")
        # WHY pad + yuv420p: libx264 refuses odd sizes, and browsers only play yuv420p.
        writer: Any = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-crf",
                "20",
            ],
        )
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError(
            f"animate writes .mp4 or .gif, not {suffix or 'a path without extension'!r}"
        )
    scene = _build_scene(joints, pose, dpi)
    seconds = (scene.end - scene.start) if duration is None else float(duration)
    frames = max(2, round(seconds * fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with writer.saving(scene.figure, str(path), dpi):
            for time in np.linspace(scene.start, scene.end, frames):
                scene.update(float(time))
                writer.grab_frame()
    return path
