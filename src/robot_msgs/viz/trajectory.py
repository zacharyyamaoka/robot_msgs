"""Plots for trajectories (generic, joint, pose) and for a JointState."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from robot_msgs.geometry.rotations import quat_to_euler, quat_to_matrix
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory, Trajectory
from robot_msgs.viz._common import (
    AXIS_COLORS,
    MUTED,
    color,
    draw_triad,
    new_figure,
    set_equal_3d,
    style,
    tidy,
    tidy_3d,
    title_of,
)
from robot_msgs.viz._dispatch import plot

# (field on JointTrajectory, singular name, unit)
JOINT_ROWS = (
    ("positions", "position", "rad"),
    ("velocities", "velocity", "rad/s"),
    ("accelerations", "acceleration", "rad/s²"),
    ("effort", "effort", "N·m"),
)
# Limit lines per row: (JointLimits attribute, drawn at ± the value?)
LIMIT_BOUNDS = {
    "positions": (("min_position", False), ("max_position", False)),
    "velocities": (("max_velocity", True),),
    "accelerations": (("max_acceleration", True),),
    "effort": (("max_effort", True),),
}


def _plottable(values: np.ndarray) -> bool:
    """Numbers and booleans can be drawn against time; strings and objects cannot."""
    return values.dtype.kind in "biuf"


def _frame_part(trajectory: Trajectory) -> str | None:
    return trajectory.header.frame_id or None


def draw_series(
    ax: Axes,
    t: np.ndarray,
    values: np.ndarray,
    *,
    colors: Sequence[str],
    labels: Sequence[str],
    gid: str,
) -> list[Any]:
    """One line per column of `values` against `t`; non-float data is drawn as steps."""
    columns = np.asarray(values).reshape(len(t), -1)
    is_float = columns.dtype.kind == "f"
    lines = []
    for index in range(columns.shape[1]):
        ydata = columns[:, index].astype(np.float64)
        keywords = {"color": colors[index], "label": labels[index], "gid": f"{gid}:{labels[index]}"}
        if is_float:
            (line,) = ax.plot(t, ydata, marker="o" if len(t) < 2 else None, **keywords)
        else:
            (line,) = ax.step(t, ydata, where="post", **keywords)
        lines.append(line)
    if not is_float:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return lines


def _size_label(count: int, seconds: float) -> str:
    return f"{count} samples, {seconds:.3g} s"


# ── generic Trajectory ───────────────────────────────────────────────────────


def draw_trajectory(fig: Any, trajectory: Trajectory, *, title: bool = True) -> dict[str, Axes]:
    """One row per sample array (typed fields, then channels) against t."""
    names = [n for n in trajectory.sample_names if _plottable(np.asarray(trajectory[n]))]
    axes = fig.subplots(max(len(names), 1), 1, sharex=True, squeeze=False)[:, 0]
    rows: dict[str, Axes] = {}
    for ax, name in zip(axes, names, strict=False):
        tidy(ax)
        values = np.asarray(trajectory[name])
        width = int(np.prod(values.shape[1:], dtype=int))
        labels = [str(i) for i in range(width)]
        lines = draw_series(
            ax,
            trajectory.t,
            values,
            colors=[color(i) for i in range(width)],
            labels=labels,
            gid=name,
        )
        ax.set_ylabel(name)
        if 1 < width <= 12:
            ax.legend(handles=lines, loc="upper right", ncols=min(width, 6), handlelength=1.2)
        rows[name] = ax
    axes[-1].set_xlabel("t (s)")
    if title:
        fig.suptitle(
            title_of(
                type(trajectory).__name__,
                _frame_part(trajectory),
                _size_label(len(trajectory), trajectory.duration),
            ),
            fontsize=11,
        )
    return rows


@plot.register(Trajectory)
def _plot_trajectory(trajectory: Trajectory, *, ax: Any = None) -> Figure:
    """Generic trajectory: every typed field and channel as a row against t."""
    rows = len([n for n in trajectory.sample_names if _plottable(np.asarray(trajectory[n]))])
    with style():
        fig = new_figure((9, 1.7 * max(rows, 1) + 1.1))
        draw_trajectory(fig, trajectory)
    return fig


# ── JointTrajectory ──────────────────────────────────────────────────────────


@dataclass
class JointPanels:
    """What `draw_joint_trajectory` made: its rows and the joint ↔ colour mapping."""

    axes: dict[str, Axes] = field(default_factory=dict)  # row name (field or channel) → axes
    names: list[str] = field(default_factory=list)  # plotted joints
    columns: list[int] = field(default_factory=list)  # their columns in the trajectory
    colors: list[str] = field(default_factory=list)


def _limit_values(limits: JointLimits, bound: str, columns: Sequence[int]) -> np.ndarray | None:
    values = getattr(limits, bound)
    return None if values is None else np.asarray(values)[list(columns)]


def _draw_limits(
    ax: Axes, row: str, panels: JointPanels, limits: JointLimits, data: np.ndarray, unit: str
) -> None:
    """Dashed limit lines in each joint's colour; the view includes only limits near the data.

    WHY: a ±2π position limit would squash a 3 rad motion into a quarter of the row. A limit that
    does not fit stays drawn (clipped) and is named in a corner note, so nothing is hidden.
    """
    lines: list[tuple[str, float]] = []
    # WHY staggered dashes: joints usually share a limit, and six lines on one level would show
    # only the last colour; offset by joint, the dashes interleave into a multicolour line.
    on, off = 3.0, max(3.0, 3.0 * (len(panels.names) - 1))
    for bound, symmetric in LIMIT_BOUNDS[row]:
        values = _limit_values(limits, bound, panels.columns)
        if values is None:
            continue
        for rank, (name, line_color, value) in enumerate(
            zip(panels.names, panels.colors, values, strict=True)
        ):
            if not np.isfinite(value):
                continue
            for level in (value, -value) if symmetric else (value,):
                ax.axhline(
                    level,
                    color=line_color,
                    linestyle=(-rank * on, (on, off)),
                    linewidth=1.1,
                    alpha=0.75,
                    zorder=1,
                    gid=f"limit:{bound}:{name}",
                )
                lines.append((bound, float(level)))
    finite = data[np.isfinite(data)]
    if not lines or finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-9)
    near = [v for _, v in lines if low - span <= v <= high + span]
    bottom, top = min([low, *near]), max([high, *near])
    pad = 0.06 * max(top - bottom, 1e-9)
    ax.set_ylim(bottom - pad, top + pad)
    far = {}
    for bound, value in lines:
        if value not in near:
            far.setdefault(bound, []).append(value)
    if far:
        notes = []
        for bound, levels in far.items():
            low_level, high_level = min(levels), max(levels)
            shown = (
                f"{low_level:.3g}"
                if low_level == high_level
                else f"{low_level:.3g} … {high_level:.3g}"
            )
            notes.append(f"{bound} {shown} {unit} off scale")
        ax.text(
            0.995,
            0.04,
            "\n".join(notes),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=MUTED,
            linespacing=1.3,
        )


def _draw_waypoints(panels: JointPanels, waypoints: JointTrajectory) -> None:
    """Markers on the position row, a faint guide on every row, names on the top edge."""
    ax = panels.axes["positions"]
    for name, line_color in zip(panels.names, panels.colors, strict=True):
        if name not in waypoints.joint_names:
            continue
        column = waypoints.joint_index(name)
        ax.plot(
            waypoints.t,
            waypoints.positions[:, column],
            linestyle="none",
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=1.5,
            zorder=4,
            gid=f"waypoints:{name}",
        )
    for row in panels.axes.values():
        for time in waypoints.t:
            row.axvline(time, color="#c8c8c8", linewidth=0.7, linestyle=":", zorder=0)
    labels = waypoints.channels.get("name")
    if labels is not None and np.asarray(labels).ndim == 1:
        top = ax.secondary_xaxis("top")
        top.set_xticks(list(waypoints.t))
        top.set_xticklabels([str(label) for label in labels], fontsize=8)
        top.tick_params(length=0, colors="#555555", pad=2)
        top.spines["top"].set_visible(False)


def draw_joint_trajectory(
    fig: Any,
    trajectory: JointTrajectory,
    *,
    joints: Sequence[str] | None = None,
    waypoints: JointTrajectory | None = None,
    fields: Sequence[str] | None = None,
    channels: bool = True,
    title: bool = True,
) -> JointPanels:
    """Rows of position / velocity / acceleration / effort (those set), then channels."""
    columns = (
        list(range(trajectory.n_joints))
        if joints is None
        else [trajectory.joint_index(name) for name in joints]
    )
    panels = JointPanels(
        names=[trajectory.joint_names[c] for c in columns],
        columns=columns,
        colors=[color(c) for c in columns],
    )
    rows: list[tuple[str, str, str | None, np.ndarray]] = []  # row, label, unit, values
    for row, label, unit in JOINT_ROWS:
        values = getattr(trajectory, row)
        if values is not None and (fields is None or row in fields):
            rows.append((row, label, unit, values))
    if channels:
        rows += [
            (n, n, None, np.asarray(v))
            for n, v in trajectory.channels.items()
            if _plottable(np.asarray(v))
        ]
    axes = fig.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
    t = trajectory.t
    joint_lines: list[Any] = []  # the first typed row's lines: the figure legend's handles
    for ax, (row, label, unit, values) in zip(axes, rows, strict=True):
        tidy(ax)
        panels.axes[row] = ax
        if unit is not None:  # a typed joint field: (N, J), one line per plotted joint
            lines = draw_series(
                ax, t, values[:, columns], colors=panels.colors, labels=panels.names, gid=row
            )
            joint_lines = joint_lines or lines
            ax.set_ylabel(f"{label} ({unit})")
            if trajectory.limits is not None and row in LIMIT_BOUNDS:
                _draw_limits(ax, row, panels, trajectory.limits, values[:, columns], unit)
            continue
        flat = values.reshape(
            len(t), -1
        )  # a channel: joint colours when it has one column per joint
        if flat.shape[1] == trajectory.n_joints:
            flat, names, colors = flat[:, columns], panels.names, panels.colors
        else:
            names = [str(i) for i in range(flat.shape[1])]
            colors = [color(i) for i in range(flat.shape[1])]
        lines = draw_series(ax, t, flat, colors=colors, labels=names, gid=row)
        ax.set_ylabel(label)
        if flat.shape[1] > 1 and names is not panels.names:
            ax.legend(handles=lines, loc="upper right", ncols=min(len(lines), 6), handlelength=1.2)
    axes[-1].set_xlabel("t (s)")
    heading = title_of(
        "JointTrajectory",
        _frame_part(trajectory),
        _size_label(len(trajectory), trajectory.duration),
    )
    if joint_lines:  # WHY the title is the legend's title: a suptitle would sit under the legend
        crowded = max(len(name) for name in panels.names) > 10
        legend = fig.legend(
            handles=joint_lines,
            labels=panels.names,
            loc="outside upper center",
            ncols=min(len(joint_lines), 3 if crowded else 6),
            handlelength=1.4,
            title=heading if title else None,
        )
        if title:
            legend.get_title().set_fontsize(11)
            legend.get_title().set_fontweight("regular")
    elif title:
        fig.suptitle(heading, fontsize=11)
    if waypoints is not None and "positions" in panels.axes:
        _draw_waypoints(panels, waypoints)
    return panels


@plot.register(JointTrajectory)
def _plot_joint_trajectory(
    trajectory: JointTrajectory,
    *,
    ax: Any = None,
    joints: Sequence[str] | None = None,
    waypoints: JointTrajectory | None = None,
) -> Figure:
    """Joint rows vs t, one colour per joint; limits dashed when attached.

    Options: ``joints=[names]`` to show a subset; ``waypoints=JointTrajectory`` to overlay its
    samples on the position row, labelled by its ``name`` channel when it has one.
    """
    rows = sum(getattr(trajectory, row) is not None for row, _, _ in JOINT_ROWS) + sum(
        _plottable(np.asarray(v)) for v in trajectory.channels.values()
    )
    with style():
        fig = new_figure((9, 1.9 * rows + 1.4))
        draw_joint_trajectory(fig, trajectory, joints=joints, waypoints=waypoints)
    return fig


# ── PoseTrajectory ───────────────────────────────────────────────────────────


@dataclass
class PosePanels:
    """What `draw_pose_trajectory` made: the 3-D path and the two time-series rows."""

    ax3d: Any
    position_ax: Axes
    orientation_ax: Axes
    rpy: np.ndarray  # (N, 3) roll, pitch, yaw, unwrapped
    triad_length: float


def draw_path_3d(
    fig: Any, ax3d: Any, trajectory: PoseTrajectory, *, n_triads: int = 10, triad_alpha: float = 1.0
) -> float:
    """The path coloured by time with start/end marks and triads; returns the triad length."""
    tidy_3d(ax3d)
    t, positions = trajectory.t, trajectory.positions
    extent = float((positions.max(axis=0) - positions.min(axis=0)).max()) if len(t) else 0.0
    length = 0.12 * extent if extent > 0.0 else 0.1
    if len(t) > 1:
        segments = np.stack([positions[:-1], positions[1:]], axis=1)
        path = Line3DCollection(segments, cmap="viridis", linewidths=2.2, gid="path")
        path.set_array(0.5 * (t[:-1] + t[1:]))
        path.set_clim(float(t[0]), float(t[-1]))
        ax3d.add_collection3d(path)
        colorbar = fig.colorbar(path, ax=ax3d, shrink=0.55, pad=0.1, aspect=18)
        colorbar.set_label("t (s)")
        colorbar.outline.set_visible(False)
    if len(t):
        ax3d.plot(
            *positions[0], marker="o", color="#333333", markersize=4, linestyle="none", gid="start"
        )
        ax3d.plot(
            *positions[-1], marker="s", color="#333333", markersize=4, linestyle="none", gid="end"
        )
        rotations = quat_to_matrix(trajectory.orientations)
        for index in np.unique(np.linspace(0, len(t) - 1, n_triads).round().astype(int)):
            draw_triad(
                ax3d, positions[index], rotations[index], length, alpha=triad_alpha, gid="triad"
            )
    set_equal_3d(ax3d, positions, pad=0.12, zoom=0.92)
    wide_in_x = np.ptp(positions[:, 0]) >= np.ptp(positions[:, 1]) if len(t) else True
    ax3d.view_init(elev=24, azim=-65 if wide_in_x else -35)  # face the longer horizontal axis
    return length


def draw_pose_trajectory(
    fig: Any,
    trajectory: PoseTrajectory,
    *,
    n_triads: int = 10,
    triad_alpha: float = 1.0,
    title: bool = True,
) -> PosePanels:
    """Left: 3-D path coloured by time with triads. Right: xyz and roll/pitch/yaw vs t."""
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    position_ax = tidy(fig.add_subplot(grid[0, 1]))
    orientation_ax = tidy(fig.add_subplot(grid[1, 1], sharex=position_ax))
    t, positions = trajectory.t, trajectory.positions
    length = draw_path_3d(fig, ax3d, trajectory, n_triads=n_triads, triad_alpha=triad_alpha)
    rpy = np.unwrap(quat_to_euler(trajectory.orientations), axis=0) if len(t) else np.zeros((0, 3))
    for axis, name in enumerate("xyz"):
        position_ax.plot(
            t, positions[:, axis], color=AXIS_COLORS[name], label=name, gid=f"position:{name}"
        )
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        orientation_ax.plot(
            t, rpy[:, axis], color=AXIS_COLORS["xyz"[axis]], label=name, gid=f"rpy:{name}"
        )
    position_ax.set_ylabel("position (m)")
    orientation_ax.set_ylabel("orientation (rad)")
    orientation_ax.set_xlabel("t (s)")
    position_ax.tick_params(labelbottom=False)
    for row in (position_ax, orientation_ax):
        row.legend(loc="best", ncols=3, handlelength=1.2)
        row.margins(x=0.02)
    if title:
        frame = trajectory.header.frame_id
        where = (
            f"{trajectory.child_frame_id or 'pose'} in {frame}"
            if frame
            else (trajectory.child_frame_id or None)
        )
        fig.suptitle(
            title_of(
                "PoseTrajectory",
                where,
                f"{trajectory.path_length():.3g} m path",
                _size_label(len(trajectory), trajectory.duration),
            ),
            fontsize=11,
        )
    return PosePanels(ax3d, position_ax, orientation_ax, rpy, length)


@plot.register(PoseTrajectory)
def _plot_pose_trajectory(trajectory: PoseTrajectory, *, ax: Any = None) -> Figure:
    """3-D path coloured by time with ~10 orientation triads; xyz and roll/pitch/yaw vs t."""
    with style():
        fig = new_figure((11, 5.2))
        draw_pose_trajectory(fig, trajectory)
    return fig


# ── JointState ───────────────────────────────────────────────────────────────


@plot.register(JointState)
def _plot_joint_state(
    state: JointState, *, ax: Any = None, limits: JointLimits | None = None
) -> Figure:
    """Horizontal bars of position per joint; ``limits=`` shades each joint's min…max range."""
    with style():
        if ax is None:
            ax = new_figure((7, 0.42 * len(state) + 1.5)).add_subplot()
        tidy(ax, grid=False)
        ax.grid(True, axis="x", color="#e4e4e4", linewidth=0.7)
        rows = np.arange(len(state))
        if limits is not None:
            for row, name in enumerate(state.name):
                if name not in limits.joint_names:
                    continue
                index = limits.joint_names.index(name)
                low = None if limits.min_position is None else float(limits.min_position[index])
                high = None if limits.max_position is None else float(limits.max_position[index])
                if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
                    ax.barh(
                        row,
                        high - low,
                        left=low,
                        height=0.78,
                        color="#eeeeee",
                        zorder=0,
                        gid="limit_range",
                    )
        bars = ax.barh(rows, state.position, height=0.5, color=[color(i) for i in rows], zorder=2)
        for bar in bars:
            bar.set_gid("position")
        for row, value in zip(rows, state.position, strict=True):
            ax.annotate(
                f"{value:+.3f}",
                (value, row),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
                color="#444444",
            )
        ax.axvline(0.0, color="#9a9a9a", linewidth=0.8, zorder=3)
        ax.set_yticks(rows, state.name)
        ax.invert_yaxis()
        ax.margins(x=0.12)
        ax.set_xlabel("position (rad)")
        ax.set_title(
            title_of(
                "JointState",
                state.header.frame_id or None,
                "grey: position limits" if limits is not None else None,
            ),
            loc="left",
        )
    return ax.figure
