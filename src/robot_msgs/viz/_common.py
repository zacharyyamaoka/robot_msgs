"""Shared look and small drawing helpers: palette, rc style, triads, equal 3-D axes."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes

# WHY one qualitative palette (Okabe–Ito minus black/yellow): readable for colour-blind viewers,
# and the same colour always means the same joint / class across every plot and video.
PALETTE = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#7A6F9B", "#8C8C8C",
)  # fmt: skip
# Orientation triads: x red, y green, z blue (the robotics convention).
AXIS_COLORS = {"x": "#D1453B", "y": "#2F9E5A", "z": "#2F6FD1"}
MUTED = "#8a8a8a"
GRID = "#e4e4e4"

RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.8,
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


@contextlib.contextmanager
def style() -> Iterator[None]:
    """Apply the package look to the figures and axes created inside the block."""
    with matplotlib.rc_context(RC):
        yield


def color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def tidy(ax: Axes, *, grid: bool = True) -> Axes:
    """Light grid, no top/right frame: the calm default for every 2-D axes."""
    if grid:
        ax.grid(True, color=GRID, linewidth=0.7)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def tidy_3d(ax: Any) -> Any:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID  # no public API for a 3-D grid colour
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(labelsize=7, colors="#555555")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return ax


def require_3d(ax: Axes, what: str) -> None:
    if getattr(ax, "name", "") != "3d":
        raise ValueError(f"plotting a {what} needs a 3-D axes: fig.add_subplot(projection='3d')")


def new_figure(size: tuple[float, float]) -> Any:
    from matplotlib import pyplot

    return pyplot.figure(figsize=size, layout="constrained")


def axes_for(ax: Axes | None, size: tuple[float, float], *, projection: str | None = None) -> Axes:
    """The given axes, or a fresh single-axes pyplot figure."""
    if ax is not None:
        return ax
    return new_figure(size).add_subplot(projection=projection)


def title_of(kind: str, *parts: str | None) -> str:
    """``kind · part · part`` with the empty parts left out."""
    return " · ".join([kind, *[p for p in parts if p]])


def set_equal_3d(
    ax: Any, points: np.ndarray, *, pad: float = 0.08, min_half: float = 0.05, zoom: float = 1.0
) -> None:
    """Limits around `points` with one metre the same length on all three axes.

    WHY not a cube: a table-top cloud is 1.4 m wide and 0.1 m tall, and a cube would spend most of
    the plot on empty height. Each axis spans its data (at least 30% of the widest axis) and the
    box takes the same proportions, so scale stays equal.
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if len(cloud) == 0:
        cloud = np.zeros((1, 3))
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    centre = (low + high) / 2.0
    half = np.maximum((high - low) / 2.0 * (1.0 + pad), min_half)
    half = np.maximum(half, 0.3 * half.max())
    ax.set_xlim(centre[0] - half[0], centre[0] + half[0])
    ax.set_ylim(centre[1] - half[1], centre[1] + half[1])
    ax.set_zlim(centre[2] - half[2], centre[2] + half[2])
    ax.set_box_aspect(tuple(half / half.max()), zoom=zoom)


def draw_triad(
    ax: Any,
    origin: Sequence[float],
    rotation: np.ndarray,
    length: float,
    *,
    label: str | None = None,
    alpha: float = 1.0,
    linewidth: float = 2.0,
    gid: str = "triad",
) -> list[Any]:
    """Three coloured axis lines from `origin`; the columns of `rotation` are the axes."""
    origin = np.asarray(origin, dtype=np.float64)
    lines = []
    for index, name in enumerate("xyz"):
        end = origin + np.asarray(rotation)[:, index] * length
        (line,) = ax.plot(
            *zip(origin, end, strict=True),
            color=AXIS_COLORS[name],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            gid=f"{gid}:{name}",
        )
        lines.append(line)
    if label:
        ax.text(*origin, f" {label}", fontsize=8, color="#333333", alpha=max(alpha, 0.6))
    return lines


def class_colors(class_ids: Sequence[str]) -> dict[str, str]:
    """A stable colour per class: first-seen order into the palette."""
    mapping: dict[str, str] = {}
    for class_id in class_ids:
        mapping.setdefault(class_id, color(len(mapping)))
    return mapping


def detection_label(class_id: str, score: float) -> str:
    return f"{class_id} {score:.2f}".strip()
