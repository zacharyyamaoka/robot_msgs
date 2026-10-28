"""Plots in 3-D space: poses and transforms as triads, point clouds, 3-D boxes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from robot_msgs.geometry import Pose, PoseStamped, Transform, TransformStamped
from robot_msgs.sensor import PointCloud
from robot_msgs.vision import BoundingBox3D, Detection3D, Detection3DArray
from robot_msgs.viz._common import (
    axes_for,
    class_colors,
    color,
    detection_label,
    draw_triad,
    require_3d,
    set_equal_3d,
    style,
    tidy_3d,
    title_of,
)
from robot_msgs.viz._dispatch import plot

MAX_POINTS = 20_000


# ── Pose, PoseStamped, Transform, TransformStamped ───────────────────────────


def _draw_frame(
    ax: Any, position: np.ndarray, rotation: np.ndarray, *, label: str, parent: str
) -> None:
    """A triad at `position` with its parent frame's faint triad at the origin."""
    length = max(0.05, 0.3 * float(np.linalg.norm(position)))
    draw_triad(
        ax, np.zeros(3), np.eye(3), length, label=parent, alpha=0.35, linewidth=1.6, gid="parent"
    )
    ax.plot(
        *zip(np.zeros(3), position, strict=True),
        color="#b0b0b0",
        linestyle=":",
        linewidth=1.0,
        gid="offset",
    )
    draw_triad(ax, position, rotation, length, label=label, gid="triad")
    reach = [
        np.zeros(3),
        position,
        *[position + rotation[:, i] * length for i in range(3)],
        np.eye(3)[0] * length,
        np.eye(3)[1] * length,
        np.eye(3)[2] * length,
    ]
    set_equal_3d(ax, np.array(reach), pad=0.15)


def _plot_frame(
    ax: Any, kind: str, position: Any, rotation: Any, label: str, parent: str, frame: str | None
) -> Figure:
    with style():
        ax = axes_for(ax, (5.2, 5.0), projection="3d")
        require_3d(ax, kind)
        tidy_3d(ax)
        _draw_frame(
            ax,
            np.asarray(position, dtype=np.float64),
            np.asarray(rotation),
            label=label,
            parent=parent,
        )
        where = f"{label} in {parent}" if frame is not None and parent else None
        ax.set_title(title_of(kind, where if where else frame), loc="left")
    return ax.figure


@plot.register(Pose)
def _plot_pose(pose: Pose, *, ax: Any = None) -> Figure:
    """A 3-D triad (x red, y green, z blue) at the pose, with the origin's triad faint."""
    return _plot_frame(
        ax, "Pose", pose.position.as_array(), pose.orientation.to_matrix(), "pose", "origin", None
    )


@plot.register(PoseStamped)
def _plot_pose_stamped(pose: PoseStamped, *, ax: Any = None) -> Figure:
    """A 3-D triad at the pose, named by `header.frame_id`."""
    frame = pose.header.frame_id
    return _plot_frame(
        ax,
        "PoseStamped",
        pose.pose.position.as_array(),
        pose.pose.orientation.to_matrix(),
        "pose",
        frame or "origin",
        frame or None,
    )


@plot.register(Transform)
def _plot_transform(transform: Transform, *, ax: Any = None) -> Figure:
    """A 3-D triad of the transformed frame, with the parent frame's triad faint."""
    return _plot_frame(
        ax,
        "Transform",
        transform.translation.as_array(),
        transform.rotation.to_matrix(),
        "child",
        "parent",
        None,
    )


@plot.register(TransformStamped)
def _plot_transform_stamped(transform: TransformStamped, *, ax: Any = None) -> Figure:
    """A 3-D triad labelled `child_frame_id`, with `header.frame_id` faint at the origin."""
    parent = transform.header.frame_id or "parent"
    return _plot_frame(
        ax,
        "TransformStamped",
        transform.transform.translation.as_array(),
        transform.transform.rotation.to_matrix(),
        transform.child_frame_id or "child",
        parent,
        parent,
    )


# ── PointCloud ───────────────────────────────────────────────────────────────


def _point_colors(cloud: PointCloud, index: np.ndarray) -> np.ndarray | None:
    """Per-point RGB from the `rgb` channel (uint8 → 0..1), or None when there is none."""
    rgb = cloud.channels.get("rgb")
    if rgb is None or rgb.ndim != 2 or rgb.shape[1] not in (3, 4) or rgb.dtype.kind not in "uf":
        return None
    rgb = rgb[index]
    if rgb.dtype.kind == "u":
        return rgb.astype(np.float64) / 255.0
    return np.clip(rgb.astype(np.float64), 0.0, 1.0)


@plot.register(PointCloud)
def _plot_point_cloud(cloud: PointCloud, *, ax: Any = None, max_points: int = MAX_POINTS) -> Figure:
    """3-D scatter, at most ``max_points`` (default 20 000, a seeded subsample so it is repeatable);
    coloured by the ``rgb`` channel when there is one, else by z."""
    count = len(cloud)
    # WHY a seeded draw, not every k-th point: a stride through row-major depth pixels aliases into
    # visible stripes; a fixed seed keeps the same cloud drawing the same picture every time.
    index = (
        np.arange(count)
        if count <= max_points
        else np.sort(np.random.default_rng(0).choice(count, size=max_points, replace=False))
    )
    points = cloud.points[index].astype(np.float64)
    with style():
        ax = axes_for(ax, (7.0, 6.0), projection="3d")
        require_3d(ax, "PointCloud")
        tidy_3d(ax)
        rgb = _point_colors(cloud, index)
        options: dict[str, Any] = (
            {"c": rgb} if rgb is not None else {"c": points[:, 2], "cmap": "viridis"}
        )
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2.5,
            depthshade=False,
            linewidths=0,
            rasterized=True,
            gid="points",
            **options,
        )
        if rgb is None and len(points):
            colorbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.55, pad=0.02, aspect=18)
            colorbar.set_label("z (m)")
            colorbar.outline.set_visible(False)
        set_equal_3d(ax, points)
        shown = (
            f"{count:,} points" if len(index) == count else f"{len(index):,} of {count:,} points"
        )
        ax.set_title(title_of("PointCloud", cloud.header.frame_id or None, shown), loc="left")
    return ax.figure


# ── 3-D boxes ────────────────────────────────────────────────────────────────

# Corners are in binary counting order (z fastest); an edge joins corners that differ in one bit.
BOX_EDGES = tuple((i, i | bit) for bit in (1, 2, 4) for i in range(8) if not i & bit)


def draw_boxes_3d(ax: Any, boxes: Sequence[tuple[BoundingBox3D, str | None, str]]) -> np.ndarray:
    """Wireframes (one Line3DCollection each) with optional labels; returns all corners."""
    all_corners = []
    for box, label, line_color in boxes:
        corners = box.corners()
        all_corners.append(corners)
        segments = np.array([[corners[a], corners[b]] for a, b in BOX_EDGES])
        ax.add_collection3d(
            Line3DCollection(segments, colors=line_color, linewidths=1.6, gid="box3d")
        )
        if label:
            top = corners[:, 2].max() + 0.03 * max(float(np.ptp(corners, axis=0).max()), 1e-3)
            ax.text(
                *corners[:, :2].mean(axis=0),
                top,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=line_color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
            )
    return np.concatenate(all_corners) if all_corners else np.zeros((0, 3))


def _plot_boxes(
    ax: Any,
    kind: str,
    boxes: Sequence[tuple[BoundingBox3D, str | None, str]],
    frame: str | None,
    extra: str | None = None,
) -> Figure:
    with style():
        ax = axes_for(ax, (6.0, 6.0), projection="3d")
        require_3d(ax, kind)
        tidy_3d(ax)
        corners = draw_boxes_3d(ax, boxes)
        set_equal_3d(ax, corners, pad=0.15)
        ax.set_title(title_of(kind, frame, extra), loc="left")
    return ax.figure


def _detection_boxes(
    detections: Sequence[Detection3D],
) -> list[tuple[BoundingBox3D, str | None, str]]:
    classes = class_colors([d.class_id for d in detections])
    boxes = []
    for detection in detections:
        label = (
            detection_label(detection.class_id, detection.score)
            if detection.results
            else detection.id
        )
        boxes.append((detection.bbox, label or None, classes[detection.class_id]))
    return boxes


@plot.register(BoundingBox3D)
def _plot_bounding_box_3d(box: BoundingBox3D, *, ax: Any = None) -> Figure:
    """A 3-D wireframe from the box's corners."""
    return _plot_boxes(ax, "BoundingBox3D", [(box, None, color(0))], None)


@plot.register(Detection3D)
def _plot_detection_3d(detection: Detection3D, *, ax: Any = None) -> Figure:
    """The detection's box as a wireframe labelled ``class score``."""
    return _plot_boxes(
        ax, "Detection3D", _detection_boxes([detection]), detection.header.frame_id or None
    )


@plot.register(Detection3DArray)
def _plot_detection_3d_array(detections: Detection3DArray, *, ax: Any = None) -> Figure:
    """One wireframe per detection, coloured by class and labelled ``class score``."""
    return _plot_boxes(
        ax,
        "Detection3DArray",
        _detection_boxes(detections.detections),
        detections.header.frame_id or None,
        f"{len(detections)} detections",
    )
