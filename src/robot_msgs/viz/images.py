"""Plots in pixel coordinates: images, 2-D detections, camera intrinsics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from robot_msgs.sensor import CameraInfo, Image
from robot_msgs.vision import BoundingBox2D, Detection2D, Detection2DArray
from robot_msgs.viz._common import (
    MUTED,
    axes_for,
    class_colors,
    color,
    detection_label,
    style,
    tidy,
    title_of,
)
from robot_msgs.viz._dispatch import plot


def _pixel_axes(ax: Any) -> None:
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_aspect("equal")


def draw_boxes_2d(ax: Any, detections: Sequence[Detection2D]) -> list[Rectangle]:
    """A rectangle per detection (rotated by `bbox.center.theta`) with a ``class score`` tag."""
    classes = class_colors([d.class_id for d in detections])
    patches = []
    for detection in detections:
        box = detection.bbox
        edge = classes[detection.class_id]
        patch = Rectangle(
            (box.center.x - box.size_x / 2.0, box.center.y - box.size_y / 2.0),
            box.size_x,
            box.size_y,
            angle=math.degrees(box.center.theta),
            rotation_point="center",
            fill=False,
            edgecolor=edge,
            linewidth=2.0,
            gid="box",
        )
        ax.add_patch(patch)
        patches.append(patch)
        label = (
            detection_label(detection.class_id, detection.score)
            if detection.results
            else detection.id
        )
        if label:
            corners = patch.get_corners()
            corner = corners[
                np.lexsort((corners[:, 0], corners[:, 1]))[0]
            ]  # top-most, then left-most
            ax.annotate(
                label,
                corner,
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                gid="label",
                bbox={"boxstyle": "square,pad=0.2", "facecolor": edge, "edgecolor": "none"},
            )
    return patches


def _show_image(ax: Any, image: Image) -> None:
    ax.imshow(image.to_rgb(), interpolation="nearest")
    ax.grid(False)


def _image_title(image: Image) -> str:
    return title_of(
        "Image", image.header.frame_id or None, f"{image.encoding} {image.width}×{image.height}"
    )


@plot.register(Image)
def _plot_image(
    image: Image, *, ax: Any = None, detections: Detection2DArray | None = None
) -> Figure:
    """The picture (``to_rgb()``), titled with encoding and size.

    Option ``detections=Detection2DArray`` overlays boxes labelled ``class score``.
    """
    with style():
        ax = axes_for(ax, (6.4, 5.0))
        tidy(ax)
        _show_image(ax, image)
        if detections is not None:
            draw_boxes_2d(ax, detections.detections)
        _pixel_axes(ax)
        extra = f" · {len(detections)} detections" if detections is not None else ""
        ax.set_title(_image_title(image) + extra, loc="left")
    return ax.figure


@plot.register(Detection2DArray)
def _plot_detection_2d_array(
    detections: Detection2DArray, *, ax: Any = None, image: Image | None = None
) -> Figure:
    """Boxes labelled ``class score`` in pixel coordinates; ``image=`` draws them over a picture."""
    with style():
        ax = axes_for(ax, (6.4, 5.0))
        tidy(ax)
        if image is not None:
            _show_image(ax, image)
        patches = draw_boxes_2d(ax, detections.detections)
        if image is None:
            corners = (
                np.concatenate([p.get_corners() for p in patches]) if patches else np.zeros((1, 2))
            )
            low, high = corners.min(axis=0), corners.max(axis=0)
            pad = 0.1 * max(float((high - low).max()), 1.0)
            ax.set_xlim(low[0] - pad, high[0] + pad)
            ax.set_ylim(high[1] + pad, low[1] - pad)
        _pixel_axes(ax)
        ax.set_title(
            title_of(
                "Detection2DArray",
                detections.header.frame_id or None,
                f"{len(detections)} detections",
            ),
            loc="left",
        )
    return ax.figure


@plot.register(Detection2D)
def _plot_detection_2d(
    detection: Detection2D, *, ax: Any = None, image: Image | None = None
) -> Figure:
    """One box, as a one-element Detection2DArray."""
    return _plot_detection_2d_array(
        Detection2DArray(header=detection.header, detections=(detection,)), ax=ax, image=image
    )


@plot.register(BoundingBox2D)
def _plot_bounding_box_2d(box: BoundingBox2D, *, ax: Any = None) -> Figure:
    """One unlabelled box in pixel coordinates."""
    return _plot_detection_2d_array(Detection2DArray(detections=(Detection2D(bbox=box),)), ax=ax)


@plot.register(CameraInfo)
def _plot_camera_info(camera: CameraInfo, *, ax: Any = None) -> Figure:
    """The image rectangle and principal point in pixels; fx, fy in the title."""
    with style():
        ax = axes_for(ax, (6.0, 4.8))
        tidy(ax, grid=False)
        ax.add_patch(
            Rectangle(
                (0, 0),
                camera.width,
                camera.height,
                fill=False,
                edgecolor="#555555",
                linewidth=1.4,
                gid="image_rect",
            )
        )
        ax.axhline(camera.cy, color=color(0), linewidth=0.8, linestyle=":")
        ax.axvline(camera.cx, color=color(0), linewidth=0.8, linestyle=":")
        ax.plot(
            [camera.width / 2],
            [camera.height / 2],
            marker="+",
            color=MUTED,
            markersize=9,
            linestyle="none",
            gid="image_centre",
        )
        ax.plot(
            [camera.cx],
            [camera.cy],
            marker="o",
            color=color(0),
            markersize=7,
            linestyle="none",
            gid="principal_point",
        )
        ax.annotate(
            f"principal point ({camera.cx:g}, {camera.cy:g})",
            (camera.cx, camera.cy),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color(0),
        )
        margin = 0.06 * max(camera.width, camera.height)
        ax.set_xlim(-margin, camera.width + margin)
        ax.set_ylim(camera.height + margin, -margin)
        _pixel_axes(ax)
        ax.set_title(
            title_of(
                "CameraInfo",
                camera.header.frame_id or None,
                f"fx={camera.fx:g} fy={camera.fy:g} px",
                f"{camera.width}×{camera.height}",
            ),
            loc="left",
        )
    return ax.figure
