"""Plots of shapes: triangle meshes, solid primitives (as their mesh), planes and colours."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robot_msgs.shape import Mesh, Plane, SolidPrimitive
from robot_msgs.std import ColorRGBA
from robot_msgs.viz._common import (
    MUTED,
    axes_for,
    color,
    require_3d,
    set_equal_3d,
    style,
    tidy_3d,
    title_of,
)
from robot_msgs.viz._dispatch import plot

SOLID_NAMES = {1: "BOX", 2: "SPHERE", 3: "CYLINDER", 4: "CONE", 5: "PRISM"}


def _draw_mesh(ax: Any, mesh: Mesh) -> None:
    """One polygon per triangle, shaded by its normal against a fixed light, so the form reads."""
    faces = mesh.vertices[mesh.triangles].astype(np.float64)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.4 + 0.6 * np.abs(normals @ light)  # abs: a flipped triangle is still visible
    base = np.array(to_rgb(color(0)))
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=np.clip(base[None, :] * shade[:, None], 0.0, 1.0),
            edgecolors="#ffffff" if len(faces) < 600 else "none",
            linewidths=0.4,
            gid="mesh",
        )
    )


def _plot_mesh_in(ax: Any, kind: str, mesh: Mesh, *parts: str | None) -> Figure:
    with style():
        ax = axes_for(ax, (6.0, 6.0), projection="3d")
        require_3d(ax, kind)
        tidy_3d(ax)
        _draw_mesh(ax, mesh)
        set_equal_3d(ax, mesh.vertices, pad=0.15)
        ax.set_title(title_of(kind, *parts), loc="left")
    return ax.figure


@plot.register(Mesh)
def _plot_mesh(mesh: Mesh, *, ax: Any = None) -> Figure:
    """Every triangle as a shaded polygon, equal axes."""
    return _plot_mesh_in(ax, "Mesh", mesh, f"{len(mesh)} triangles")


@plot.register(SolidPrimitive)
def _plot_solid_primitive(solid: SolidPrimitive, *, ax: Any = None, resolution: int = 48) -> Figure:
    """The primitive's `to_mesh(resolution)`, titled with its type and dimensions."""
    dimensions = " × ".join(f"{value:g}" for value in solid.dimensions)
    return _plot_mesh_in(
        ax,
        "SolidPrimitive",
        solid.to_mesh(resolution),
        f"{SOLID_NAMES[solid.type]} {dimensions} m",
    )


@plot.register(Plane)
def _plot_plane(plane: Plane, *, ax: Any = None, size: float = 1.0) -> Figure:
    """A square of side `size` (m) in the plane around the point nearest the origin, with the
    unit normal as an arrow and the origin marked, so the plane's offset reads."""
    unit = plane.normal.as_array()
    centre = -plane.coef[3] / np.linalg.norm(plane.coef[:3]) * unit
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(unit, helper)
    u /= np.linalg.norm(u)
    v = np.cross(unit, u)
    half = size / 2.0
    corners = np.array(
        [centre + a * half * u + b * half * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )
    tip = centre + unit * size / 3.0
    with style():
        ax = axes_for(ax, (6.0, 6.0), projection="3d")
        require_3d(ax, "Plane")
        tidy_3d(ax)
        ax.add_collection3d(
            Poly3DCollection(
                [corners], facecolors=color(0), alpha=0.3, edgecolors=color(0), gid="plane"
            )
        )
        ax.plot(*zip(centre, tip, strict=True), color=color(5), linewidth=2.0, gid="normal")
        ax.plot([0.0], [0.0], [0.0], "o", color=MUTED, markersize=4, gid="origin")
        set_equal_3d(ax, np.vstack([corners, tip, np.zeros(3)]), pad=0.15)
        a, b, c, d = plane.coef.tolist()
        ax.set_title(title_of("Plane", f"{a:g}x + {b:g}y + {c:g}z + {d:g} = 0"), loc="left")
    return ax.figure


@plot.register(ColorRGBA)
def _plot_color(rgba: ColorRGBA, *, ax: Any = None) -> Figure:
    """A swatch of the colour over a light checker (so alpha shows), labelled with its hex."""
    with style():
        ax = axes_for(ax, (2.6, 1.6))
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(8):
            for j in range(4):
                ax.add_patch(
                    Rectangle(
                        (i / 4, j / 4),
                        0.25,
                        0.25,
                        facecolor="#e4e4e4" if (i + j) % 2 else "#ffffff",
                        linewidth=0,
                    )
                )
        ax.add_patch(
            Rectangle(
                (0.0, 0.0), 2.0, 1.0, facecolor=tuple(rgba.as_array()), linewidth=0, gid="swatch"
            )
        )
        luminance = 0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b
        ax.text(
            1.0,
            0.5,
            rgba.to_hex(),
            ha="center",
            va="center",
            fontsize=11,
            color="black" if luminance > 0.5 or rgba.a < 0.5 else "white",
        )
        ax.set_title(title_of("ColorRGBA"), loc="left")
    return ax.figure
