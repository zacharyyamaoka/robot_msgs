"""Viser: put any spatial robot_msgs message into a 3-D Viser scene (``robot-msgs[viser]``).

    from robot_msgs.viz import viser as viz_viser
    viz_viser.show(grasp=grasp, cloud=cloud)        # a live server; open the printed URL
    viz_viser.save("scene.viser", grasp=grasp)       # a recording any Viser client can play
    html = viz_viser.html(grasp=grasp)               # one self-contained HTML page

Viser is 3-D only: a message with nothing spatial in it (a joint vector, a time series) has no
logger here and `supports(type)` is False for it; its 2-D view is `msg.plot()` and Rerun.
One logger per base class (`singledispatch`), as robot_msgs.viz.rerun and robot_msgs.viz.plot.
Positions are metres in the scene's world frame; Viser quaternions are wxyz, robot_msgs' xyzw.
"""

from __future__ import annotations

import socket
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np

try:
    import viser
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "robot_msgs.viz.viser needs viser: pip install 'robot-msgs[viser]'"
    ) from error

from robot_msgs._core import Msg
from robot_msgs.geometry import (
    Accel,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
    Vector3,
    Wrench,
)
from robot_msgs.manipulation import GraspPose, Waypoint, WaypointAction
from robot_msgs.sensor import CameraInfo, Image, PointCloud
from robot_msgs.shape import Mesh, SolidPrimitive
from robot_msgs.trajectory import PoseTrajectory
from robot_msgs.vision import BoundingBox3D, Detection3D, Detection3DArray

VISER_VERSION = "1.1.1"
AXIS_LENGTH = 0.1
AXIS_RADIUS = 0.004
ACCENT = (34, 197, 94)
MUTED = (148, 163, 184)


def _wxyz(q: Quaternion) -> tuple[float, float, float, float]:
    return (q.w, q.x, q.y, q.z)


def _xyz(p: Any) -> tuple[float, float, float]:
    return (float(p.x), float(p.y), float(p.z))


# Points the loggers drew, collected while a scene is filled, so the camera can frame them.
_EXTENT: list[np.ndarray] = []


def _note(points: Any) -> None:
    _EXTENT.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def _part(name: object) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)) or "_"


@singledispatch
def log(msg: Any, name: str | None = None, *, server: viser.ViserServer) -> None:
    """Add `msg` to `server`'s scene under ``/<name>``."""
    if not isinstance(msg, Msg):
        raise TypeError(f"robot_msgs.viz.viser logs robot_msgs messages, got {type(msg).__name__}")
    raise TypeError(f"{type(msg).__name__} has nothing 3-D to show in Viser; use .plot() or Rerun")


def supports(message_type: type) -> bool:
    """True when `message_type` (or a base class) has a Viser logger."""
    return log.dispatch(message_type) is not log.dispatch(object)


def _frame(server: viser.ViserServer, path: str, pose: Pose, *, label: str | None = None) -> None:
    _note(_xyz(pose.position))
    server.scene.add_frame(
        path,
        wxyz=_wxyz(pose.orientation),
        position=_xyz(pose.position),
        axes_length=AXIS_LENGTH,
        axes_radius=AXIS_RADIUS,
    )
    if label:
        server.scene.add_label(f"{path}/label", label, position=(0.0, 0.0, AXIS_LENGTH * 1.2))


@log.register
def _(pose: Pose, name: str | None = None, *, server: viser.ViserServer) -> None:
    _frame(server, f"/{_part(name or 'Pose')}", pose)


@log.register
def _(stamped: PoseStamped, name: str | None = None, *, server: viser.ViserServer) -> None:
    _frame(server, f"/{_part(name or 'PoseStamped')}", stamped.pose, label=stamped.header.frame_id)


@log.register
def _(transform: Transform, name: str | None = None, *, server: viser.ViserServer) -> None:
    pose = Pose(Point(*_xyz(transform.translation)), transform.rotation)
    _frame(server, f"/{_part(name or 'Transform')}", pose)


@log.register
def _(stamped: TransformStamped, name: str | None = None, *, server: viser.ViserServer) -> None:
    t = stamped.transform
    pose = Pose(Point(*_xyz(t.translation)), t.rotation)
    label = f"{stamped.header.frame_id} → {stamped.child_frame_id}"
    _frame(server, f"/{_part(name or 'TransformStamped')}", pose, label=label)


@log.register
def _(pose: Pose2D, name: str | None = None, *, server: viser.ViserServer) -> None:
    _frame(
        server,
        f"/{_part(name or 'Pose2D')}",
        Pose.from_xyz_rpy(pose.x, pose.y, 0, 0, 0, pose.theta),
    )


@log.register
def _(q: Quaternion, name: str | None = None, *, server: viser.ViserServer) -> None:
    _frame(server, f"/{_part(name or 'Quaternion')}", Pose(Point(), q))


@log.register
def _(point: Point, name: str | None = None, *, server: viser.ViserServer) -> None:
    _note(_xyz(point))
    server.scene.add_icosphere(
        f"/{_part(name or 'Point')}", radius=0.01, color=ACCENT, position=_xyz(point)
    )


def _arrows(server, path: str, vectors: list[tuple[str, Vector3]], colors) -> None:
    starts, ends, rgb = [], [], []
    for (_, vector), color in zip(vectors, colors, strict=True):
        end = np.array(_xyz(vector))
        norm = float(np.linalg.norm(end))
        if norm == 0.0:
            continue
        starts.append((0.0, 0.0, 0.0))
        ends.append(tuple(end / max(norm, 1e-12) * min(norm, 0.5) if norm > 0.5 else end))
        rgb.append(color)
    if starts:
        _note(ends)
        _note([(0.0, 0.0, 0.0)])
        server.scene.add_arrows(
            path,
            points=np.array([starts, ends]).transpose(1, 0, 2),
            colors=np.array(rgb),
            shaft_radius=0.004,
        )


@log.register
def _(vector: Vector3, name: str | None = None, *, server: viser.ViserServer) -> None:
    _arrows(server, f"/{_part(name or 'Vector3')}", [("v", vector)], [ACCENT])


@log.register
def _(twist: Twist, name: str | None = None, *, server: viser.ViserServer) -> None:
    pairs = [("linear", twist.linear), ("angular", twist.angular)]
    _arrows(server, f"/{_part(name or 'Twist')}", pairs, [ACCENT, (59, 130, 246)])


@log.register
def _(accel: Accel, name: str | None = None, *, server: viser.ViserServer) -> None:
    pairs = [("linear", accel.linear), ("angular", accel.angular)]
    _arrows(server, f"/{_part(name or 'Accel')}", pairs, [ACCENT, (59, 130, 246)])


@log.register
def _(wrench: Wrench, name: str | None = None, *, server: viser.ViserServer) -> None:
    pairs = [("force", wrench.force), ("torque", wrench.torque)]
    _arrows(server, f"/{_part(name or 'Wrench')}", pairs, [(239, 68, 68), (168, 85, 247)])


@log.register
def _(traj: PoseTrajectory, name: str | None = None, *, server: viser.ViserServer) -> None:
    base = f"/{_part(name or 'PoseTrajectory')}"
    positions = np.asarray(traj.positions, dtype=np.float64)
    _note(positions)
    if len(positions) >= 2:
        server.scene.add_spline_catmull_rom(
            f"{base}/path", positions, color=ACCENT, thickness=3.0, thickness_units="screen"
        )
    # WHY a handful of frames, not every sample: orientation along the path reads at a glance
    # without a forest of axes.
    for i in np.unique(np.linspace(0, len(traj) - 1, min(len(traj), 6)).round().astype(int)):
        q = traj.orientations[i]
        server.scene.add_frame(
            f"{base}/frame_{i}", wxyz=(q[3], q[0], q[1], q[2]), position=tuple(positions[i]),
            axes_length=AXIS_LENGTH * 0.6, axes_radius=AXIS_RADIUS * 0.7,
        )  # fmt: skip


@log.register
def _(mesh: Mesh, name: str | None = None, *, server: viser.ViserServer) -> None:
    _note(mesh.vertices)
    server.scene.add_mesh_simple(
        f"/{_part(name or 'Mesh')}",
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.triangles, dtype=np.uint32),
        color=MUTED,
    )


@log.register
def _(solid: SolidPrimitive, name: str | None = None, *, server: viser.ViserServer) -> None:
    log(solid.to_mesh(), name or f"SolidPrimitive_{solid.type}", server=server)


@log.register
def _(cloud: PointCloud, name: str | None = None, *, server: viser.ViserServer) -> None:
    points = np.asarray(cloud.points, dtype=np.float32)
    _note(points)
    colors: Any = ACCENT
    rgb = cloud.channels.get("rgb") if cloud.channels else None
    if rgb is not None:
        rgb = np.asarray(rgb)
        colors = (rgb * 255 if rgb.dtype.kind == "f" and rgb.max() <= 1.0 else rgb).astype(np.uint8)
    server.scene.add_point_cloud(f"/{_part(name or 'PointCloud')}", points=points, colors=colors,
                                 point_size=0.005)  # fmt: skip


@log.register
def _(info: CameraInfo, name: str | None = None, *, server: viser.ViserServer) -> None:
    k = np.asarray(info.k).reshape(3, 3)
    fov = 2.0 * np.arctan2(info.height / 2.0, k[1, 1])
    _note([(0, 0, 0), (0.15, 0.15, 0.15), (-0.15, -0.15, 0)])
    server.scene.add_camera_frustum(
        f"/{_part(name or 'CameraInfo')}", fov=float(fov), aspect=info.width / info.height,
        scale=0.15, color=ACCENT,
    )  # fmt: skip


@log.register
def _(image: Image, name: str | None = None, *, server: viser.ViserServer) -> None:
    data = np.asarray(image.data)
    if data.ndim == 2:  # mono or depth: normalise to 8-bit grey
        finite = data[np.isfinite(data)] if data.dtype.kind == "f" else data
        span = float(finite.max() - finite.min()) if finite.size else 1.0
        data = ((data - finite.min()) / (span or 1.0) * 255).clip(0, 255).astype(np.uint8)
        data = np.repeat(data[..., None], 3, axis=2)
    if image.encoding.startswith("bgr"):
        data = data[..., ::-1]
    data = np.ascontiguousarray(data[..., :3]).astype(np.uint8)
    height = 0.3 * data.shape[0] / max(data.shape[1], 1)
    _note([(-0.15, -height / 2, 0), (0.15, height / 2, 0)])
    server.scene.add_image(
        f"/{_part(name or 'Image')}", data, render_width=0.3, render_height=height
    )


def _box(server, path: str, box: BoundingBox3D, label: str | None) -> None:
    half = np.array(_xyz(box.size)) / 2
    _note([np.array(_xyz(box.center.position)) + half, np.array(_xyz(box.center.position)) - half])
    server.scene.add_box(
        path, dimensions=_xyz(box.size), wxyz=_wxyz(box.center.orientation),
        position=_xyz(box.center.position), color=ACCENT, wireframe=True,
    )  # fmt: skip
    if label:
        server.scene.add_label(
            f"{path}/label", label, position=(0.0, 0.0, float(box.size.z) / 2 + 0.03)
        )


def _best(detection: Detection3D) -> str:
    if not detection.results:
        return detection.id
    best = max(detection.results, key=lambda r: r.hypothesis.score)
    return f"{best.hypothesis.class_id} {best.hypothesis.score:.2f}"


@log.register
def _(box: BoundingBox3D, name: str | None = None, *, server: viser.ViserServer) -> None:
    _box(server, f"/{_part(name or 'BoundingBox3D')}", box, None)


@log.register
def _(detection: Detection3D, name: str | None = None, *, server: viser.ViserServer) -> None:
    _box(server, f"/{_part(name or 'Detection3D')}", detection.bbox, _best(detection))


@log.register
def _(array: Detection3DArray, name: str | None = None, *, server: viser.ViserServer) -> None:
    base = f"/{_part(name or 'Detection3DArray')}"
    for i, detection in enumerate(array.detections):
        _box(server, f"{base}/{i}", detection.bbox, _best(detection))


def _gripper(server, path: str, width: float, *, color=ACCENT) -> None:
    """A two-finger gripper glyph in the tool frame: jaws open along x, approach along +z."""
    half = max(width, 0.004) / 2.0
    depth = 0.04
    segments = np.array(
        [
            [(-half, 0, 0), (-half, 0, -depth)],
            [(half, 0, 0), (half, 0, -depth)],
            [(-half, 0, -depth), (half, 0, -depth)],
            [(0, 0, -depth), (0, 0, -depth - 0.05)],
        ],
        dtype=np.float32,
    )
    server.scene.add_line_segments(
        path, points=segments, colors=color, thickness=3.0, thickness_units="screen"
    )


@log.register
def _(grasp: GraspPose, name: str | None = None, *, server: viser.ViserServer) -> None:
    base = f"/{_part(name or 'GraspPose')}"
    label = f"{grasp.object_id or 'grasp'}  {grasp.width * 1000:.0f} mm"
    if grasp.score is not None:
        label += f"  ({grasp.score:.2f})"
    _frame(server, base, grasp.pose, label=label)
    _gripper(server, f"{base}/jaws", grasp.width)
    if grasp.pre_grasp_width is not None:
        _gripper(server, f"{base}/pre_grasp_jaws", grasp.pre_grasp_width, color=MUTED)


@log.register
def _(waypoint: Waypoint, name: str | None = None, *, server: viser.ViserServer) -> None:
    _frame(
        server,
        f"/{_part(name or waypoint.name)}",
        waypoint.pose,
        label=waypoint.label or waypoint.name,
    )


@log.register
def _(action: WaypointAction, name: str | None = None, *, server: viser.ViserServer) -> None:
    base = f"/{_part(name or 'WaypointAction')}"
    points = np.array([_xyz(w.pose.position) for w in action.waypoints])
    if len(points) >= 2:
        server.scene.add_line_segments(
            f"{base}/path",
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=MUTED,
            thickness=2.0,
            thickness_units="screen",
        )
    for i, w in enumerate(action.waypoints):
        _frame(server, f"{base}/{i}_{_part(w.name)}", w.pose, label=f"{i} {w.label or w.name}")
        if w.gripper_width is not None:
            _gripper(server, f"{base}/{i}_{_part(w.name)}/jaws", w.gripper_width)


# ── entry points ─────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _named(messages: tuple[Msg, ...], named: dict[str, Msg]) -> dict[str, Msg]:
    out = {f"{type(m).__name__}_{i}" if len(messages) > 1 else type(m).__name__: m
           for i, m in enumerate(messages)}  # fmt: skip
    out.update(named)
    return out


def _fill(server: viser.ViserServer, messages: tuple[Msg, ...], named: dict[str, Msg]) -> None:
    server.scene.set_up_direction("+z")
    _EXTENT.clear()
    for key, msg in _named(messages, named).items():
        log(msg, key, server=server)
    _frame_camera(server)


def _frame_camera(server: viser.ViserServer) -> None:
    """Point the opening camera at what was drawn, and size the floor grid to it.

    WHY: Viser opens at (3, 3, 3) looking at the origin, where a 5 cm grasp is a speck.
    """
    points = np.concatenate(_EXTENT) if _EXTENT else np.zeros((1, 3))
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        points = np.zeros((1, 3))
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = max(float(np.max(high - low)), 2 * AXIS_LENGTH)
    cell = 10 ** np.floor(np.log10(span / 2))
    size = float(np.ceil(span * 1.5 / cell) * cell)
    server.scene.add_grid("/grid", width=size, height=size, cell_size=float(cell), plane="xy",
                          position=(float(center[0]), float(center[1]), float(low[2])))  # fmt: skip
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.position = tuple(center + np.array([1.0, -1.3, 0.9]) * span * 1.1)


def scene(*messages: Msg, port: int | None = None, **named: Msg) -> viser.ViserServer:
    """A new Viser server holding every message given (keyword name = scene path)."""
    server = viser.ViserServer(host="127.0.0.1", port=port or _free_port(), verbose=False)
    _fill(server, messages, named)
    return server


_OFFLINE: viser.ViserServer | None = None


def _offline(messages: tuple[Msg, ...], named: dict[str, Msg]) -> viser.ViserServer:
    """One private server reused by `save` and `html`, its scene reset each call.

    WHY reuse: starting a ViserServer takes ~4.6 s and stopping it ~0.7 s; robotmsgs.com saves a
    scene per type, so a fresh server each time made the site build take four minutes.
    """
    global _OFFLINE
    if _OFFLINE is None:
        _OFFLINE = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    else:
        _OFFLINE.scene.reset()
    try:
        _fill(_OFFLINE, messages, named)
    except BaseException:
        _OFFLINE.scene.reset()
        raise
    return _OFFLINE


def show(*messages: Msg, port: int = 8080, **named: Msg) -> viser.ViserServer:
    """Serve the messages on `port` until the process exits; open the URL Viser prints."""
    return scene(*messages, port=port, **named)


def save(path: str | Path, *messages: Msg, **named: Msg) -> str:
    """Write the scene as a Viser recording (bytes any Viser client can play back)."""
    server = _offline(messages, named)
    Path(path).write_bytes(server.get_scene_serializer().serialize())
    return str(path)


def html(*messages: Msg, dark_mode: bool = False, **named: Msg) -> str:
    """One self-contained HTML page with the Viser client and this scene inside it."""
    return _offline(messages, named).scene.as_html(dark_mode=dark_mode)


__all__ = ["VISER_VERSION", "html", "log", "save", "scene", "show", "supports"]
