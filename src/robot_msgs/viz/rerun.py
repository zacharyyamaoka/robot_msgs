"""Every standard message in Rerun: ``msg.show()`` opens it, ``log`` composes several.

    traj.show()                                    # opens the Rerun viewer on this trajectory
    show(target=planned, real=measured)            # overlay: one plot per joint, one line per name
    save("pick.rrd", target=planned, real=measured)

Needs ``pip install 'robot-msgs[rerun]'``; `import robot_msgs` never imports this module.

WHY Rerun (Zach, 2026-11-11): a great stock viewer for time series, 3D, images and video, so we
log into it instead of building a viewer. One logger per base class (`functools.singledispatch`, as
`robot_msgs.viz.plot` does), so a subclass falls back to its parent and a new type is one function.

Entity layout, shared with traj_compare's Rerun projection so the two overlay:

    /<field>/<column>/<name>     sampled data on the ``t`` timeline (seconds from the first sample)
    /world/<name>/...            anything spatial: poses, paths, clouds, solids, boxes
    /image/<name>, /camera/<name>, /notes/<name>

``<name>`` is the keyword you passed to `show`/`log` (``real``, ``target``), or the type name.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rerun as rr
except ModuleNotFoundError as error:  # pragma: no cover - exercised only without the extra
    if error.name != "rerun":
        raise
    raise ImportError(
        "robot_msgs.viz.rerun needs Rerun: pip install 'robot-msgs[rerun]'"
    ) from error

from robot_msgs._core import Msg
from robot_msgs.compare.comparison import Comparison
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
from robot_msgs.sensor import CameraInfo, Image, JointState, PointCloud
from robot_msgs.shape import Mesh, SolidPrimitive
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, Trajectory
from robot_msgs.trajectory.trajectory import sample_fields
from robot_msgs.vision import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
)

# WHY one pin, the same as traj_compare's RERUN_SDK_VERSION: an .rrd must be opened by a viewer at
# least as new as the SDK that wrote it.
RERUN_SDK_VERSION = "0.40.0"
APPLICATION_ID = "robot_msgs"
TIMELINE = "t"
AXIS_LENGTH = 0.1  # metres: the triad drawn at every pose
_DEPTH_ENCODINGS = {"16UC1", "32FC1"}
_COLOR_MODELS = {"rgb8": "RGB", "bgr8": "BGR", "rgba8": "RGBA", "bgra8": "BGRA", "8UC3": "RGB"}


def _part(name: object) -> str:
    """One entity-path segment: Rerun paths split on '/', so keep names to safe characters."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)) or "_"


def path_part(name: object) -> str:
    """The entity-path segment `log` writes for ``name`` (a joint, a run), for building views on it."""
    return _part(name)


# Every entity path written into each live stream, so `_blueprint` can lay out what was logged.
_PATHS: dict[int, list[str]] = {}


def _log(stream: rr.RecordingStream, path: str, *archetypes: Any, **options: Any) -> None:
    _PATHS.setdefault(id(stream), []).append(path)
    stream.log(path, *archetypes, **options)


def _send(stream: rr.RecordingStream, path: str, **options: Any) -> None:
    _PATHS.setdefault(id(stream), []).append(path)
    stream.send_columns(path, **options)


_SPECIAL_ROOTS = ("world", "image", "camera", "notes")


def _blueprint(paths: list[str]) -> Any:
    """The default dashboard for whatever was logged.

    WHY not Rerun's automatic layout: it gives every `/<field>/<column>/<name>` its own plot,
    so `show(target=a, real=b)` lost the overlay. Here one plot per `<field>/<column>` holds every
    name, each field is a tab, and anything spatial gets one 3D view beside them.
    """
    import rerun.blueprint as rrb

    unique = list(dict.fromkeys(paths))
    roots = {path.split("/")[0] for path in unique}
    series: dict[str, list[str]] = {}
    for path in unique:
        parts = path.split("/")
        if parts[0] in _SPECIAL_ROOTS or len(parts) < 3:
            continue
        field_name, column = "/".join(parts[:-2]), "/".join(parts[:-1])
        if column not in series.setdefault(field_name, []):
            series[field_name].append(column)
    tabs = [
        rrb.Grid(
            *(
                rrb.TimeSeriesView(origin=f"/{column}", name=column.split("/")[-1])
                for column in columns
            ),
            name=field_name,
        )
        for field_name, columns in series.items()
    ]
    side: list[Any] = []
    if "world" in roots:
        side.append(rrb.Spatial3DView(origin="/world", name="world"))
    side += [
        rrb.Spatial2DView(origin=f"/{p}", name=p.split("/")[-1])
        for p in unique
        if p.startswith("image/") and p.count("/") == 1
    ]
    side += [
        rrb.TextDocumentView(origin=f"/{p}", name=p.split("/")[-1])
        for p in unique
        if p.startswith("notes/")
    ]
    parts: list[Any] = []
    if side:
        parts.append(side[0] if len(side) == 1 else rrb.Vertical(*side))
    if tabs:
        parts.append(tabs[0] if len(tabs) == 1 else rrb.Tabs(*tabs))
    if not parts:
        return None
    root = parts[0] if len(parts) == 1 else rrb.Horizontal(*parts, column_shares=[2, 3])
    return rrb.Blueprint(root, collapse_panels=False)


def _finish(stream: rr.RecordingStream) -> None:
    """Send the default dashboard for everything logged into `stream`."""
    blueprint = _blueprint(_PATHS.pop(id(stream), []))
    if blueprint is not None:
        stream.send_blueprint(blueprint)


# ── the generic entry points ─────────────────────────────────────────────────


@singledispatch
def log(msg: Any, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    """Log one message into `recording` under `name` (default: its type name).

    Implementations are registered per base class below; an unregistered `Msg` is logged as a
    text document of its repr, so every message shows *something*.
    """
    if not isinstance(msg, Msg):
        raise TypeError(f"robot_msgs.viz.rerun can only log messages, not {type(msg).__name__}")
    name = name or type(msg).__name__
    _log(
        recording,
        f"notes/{_part(name)}",
        rr.TextDocument(f"```\n{msg!r}\n```", media_type="text/markdown"),
        static=True,
    )


def recording(
    *messages: Msg, application_id: str = APPLICATION_ID, **named: Msg
) -> rr.RecordingStream:
    """A fresh recording holding every message: positional ones under their type name."""
    stream = rr.RecordingStream(application_id)
    for msg in messages:
        log(msg, recording=stream)
    for name, msg in named.items():
        log(msg, name, recording=stream)
    _finish(stream)
    return stream


def show(*messages: Msg, **named: Msg) -> rr.RecordingStream:
    """Open the Rerun viewer on these messages (a notebook cell shows it inline).

    Keyword names become the line names, so ``show(target=a, real=b)`` overlays the two.
    """
    return view(recording(*messages, **named))


def compose(
    *entries: tuple[str, Msg], application_id: str = APPLICATION_ID, into: str | None = None
) -> rr.RecordingStream:
    """One recording of ``(name, message)`` pairs, its default dashboard sent; a name may repeat.

    WHY pairs and not keywords (2026-11-11): a run's recording holds each trajectory and its
    Comparison under one name (``("real", real), ("real", compare(target, real))``), which keyword
    arguments cannot. ``into`` saves the recording to that ``.rrd`` path as it is logged.
    """
    stream = rr.RecordingStream(application_id)
    if into is not None:
        stream.save(into)
    for name, msg in entries:
        log(msg, name, recording=stream)
    _finish(stream)
    return stream


def view(stream: rr.RecordingStream) -> rr.RecordingStream:
    """Open the Rerun viewer on a recording (a notebook cell shows it inline)."""
    if _in_notebook():
        stream.notebook_show()
    else:
        stream.spawn()
    return stream


def save(path: str, *messages: Msg, **named: Msg) -> str:
    """Write the messages to an ``.rrd`` file; open it with ``rerun <path>``."""
    stream = rr.RecordingStream(APPLICATION_ID)
    stream.save(path)
    for msg in messages:
        log(msg, recording=stream)
    for name, msg in named.items():
        log(msg, name, recording=stream)
    _finish(stream)
    stream.flush()
    stream.disconnect()
    return path


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


# ── trajectories: one scalar series per column, on the t timeline ────────────


def _columns(traj: Trajectory, field_name: str, values: np.ndarray) -> list[str]:
    width = values.shape[1] if values.ndim > 1 else 1
    if isinstance(traj, JointTrajectory) and field_name in (
        "positions",
        "velocities",
        "accelerations",
        "effort",
    ):
        return [str(n) for n in traj.joint_names]
    if isinstance(traj, PoseTrajectory) and width == 4 and field_name == "orientations":
        return ["x", "y", "z", "w"]
    if width == 3:
        return ["x", "y", "z"]
    return [str(i) for i in range(width)]


_ANGULAR_FIELDS = ("positions", "velocities", "accelerations")


def _degree_joints(traj: Trajectory, degrees: bool | Collection[str]) -> set[str]:
    """The joints whose angular columns are shown in degrees.

    robot_msgs carries no joint type (JointLimits holds bounds only), so ``degrees=True`` treats
    every joint as revolute; pass the revolute joints' names instead when some are prismatic."""
    if not degrees or not isinstance(traj, JointTrajectory):
        return set()
    names = set(traj.joint_names) if degrees is True else {str(name) for name in degrees}
    unknown = names - set(traj.joint_names)
    if unknown:
        raise ValueError(f"degrees names joints this trajectory lacks: {sorted(unknown)}")
    return names


def _series(
    traj: Trajectory,
    name: str,
    stream: rr.RecordingStream,
    prefix: str = "",
    *,
    degrees: bool | Collection[str] = False,
    color: Any = None,
    t0: float | None = None,
    label: str | None = None,
) -> None:
    t = np.asarray(traj.t, dtype=np.float64)
    start = t0 if t0 is not None else (t[0] if t.size else 0.0)
    seconds = t - start
    in_degrees = _degree_joints(traj, degrees)
    arrays: dict[str, np.ndarray] = {}
    for f in sample_fields(type(traj)):
        value = getattr(traj, f.name)
        if value is not None:
            arrays[f.name] = np.asarray(value)
    for key, value in (traj.channels or {}).items():
        arrays[f"channels/{key}"] = np.asarray(value)
    for field_name, values in arrays.items():
        if values.dtype.kind not in "fiub" or values.shape[:1] != t.shape:
            continue
        flat = values.reshape(len(t), -1).astype(np.float64)
        for column, column_name in enumerate(_columns(traj, field_name.split("/")[-1], values)):
            path = f"{prefix}{field_name}/{_part(column_name)}/{_part(name)}"
            scale = 180.0 / np.pi if field_name in _ANGULAR_FIELDS and column_name in in_degrees else 1.0
            legend = label or name
            style = rr.SeriesLines(names=legend) if color is None else rr.SeriesLines(names=legend, colors=color)
            _log(stream, path, style, static=True)
            _send(
                stream,
                path,
                indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
                columns=rr.Scalars.columns(scalars=flat[:, column] * scale),
            )


@log.register
def _(
    traj: Trajectory,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = False,
    color: Any = None,
    prefix: str = "",
    t0: float | None = None,
    label: str | None = None,
) -> None:
    """One series per column at ``<prefix><field>/<column>/<name>``.

    ``degrees`` shows joint angles (and their rates) in degrees: True for every joint, or the
    revolute joints' names; the message itself stays SI. ``color`` colours every series (one colour
    per name keeps a run recognisable across plots); ``prefix`` roots the series (``"error/"``);
    ``t0`` is the time shown as zero (default: this trajectory's first sample), so several
    trajectories on one clock share it; ``label`` is the legend text (default: ``name``), so a path can
    stay short (``real_1``) while the legend reads ``Real #1``."""
    _series(traj, name or type(traj).__name__, recording, prefix, degrees=degrees, color=color, t0=t0, label=label)


@log.register
def _(traj: PoseTrajectory, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or type(traj).__name__
    _series(traj, name, recording)
    positions = np.asarray(traj.positions, dtype=np.float64)
    quaternions = np.asarray(traj.orientations, dtype=np.float64)  # xyzw, as Rerun takes it
    t = np.asarray(traj.t, dtype=np.float64)
    seconds = t - t[0] if t.size else t
    base = f"world/{_part(name)}"
    _log(recording, f"{base}/path", rr.LineStrips3D([positions]), static=True)
    _log(recording, f"{base}/frame", rr.TransformAxes3D(AXIS_LENGTH), static=True)
    _send(
        recording,
        f"{base}/frame",
        indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
        columns=rr.Transform3D.columns(translation=positions, quaternion=quaternions),
    )


@log.register
def _(comparison: Comparison, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "error"
    # WHY under error/ (2026-11-11): a recording often holds the compared trajectory under the same name (target / sim / real and each
    # one's Comparison); on the trajectory's own path the error would share its plot and its series. error/<field>/<column>/<name> is
    # its own tab, as traj_compare's Rerun projection keeps /error/...
    _series(comparison.error, name, recording, prefix="error/")
    rows = ["| metric | rms | p95 abs | max abs | final |", "|---|---|---|---|---|"]
    for metric, stats in comparison.metrics.items():
        values = [getattr(stats, key, None) for key in ("rms", "p95_abs", "max_abs", "final")]
        cells = [f"{value:.6g}" if isinstance(value, float) else "" for value in values]
        rows.append(f"| {metric} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append(
        f"time offset {comparison.time_offset_s:.6g} s · dropped {comparison.dropped}"
        f" · coverage {comparison.coverage:.1%}"
    )
    _log(
        recording,
        f"notes/{_part(name)}",
        rr.TextDocument("\n".join(rows), media_type="text/markdown"),
        static=True,
    )


# ── geometry: a triad at a pose, an arrow for a vector ───────────────────────


def _frame(path: str, translation: Any, quaternion_xyzw: Any, stream: rr.RecordingStream) -> None:
    _log(
        stream,
        path,
        rr.Transform3D(translation=translation, quaternion=quaternion_xyzw),
        rr.TransformAxes3D(AXIS_LENGTH),
        static=True,
    )


def _xyzw(q: Quaternion) -> list[float]:
    return [q.x, q.y, q.z, q.w]


@log.register
def _(pose: Pose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    p = pose.position
    _frame(f"world/{_part(name or 'Pose')}", [p.x, p.y, p.z], _xyzw(pose.orientation), recording)


@log.register
def _(stamped: PoseStamped, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    log(stamped.pose, name or stamped.header.frame_id or "PoseStamped", recording=recording)


@log.register
def _(transform: Transform, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    v = transform.translation
    _frame(
        f"world/{_part(name or 'Transform')}", [v.x, v.y, v.z], _xyzw(transform.rotation), recording
    )


@log.register
def _(stamped: TransformStamped, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    log(
        stamped.transform, name or stamped.child_frame_id or "TransformStamped", recording=recording
    )


@log.register
def _(pose: Pose2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    half = pose.theta / 2.0
    _frame(
        f"world/{_part(name or 'Pose2D')}",
        [pose.x, pose.y, 0.0],
        [0.0, 0.0, np.sin(half), np.cos(half)],
        recording,
    )


@log.register
def _(q: Quaternion, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _frame(f"world/{_part(name or 'Quaternion')}", [0.0, 0.0, 0.0], _xyzw(q), recording)


@log.register
def _(point: Point, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _log(
        recording,
        f"world/{_part(name or 'Point')}",
        rr.Points3D([[point.x, point.y, point.z]], radii=0.01),
        static=True,
    )


@log.register
def _(vector: Vector3, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _log(
        recording,
        f"world/{_part(name or 'Vector3')}",
        rr.Arrows3D(vectors=[[vector.x, vector.y, vector.z]]),
        static=True,
    )


def _pair(
    path: str, labels: tuple[str, str], first: Vector3, second: Vector3, stream: rr.RecordingStream
) -> None:
    vectors = [[first.x, first.y, first.z], [second.x, second.y, second.z]]
    _log(
        stream,
        path,
        rr.Arrows3D(vectors=vectors, labels=list(labels), colors=[[230, 120, 40], [60, 140, 230]]),
        static=True,
    )


@log.register
def _(twist: Twist, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(
        f"world/{_part(name or 'Twist')}",
        ("linear", "angular"),
        twist.linear,
        twist.angular,
        recording,
    )


@log.register
def _(accel: Accel, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(
        f"world/{_part(name or 'Accel')}",
        ("linear", "angular"),
        accel.linear,
        accel.angular,
        recording,
    )


@log.register
def _(wrench: Wrench, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(
        f"world/{_part(name or 'Wrench')}",
        ("force", "torque"),
        wrench.force,
        wrench.torque,
        recording,
    )


# ── on the clock: a shaded band, a video, a robot's links ─────────────────────


def band(
    path: str,
    t: Any,
    low: Any,
    high: Any,
    *,
    recording: rr.RecordingStream,
    color: Any = (120, 120, 120, 70),
    name: str = "tolerance",
) -> None:
    """A shaded band between ``low`` and ``high`` on the ``t`` timeline (``t`` in seconds, as logged).

    WHY Measurements: Rerun 0.40 has no fill-between, but `Measurements` shades value ± one
    standard deviation, so centre = (low + high) / 2 and variance = ((high - low) / 2)² draw
    exactly the band. A tolerance plot puts it beside the error series under the same parent."""
    t = np.asarray(t, dtype=np.float64)
    low = np.broadcast_to(np.asarray(low, dtype=np.float64), t.shape)
    high = np.broadcast_to(np.asarray(high, dtype=np.float64), t.shape)
    _log(recording, path, rr.Measurements.from_fields(colors=[color], names=[name], widths=[0.5]), static=True)
    _send(
        recording,
        path,
        indexes=[rr.TimeColumn(TIMELINE, duration=t)],
        columns=rr.Measurements.columns(values=0.5 * (low + high), variances=(0.5 * (high - low)) ** 2),
    )


def video(path: str | Path, name: str, *, recording: rr.RecordingStream, start_s: float = 0.0) -> None:
    """An MP4 at ``video/<name>`` on the ``t`` timeline: frame k at its own timestamp + ``start_s``.

    ``start_s`` is the timeline time shown at video time 0 (a camera rarely starts recording
    exactly when the trajectory does)."""
    asset = rr.AssetVideo(path=str(path))
    entity = f"video/{_part(name)}"
    _log(recording, entity, asset, static=True)
    nanos = asset.read_frame_timestamps_nanos()
    _send(
        recording,
        entity,
        indexes=[rr.TimeColumn(TIMELINE, duration=nanos / 1e9 + start_s)],
        columns=rr.VideoFrameReference.columns_nanos(nanos),
    )


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    faces = vertices[triangles]
    face_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def links(
    parts: Sequence[tuple[str, Mesh, PoseTrajectory]],
    name: str,
    *,
    recording: rr.RecordingStream,
    color: Any = None,
    t0: float | None = None,
) -> None:
    """A robot as (link name, mesh in the link's frame, the link's pose over time) at ``world/<name>/robot/<link>``.

    WHY these three and not a robot description: robot_msgs depends on nothing, so the caller
    (robot_description, traj_compare) turns its own model into meshes and pose trajectories.
    ``color`` is one RGBA for every link, or a list with one per link; ``t0`` as for trajectories."""
    colors = list(color) if isinstance(color, list) and len(color) == len(parts) else [color] * len(parts)
    for (link, mesh, poses), link_color in zip(parts, colors, strict=True):
        entity = f"world/{_part(name)}/robot/{_part(link)}"
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles)
        style = {} if link_color is None else {"albedo_factor": link_color}
        _log(recording, entity,
             rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles,
                       vertex_normals=_vertex_normals(vertices, triangles), **style),
             static=True)  # fmt: skip
        t = np.asarray(poses.t, dtype=np.float64)
        _send(
            recording,
            entity,
            indexes=[rr.TimeColumn(TIMELINE, duration=t - (t0 if t0 is not None else t[0]))],
            columns=rr.Transform3D.columns(
                translation=np.asarray(poses.positions, dtype=np.float64),
                quaternion=np.asarray(poses.orientations, dtype=np.float64),  # xyzw, as Rerun takes it
            ),
        )


# ── shapes ───────────────────────────────────────────────────────────────────


@log.register
def _(mesh: Mesh, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _log(
        recording,
        f"world/{_part(name or 'Mesh')}",
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices), triangle_indices=np.asarray(mesh.triangles)
        ),
        static=True,
    )


@log.register
def _(solid: SolidPrimitive, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    log(solid.to_mesh(), name or "SolidPrimitive", recording=recording)


# ── sensors ──────────────────────────────────────────────────────────────────


@log.register
def _(cloud: PointCloud, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    colors = None
    for key in ("rgb", "colors", "color"):
        value = (cloud.channels or {}).get(key)
        if value is not None and np.asarray(value).shape[-1:] in ((3,), (4,)):
            colors = np.asarray(value)
            break
    _log(
        recording,
        f"world/{_part(name or 'PointCloud')}",
        rr.Points3D(np.asarray(cloud.points), colors=colors),
        static=True,
    )


@log.register
def _(image: Image, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"image/{_part(name or 'Image')}"
    data = np.asarray(image.data)
    if image.encoding in _DEPTH_ENCODINGS:
        # WHY meter: 16UC1 depth is millimetres by the ROS convention, 32FC1 is metres.
        _log(
            recording,
            path,
            rr.DepthImage(data, meter=1000.0 if image.encoding == "16UC1" else 1.0),
            static=True,
        )
    else:
        _log(
            recording,
            path,
            rr.Image(data, color_model=_COLOR_MODELS.get(image.encoding)),
            static=True,
        )


@log.register
def _(info: CameraInfo, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _log(
        recording,
        f"camera/{_part(name or 'CameraInfo')}",
        rr.Pinhole(
            image_from_camera=np.asarray(info.k).reshape(3, 3), width=info.width, height=info.height
        ),
        static=True,
    )


@log.register
def _(state: JointState, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    def cell(values: Any, index: int) -> str:
        return "" if values is None else f"{float(np.asarray(values)[index]):.6g}"

    rows = ["| joint | position | velocity | effort |", "|---|---|---|---|"]
    for index, joint in enumerate(state.name):
        rows.append(
            f"| {joint} | {cell(state.position, index)} | {cell(state.velocity, index)}"
            f" | {cell(state.effort, index)} |"
        )
    _log(
        recording,
        f"notes/{_part(name or 'JointState')}",
        rr.TextDocument("\n".join(rows), media_type="text/markdown"),
        static=True,
    )


# ── vision ───────────────────────────────────────────────────────────────────


def _label(results: tuple[Any, ...], fallback: str) -> str:
    if results:
        best = max(results, key=lambda r: r.hypothesis.score)
        return f"{best.hypothesis.class_id} {best.hypothesis.score:.2f}"
    return fallback


def _boxes3d(
    boxes: list[BoundingBox3D], labels: list[str], path: str, stream: rr.RecordingStream
) -> None:
    _log(
        stream,
        path,
        rr.Boxes3D(
            centers=[
                [b.center.position.x, b.center.position.y, b.center.position.z] for b in boxes
            ],
            half_sizes=[[b.size.x / 2, b.size.y / 2, b.size.z / 2] for b in boxes],
            quaternions=[_xyzw(b.center.orientation) for b in boxes],
            labels=labels or None,
        ),
        static=True,
    )


def _boxes2d(
    boxes: list[BoundingBox2D], labels: list[str], path: str, stream: rr.RecordingStream
) -> None:
    # Rerun's 2D boxes are axis-aligned: a rotated box (theta != 0) is drawn unrotated.
    _log(
        stream,
        path,
        rr.Boxes2D(
            centers=[[b.center.x, b.center.y] for b in boxes],
            sizes=[[b.size_x, b.size_y] for b in boxes],
            labels=labels or None,
        ),
        static=True,
    )


@log.register
def _(box: BoundingBox3D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _boxes3d([box], [], f"world/{_part(name or 'BoundingBox3D')}", recording)


@log.register
def _(box: BoundingBox2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _boxes2d([box], [], f"image/{_part(name or 'BoundingBox2D')}", recording)


@log.register
def _(detection: Detection3D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _boxes3d(
        [detection.bbox],
        [_label(detection.results, detection.id)],
        f"world/{_part(name or 'Detection3D')}",
        recording,
    )


@log.register
def _(detection: Detection2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _boxes2d(
        [detection.bbox],
        [_label(detection.results, detection.id)],
        f"image/{_part(name or 'Detection2D')}",
        recording,
    )


@log.register
def _(array: Detection3DArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    labels = [_label(d.results, d.id) for d in array.detections]
    _boxes3d(
        [d.bbox for d in array.detections],
        labels,
        f"world/{_part(name or 'Detection3DArray')}",
        recording,
    )


@log.register
def _(array: Detection2DArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    labels = [_label(d.results, d.id) for d in array.detections]
    _boxes2d(
        [d.bbox for d in array.detections],
        labels,
        f"image/{_part(name or 'Detection2DArray')}",
        recording,
    )


@log.register
def _(grasp: GraspPose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{_part(name or 'GraspPose')}"
    p = grasp.pose.position
    _frame(base, [p.x, p.y, p.z], _xyzw(grasp.pose.orientation), recording)
    text = f"{grasp.object_id or 'grasp'} w={grasp.width * 1000:.0f} mm"
    if grasp.score is not None:
        text += f" score={grasp.score:.2f}"
    _log(recording, f"{base}/label", rr.Points3D([[0.0, 0.0, 0.0]], labels=[text]), static=True)


@log.register
def _(waypoint: Waypoint, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    p = waypoint.pose.position
    _frame(
        f"world/{_part(name or waypoint.name)}",
        [p.x, p.y, p.z],
        _xyzw(waypoint.pose.orientation),
        recording,
    )


@log.register
def _(action: WaypointAction, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    # WHY a path plus one labelled frame per waypoint: a request has no timing yet (the generator
    # adds it), so it is a static picture of where the tool is asked to go, in order.
    base = f"world/{_part(name or 'WaypointAction')}"
    points = [[w.pose.position.x, w.pose.position.y, w.pose.position.z] for w in action.waypoints]
    _log(recording, f"{base}/path", rr.LineStrips3D([points]), static=True)
    labels = [
        f"{i} {w.name}" + (f" [{w.label}]" if w.label and w.label != w.name else "")
        for i, w in enumerate(action.waypoints)
    ]
    _log(recording, f"{base}/points", rr.Points3D(points, labels=labels), static=True)
    for w in action.waypoints:
        p = w.pose.position
        _frame(f"{base}/{_part(w.name)}", [p.x, p.y, p.z], _xyzw(w.pose.orientation), recording)


__all__ = ["APPLICATION_ID", "RERUN_SDK_VERSION", "TIMELINE", "log", "recording", "save", "show"]
