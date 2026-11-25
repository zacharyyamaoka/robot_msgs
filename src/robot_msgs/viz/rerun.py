"""Every standard message in Rerun: ``msg.show()`` opens it, ``log`` composes several.

    traj.show()                                    # opens the Rerun viewer on this trajectory
    show(target=planned, real=measured)            # overlay: one plot per joint, one line per name
    save("pick.rrd", target=planned, real=measured)
    traj.show(robot="ur5e.urdf", tool=tcp)          # the URDF arm + tabs of joint and TCP plots

Needs ``pip install 'robot-msgs[rerun]'``; `import robot_msgs` never imports this module.

WHY Rerun (Zach, 2026-11-11): a great stock viewer for time series, 3D, images and video, so we
log into it instead of building a viewer. One logger per base class (`functools.singledispatch`, as
`robot_msgs.viz.plot` does), so a subclass falls back to its parent and a new type is one function.

Entity layout, shared with traj_compare's Rerun projection so the two overlay:

    /<field>/<column>/<name>     sampled data on the ``t`` timeline (seconds from the first sample)
    /world/<name>/...            anything spatial: poses, paths, clouds, solids, boxes
    /image/<name>, /camera/<name>, /notes/<name>

``<name>`` is the keyword you passed to `show`/`log` (``real``, ``target``), or the type name.
A JointTrajectory with ``robot=`` adds its arm at ``/world/<name>/robot`` (Rerun's own URDF loader),
its derived signals beside the stock ones (``/jerks/...``, ``/tcp/...``, ``/limit_pct/...``) and the
playhead table at ``/notes/<name>`` (see "a JointTrajectory with its robot" below).
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
    *messages: Msg,
    application_id: str = APPLICATION_ID,
    robot: str | Path | None = None,
    ghosts: bool = True,
    **named: Msg,
) -> rr.RecordingStream:
    """A fresh recording holding every message: positional ones under their type name.

    ``robot`` (a URDF path) draws ONE JointTrajectory as its arm, with ``tool=`` its tool's
    PoseTrajectory and ``ghosts`` the faint arms at its start and end poses (see `robot_blueprint`).
    Without ``robot``, ``tool`` is a message name like any other (``show(target=a, tool=path)``)."""
    stream = rr.RecordingStream(application_id)
    if robot is not None:
        traj, name, tool = _robot_inputs(messages, named)
        plan = _robot_plan(traj, name, robot, tool)
        # WHY first: the viewer opens on the dashboard, paused at the start, before any data arrives
        stream.send_blueprint(_robot_dashboard(plan, ghosts=ghosts))
        _log_robot(traj, plan, stream, ghosts=ghosts)
        _PATHS.pop(id(stream), None)
        return stream
    for msg in messages:
        log(msg, recording=stream)
    for name, msg in named.items():
        log(msg, name, recording=stream)
    _finish(stream)
    return stream


def show(
    *messages: Msg, robot: str | Path | None = None, ghosts: bool = True, **named: Msg
) -> rr.RecordingStream:
    """Open the Rerun viewer on these messages (a notebook cell shows it inline).

    Keyword names become the line names, so ``show(target=a, real=b)`` overlays the two.
    ``show(plan=traj, robot="ur5e.urdf", tool=tcp)`` draws the trajectory as its URDF arm.
    """
    return view(recording(*messages, robot=robot, ghosts=ghosts, **named))


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


def save(
    path: str, *messages: Msg, robot: str | Path | None = None, ghosts: bool = True, **named: Msg
) -> str:
    """Write the messages to an ``.rrd`` file; open it with ``rerun <path>``.

    ``save("plan.rrd", plan=traj, robot="ur5e.urdf", tool=tcp)`` writes the URDF dashboard
    (`robot_blueprint`); ``ghosts=False`` leaves out the start and end arms, for a web bundle of
    many trajectories: each is another copy of every mesh (the UR5e's: 0.3 MB each, 28 % of a
    3 s trajectory's 2.2 MB)."""
    stream = rr.RecordingStream(APPLICATION_ID)
    if robot is not None:
        traj, name, tool = _robot_inputs(messages, named)
        plan = _robot_plan(traj, name, robot, tool)
        # WHY default_blueprint at save (the round's R4): sent after the data, the playhead pinned
        # at the first chunk's time; written first, the file opens paused at t = 0
        stream.save(path, default_blueprint=_robot_dashboard(plan, ghosts=ghosts))
        _log_robot(traj, plan, stream, ghosts=ghosts)
        _PATHS.pop(id(stream), None)
        stream.flush()
        stream.disconnect()
        return path
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


@log.register
def _(
    traj: JointTrajectory,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = False,
    color: Any = None,
    prefix: str = "",
    t0: float | None = None,
    label: str | None = None,
    robot: str | Path | None = None,
    tool: PoseTrajectory | None = None,
    ghosts: bool = True,
) -> None:
    """The stock series (as every Trajectory); with ``robot`` (a URDF path), also its arm.

    ``robot`` draws the trajectory as Rerun's own URDF arm at ``/world/<name>/robot`` on the ``t``
    timeline, faint ``ghosts`` at its start and end poses, the tool's path, the derived jerk and TCP
    signals and the playhead table; ``tool`` is the tool's PoseTrajectory (its ``header.frame_id`` a
    URDF link, empty = the root link; without it the tool is the URDF's last link, by forward
    kinematics). The series are then in degrees for the URDF's revolute joints; `robot_blueprint`
    lays the recording out. Without ``robot`` nothing changes."""
    name = name or type(traj).__name__
    if robot is None:
        if tool is not None:
            raise TypeError("tool= is the tool path of robot=: pass robot= (a URDF path) too")
        _series(traj, name, recording, prefix, degrees=degrees, color=color, t0=t0, label=label)
        return
    given = {"degrees": degrees is not False, "color": color is not None, "prefix": prefix != "",
             "t0": t0 is not None, "label": label is not None}
    styled = [key for key, value in given.items() if value]
    if styled:
        raise TypeError(f"with robot= the URDF decides the series' units and style; drop {styled}")
    _log_robot(traj, _robot_plan(traj, name, robot, tool), recording, ghosts=ghosts)


# ── a JointTrajectory with its robot: the URDF arm, and tabs of joint and TCP plots ───────────────────
#
# WHY (Zach, Daily Note 2026-11-24): "this should basically be the stock trajectory rerun viewer... 100% there should
# be a urdf showing". The babble round's pick (traj_compare docs/proposals/rerun-joint-trajectory-2026-11-24/), R3
# "Solo run dashboard", with Zach's one change: "add tabs Similar to R1 so that you can switch between what information
# you are showing on the right hand side ... we are always showing the 3d robot". So the page is split in half: the arm
# on the left, always (start/end ghosts, the tool path solid up to the playhead and faint after it); tabs on the right:
# q, qd, qdd, qddd (one strip per joint, degrees, limit lines), "q / qd / qdd" (R5's cards: all three as % of each
# joint's own limits, under R5's at-the-playhead table) and TCP (position, orientation, speed, angular speed). Opens
# paused at t = 0 on q; every time cursor is the same playhead as the arm.
#
#     /positions|velocities|accelerations/<joint>/<name>   the stock series, degrees for revolute joints
#     /jerks/<joint>/<name>                                the slope of the accelerations (the message carries no jerk)
#     /<field>/<joint>/lower_limit, upper_limit            the joint's limits in the strip's units
#     /limit_pct/<joint>/<name>                            q / qd / qdd as % of the joint's limits (the cards)
#     /tcp/position_mm|orientation_deg/<axis>/<name>       the tool point, and its roll / pitch / yaw
#     /tcp/speed_mm_s/<name>, /tcp/angular_speed_deg_s/<name>
#     /notes/<name>                                        the playhead table, one per instant
#     /world/<name>/robot|ghost_start|ghost_end/...        the URDF (Rerun's loader), frames prefixed <name>/
#     /world/<name>/tool/trail|path|ends|stick             the tool path travelled, still to go, start / end; flange → TCP

_DEG = 180.0 / np.pi
# WHY calm (R3): one ink colour for every joint line (the strip's title names the joint), grey limits, and one accent
# (Solo Run's orange) for the tool, so nothing reads as an alarm (Zach: amber/red only for problems)
_INK = (37, 99, 160)
_LIMIT = (150, 150, 150)
_ACCENT = (240, 140, 0)
_PATH_FAINT = (250, 208, 160)  # WHY opaque: a line colour with alpha drew darker (brown), not fainter, in 0.40
_ARM = (175, 180, 190, 255)  # a neutral steel, so the faint ghosts read against it
_GHOST_START = (90, 140, 230, 60)
_GHOST_END = (240, 150, 60, 60)
_LABEL_START = (90, 140, 230)
_LABEL_END = (220, 120, 20)
_STICK = (90, 90, 100)  # the flange-to-TCP stick
_AXIS_COLORS = ((214, 62, 62), (52, 150, 72), (48, 98, 214))  # x / y / z (roll / pitch / yaw): Rerun's own axis colours
# the cards: q strong, qd a second hue (not the tool's orange: the two are unrelated), qdd quietest (the noisiest)
_CARD_STYLE = {"q": ((38, 110, 214), 2.5), "qd": ((20, 150, 140), 1.6), "qdd": ((140, 140, 150), 1.0)}
_CARD_BAND = (70, 160, 90, 60)  # ±100 %: a light band whose edges are the limits
_DISTANCE_RADII = 2.35  # camera distance in radii of the motion's bounding box: fits Rerun's default field of view
# the tool path is two polylines per instant (travelled, still to go), and the table one document per instant: at most
# this many instants × vertices, so a 10 000-sample trajectory stays a few MB; between instants the latest one holds
_ROWS, _VERTICES = 400, 256
# WHY a list: Rerun 0.40 cannot draw a Collada (.dae) mesh, so a link whose visuals are .dae shows its collision geometry
_DRAWABLE_MEDIA = {"model/stl", "model/obj", "model/gltf-binary", "model/gltf+json"}
_FIELDS = (  # (short, tab, stock field, rate suffix)
    ("q", "q", "positions", ""),
    ("qd", "qd", "velocities", "/s"),
    ("qdd", "qdd", "accelerations", "/s²"),
    ("qddd", "qddd", "jerks", "/s³"),
)


def _robot_inputs(messages: tuple, named: dict) -> tuple[JointTrajectory, str, PoseTrajectory | None]:
    """The one JointTrajectory (and its name) and ``tool`` that ``robot=`` draws."""
    named = dict(named)
    tool = named.pop("tool", None)
    entries = [(None, msg) for msg in messages] + list(named.items())
    if len(entries) != 1 or not isinstance(entries[0][1], JointTrajectory):
        kinds = [type(msg).__name__ for _, msg in entries]
        raise ValueError(
            f"robot= draws one JointTrajectory (tool= is its tool's PoseTrajectory), got {kinds}; "
            "overlays of several trajectories on one arm are not designed yet"
        )
    if tool is not None and not isinstance(tool, PoseTrajectory):
        raise TypeError(f"tool= is the tool's PoseTrajectory, got {type(tool).__name__}")
    name, traj = entries[0]
    return traj, name or type(traj).__name__, tool


def _slope(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d values / dt along axis 0 on the trajectory's own clock: central differences inside, one-sided at the ends.

    WHY first order at the ends (traj_compare's coverage `_slope`, 2026-11-23): numpy's second-order edge stencil
    overshoots when the slope changes right after the first sample, a false peak at sample 0."""
    if len(t) < 2:
        return np.zeros_like(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(values, t, axis=0, edge_order=1)


def _angular_rate(rotations: np.ndarray, t: np.ndarray) -> np.ndarray:
    """|ω| per sample from rotation matrices: the angle turned between the neighbours over their time apart."""
    from robot_msgs.geometry.rotations import quat_angle, quat_from_matrix

    rate = np.zeros(len(t))
    if len(t) < 2:
        return rate
    q = quat_from_matrix(rotations)
    rate[1:-1] = quat_angle(q[:-2], q[2:]) / (t[2:] - t[:-2])
    rate[0] = quat_angle(q[0], q[1]) / (t[1] - t[0])
    rate[-1] = quat_angle(q[-2], q[-1]) / (t[-1] - t[-2])
    return rate


def _joint_values(tree: Any, traj: JointTrajectory) -> dict[str, np.ndarray]:
    """Every URDF joint the trajectory moves (directly, or as a mimic of one): its value at each sample (rad or m)."""
    q = np.asarray(traj.positions, dtype=np.float64)
    values = {str(name): q[:, index] for index, name in enumerate(traj.joint_names)}
    for joint in tree.joints():
        mimic = joint.mimic
        if mimic is not None and mimic.joint in values and joint.name not in values:
            values[joint.name] = mimic.multiplier * values[mimic.joint] + mimic.offset
    return values


def _forward_kinematics(tree: Any, values: dict[str, np.ndarray], samples: int) -> dict[str, np.ndarray]:
    """Every link's pose in the root link's frame, ``(samples, 4, 4)``: for the camera, the tool path and the TCP tab.

    WHY our own: Rerun poses the links inside the viewer only, and the camera is set before anything is drawn. Built from
    the URDF's own joint origins and axes; Rerun draws the arm from its own joint transforms."""
    from robot_msgs.geometry.rotations import euler_to_matrix, quat_from_axis_angle, quat_to_matrix

    poses = {tree.root_link().name: np.broadcast_to(np.eye(4), (samples, 4, 4))}
    pending = list(tree.joints())
    while pending:  # parents first, whatever order the URDF lists its joints in
        progressed = False
        for joint in list(pending):
            if joint.parent_link not in poses:
                continue
            origin = np.eye(4)
            origin[:3, :3] = euler_to_matrix(*joint.origin_rpy)
            origin[:3, 3] = joint.origin_xyz
            motion = np.broadcast_to(np.eye(4), (samples, 4, 4)).copy()
            value = values.get(joint.name)
            if value is not None and joint.joint_type in ("revolute", "continuous"):
                motion[:, :3, :3] = quat_to_matrix(quat_from_axis_angle(np.asarray(joint.axis), value))
            elif value is not None and joint.joint_type == "prismatic":
                motion[:, :3, 3] = np.asarray(joint.axis, dtype=np.float64)[None, :] * value[:, None]
            poses[joint.child_link] = poses[joint.parent_link] @ origin @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f"URDF joints not connected to the root link: {[joint.name for joint in pending]}")
    return poses


def _tip_link(tree: Any, traj: JointTrajectory) -> str:
    """The deepest link the trajectory moves, followed down single fixed joints (to the flange, tool0, a tip ...)."""
    parent = {joint.child_link: joint.parent_link for joint in tree.joints()}

    def depth(link: str) -> int:
        count = 0
        while link in parent:
            link, count = parent[link], count + 1
        return count

    moved = [tree.get_joint_by_name(str(name)).child_link for name in traj.joint_names]
    link = max(moved, key=depth) if moved else tree.root_link().name
    while True:
        children = [joint for joint in tree.joints() if joint.parent_link == link and joint.joint_type == "fixed"]
        if len(children) != 1:
            return link
        link = children[0].child_link


def _limit_arrays(tree: Any, traj: JointTrajectory) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per short field, (low, high) per trajectory joint in SI (NaN = none): the message's JointLimits first, else the
    URDF's (position and velocity; a URDF states no acceleration or jerk). A ``max_*`` bound is a magnitude."""
    names = [str(name) for name in traj.joint_names]
    count = len(names)
    out = {short: (np.full(count, np.nan), np.full(count, np.nan)) for short, *_ in _FIELDS}
    limits = traj.limits
    by_name = {} if limits is None else {str(name): index for index, name in enumerate(limits.joint_names)}

    def given(field_name: str, joint: str) -> float:
        values = None if limits is None else getattr(limits, field_name)
        if values is None or joint not in by_name:
            return np.nan
        value = float(np.asarray(values, dtype=np.float64)[by_name[joint]])
        return value if np.isfinite(value) else np.nan

    for index, joint in enumerate(names):
        urdf = tree.get_joint_by_name(joint)
        low, high = given("min_position", joint), given("max_position", joint)
        if (np.isnan(low) or np.isnan(high)) and urdf.joint_type in ("revolute", "prismatic") \
                and urdf.limit_lower < urdf.limit_upper:
            low = urdf.limit_lower if np.isnan(low) else low
            high = urdf.limit_upper if np.isnan(high) else high
        out["q"][0][index], out["q"][1][index] = low, high
        for short, field_name in (("qd", "max_velocity"), ("qdd", "max_acceleration"), ("qddd", "max_jerk")):
            bound = given(field_name, joint)
            if short == "qd" and np.isnan(bound) and urdf.limit_velocity > 0:
                bound = urdf.limit_velocity
            if bound > 0:
                out[short][0][index], out[short][1][index] = -bound, bound
    return out


def _nice(step: float) -> float:
    exponent = 10.0 ** np.floor(np.log10(step))
    return next(m * exponent for m in (1.0, 2.0, 5.0, 10.0) if m * exponent >= step)


def _snap(low: float, high: float) -> tuple[float, float]:
    """Outward to round numbers, then 6 % air, so no tick label sits on the edge (it overprinted the time axis's +0s)."""
    step = _nice(max(high - low, 1e-9) / 2.0)
    low, high = float(np.floor(low / step) * step), float(np.ceil(high / step) * step)
    air = 0.06 * (high - low)
    return low - air, high + air


def _strip_range(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """A strip's y range: the motion padded, stretched to a limit only when that limit is within one motion span,
    snapped outward to round numbers.

    WHY not always to the limits (R3): the UR5e's are ±360°, so a 60° move would be a flat line in a strip this small;
    the title carries the distance to the limit instead. WHY snapped: a ~90 px plot gets ticks 50 or 100 apart, and a
    range like 14…96 holds none."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vmin, vmax = float(values.min()), float(values.max())
    span = max(vmax - vmin, 1e-9)
    lo, hi = vmin - 0.08 * span, vmax + 0.08 * span
    if np.isfinite(low) and vmin - low <= span:
        lo = min(lo, low)
    if np.isfinite(high) and high - vmax <= span:
        hi = max(hi, high)
    return _snap(lo, hi)


def _eye(points: np.ndarray) -> Any:
    """A camera on the whole motion: the arm seen in profile, from above, z up (R3).

    Profile = across the main horizontal direction of everything that moves (link origins + tool path), from the side
    nearer traj_compare's front-right default. WHY not that fixed front-right: a UR5e pointing along it is seen end-on,
    foreshortened into a blob; in profile the elbow and wrist read as angles. A fixed ~29° elevation, so every trajectory
    opens from a familiar height."""
    import rerun.blueprint as rrb

    points = points[np.all(np.isfinite(points), axis=1)]
    low, high = points.min(axis=0), points.max(axis=0)
    look = 0.5 * (low + high)
    radius = 0.5 * float(np.linalg.norm(high - low))
    front = np.array([1.5, -1.9, 0.0]) / np.linalg.norm([1.5, -1.9])
    direction = front.copy()
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    if len(xy) > 2 and np.ptp(xy, axis=0).max() > 1e-3:
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        across = np.array([-vt[0, 1], vt[0, 0], 0.0])
        across = across if across @ front >= 0 else -across
        direction = across + 0.4 * front
    direction = direction / np.linalg.norm(direction[:2])
    direction[2] = 0.55
    direction = direction / np.linalg.norm(direction)
    distance = _DISTANCE_RADII * max(radius, 0.1)  # a 1-DOF rig's link origins span centimetres
    return rrb.EyeControls3D(kind="Orbital", look_target=look.tolist(), eye_up=[0.0, 0.0, 1.0],
                             position=(look + distance * direction).tolist())  # fmt: skip


def _percent(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """[low, high] onto [-100, +100] % (R5): the limit lines are ±100 % for every joint, symmetric range or not."""
    return (values - 0.5 * (low + high)) / (0.5 * (high - low)) * 100.0


def _short_joint(joint: str) -> str:
    """`shoulder_pan_joint` → `shoulder_pan`: the playhead table is narrow, and every row is a joint."""
    return joint[: -len("_joint")] if joint.endswith("_joint") and len(joint) > len("_joint") else joint


class _Strip:
    """One plot of a tab: its title, the entities it shows, its y range (None = Rerun's own)."""

    def __init__(self, title: str, contents: list[str], y_range: tuple[float, float] | None = None,
                 legend: bool = False) -> None:  # fmt: skip
        self.title, self.contents, self.y_range, self.legend = title, contents, y_range, legend


class _RobotPlan:
    """A JointTrajectory + its URDF, worked out once: what the logger writes where and how the dashboard frames it.
    Pure: nothing is logged here, so the blueprint can be built (and written) before the data."""

    def __init__(self, traj: JointTrajectory, name: str, robot: str | Path, tool: PoseTrajectory | None) -> None:
        import rerun.urdf as rurdf

        self.name, self.part = name, _part(name)
        self.urdf = Path(robot).expanduser().resolve()
        tree = rurdf.UrdfTree.from_file_path(self.urdf)
        self.robot_name = tree.name
        joints = [str(joint) for joint in traj.joint_names]
        missing = [joint for joint in joints if tree.get_joint_by_name(joint) is None]
        if missing:
            known = sorted(joint.name for joint in tree.joints() if joint.joint_type != "fixed")
            raise ValueError(f"{self.urdf.name} has no joint(s) {missing}; its joints: {known}")
        self.joints = joints
        self.revolute = [j for j in joints if tree.get_joint_by_name(j).joint_type in ("revolute", "continuous")]
        t = np.asarray(traj.t, dtype=np.float64)
        self.seconds = t - (t[0] if t.size else 0.0)
        self.values = _joint_values(tree, traj)
        self.scale = np.array([_DEG if joint in self.revolute else 1.0 for joint in joints])

        # the joint signals in SI, (N, J) each; a missing field stays missing (its tab is left out); the jerk is the
        # slope of the accelerations, the message has no jerk field
        accelerations = None if traj.accelerations is None else np.asarray(traj.accelerations, dtype=np.float64)
        self.signals = {
            "q": np.asarray(traj.positions, dtype=np.float64),
            "qd": None if traj.velocities is None else np.asarray(traj.velocities, dtype=np.float64),
            "qdd": accelerations,
            "qddd": None if accelerations is None else _slope(accelerations, self.seconds),
        }
        self.limits = _limit_arrays(tree, traj)
        self.tabs: dict[str, list[_Strip]] = {}
        for short, tab, field_name, rate in _FIELDS:
            if self.signals[short] is not None:
                self.tabs[tab] = [self._strip(short, field_name, rate, index) for index in range(len(joints))]
        self.cards = self._cards()
        if self.cards:
            self.tabs["q / qd / qdd"] = self.cards

        # the tool: the given PoseTrajectory in the link its header names, else the URDF's last link by FK
        from robot_msgs.geometry.rotations import matrix_to_euler, quat_to_matrix

        links = _forward_kinematics(tree, self.values, len(t))
        root = tree.root_link().name
        self.root = root
        self.flange: np.ndarray | None = None  # the URDF's last link, when the tool sits beyond it (the stick)
        if tool is not None:
            frame = tool.header.frame_id or root
            if frame not in links:
                raise ValueError(f"the tool's header.frame_id {frame!r} is not a link of {self.urdf.name} "
                                 f"(empty = its root link {root!r}); its links: {sorted(links)}")  # fmt: skip
            self.tool_t = np.asarray(tool.t, dtype=np.float64) - (t[0] if t.size else 0.0)
            self.tool_xyz = np.asarray(tool.positions, dtype=np.float64)
            rotations = quat_to_matrix(np.asarray(tool.orientations, dtype=np.float64))
            self.tool_frame, self.tool_label = frame, tool.child_frame_id or "tool"
            source = f"tool PoseTrajectory {self.tool_label} in {frame}"
            # WHY a stick from the flange (R1, R5): a URDF usually ends at the flange and the TCP sits further out (the
            # UR5e's 145 mm tool); without it the tool point floats beside the wrist and reads as a frame bug
            tip = links[_tip_link(tree, traj)][:, :3, 3]
            if frame == root and len(self.tool_xyz) == len(tip) and len(tip) \
                    and float(np.linalg.norm(self.tool_xyz - tip, axis=1).max()) > 0.001:
                self.flange = tip
        else:
            tip = _tip_link(tree, traj)
            self.tool_t, self.tool_xyz, rotations = self.seconds, links[tip][:, :3, 3], links[tip][:, :3, :3]
            self.tool_frame, self.tool_label = root, tip
            source = f"FK of {tip} in {root}"
        speed = np.linalg.norm(_slope(self.tool_xyz, self.tool_t), axis=1) * 1000.0
        rpy = matrix_to_euler(rotations)
        self.tcp = {
            "position_mm": self.tool_xyz * 1000.0,
            # WHY unwrapped: a tool pointing down sits at roll ±180°, and the wrapped value flips sign every sample
            "orientation_deg": np.degrees(np.unwrap(rpy, axis=0)),
            "speed_mm_s": speed,
            "angular_speed_deg_s": np.degrees(_angular_rate(rotations, self.tool_t)),
        }
        # WHY say where: near |pitch| = 90° roll and yaw trade off fast (gimbal lock) though the tool turns smoothly;
        # every 3-angle set has such a place (tilt + yaw: pointing straight up), so the title names it instead
        near = np.flatnonzero(np.abs(rpy[:, 1]) > np.radians(80.0)) if len(rpy) else np.zeros(0, dtype=int)
        self.gimbal = (f" · near gimbal lock {self.tool_t[near[0]]:.2f}–{self.tool_t[near[-1]]:.2f} s" if near.size
                       else "")  # fmt: skip
        self.tabs["TCP"] = self._tcp_strips(source)

        # the camera: every link origin at every sample, and the tool path (in the root frame; framing only)
        mount = links[self.tool_frame][0] if self.tool_frame != root else np.eye(4)
        tool_root = self.tool_xyz @ mount[:3, :3].T + mount[:3, 3]
        self.eye = _eye(np.vstack([*(pose[:, :3, 3] for pose in links.values()), tool_root]))
        length = float(np.linalg.norm(np.diff(self.tool_xyz, axis=0), axis=1).sum())
        self.title = (f"{name} · {tree.name} · {self.seconds[-1]:.2f} s · {self.tool_label} path {length:.2f} m")
        self.tables = self._tables()

    def unit(self, index: int, rate: str) -> str:
        return ("°" if self.joints[index] in self.revolute else " m") + rate

    def _strip(self, short: str, field_name: str, rate: str, index: int) -> _Strip:
        joint, scale = self.joints[index], self.scale[index]
        unit = self.unit(index, rate)
        digits = 0 if joint in self.revolute else 3
        shown = self.signals[short][:, index] * scale
        low, high = (bound[index] * scale for bound in self.limits[short])
        contents = [f"/{field_name}/{_part(joint)}/{self.part}"]
        contents += [f"/{field_name}/{_part(joint)}/{side}" for side, bound in (("lower_limit", low), ("upper_limit", high))
                     if np.isfinite(bound)]  # fmt: skip
        title = f"{joint} [{unit.strip()}]"
        if short == "q":
            title += f" · travel {np.ptp(shown):.{digits}f}{unit}"
            gaps = [gap for gap in (shown.min() - low, high - shown.max()) if np.isfinite(gap)]
            if gaps:
                headroom = min(gaps)
                title += f" · {abs(headroom):.{digits}f}{unit} {'to limit' if headroom >= 0 else 'PAST limit'}"
        else:
            peak = float(np.nanmax(np.abs(shown))) if shown.size else 0.0
            title += f" · peak {peak:.{digits}f}{unit}"
            if short == "qddd":
                title += " · slope of qdd"
            if np.isfinite(high):
                title += f" · {peak / high * 100.0:.0f} % of limit"
        return _Strip(title, contents, _strip_range(shown, low, high))

    def _cards(self) -> list[_Strip]:
        """R5's joint cards: q / qd / qdd as % of the joint's own limits, three instances of one series entity (the
        legend reads q, qd, qdd in order), on one shared ±span axis. A joint with no limit at all has no card."""
        self.card_data: list[tuple[str, list[str], list[Any], list[float], np.ndarray]] = []
        self.card_scales: dict[str, str] = {}  # joint: what 100 % is for q · qd · qdd, for the table
        cards, peak = [], 100.0
        for index, joint in enumerate(self.joints):
            shown, absent, columns, names = [], [], [], []
            for short in ("q", "qd", "qdd"):
                values = self.signals[short]
                if values is None:
                    continue
                low, high = (bound[index] for bound in self.limits[short])
                if not (np.isfinite(low) and np.isfinite(high)):
                    absent.append(short)
                    continue
                columns.append(_percent(values[:, index], low, high))
                scale, rate = self.scale[index], {"q": "", "qd": "/s", "qdd": "/s²"}[short]
                unit = self.unit(index, rate).strip()
                digits = 0 if joint in self.revolute else 3
                if short == "q" and not np.isclose(low, -high):
                    names.append(f"{short} {low * scale:.{digits}f}…{high * scale:.{digits}f}{unit}")
                else:
                    names.append(f"{short} ±{high * scale:.{digits}f}{unit}")
                shown.append(short)
            if not columns:
                continue
            percent = np.stack(columns, axis=1)
            peak = max(peak, float(np.nanmax(np.abs(percent))))
            # WHY no legend: a legend box hid a third of every card's time axis (R5's open issue, worse in half the
            # page); the colour key is one line above the cards and each card's 100 % is a column of the table
            title = f"{joint} · % of limit"
            if absent:
                title += f" ({' / '.join(absent)}: no limit)"
            self.card_scales[joint] = " · ".join(name.split(" ", 1)[1] for name in names)
            entity = f"/limit_pct/{_part(joint)}/{self.part}"
            self.card_data.append((entity, names, [_CARD_STYLE[s][0] for s in shown], [_CARD_STYLE[s][1] for s in shown],
                                   percent))  # fmt: skip
            cards.append(_Strip(title, [f"{entity}/**"], None))
        # WHY one axis a quarter past ±100 % (R5): every card reads the same and the band's edges sit inside the plot;
        # a trajectory past a limit widens every card the same amount instead of clipping it
        self.card_span = max(125.0, float(np.ceil(peak / 10.0) * 10.0 + 15.0))
        for card in cards:
            card.y_range = (-self.card_span, self.card_span)
        return cards

    def _tcp_strips(self, source: str) -> list[_Strip]:
        speed, angular = self.tcp["speed_mm_s"], self.tcp["angular_speed_deg_s"]
        peak, turn = int(np.argmax(speed)), int(np.argmax(angular))
        return [
            _Strip(f"TCP position [mm] · x / y / z · {source}", [f"/tcp/position_mm/{self.part}"], legend=True),
            _Strip(f"TCP orientation [°] · roll / pitch / yaw (ROS rpy, unwrapped){self.gimbal} · {source}",
                   [f"/tcp/orientation_deg/{self.part}"], legend=True),
            _Strip(f"TCP speed [mm/s] · peak {speed[peak]:.0f} at {self.tool_t[peak]:.2f} s · {source}",
                   [f"/tcp/speed_mm_s/{self.part}"], _snap(0.0, max(float(speed.max()), 1.0))),
            _Strip(f"TCP angular speed [°/s] · peak {angular[turn]:.0f} at {self.tool_t[turn]:.2f} s · {source}",
                   [f"/tcp/angular_speed_deg_s/{self.part}"], _snap(0.0, max(float(angular.max()), 1.0))),
        ]  # fmt: skip

    def _tables(self) -> tuple[np.ndarray, list[str]]:
        """R5's playhead table, one markdown document per instant (at most `_ROWS`): each joint's q, qd, qdd in degrees
        (metres for a prismatic joint, as its strips) and its worst % of limit, with which quantity it is."""
        shorts = [short for short in ("q", "qd", "qdd") if self.signals[short] is not None]
        rates = {"q": "", "qd": "/s", "qdd": "/s²"}
        # every joint revolute (the usual arm): the unit sits once in the header; a prismatic joint in the mix puts
        # the unit in each cell instead, so no row reads metres as degrees
        mixed = len(self.revolute) != len(self.joints)
        scales = bool(self.card_scales)
        header = "| joint | " + " | ".join(short if mixed else f"{short} °{rates[short]}" for short in shorts) \
            + (" | card 100 % |" if scales else " |") + " worst, % of limit |"  # fmt: skip
        rule = "|---|" + "---:|" * len(shorts) + ("---|" if scales else "") + "---:|"
        last = len(self.seconds) - 1
        picked = np.unique(np.linspace(0, last, min(last + 1, _ROWS)).round().astype(int)) if last >= 0 else []
        tables = []
        key = ("" if not self.cards else
               " · the cards below: q thick blue, qd teal, qdd thin grey, as % of each joint's own limits (the green "
               "band is ±100 %)")  # fmt: skip
        for k in picked:
            lines = [f"**t = {self.seconds[k]:.2f} s**{key}", "", header, rule]
            for index, joint in enumerate(self.joints):
                revolute = joint in self.revolute
                cells, worst = [], None
                for short in shorts:
                    value = float(self.signals[short][k, index])
                    shown = f"{value * _DEG:+.1f}" if revolute else f"{value:+.3f}"
                    cells.append(f"{shown}{'°' if revolute else ' m'}{rates[short]}" if mixed else shown)
                    low, high = (bound[index] for bound in self.limits[short])
                    if np.isfinite(low) and np.isfinite(high):
                        percent = abs(float(_percent(np.asarray(value), low, high)))
                        if worst is None or percent > worst[0]:
                            worst = (percent, short)
                text = "–" if worst is None else f"{worst[0]:.0f} {worst[1]}"
                scale = f" | {self.card_scales.get(joint, '–')}" if scales else ""
                lines.append(f"| {_short_joint(joint)} | " + " | ".join(cells) + f"{scale} | {text} |")
            tables.append("\n".join(lines))
        return self.seconds[np.asarray(picked, dtype=int)], tables


def _robot_plan(traj: JointTrajectory, name: str, robot: str | Path, tool: PoseTrajectory | None) -> _RobotPlan:
    return _RobotPlan(traj, name, robot, tool)


def _geometry_kinds(chunks: list[Any]) -> dict[str, tuple[str, str | None]]:
    """{entity: (archetype, media type)} of every geometry the URDF loader wrote (Asset3D, Boxes3D, Mesh3D, ...)."""
    kinds: dict[str, tuple[str, str | None]] = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        for column in batch.schema.names:
            archetype = column.split(":")[0] if ":" in column else ""
            if archetype and archetype not in ("InstancePoses3D", "CoordinateFrame"):
                media = None
                if "Asset3D:media_type" in batch.schema.names:
                    rows = batch.column("Asset3D:media_type").to_pylist()
                    media = rows[0][0] if rows and rows[0] else None
                kinds[chunk.entity_path] = (archetype, media)
    return kinds


def _urdf_copy(plan: _RobotPlan, recording: rr.RecordingStream, copy: str, colour: Any) -> Any:
    """One copy of the URDF at ``/world/<name>/<copy>`` (meshes, no joint transforms), its frames prefixed, every
    geometry tinted ``colour``; returns the loaded tree, whose joints carry the prefixed frames.

    WHY move the loader's chunks: `UrdfTree.from_file_path(entity_path_prefix=…)` takes ONE path segment (a '/' is
    escaped), so the robot is loaded at its default root and each chunk re-pathed (`Chunk.with_entity_path`).
    WHY no loader joint transforms: every joint gets exactly one transform entity below (moving = on t, the rest
    static); a second static transform for the same child frame is a viewer ERROR (the round's R3 / R4).
    WHY a mount: the loader's frames hang off the URDF's root link, which nothing connects to the view; one identity
    edge from `tf#/world/<name>` (the entity's implicit frame) to the root link puts the robot in the scene.
    WHY visuals first: a link draws its visual geometry when Rerun can draw it, else its collision geometry, never both
    on top of each other."""
    import rerun.urdf as rurdf

    prefix = f"{plan.part}/" if copy == "robot" else f"{plan.part}/{copy}/"
    tree = rurdf.UrdfTree.from_file_path(plan.urdf, frame_prefix=prefix)
    entity, old_root = f"/world/{plan.part}/{copy}", f"/{tree.name}"
    chunks = list(tree.stream(include_joint_transforms=False))
    kinds = _geometry_kinds(chunks)

    def drawable(path: str) -> bool:
        archetype, media = kinds.get(path, ("", None))
        return bool(archetype) and (archetype != "Asset3D" or media in _DRAWABLE_MEDIA)

    keep: set[str] = set()
    for link in [tree.root_link().name, *(joint.child_link for joint in tree.joints())]:
        visuals = [path for path in tree.get_visual_geometry_paths(link) if drawable(path)]
        keep.update(visuals or [path for path in tree.get_collision_geometry_paths(link) if drawable(path)])
    geometry = set(kinds)
    recording.send_chunks([
        chunk.with_entity_path(entity + chunk.entity_path[len(old_root):])
        for chunk in chunks
        if not any(chunk.entity_path == path or chunk.entity_path.startswith(path + "/") for path in geometry - keep)
    ])  # fmt: skip
    for path in sorted(keep):
        archetype = kinds[path][0]
        tint = (rr.Asset3D.from_fields(albedo_factor=colour) if archetype == "Asset3D"
                else rr.Mesh3D.from_fields(albedo_factor=colour) if archetype == "Mesh3D"
                else getattr(rr, archetype).from_fields(colors=[colour]))  # Boxes3D, Cylinders3D, Ellipsoids3D, ...
        recording.log(entity + path[len(old_root):], tint, static=True)
    recording.log(f"{entity}/mount", rr.Transform3D(parent_frame=f"tf#/world/{plan.part}",
                                                    child_frame=prefix + tree.root_link().name), static=True)
    return tree


def _log_robot(traj: JointTrajectory, plan: _RobotPlan, recording: rr.RecordingStream, *, ghosts: bool) -> None:
    """Log one JointTrajectory with its robot (see the section's entity layout)."""
    part, seconds = plan.part, plan.seconds
    on_t = [rr.TimeColumn(TIMELINE, duration=seconds)]

    # 1. the stock series: robot_msgs' own logger, degrees for the URDF's revolute joints (the message stays SI); the
    #    jerk beside them, in the same units per second
    _series(traj, plan.name, recording, degrees=plan.revolute, color=_INK, label=plan.name)
    if plan.signals["qddd"] is not None:
        for index, joint in enumerate(plan.joints):
            entity = f"/jerks/{_part(joint)}/{part}"
            recording.log(entity, rr.SeriesLines(names=plan.name, colors=_INK), static=True)
            recording.send_columns(entity, indexes=on_t,
                                   columns=rr.Scalars.columns(scalars=plan.signals["qddd"][:, index] * plan.scale[index]))

    # 2. limits: two grey lines per joint and field, in the units the strip shows
    ends = [rr.TimeColumn(TIMELINE, duration=[seconds[0], seconds[-1]])]
    for short, _, field_name, _ in _FIELDS:
        if plan.signals[short] is None:
            continue
        for index, joint in enumerate(plan.joints):
            for side, bound in zip(("lower_limit", "upper_limit"), plan.limits[short], strict=True):
                value = float(bound[index] * plan.scale[index])
                if np.isfinite(value):
                    entity = f"/{field_name}/{_part(joint)}/{side}"
                    recording.log(entity, rr.SeriesLines(colors=[_LIMIT], widths=[1.0], names=[side.replace("_", " ")]),
                                  static=True)  # fmt: skip
                    recording.send_columns(entity, indexes=ends, columns=rr.Scalars.columns(scalars=[value, value]))

    # 3. the cards: q / qd / qdd as % of limit, three instances of one entity, over the ±100 % band (the stock `band`)
    for entity, names, colours, widths, percent in plan.card_data:
        recording.log(entity, rr.SeriesLines(names=names, colors=colours, widths=widths), static=True)
        recording.send_columns(entity, indexes=on_t, columns=rr.Scalars.columns(scalars=percent.reshape(-1))
                               .partition([percent.shape[1]] * len(seconds)))  # fmt: skip
        band(f"{entity}/limit", [0.0, float(seconds[-1])], -100.0, 100.0, recording=recording, color=_CARD_BAND,
             name="limit ±100 %")

    # 4. the TCP: position and orientation, each one entity whose three instances are its axes (so the legend reads
    #    x, y, z and roll, pitch, yaw in that order: separate entities sort by path, pitch first), and its speeds
    tool_t = [rr.TimeColumn(TIMELINE, duration=plan.tool_t)]
    for group, axes in (("position_mm", ["x", "y", "z"]), ("orientation_deg", ["roll", "pitch", "yaw"])):
        entity = f"/tcp/{group}/{part}"
        recording.log(entity, rr.SeriesLines(names=axes, colors=list(_AXIS_COLORS)), static=True)
        recording.send_columns(entity, indexes=tool_t, columns=rr.Scalars.columns(scalars=plan.tcp[group].reshape(-1))
                               .partition([3] * len(plan.tool_t)))  # fmt: skip
    for group, colour in (("speed_mm_s", _ACCENT), ("angular_speed_deg_s", _INK)):
        entity = f"/tcp/{group}/{part}"
        recording.log(entity, rr.SeriesLines(names=group.split("_")[0], colors=colour, widths=1.6), static=True)
        recording.send_columns(entity, indexes=tool_t, columns=rr.Scalars.columns(scalars=plan.tcp[group]))

    # 5. the playhead table: one markdown document per instant (latest-at holds it until the next)
    table_t, tables = plan.tables
    recording.send_columns(f"/notes/{part}", indexes=[rr.TimeColumn(TIMELINE, duration=table_t)],
                           columns=rr.TextDocument.columns(text=tables, media_type=["text/markdown"] * len(tables)))

    # 6. the arm: Rerun's URDF loader, every joint one transform entity (moving joints on t, the rest static)
    copies = [("robot", None, _ARM)] + ([("ghost_start", 0, _GHOST_START), ("ghost_end", -1, _GHOST_END)] if ghosts else [])
    for copy, sample, colour in copies:
        tree = _urdf_copy(plan, recording, copy, colour)
        for joint in tree.joints():
            entity = f"/world/{part}/{copy}/joint_transforms/{_part(joint.name)}"
            values = plan.values.get(joint.name)
            if sample is None and values is not None and joint.joint_type != "fixed":
                recording.send_columns(entity, indexes=on_t,
                                       columns=joint.compute_transform_columns(values.tolist(), clamp=False))
            else:
                value = 0.0 if values is None or joint.joint_type == "fixed" else float(values[sample])
                recording.log(entity, joint.compute_transform(value, clamp=False), static=True)

    # 7. the tool, in its own frame: at every instant the travelled part solid (start → now) and the rest faint (now →
    #    end), so the whole path is always there. WHY two polylines per instant, latest-at (R3): one static faint path
    #    under a solid trail z-fought into beads, and a time range on the whole 3D view dropped the ghosts' colours.
    frame = rr.CoordinateFrame(f"{part}/{plan.tool_frame}")
    xyz = plan.tool_xyz
    last = len(xyz) - 1
    picked = np.unique(np.linspace(0, last, min(last + 1, _ROWS)).round().astype(int))

    def thin(first: int, end: int) -> np.ndarray:
        return xyz[np.unique(np.linspace(first, end, min(end - first + 1, _VERTICES)).round().astype(int))]

    at = [rr.TimeColumn(TIMELINE, duration=plan.tool_t[picked])]
    for entity, colour, radius, strips in (
        ("path", _PATH_FAINT, 0.003, [thin(row, last) for row in picked]),
        ("trail", _ACCENT, 0.005, [thin(0, row) for row in picked]),
    ):
        path = f"/world/{part}/tool/{entity}"
        recording.log(path, rr.LineStrips3D.from_fields(colors=[colour], radii=[radius]), frame, static=True)
        recording.send_columns(path, indexes=at, columns=rr.LineStrips3D.columns(strips=strips))
    if plan.flange is not None:
        stick = f"/world/{part}/tool/stick"
        recording.log(stick, rr.LineStrips3D.from_fields(colors=[_STICK], radii=[0.004]), frame, static=True)
        recording.send_columns(stick, indexes=at, columns=rr.LineStrips3D.columns(
            strips=[np.stack([plan.flange[row], xyz[row]]) for row in picked]))
    recording.log(f"/world/{part}/tool/ends", rr.Points3D([xyz[0], xyz[-1]], labels=["start", "end"],
                                                          colors=[_LABEL_START, _LABEL_END], radii=[0.009]),
                  frame, static=True)  # fmt: skip


def _robot_dashboard(plan: _RobotPlan, *, ghosts: bool = True) -> Any:
    """The page split in half: the arm on the left, always; tabs of plots on the right, q open (see the section)."""
    import rerun.blueprint as rrb

    title = plan.title + (" · ghosts: blue = start, orange = end" if ghosts else "")
    world = rrb.Spatial3DView(origin="/", contents=[f"/world/{plan.part}/**"], name=title, eye_controls=plan.eye)

    def plot(strip: _Strip) -> Any:
        axis_y = rrb.ScalarAxis(range=strip.y_range, zoom_lock=False) if strip.y_range else None
        return rrb.TimeSeriesView(
            origin="/", contents=strip.contents, name=strip.title,
            # WHY bottom right: a rest-to-rest motion ends quiet there (qd, qdd back to 0); at the top left the legend
            # hid the first third of every card (R5's open issue, worse in half the width)
            plot_legend=rrb.PlotLegend(visible=strip.legend, corner="RightBottom"),
            axis_x=rrb.TimeAxis(link="LinkToGlobal"),  # zoom one plot's time, every plot follows
            axis_y=axis_y,
        )  # fmt: skip

    tabs = []
    for name, strips in plan.tabs.items():
        if name == "q / qd / qdd":
            table = rrb.TextDocumentView(origin=f"/notes/{plan.part}", name="at the playhead")
            columns = 1 if len(strips) <= 3 else 2
            # WHY these shares: the table needs a line per joint + 5 (title, time, header), a card about 8, so six
            # joints show every row unscrolled at 1000 px (the first cut, 1 : 3, hid the last joint)
            card_rows = -(-len(strips) // columns)
            tabs.append(rrb.Vertical(table, rrb.Grid(*(plot(s) for s in strips), grid_columns=columns),
                                     row_shares=[len(plan.joints) + 5, 8 * card_rows], name=name))  # fmt: skip
        else:
            tabs.append(rrb.Vertical(*(plot(s) for s in strips), name=name))
    return rrb.Blueprint(
        rrb.Horizontal(world, rrb.Tabs(*tabs, active_tab=0), column_shares=[1, 1]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        # WHY paused (R3): the viewer opens on the start pose and every still lands where it is asked; play loops
        rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="paused", loop_mode="all"),
        collapse_panels=False,
    )


def robot_blueprint(
    traj: JointTrajectory,
    name: str | None = None,
    *,
    robot: str | Path,
    tool: PoseTrajectory | None = None,
    ghosts: bool = True,
) -> Any:
    """The dashboard of ``log(traj, name, robot=robot, tool=tool)``, for a recording you make yourself
    (``stream.save(path, default_blueprint=robot_blueprint(...))``): the arm on the left half, and tabs on the right
    half: q, qd, qdd, qddd (a strip per joint in degrees with its limits), q / qd / qdd (each joint's three as % of its
    limits, under a table of the values at the playhead) and TCP. A tab whose data the message lacks is left out."""
    return _robot_dashboard(_robot_plan(traj, name or type(traj).__name__, robot, tool), ghosts=ghosts)


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


__all__ = ["APPLICATION_ID", "RERUN_SDK_VERSION", "TIMELINE", "log", "recording", "robot_blueprint", "save", "show"]
