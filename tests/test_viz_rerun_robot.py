"""robot_msgs.viz.rerun with ``robot=``: a JointTrajectory drawn as its URDF arm, the stock trajectory viewer.

WHY (Zach, Daily Note 2026-11-24): "this should basically be the stock trajectory rerun viewer... 100% there should be
a urdf showing". The layout is the babble round's pick, R3 "Solo run dashboard", with R5's at-the-playhead table
(traj_compare docs/proposals/rerun-joint-trajectory-2026-11-24/). Every check reads the .rrd back with Rerun's own
reader; the arm's kinematics are checked against a closed form (the tiny arm) and against arm_motion_bench's UR5e
forward kinematics (the round's data), never against our own forward kinematics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")

from robot_msgs.geometry.rotations import quat_from_matrix  # noqa: E402
from robot_msgs.std.header import Header  # noqa: E402
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory  # noqa: E402
from robot_msgs.viz import rerun as viz_rerun  # noqa: E402

ROUND = Path("/home/bam/traj_compare/docs/proposals/rerun-joint-trajectory-2026-11-24/data")

# A made-up arm: a continuous pan, a revolute elbow with limits, a prismatic slider, a fixed tip; primitives only, and
# one link (upper) with both a visual and a collision box.
TINY_URDF = """<?xml version="1.0"?>
<robot name="tiny_arm">
  <link name="base"><visual><geometry><cylinder radius="0.05" length="0.1"/></geometry></visual></link>
  <link name="upper">
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.04 0.04 0.3"/></geometry></visual>
    <collision><origin xyz="0 0 0.15"/><geometry><box size="0.05 0.05 0.3"/></geometry></collision>
  </link>
  <link name="lower"><visual><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></visual></link>
  <link name="slider"><visual><geometry><sphere radius="0.02"/></geometry></visual></link>
  <link name="tip"/>
  <joint name="pan" type="continuous"><parent link="base"/><child link="upper"/><origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="lower"/><origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" effort="5" velocity="3"/></joint>
  <joint name="extend" type="prismatic"><parent link="lower"/><child link="slider"/><origin xyz="0 0 0.2"/>
    <axis xyz="0 0 1"/><limit lower="0.0" upper="0.1" effort="5" velocity="0.2"/></joint>
  <joint name="tip_fixed" type="fixed"><parent link="slider"/><child link="tip"/><origin xyz="0 0 0.02"/></joint>
</robot>
"""


@pytest.fixture
def tiny(tmp_path) -> Path:
    path = tmp_path / "tiny_arm.urdf"
    path.write_text(TINY_URDF)
    return path


def tiny_trajectory(n: int = 41) -> JointTrajectory:
    t = np.linspace(0.0, 1.0, n)
    s = 0.5 - 0.5 * np.cos(np.pi * t)  # rest to rest
    positions = np.stack([1.0 * s, 0.2 + 1.3 * s, 0.05 * s], 1)
    return JointTrajectory(t=t, joint_names=("pan", "elbow", "extend"), positions=positions).with_derivatives()


def tiny_tip(q: np.ndarray) -> np.ndarray:
    """The tip of the made-up arm, closed form: base → pan (z 0.1, about z) → elbow (z 0.3, about y) → slider."""
    pan, elbow, extend = q[:, 0], q[:, 1], q[:, 2]
    reach = 0.2 + extend + 0.02  # lower link along its z, the slider, the fixed tip
    upper = np.stack([np.sin(elbow) * reach, np.zeros_like(pan), 0.3 + np.cos(elbow) * reach], 1)
    rotated = np.stack([np.cos(pan) * upper[:, 0] - np.sin(pan) * upper[:, 1],
                        np.sin(pan) * upper[:, 0] + np.cos(pan) * upper[:, 1], upper[:, 2]], 1)
    return rotated + [0.0, 0.0, 0.1]


# ── reading the .rrd back with Rerun's own reader ────────────────────────────


def rows(path: Path, entity: str, column: str) -> list[tuple[float | None, object]]:
    """(t in seconds or None for static, value) of one component column at one entity, in time order."""
    from rerun.chunk import RrdReader

    found = []
    for chunk in RrdReader(path).stream():
        if chunk.entity_path != entity:
            continue
        batch = chunk.to_record_batch()
        if column not in batch.schema.names:
            continue
        values = batch.column(column).to_pylist()
        times = (batch.column("t").cast("int64").to_pylist() if "t" in batch.schema.names else [None] * len(values))
        found += [(None if ns is None else ns / 1e9, value) for ns, value in zip(times, values, strict=True)]
    return sorted(found, key=lambda row: -1.0 if row[0] is None else row[0])


def entities(path: Path) -> set[str]:
    from rerun.chunk import RrdReader

    return {chunk.entity_path for chunk in RrdReader(path).stream()}


def scalars(path: Path, entity: str) -> np.ndarray:
    return np.asarray([value[0] for _, value in rows(path, entity, "Scalars:scalars") if value])


def instances(path: Path, entity: str) -> np.ndarray:
    """(N, k): a series entity whose k instances are k lines (the TCP's axes, a card's q / qd / qdd)."""
    return np.asarray([value for _, value in rows(path, entity, "Scalars:scalars") if value])


def views(node) -> list:
    """Every view under a blueprint container, depth first (a view's own contents are entity paths)."""
    if type(node).__name__ in ("Horizontal", "Vertical", "Grid", "Tabs"):
        return [view for child in node.contents for view in views(child)]
    return [node]


# ── the tiny arm: any URDF, primitives, a prismatic joint, URDF limits ───────


def test_robot_draws_the_urdf_arm_moving_on_the_t_timeline(tiny, tmp_path):
    traj = tiny_trajectory()
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny))
    found = entities(path)
    base = "/world/plan"
    # the arm: Rerun's own URDF loader, re-rooted under /world/<name>/robot, its moving joints on the t timeline
    assert f"{base}/robot/visual_geometries/upper/visual_0" in found
    # a link with a visual draws the visual only (its collision box is left out, not drawn on top)
    assert not any(entity.startswith(f"{base}/robot/collision_geometries") for entity in found)
    for joint in ("pan", "elbow", "extend"):
        transforms = rows(path, f"{base}/robot/joint_transforms/{joint}", "Transform3D:child_frame")
        assert len([row for row in transforms if row[0] is not None]) == len(traj.t), joint
    mount = rows(path, f"{base}/robot/mount", "Transform3D:child_frame")
    assert mount and mount[0][1] == ["plan/base"]  # the URDF's root frame, connected to the scene
    # the ghosts at the start and end poses, and the tool path
    assert {f"{base}/ghost_start/visual_geometries/upper/visual_0", f"{base}/ghost_end/visual_geometries/upper/visual_0",
            f"{base}/tool/trail", f"{base}/tool/path", f"{base}/tool/ends"} <= found


def test_the_series_keep_robot_msgs_layout_in_degrees_for_revolute_joints_and_si_for_prismatic(tiny, tmp_path):
    traj = tiny_trajectory()
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny))
    np.testing.assert_allclose(scalars(path, "/positions/pan/plan"), np.degrees(traj.positions[:, 0]))
    np.testing.assert_allclose(scalars(path, "/positions/elbow/plan"), np.degrees(traj.positions[:, 1]))
    np.testing.assert_allclose(scalars(path, "/velocities/elbow/plan"), np.degrees(traj.velocities[:, 1]))
    np.testing.assert_allclose(scalars(path, "/positions/extend/plan"), traj.positions[:, 2])  # metres
    # limits from the URDF (the message has none), in the units the strip shows; a continuous joint has none
    np.testing.assert_allclose(scalars(path, "/positions/elbow/lower_limit"), np.degrees([-2.0, -2.0]))
    np.testing.assert_allclose(scalars(path, "/positions/extend/upper_limit"), [0.1, 0.1])
    assert "/positions/pan/lower_limit" not in entities(path)


def test_the_tool_path_without_a_tool_is_the_tip_from_the_urdf(tiny, tmp_path):
    traj = tiny_trajectory()
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny))
    trail = rows(path, "/world/plan/tool/trail", "LineStrips3D:strips")
    last = np.asarray(trail[-1][1][0])  # the travelled part at the last instant: the whole path
    np.testing.assert_allclose(last, tiny_tip(np.asarray(traj.positions)), atol=1e-9)
    first = np.asarray(trail[0][1][0])
    assert len(first) == 1  # at t = 0 nothing is travelled yet
    frame = rows(path, "/world/plan/tool/trail", "CoordinateFrame:frame")
    assert frame[0][1] == ["plan/base"]
    assert "/world/plan/tool/stick" not in entities(path)  # the tool IS the tip: nothing to bridge


def test_every_plotted_series_and_the_tool_path_show_a_dot_per_sample(tiny, tmp_path):
    """WHY (Zach, 2026-11-25): "right now its just a continue line you cannot see like the time sample"."""
    traj = tiny_trajectory(600)  # more samples than the tool polylines keep (_VERTICES): the dots keep every one
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny))
    series = [f"/{field}/{joint}/plan" for field in ("positions", "velocities", "accelerations", "jerks")
              for joint in traj.joint_names] + [f"/tcp/{group}/plan" for group in
                                                ("position_mm", "orientation_deg", "speed_mm_s", "angular_speed_deg_s")]
    for entity in series:
        line = rows(path, entity, "SeriesLines:names")
        dots = rows(path, entity, "SeriesPoints:names")
        assert dots and dots[0][1] == line[0][1], entity  # the same legend entry as its line
        assert rows(path, entity, "SeriesPoints:markers")[0][1] == [1], entity  # a circle
    samples = np.asarray(rows(path, "/world/plan/tool/samples", "Points3D:positions")[0][1])
    np.testing.assert_allclose(samples, tiny_tip(np.asarray(traj.positions)), atol=1e-9)
    assert len(np.asarray(rows(path, "/world/plan/tool/trail", "LineStrips3D:strips")[-1][1][0])) < len(samples)
    # the limit lines and the cards stay lines
    assert not rows(path, "/positions/elbow/lower_limit", "SeriesPoints:names")
    # the URDF-free tabs share the path, so they get the dots too
    import rerun as rr

    stream = rr.RecordingStream("dots")
    stream.save(str(tmp_path / "tabs.rrd"))
    viz_rerun.log_joint_tabs(traj, "plan", recording=stream)
    stream.flush()
    stream.disconnect()
    assert rows(tmp_path / "tabs.rrd", "/positions/elbow/plan", "SeriesPoints:names")


def tabs_of(blueprint) -> dict[str, list]:
    """{tab name: its views} of the right half's Tabs."""
    _, right = blueprint.root_container.contents
    assert type(right).__name__ == "Tabs"
    return {tab.name: views(tab) for tab in right.contents}


def test_the_dashboard_is_the_arm_left_and_tabs_of_joint_and_tcp_plots_right_paused_at_the_start(tiny):
    # WHY (Zach 2026-11-24, after the round): "I like R3 ... But Please add tabs Similar to R1 so that you can switch
    # between what information you are showing on the right hand side ... we are always showing the 3d robot"
    traj = tiny_trajectory()
    blueprint = viz_rerun.robot_blueprint(traj, "plan", robot=tiny)
    root = blueprint.root_container
    assert type(root).__name__ == "Horizontal" and root.column_shares == [1, 1]  # the page split in half
    world = root.contents[0]
    assert type(world).__name__ == "Spatial3DView" and "/world/plan/**" in world.contents
    tabs = tabs_of(blueprint)
    assert list(tabs) == ["q", "qd", "qdd", "qddd", "q / qd / qdd", "TCP"]
    assert root.contents[1].active_tab in (0, "q")  # q opens
    names = {tab: [view.name for view in found if type(view).__name__ == "TimeSeriesView"] for tab, found in tabs.items()}
    assert names["q"] == ["pan [°] · travel 57°", "elbow [°] · travel 74° · 29° to limit",
                          "extend [m] · travel 0.050 m · 0.000 m to limit"]
    assert names["qd"][1].startswith("elbow [°/s] · peak ") and names["qd"][1].endswith("% of limit")
    assert names["qd"][0].startswith("pan [°/s] · peak ") and "limit" not in names["qd"][0]  # continuous: none
    assert names["qdd"][1].startswith("elbow [°/s²] · peak ")
    assert names["qddd"][1].startswith("elbow [°/s³] · peak ") and "slope of qdd" in names["qddd"][1]
    assert len(names["q / qd / qdd"]) == 2  # a card per joint with a limit: elbow, extend (pan has none)
    assert [type(view).__name__ for view in tabs["q / qd / qdd"]][0] == "TextDocumentView"  # the table on top
    assert tabs["q / qd / qdd"][0].origin == "/notes/plan"
    tcp = names["TCP"]
    assert [name.split(" · ")[0] for name in tcp] == [
        "TCP position [mm]", "TCP orientation [°]", "TCP speed [mm/s]", "TCP angular speed [°/s]"]
    assert "roll / pitch / yaw" in tcp[1] and all("FK of tip" in name for name in tcp)
    panel = blueprint.time_panel
    assert (panel.play_state, panel.loop_mode, panel.timeline) == ("paused", "all", "t")


def test_a_tab_whose_data_is_missing_is_left_out(tiny):
    full = tiny_trajectory()
    positions_only = JointTrajectory(t=full.t, joint_names=full.joint_names, positions=full.positions)
    assert list(tabs_of(viz_rerun.robot_blueprint(positions_only, "plan", robot=tiny))) == ["q", "q / qd / qdd", "TCP"]


def test_jerk_is_the_slope_of_qdd_on_the_trajectorys_own_clock_and_jerk_limits_are_drawn(tiny, tmp_path):
    full = tiny_trajectory()
    limits = JointLimits(joint_names=full.joint_names, max_velocity=[2.0, 3.0, 0.2],
                         max_acceleration=[10.0, 10.0, 1.0], max_jerk=[100.0, 200.0, 5.0])
    traj = full.replace(limits=limits)
    path = Path(viz_rerun.save(str(tmp_path / "jerk.rrd"), plan=traj, robot=tiny))
    qdd, t = np.asarray(traj.accelerations)[:, 1], np.asarray(traj.t)
    slope = np.empty_like(qdd)  # central differences inside, one-sided at the ends (traj_compare's _slope)
    slope[1:-1] = (qdd[2:] - qdd[:-2]) / (t[2:] - t[:-2])
    slope[0], slope[-1] = (qdd[1] - qdd[0]) / (t[1] - t[0]), (qdd[-1] - qdd[-2]) / (t[-1] - t[-2])
    np.testing.assert_allclose(scalars(path, "/jerks/elbow/plan"), np.degrees(slope), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scalars(path, "/jerks/elbow/upper_limit"), np.degrees([200.0, 200.0]))
    np.testing.assert_allclose(scalars(path, "/jerks/extend/lower_limit"), [-5.0, -5.0])  # prismatic: m/s³
    np.testing.assert_allclose(scalars(path, "/velocities/pan/upper_limit"), np.degrees([2.0, 2.0]))  # the message's
    np.testing.assert_allclose(scalars(path, "/accelerations/elbow/lower_limit"), np.degrees([-10.0, -10.0]))
    names = [view.name for view in tabs_of(viz_rerun.robot_blueprint(traj, "plan", robot=tiny))["qddd"]]
    assert names[1].endswith("% of limit")


def test_the_tcp_tab_is_the_tips_position_orientation_and_speeds(tiny, tmp_path):
    traj = tiny_trajectory()
    path = Path(viz_rerun.save(str(tmp_path / "tcp.rrd"), plan=traj, robot=tiny))
    q, t = np.asarray(traj.positions), np.asarray(traj.t)
    tip = tiny_tip(q)
    np.testing.assert_allclose(instances(path, "/tcp/position_mm/plan"), tip * 1000.0, atol=1e-9)  # x, y, z
    # the tip's rotation is Rz(pan) · Ry(elbow): roll 0, pitch = elbow, yaw = pan (elbow stays inside ±90°)
    rpy = instances(path, "/tcp/orientation_deg/plan")
    np.testing.assert_allclose(rpy, np.degrees(np.stack([0 * q[:, 0], q[:, 1], q[:, 0]], 1)), atol=1e-9)
    # |omega| of Rz(pan) Ry(elbow) is sqrt(pan' ** 2 + elbow' ** 2): the rest-to-rest profile's own derivative
    rate = 0.5 * np.pi * np.sin(np.pi * t)
    omega = np.degrees(np.hypot(1.0 * rate, 1.3 * rate))
    angular = scalars(path, "/tcp/angular_speed_deg_s/plan")
    np.testing.assert_allclose(angular[1:-1], omega[1:-1], rtol=2e-3, atol=0.05)
    linear = scalars(path, "/tcp/speed_mm_s/plan")
    np.testing.assert_allclose(linear, np.linalg.norm(np.gradient(tip, t, axis=0, edge_order=1), axis=1) * 1000.0)


def test_the_playhead_table_gives_every_joint_in_degrees_and_its_worst_percent_of_limit(tiny, tmp_path):
    traj = tiny_trajectory()
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny))
    tables = rows(path, "/notes/plan", "TextDocument:text")
    assert len(tables) == len(traj.t)
    t, text = tables[20]
    assert t == pytest.approx(traj.t[20])
    assert f"t = {traj.t[20]:.2f} s" in text[0]
    lines = {line.split("|")[1].strip(): line for line in text[0].splitlines() if line.startswith("| ") and "---" not in line}
    assert f"{np.degrees(traj.positions[20, 0]):+.1f}°" in lines["pan"]
    assert f"{np.degrees(traj.positions[20, 1]):+.1f}°" in lines["elbow"]
    assert f"{traj.positions[20, 2]:+.3f} m" in lines["extend"]  # prismatic: metres, as its strip
    # elbow at 0.85 rad in ±2 rad: 42 % of the way to a limit; its URDF speed limit is 3 rad/s
    percent = abs(traj.positions[20, 1]) / 2.0 * 100.0
    speed = abs(traj.velocities[20, 1]) / 3.0 * 100.0
    assert lines["elbow"].rstrip(" |").endswith(f"{max(percent, speed):.0f} {'q' if percent >= speed else 'qd'}")


def test_ghosts_false_leaves_out_the_start_and_end_arms_and_is_smaller(tiny, tmp_path):
    traj = tiny_trajectory()
    with_ghosts = Path(viz_rerun.save(str(tmp_path / "g.rrd"), plan=traj, robot=tiny))
    without = Path(viz_rerun.save(str(tmp_path / "n.rrd"), plan=traj, robot=tiny, ghosts=False))
    assert not any("/ghost_" in entity for entity in entities(without))
    assert "/world/plan/robot/joint_transforms/pan" in entities(without)
    assert without.stat().st_size < with_ghosts.stat().st_size


def test_a_tool_pose_trajectory_is_drawn_in_the_link_its_header_names(tiny, tmp_path):
    traj = tiny_trajectory()
    tip = tiny_tip(np.asarray(traj.positions))
    tool = PoseTrajectory(t=traj.t, positions=tip, orientations=np.tile([0.0, 0.0, 0.0, 1.0], (len(tip), 1)),
                          header=Header(frame_id="base"), child_frame_id="tcp")
    path = Path(viz_rerun.save(str(tmp_path / "tool.rrd"), plan=traj, robot=tiny, tool=tool))
    np.testing.assert_allclose(np.asarray(rows(path, "/world/plan/tool/trail", "LineStrips3D:strips")[-1][1][0]), tip)
    tcp = [view.name for view in tabs_of(viz_rerun.robot_blueprint(traj, "plan", robot=tiny, tool=tool))["TCP"]]
    assert all("tool PoseTrajectory tcp in base" in name for name in tcp), tcp
    elsewhere = PoseTrajectory(t=traj.t, positions=tip, orientations=tool.orientations, header=Header(frame_id="cell"))
    with pytest.raises(ValueError, match="cell"):
        viz_rerun.save(str(tmp_path / "x.rrd"), plan=traj, robot=tiny, tool=elsewhere)


def test_robot_refuses_what_it_cannot_draw(tiny, tmp_path):
    traj = tiny_trajectory()
    wrist = JointTrajectory(t=traj.t, joint_names=("pan", "wrist"), positions=np.asarray(traj.positions)[:, :2])
    with pytest.raises(ValueError, match="wrist"):
        viz_rerun.save(str(tmp_path / "a.rrd"), plan=wrist, robot=tiny)
    with pytest.raises(ValueError, match="one JointTrajectory"):
        viz_rerun.save(str(tmp_path / "b.rrd"), target=traj, real=traj, robot=tiny)
    with pytest.raises(ValueError, match="one JointTrajectory"):
        viz_rerun.save(str(tmp_path / "c.rrd"), plan=traj, other=PoseTrajectory(
            t=traj.t, positions=np.zeros((len(traj.t), 3)), orientations=np.tile([0.0, 0.0, 0.0, 1.0], (len(traj.t), 1))),
            robot=tiny)


def test_log_with_robot_refuses_the_stock_series_options(tiny, tmp_path):
    import rerun as rr

    stream = rr.RecordingStream("robot_msgs_test", recording_id="t")
    with pytest.raises(TypeError, match="t0"):
        viz_rerun.log(tiny_trajectory(), "plan", recording=stream, robot=tiny, t0=0.0)
    with pytest.raises(TypeError, match="robot="):
        viz_rerun.log(tiny_trajectory(), "plan", recording=stream, tool=PoseTrajectory(
            t=[0.0, 1.0], positions=np.zeros((2, 3)), orientations=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))))


def test_log_composes_into_any_recording_and_robot_blueprint_lays_it_out(tiny, tmp_path):
    import rerun as rr

    traj = tiny_trajectory()
    path = tmp_path / "log.rrd"
    stream = rr.RecordingStream("robot_msgs_test", recording_id="t")
    stream.save(str(path), default_blueprint=viz_rerun.robot_blueprint(traj, "plan", robot=tiny))
    viz_rerun.log(traj, "plan", recording=stream, robot=tiny, ghosts=False)
    stream.flush()
    stream.disconnect()
    found = entities(path)
    assert {"/positions/elbow/plan", "/world/plan/robot/joint_transforms/elbow", "/notes/plan"} <= found


def test_show_and_recording_take_robot_and_tool(tiny, tmp_path, monkeypatch):
    traj = tiny_trajectory()
    monkeypatch.setattr(viz_rerun, "view", lambda stream: stream)  # never spawn a viewer in a test
    stream = viz_rerun.show(plan=traj, robot=tiny)
    path = tmp_path / "shown.rrd"
    stream.save(str(path))  # the buffered recording, blueprint first
    stream.flush()
    stream.disconnect()
    from rerun.chunk import RrdReader

    assert RrdReader(path).blueprints()
    assert "/world/plan/robot/joint_transforms/pan" in entities(path)


def test_joint_trajectory_show_forwards_robot_and_tool_to_the_rerun_viewer(tiny, monkeypatch):
    opened = []
    monkeypatch.setattr(viz_rerun, "show", lambda **named: opened.append(named) or "stream")
    traj = tiny_trajectory()
    assert traj.show(robot=tiny) == "stream"
    assert opened[-1] == {"JointTrajectory": traj, "robot": tiny}
    traj.show("plan", robot=tiny, tool=None)
    assert opened[-1] == {"plan": traj, "robot": tiny, "tool": None}


# ── the UR5e: the round's URDF and its real dataset trajectories ─────────────


def ur5e(task: str) -> tuple[JointTrajectory, PoseTrajectory, dict]:
    urdf = ROUND / "robot" / "ur5e_collision.urdf"
    npz = ROUND / f"{task}.npz"
    if not (urdf.exists() and npz.exists()):
        pytest.skip(f"the round's UR5e data is not on this machine ({ROUND})")
    data = dict(np.load(npz))
    names = tuple(str(name) for name in data["names"])
    limits = JointLimits(joint_names=names, min_position=data["lo"], max_position=data["hi"],
                         max_velocity=data["vmax"], max_acceleration=data["amax"])
    traj = JointTrajectory(t=data["t"], joint_names=names, positions=data["q"], velocities=data["qd"],
                           accelerations=data["qdd"], limits=limits)
    tool = PoseTrajectory(t=data["t"], positions=data["tool_xyz"], orientations=quat_from_matrix(data["tool_rot"]),
                          header=Header(frame_id="base_link_ur"), child_frame_id="tool")
    return traj, tool, {"urdf": urdf, **data}


def test_ur5e_the_flange_path_matches_arm_motion_bench_forward_kinematics(tmp_path):
    traj, _, data = ur5e("workspace-dev-003")  # 177 samples: the trail keeps every one
    path = Path(viz_rerun.save(str(tmp_path / "ur.rrd"), plan=traj, robot=data["urdf"]))
    flange = np.asarray(rows(path, "/world/plan/tool/trail", "LineStrips3D:strips")[-1][1][0])
    # the dataset's TCP (arm_motion_bench's own FK) sits a fixed distance out along the flange's z axis
    out = np.einsum("nij,nj->ni", np.transpose(data["tool_rot"], (0, 2, 1)), data["tool_xyz"] - flange)
    assert np.ptp(out, axis=0).max() < 0.002, out.min(axis=0)
    assert abs(np.median(out[:, 2]) - 0.1447) < 0.002 and np.abs(out[:, :2]).max() < 0.002


def test_ur5e_limits_come_from_the_message_and_the_whole_dashboard_is_there(tmp_path):
    traj, tool, data = ur5e("workspace-dev-029")
    path = Path(viz_rerun.save(str(tmp_path / "ur.rrd"), plan=traj, robot=data["urdf"], tool=tool))
    found = entities(path)
    for joint in traj.joint_names:
        assert f"/world/plan/robot/joint_transforms/{joint}" in found
        np.testing.assert_allclose(scalars(path, f"/positions/{joint}/plan"),
                                   np.degrees(traj.positions[:, list(traj.joint_names).index(joint)]))
    # the message's ±360° pan, not the URDF's ±180°
    np.testing.assert_allclose(scalars(path, "/positions/shoulder_pan_joint/upper_limit"), np.degrees(data["hi"][[0, 0]]))
    assert any(entity.startswith("/world/plan/robot/collision_geometries/mesh/") for entity in found)
    np.testing.assert_allclose(np.asarray(rows(path, "/world/plan/tool/trail", "LineStrips3D:strips")[-1][1][0])[-1],
                               data["tool_xyz"][-1])
    # the stick from the flange to the TCP (the URDF ends at the flange; the tool is 145 mm further out)
    sticks = [np.asarray(value[0]) for t, value in rows(path, "/world/plan/tool/stick", "LineStrips3D:strips") if t is not None]
    assert len(sticks) == len(traj.t)
    lengths = [np.linalg.norm(stick[1] - stick[0]) for stick in sticks]
    assert abs(np.median(lengths) - 0.1447) < 0.002 and np.ptp(lengths) < 0.003
    tabs = tabs_of(viz_rerun.robot_blueprint(traj, "plan", robot=data["urdf"], tool=tool))
    assert list(tabs) == ["q", "qd", "qdd", "qddd", "q / qd / qdd", "TCP"]
    assert len(tabs["q"]) == 6 and all(" · travel " in view.name for view in tabs["q"])
    assert len([view for view in tabs["q / qd / qdd"] if type(view).__name__ == "TimeSeriesView"]) == 6
    for field_name in ("velocities", "accelerations"):
        assert f"/{field_name}/wrist_1_joint/upper_limit" in found


def test_ur5e_each_ghost_is_another_copy_of_the_meshes(tmp_path):
    traj, tool, data = ur5e("workspace-dev-029")
    sizes = {}
    for ghosts in (True, False):
        path = Path(viz_rerun.save(str(tmp_path / f"{ghosts}.rrd"), plan=traj, robot=data["urdf"], tool=tool,
                                   ghosts=ghosts))
        sizes[ghosts] = path.stat().st_size
    meshes = sum(path.stat().st_size for path in (data["urdf"].parent / "meshes").glob("*.stl"))
    print(json.dumps({"with_ghosts_bytes": sizes[True], "without_ghosts_bytes": sizes[False], "stl_bytes": meshes}))
    assert sizes[True] - sizes[False] > meshes  # two more copies, compressed
