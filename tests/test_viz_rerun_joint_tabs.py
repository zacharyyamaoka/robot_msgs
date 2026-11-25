"""robot_msgs.viz.rerun `log_joint_tabs` / `joint_tabs_blueprint`: the stock trajectory viewer's tabs without a URDF.

WHY (Zach, 2026-11-25): "when we are look at just a normal trajectory without comparison than I think it makes sense to
just keep with V1. that should basically just be the stock like robot msg viz viewer for a joint trajectory". A
JointTrajectory with no URDF gets V1's right-hand tabs (q, qd, qdd, qddd, q / qd / qdd) at full width. They must be the
SAME tabs as ``robot=`` draws (one code path, never a copy), and ``log`` without ``robot=`` stays frozen (the golden
digest test). Every check reads the .rrd back with Rerun's own reader, or the blueprint object itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

from robot_msgs.trajectory import JointLimits, JointTrajectory  # noqa: E402
from robot_msgs.viz import rerun as viz_rerun  # noqa: E402

from test_viz_rerun_robot import rows  # noqa: E402

# Two revolute joints only, so with the message's full limits ``robot=`` reads them exactly as the URDF-free tabs do.
TWO_URDF = """<?xml version="1.0"?>
<robot name="two_arm">
  <link name="base"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="upper"><visual><origin xyz="0 0 0.15"/><geometry><box size="0.04 0.04 0.3"/></geometry></visual></link>
  <link name="lower"><visual><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></visual></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/><origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="5" velocity="9"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="lower"/><origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="5" velocity="9"/></joint>
</robot>
"""


def two_joint(limits: bool = True) -> JointTrajectory:
    t = np.linspace(0.0, 2.0, 81)
    s = 0.5 - 0.5 * np.cos(np.pi * t / 2.0)
    names = ("shoulder", "elbow")
    bounds = JointLimits(joint_names=names, min_position=[-1.0, -2.0], max_position=[1.0, 2.0],
                         max_velocity=[1.5, 2.0], max_acceleration=[4.0, 6.0]) if limits else None  # fmt: skip
    return JointTrajectory(t=t, joint_names=names, positions=np.stack([0.9 * s, -0.4 + 1.6 * s], 1),
                           limits=bounds).with_derivatives()  # fmt: skip


def save(path: Path, traj: JointTrajectory, name: str = "plan") -> Path:
    stream = rr.RecordingStream("joint_tabs")
    stream.save(str(path), default_blueprint=viz_rerun.joint_tabs_blueprint(traj, name))
    viz_rerun.log_joint_tabs(traj, name, recording=stream)
    stream.flush()
    stream.disconnect()
    return path


def views(container) -> list:
    if isinstance(container, rrb.View):
        return [container]
    return [view for child in (getattr(container, "contents", None) or []) for view in views(child)]


def test_the_tabs_are_V1s_right_half_at_full_width():
    blueprint = viz_rerun.joint_tabs_blueprint(two_joint(), "plan")
    root = blueprint.root_container
    assert isinstance(root, rrb.Tabs), "the tabs ARE the page: no arm, no split"
    assert [tab.name for tab in root.contents] == ["q", "qd", "qdd", "qddd", "q / qd / qdd"]
    assert not [v for v in views(root) if isinstance(v, rrb.Spatial3DView)], "no 3D view without a URDF"


def test_the_tabs_are_the_same_strips_robot_draws(tmp_path):
    """One code path: the joint tabs of `robot_blueprint` on an all-revolute URDF, minus TCP, are these tabs."""
    urdf = tmp_path / "two.urdf"
    urdf.write_text(TWO_URDF)
    traj = two_joint()
    alone = viz_rerun.joint_tabs_blueprint(traj, "plan").root_container
    with_arm = viz_rerun.robot_blueprint(traj, "plan", robot=urdf).root_container.contents[1]

    def strips(tabs) -> list:
        return [(tab.name, [(v.name, str(v.properties.get("ScalarAxis"))) for v in views(tab)])
                for tab in tabs.contents if tab.name != "TCP"]  # fmt: skip

    assert strips(alone) == strips(with_arm)


def test_the_recording_holds_every_tabs_data_in_degrees(tmp_path):
    traj = two_joint()
    path = save(tmp_path / "tabs.rrd", traj)
    q = rows(path, "/positions/elbow/plan", "Scalars:scalars")
    values = np.array([value[0] for at, value in q if at is not None])
    np.testing.assert_allclose(values, np.degrees(traj.positions[:, 1]), rtol=1e-9)
    for entity in ("/velocities/shoulder/plan", "/accelerations/elbow/plan", "/jerks/shoulder/plan",
                   "/limit_pct/elbow/plan", "/positions/shoulder/upper_limit"):  # fmt: skip
        assert rows(path, entity, "Scalars:scalars"), f"nothing at {entity}"
    tables = [value for at, value in rows(path, "/notes/plan", "TextDocument:text") if at is not None]
    assert len(tables) >= 2, "the playhead table changes with the playhead"
    assert not rows(path, "/world/plan/robot", "Transform3D:translation"), "no arm"


def test_without_limits_there_are_no_cards_and_no_limit_lines(tmp_path):
    traj = two_joint(limits=False)
    names = [tab.name for tab in viz_rerun.joint_tabs_blueprint(traj).root_container.contents]
    assert names == ["q", "qd", "qdd", "qddd"]
    path = save(tmp_path / "bare.rrd", traj, "JointTrajectory")
    assert rows(path, "/positions/elbow/JointTrajectory", "Scalars:scalars")
    assert not rows(path, "/positions/elbow/upper_limit", "Scalars:scalars")


def test_positions_alone_leave_out_the_rate_tabs():
    t = np.linspace(0.0, 1.0, 11)
    traj = JointTrajectory(t=t, joint_names=("a",), positions=t[:, None])
    assert [tab.name for tab in viz_rerun.joint_tabs_blueprint(traj).root_container.contents] == ["q"]
