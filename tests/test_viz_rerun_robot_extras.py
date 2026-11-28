"""robot_msgs.viz.rerun with ``robot=`` and the opt-in extras: ``wrenches=`` (a force arrow at the tool), ``events=``
(moments on the timeline) and ``plots=`` (more tabs of series on the trajectory's clock).

WHY (Zach, 2026-11-27, the gravity compensation box): "we could show this in rerun no? you can show an external force
being applied as like a vector arrow?". Every check reads the .rrd back with Rerun's own reader; the tool point is the
tiny arm's closed form (tests/test_viz_rerun_robot.py), never our own forward kinematics. Without the extras nothing
changes (the existing robot tests and the stock golden hold that).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")

from robot_msgs.std.header import Header  # noqa: E402
from robot_msgs.trajectory import JointTrajectory, WrenchTrajectory  # noqa: E402
from robot_msgs.viz import rerun as viz_rerun  # noqa: E402

from test_viz_rerun_robot import TINY_URDF, entities, rows, scalars, tabs_of, tiny_tip, tiny_trajectory, views  # noqa: E402


@pytest.fixture
def tiny(tmp_path) -> Path:
    path = tmp_path / "tiny_arm.urdf"
    path.write_text(TINY_URDF)
    return path


def push(traj: JointTrajectory, frame: str = "") -> WrenchTrajectory:
    """A push along +x that ramps 0 → 20 N and back over the trajectory, in the URDF's root frame."""
    t = np.asarray(traj.t)
    force = np.zeros((len(t), 3))
    force[:, 0] = 20.0 * np.sin(np.pi * t / t[-1])
    return WrenchTrajectory(t=t, forces=force, torques=np.zeros_like(force), header=Header(frame_id=frame))


def test_a_wrench_is_an_arrow_at_the_tool_point_scaled_and_labelled_with_its_magnitude(tiny, tmp_path):
    traj = tiny_trajectory()
    wrench = push(traj)
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny, wrenches={"hand": wrench}))
    entity = "/world/plan/wrenches/hand"
    assert entity in entities(path)
    origins = [(t, value) for t, value in rows(path, entity, "Arrows3D:origins") if t is not None]
    vectors = [(t, value) for t, value in rows(path, entity, "Arrows3D:vectors") if t is not None]
    labels = [(t, value) for t, value in rows(path, entity, "Arrows3D:labels") if t is not None]
    assert len(origins) == len(vectors) == len(labels) == len(traj.t)  # 41 samples: every one drawn
    times = np.array([t for t, _ in origins])
    np.testing.assert_allclose(times, np.asarray(traj.t), atol=1e-9)
    # the arrow starts at the tool point (the tiny arm's tip, closed form)
    np.testing.assert_allclose(np.array([v[0] for _, v in origins]), tiny_tip(np.asarray(traj.positions)), atol=1e-6)
    # its length is the force times one scale, the same for every sample; the scale makes the peak 0.1-0.25 m
    drawn = np.array([v[0] for _, v in vectors])
    scale = drawn[:, 0].max() / 20.0
    np.testing.assert_allclose(drawn, np.asarray(wrench.forces) * scale, rtol=1e-6, atol=1e-9)
    assert 0.1 <= 20.0 * scale <= 0.25
    # its label names the wrench and the TRUE magnitude in N (not the drawn length)
    peak = int(np.argmax(np.asarray(wrench.forces)[:, 0]))
    assert labels[peak][1] == ["hand 20.0 N"]
    # and a Wrench tab plots each axis and the magnitude, in N
    blueprint = viz_rerun.robot_blueprint(traj, "plan", robot=tiny, wrenches={"hand": wrench})
    tabs = tabs_of(blueprint)
    assert list(tabs)[-1] == "Wrench"
    (strip,) = [view for view in tabs["Wrench"] if type(view).__name__ == "TimeSeriesView"]
    assert strip.name.startswith("hand: force [N] · peak 20.0 N · arrow 1 N = ")
    np.testing.assert_allclose(scalars(path, "/wrench/hand/magnitude"), np.abs(np.asarray(wrench.forces)[:, 0]), atol=1e-9)


def test_a_wrench_must_be_in_the_tools_frame_and_on_the_trajectorys_clock(tiny, tmp_path):
    traj = tiny_trajectory()
    with pytest.raises(ValueError, match="frame"):
        viz_rerun.save(str(tmp_path / "a.rrd"), plan=traj, robot=tiny, wrenches={"hand": push(traj, frame="upper")})
    with pytest.raises(TypeError, match="WrenchTrajectory"):
        viz_rerun.save(str(tmp_path / "b.rrd"), plan=traj, robot=tiny, wrenches={"hand": np.zeros((41, 3))})
    with pytest.raises(TypeError, match="robot="):
        viz_rerun.save(str(tmp_path / "c.rrd"), plan=traj, wrenches={"hand": push(traj)})


def test_events_are_text_log_entries_on_the_timeline_and_an_events_tab(tiny, tmp_path):
    traj = tiny_trajectory()
    events = [(0.75, "release"), (0.25, "hand grabs")]
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny, events=events))
    logged = [(t, value) for t, value in rows(path, "/events", "TextLog:text") if t is not None]
    assert logged == [(0.25, ["hand grabs"]), (0.75, ["release"])]
    tabs = tabs_of(viz_rerun.robot_blueprint(traj, "plan", robot=tiny, events=events))
    assert [type(view).__name__ for view in tabs["Events"]] == ["TextLogView"]
    assert tabs["Events"][0].origin == "/events"


def test_plots_add_tabs_of_series_on_the_trajectorys_clock(tiny, tmp_path):
    traj = tiny_trajectory()
    n = len(traj.t)
    plots = {"Model check": {"elbow: gravity residual [N·m]": {"reported": np.linspace(0, 1, n),
                                                                 "true": np.linspace(0, 2, n)}}}
    path = Path(viz_rerun.save(str(tmp_path / "arm.rrd"), plan=traj, robot=tiny, plots=plots))
    np.testing.assert_allclose(scalars(path, "/plots/model_check/0/true"), np.linspace(0, 2, n))
    np.testing.assert_allclose(scalars(path, "/plots/model_check/0/reported"), np.linspace(0, 1, n))
    tabs = tabs_of(viz_rerun.robot_blueprint(traj, "plan", robot=tiny, plots=plots))
    (strip,) = tabs["Model check"]
    assert strip.name == "elbow: gravity residual [N·m]"
    with pytest.raises(ValueError, match="one value per sample"):
        viz_rerun.save(str(tmp_path / "b.rrd"), plan=traj, robot=tiny, plots={"X": {"y": {"a": np.zeros(3)}}})


def test_without_extras_the_dashboard_is_unchanged(tiny):
    traj = tiny_trajectory()
    plain = viz_rerun.robot_blueprint(traj, "plan", robot=tiny)
    empty = viz_rerun.robot_blueprint(traj, "plan", robot=tiny, wrenches={}, events=(), plots={})
    assert list(tabs_of(plain)) == list(tabs_of(empty)) == ["q", "qd", "qdd", "qddd", "q / qd / qdd", "TCP"]
    assert [view.name for view in views(plain.root_container)] == [view.name for view in views(empty.root_container)]
