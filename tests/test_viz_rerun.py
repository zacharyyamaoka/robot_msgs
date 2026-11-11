"""robot_msgs.viz.rerun: every catalog message logs into Rerun, checked by Rerun's own reader.

The oracle is the ``rerun rrd print`` CLI that ships with rerun-sdk: it reads the .rrd we wrote
and lists each chunk's entity path and components, so these tests never trust our own logger.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")
pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).parent))
from catalog import EXAMPLES  # noqa: E402

from robot_msgs.geometry import Pose  # noqa: E402
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory  # noqa: E402
from robot_msgs.viz import rerun as viz_rerun  # noqa: E402

CHUNK = re.compile(r"with (\d+) rows .*? - (/\S+) - data columns: \[(.*)\]")


def rerun_cli() -> list[str]:
    beside_python = Path(sys.executable).with_name("rerun")
    if beside_python.exists():
        return [str(beside_python)]
    found = shutil.which("rerun")
    if not found:
        pytest.skip("the rerun CLI is not installed")
    return [found]


def chunks(path: Path) -> dict[str, tuple[int, str]]:
    """{entity path: (total rows, components)} as Rerun itself reads the file."""
    out = subprocess.run(
        [*rerun_cli(), "rrd", "print", str(path)], capture_output=True, text=True, check=True
    )
    found: dict[str, tuple[int, str]] = {}
    for rows, entity, columns in CHUNK.findall(out.stdout):
        if entity.startswith("/__"):
            continue
        before = found.get(entity, (0, ""))
        found[entity] = (before[0] + int(rows), f"{before[1]} {columns}".strip())
    return found


def joint_trajectory(n: int = 50) -> JointTrajectory:
    t = np.linspace(0.0, 1.0, n)
    return JointTrajectory(
        t=t, joint_names=("shoulder", "elbow"), positions=np.stack([np.sin(t), np.cos(t)], 1)
    )


def test_every_catalog_message_logs_something_rerun_can_read(tmp_path):
    # WHY one recording: one `rrd print` for the whole catalog (one per type took ~3 minutes).
    named = {
        f"m{index}_{type(example).__name__}": example for index, example in enumerate(EXAMPLES)
    }
    found = chunks(Path(viz_rerun.save(str(tmp_path / "catalog.rrd"), **named)))
    silent = [
        name
        for name in named
        if not any(entity.split("/")[-1] == name or f"/{name}/" in entity + "/" for entity in found)
    ]
    assert not silent, f"these logged nothing Rerun can read: {silent}"


def test_a_joint_trajectory_is_one_series_per_joint_on_the_t_timeline(tmp_path):
    traj = joint_trajectory(50)
    found = chunks(Path(viz_rerun.save(str(tmp_path / "j.rrd"), real=traj)))
    for joint in ("shoulder", "elbow"):
        rows, columns = found[f"/positions/{joint}/real"]
        assert "Scalars:scalars" in columns
        assert rows >= 50  # every sample, plus the static series name


def test_two_names_overlay_on_the_same_joint(tmp_path):
    target, real = joint_trajectory(), joint_trajectory()
    found = chunks(Path(viz_rerun.save(str(tmp_path / "o.rrd"), target=target, real=real)))
    assert {"/positions/elbow/target", "/positions/elbow/real"} <= set(found)


def test_a_pose_trajectory_has_a_path_and_a_moving_frame(tmp_path):
    t = np.linspace(0, 1, 20)
    positions = np.stack([t, np.zeros_like(t), np.ones_like(t)], 1)
    traj = PoseTrajectory(
        t=t, positions=positions, orientations=np.tile([0.0, 0.0, 0.0, 1.0], (20, 1))
    )
    found = chunks(Path(viz_rerun.save(str(tmp_path / "p.rrd"), tool=traj)))
    assert "LineStrips3D:strips" in found["/world/tool/path"][1]
    rows, columns = found["/world/tool/frame"]
    assert "Transform3D:translation" in columns and rows >= 20


def test_a_pose_is_a_frame_in_the_world(tmp_path):
    found = chunks(
        Path(viz_rerun.save(str(tmp_path / "g.rrd"), grasp=Pose.from_xyz_rpy(0.5, 0, 0.2, 0, 0, 0)))
    )
    assert "Transform3D:translation" in found["/world/grasp"][1]


def test_show_is_a_method_on_every_message(monkeypatch):
    opened = []
    monkeypatch.setattr(viz_rerun, "show", lambda **named: opened.append(named) or "stream")
    traj = joint_trajectory()
    assert traj.show() == "stream"
    assert opened == [{"JointTrajectory": traj}]
    traj.show("real")
    assert list(opened[-1]) == ["real"]


def test_plot_is_a_method_on_every_message():
    from matplotlib.figure import Figure

    assert isinstance(joint_trajectory().plot(), Figure)


def test_logging_a_non_message_is_refused():
    with pytest.raises(TypeError):
        viz_rerun.recording(object())  # type: ignore[arg-type]


def views(node) -> list:
    """Every view under a blueprint container, depth first."""
    contents = getattr(node, "contents", None)
    if isinstance(contents, (list, tuple)):
        return [view for child in contents for view in views(child)]
    return [node]


def test_the_default_dashboard_overlays_names_in_one_plot_per_column():
    paths = [
        "positions/elbow/target",
        "positions/elbow/real",
        "positions/shoulder/real",
        "world/tool/path",
    ]
    found = views(viz_rerun._blueprint(paths).root_container)
    kinds = sorted((type(view).__name__, str(view.origin)) for view in found)
    assert kinds == [
        ("Spatial3DView", "/world"),
        ("TimeSeriesView", "/positions/elbow"),  # target and real share it: the overlay
        ("TimeSeriesView", "/positions/shoulder"),
    ]
