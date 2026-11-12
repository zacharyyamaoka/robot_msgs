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


def test_a_comparison_logs_its_error_beside_the_trajectory_it_compares(tmp_path):
    # WHY (2026-11-11, sim2real's result.show): a run's recording holds target / sim / real AND each one's Comparison under the same
    # name; the error must sit on its own `error/` field (its own tab), never on the angles' plot under the angles' path
    from robot_msgs.compare import compare

    target, real = joint_trajectory(), joint_trajectory()
    path = tmp_path / "c.rrd"
    stream = viz_rerun.compose(("target", target), ("real", real), ("real", compare(target, real)), into=str(path))
    stream.flush()
    stream.disconnect()
    found = chunks(path)
    assert {"/positions/elbow/target", "/positions/elbow/real", "/error/positions/elbow/real"} <= set(found)
    assert "/notes/real" in found


# ── degrees, bands, video, links (2026-11-11, for traj_compare's Rerun layouts) ──


def _rrd_scalars(path: Path, entity: str) -> np.ndarray:
    """Scalars at one entity, read back with Rerun's own RrdReader."""
    from rerun.chunk import RrdReader

    values: list[float] = []
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == entity:
            batch = chunk.to_record_batch()
            if "Scalars:scalars" in batch.schema.names:
                column = batch.column(batch.schema.names.index("Scalars:scalars"))
                values += [row[0] for row in column.to_pylist() if row]
    return np.asarray(values)


def _stream(path: Path):
    import rerun as rr

    stream = rr.RecordingStream("robot_msgs_test", recording_id="t")
    stream.save(str(path))
    return stream


def _close(stream) -> None:
    stream.flush()
    stream.disconnect()


def test_degrees_shows_joint_angles_in_degrees_and_keeps_the_message_si(tmp_path):
    traj = joint_trajectory()
    path = tmp_path / "deg.rrd"
    stream = _stream(path)
    viz_rerun.log(traj, "real", recording=stream, degrees=True)
    viz_rerun.log(traj, "only_elbow", recording=stream, degrees=["elbow"])
    _close(stream)
    np.testing.assert_allclose(_rrd_scalars(path, "/positions/shoulder/real"), np.degrees(traj.positions[:, 0]))
    np.testing.assert_allclose(_rrd_scalars(path, "/positions/shoulder/only_elbow"), traj.positions[:, 0])  # left SI
    np.testing.assert_allclose(_rrd_scalars(path, "/positions/elbow/only_elbow"), np.degrees(traj.positions[:, 1]))
    assert np.allclose(traj.positions[:, 0], np.sin(traj.t))  # the message itself is untouched
    with pytest.raises(ValueError, match="wrist"):
        viz_rerun.log(traj, "x", recording=stream, degrees=["wrist"])


def test_prefix_color_and_a_shared_t0(tmp_path):
    traj = joint_trajectory()
    later = JointTrajectory(t=traj.t + 2.0, joint_names=traj.joint_names, positions=traj.positions)
    path = tmp_path / "p.rrd"
    stream = _stream(path)
    viz_rerun.log(later, "real", recording=stream, prefix="error/", color=[235, 104, 52], t0=0.0)
    _close(stream)
    found = chunks(path)
    assert "/error/positions/shoulder/real" in found
    assert "SeriesLines:colors" in found["/error/positions/shoulder/real"][1]
    # t0=0: the trajectory that starts at 2 s stays at 2 s on the timeline (not shifted to 0)
    from rerun.chunk import RrdReader

    times = []
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == "/error/positions/shoulder/real" and "t" in chunk.timeline_names:
            column = chunk.to_record_batch().column("t")
            times += [nanos / 1e9 for nanos in column.cast("int64").to_pylist()]  # a duration timeline, in ns
    assert times and min(times) == pytest.approx(2.0)


def test_band_is_a_shaded_measurement_between_low_and_high(tmp_path):
    from rerun.chunk import RrdReader

    path = tmp_path / "b.rrd"
    stream = _stream(path)
    t = np.linspace(0.0, 1.0, 11)
    viz_rerun.band("error/positions/shoulder/tolerance", t, -0.5, np.full_like(t, 0.5), recording=stream)
    _close(stream)
    found = chunks(path)
    assert "Measurements:variances" in found["/error/positions/shoulder/tolerance"][1]
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == "/error/positions/shoulder/tolerance":
            batch = chunk.to_record_batch()
            names = batch.schema.names
            if "Measurements:variances" in names:
                variances = [row[0] for row in batch.column(names.index("Measurements:variances")).to_pylist()]
                values = [row[0] for row in batch.column(names.index("Measurements:values")).to_pylist()]
                np.testing.assert_allclose(values, 0.0)
                np.testing.assert_allclose(np.sqrt(variances), 0.5)  # ± one σ = the half-width


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_lands_on_the_t_timeline_with_its_start_offset(tmp_path):
    clip = tmp_path / "cam.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:size=64x48:rate=10",
                    "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", str(clip)], check=True)
    path = tmp_path / "v.rrd"
    stream = _stream(path)
    viz_rerun.video(clip, "front", recording=stream, start_s=0.5)
    _close(stream)
    found = chunks(path)
    assert "/video/front" in found
    assert "VideoFrameReference" in found["/video/front"][1] and "AssetVideo" in found["/video/front"][1]


def test_links_put_each_mesh_on_its_own_moving_frame(tmp_path):
    from robot_msgs.shape import Mesh

    t = np.linspace(0.0, 1.0, 5)
    poses = PoseTrajectory(t=t, positions=np.column_stack([t, 0 * t, 0 * t]),
                           orientations=np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)))
    box = Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), triangles=np.array([[0, 1, 2]]))
    path = tmp_path / "l.rrd"
    stream = _stream(path)
    viz_rerun.links([("base", box, poses), ("arm", box, poses)], "real", recording=stream, color=[200, 80, 40, 255])
    _close(stream)
    found = chunks(path)
    for link in ("base", "arm"):
        entity = f"/world/real/robot/{link}"
        assert entity in found
        assert "Mesh3D:vertex_positions" in found[entity][1] and "Transform3D:translation" in found[entity][1]
