"""robot_msgs.viz.viser and robot_msgs.viewers: every spatial catalog message puts something in a
Viser scene, and the registry knows every viewer the site shows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("viser")

sys.path.insert(0, str(Path(__file__).parent))
from catalog import EXAMPLES  # noqa: E402

from robot_msgs import viewers  # noqa: E402
from robot_msgs.geometry import Pose  # noqa: E402
from robot_msgs.sensor import JointState  # noqa: E402
from robot_msgs.viz import viser as viz_viser  # noqa: E402

SPATIAL = [m for m in EXAMPLES if viz_viser.supports(type(m))]


def test_the_spatial_types_have_viser_views():
    names = {type(m).__name__ for m in SPATIAL}
    for expected in ("Pose", "PointCloud", "Mesh", "PoseTrajectory", "GraspPose", "WaypointAction",
                     "Detection3DArray", "CameraInfo", "Wrench"):  # fmt: skip
        assert expected in names
    assert not viz_viser.supports(JointState)  # no 3-D content: plot() and Rerun show it


def test_every_spatial_example_adds_scene_nodes():
    # WHY count nodes in the recording: an empty scene still serialises to a few hundred bytes.
    empty = len(viz_viser.scene().get_scene_serializer().serialize())
    server = viz_viser.scene(**{f"m{i}": m for i, m in enumerate(SPATIAL)})
    try:
        full = len(server.get_scene_serializer().serialize())
        node_names = set(server.scene._handle_from_node_name)
    finally:
        server.stop()
    assert full > empty
    silent = [
        f"m{i}" for i in range(len(SPATIAL)) if not any(n.startswith(f"/m{i}") for n in node_names)
    ]
    assert not silent, [type(SPATIAL[int(s[1:])]).__name__ for s in silent]


def test_save_writes_a_recording_and_html_embeds_the_client(tmp_path):
    pose = Pose.from_xyz_rpy(0.1, 0, 0.2, 0, 0, 0)
    path = Path(viz_viser.save(tmp_path / "p.viser", grasp=pose))
    assert path.stat().st_size > 100
    assert "__VISER_EMBED_DATA__" in viz_viser.html(grasp=pose)


def test_a_non_spatial_message_is_refused_with_a_pointer_to_plot():
    with pytest.raises(TypeError, match="plot"):
        viz_viser.save("unused.viser", state=JointState(name=("a",), position=[0.0]))


def test_the_registry_has_the_three_viewers_and_routes_show(monkeypatch):
    assert {"matplotlib", "rerun", "viser"} <= set(viewers.VIEWERS)
    assert viewers.get("viser").supports(Pose) and not viewers.get("viser").supports(JointState)
    assert viewers.get("rerun").supports(JointState)
    assert viewers.get("matplotlib").supports(JointState)
    opened = []
    monkeypatch.setattr(viz_viser, "show", lambda **named: opened.append(named) or "server")
    pose = Pose()
    assert pose.show("goal", viewer="viser") == "server" and opened == [{"goal": pose}]
    with pytest.raises(KeyError, match="no viewer"):
        pose.show(viewer="nope")
