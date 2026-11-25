"""robot_msgs.viz.rerun without ``robot=``: the recording is frozen against a golden digest.

WHY (2026-11-24): ``robot=`` grows the JointTrajectory viewer into the URDF dashboard, and every
recording made WITHOUT it (traj_compare's exports, sim2real's result.show, the Trajectory Tracking
Agent's files) must stay exactly what it was. The digest is what Rerun's own reader reads back from
the .rrd: every chunk's entity path, static flag and component values, the blueprint included.
Row ids, log times and the blueprint's random view / container ids are the only things left out
(they differ between two saves of the same code).

    /home/bam/robot_msgs/.venv/bin/python tests/test_viz_rerun_stock_golden.py --write   # re-freeze
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")

from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory  # noqa: E402
from robot_msgs.viz import rerun as viz_rerun  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "rerun_stock_without_robot.json"
# a 16-byte UUID as Arrow lists it, or as hex: the view / container / recording ids
_IDS = re.compile(
    r"\[(?:\d{1,3}, ){15}\d{1,3}\]|[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_SKIP = ("rerun.controls.RowId", "log_time", "log_tick")


def digest(path: Path) -> dict[str, list]:
    """{store kind: sorted [entity, static, [(column, values json)]]} as Rerun's RrdReader reads the file."""
    from rerun.chunk import RrdReader

    reader = RrdReader(path)
    out: dict[str, list] = {}
    for kind, stores in (("recording", reader.recordings()), ("blueprint", reader.blueprints())):
        rows = [["application_id", store.application_id] for store in stores]
        for store in stores:
            for chunk in reader.stream(store=store):
                if chunk.entity_path.startswith("/__"):
                    continue
                batch = chunk.to_record_batch()
                columns = [
                    (name, _IDS.sub("<id>", json.dumps(batch.column(name).to_pylist(), default=str)))
                    for name in batch.schema.names
                    if not any(name.startswith(skip) for skip in _SKIP)
                ]
                rows.append([_IDS.sub("<id>", chunk.entity_path), chunk.is_static, sorted(columns)])
        out[kind] = sorted(rows, key=json.dumps)
    return out


def cases(tmp: Path) -> dict[str, Path]:
    """The stock recordings frozen here: an overlay with a named "tool" message, and one positional."""
    t = np.linspace(0.0, 1.0, 6)
    names = ("shoulder", "elbow")
    traj = JointTrajectory(
        t=t,
        joint_names=names,
        positions=np.stack([np.sin(t), np.cos(t)], 1),
        limits=JointLimits(joint_names=names, min_position=[-1.0, -2.0], max_position=[1.0, 2.0],
                           max_velocity=[3.0, 3.0], max_acceleration=[9.0, 9.0]),
        channels={"phase": np.arange(6.0)},
    ).with_derivatives()
    pose = PoseTrajectory(t=t, positions=np.stack([t, 0 * t, 0 * t + 1.0], 1),
                          orientations=np.tile([0.0, 0.0, 0.0, 1.0], (6, 1)))
    # WHY a message named "tool": `tool=` is a named message unless `robot=` is given (README: show(..., tool=path))
    return {
        "named_overlay_with_tool": Path(viz_rerun.save(str(tmp / "named.rrd"), real=traj, tool=pose)),
        "positional": Path(viz_rerun.save(str(tmp / "positional.rrd"), traj)),
    }


def test_without_robot_the_recording_is_unchanged(tmp_path):
    expected = json.loads(GOLDEN.read_text())
    for case, path in cases(tmp_path).items():
        found = digest(path)
        for kind in ("recording", "blueprint"):
            want = [json.dumps(row) for row in expected[case][kind]]
            got = [json.dumps(row) for row in found[kind]]
            missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
            assert not missing and not extra, (
                f"{case} {kind} changed:\n missing {[m[:160] for m in missing][:5]}\n extra {[e[:160] for e in extra][:5]}"
            )
            assert want == got, f"{case} {kind}: the same chunks, a different count or order"


if __name__ == "__main__" and "--write" in sys.argv:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        frozen = {case: digest(path) for case, path in cases(Path(tmp)).items()}
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(frozen, indent=0, sort_keys=True) + "\n")
    print(GOLDEN, GOLDEN.stat().st_size, "bytes")
