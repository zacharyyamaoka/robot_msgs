"""robot_msgs.compare's dataset nouns: Recording, Run, Dataset → RunComparison, DatasetComparison.

The pairing rule is traj_compare's `default_pairs` (the reference against each source, then every
source against each later one); every number below is a constant offset built into the fixture.
"""

import json

import numpy as np
import pytest

from robot_msgs import load_npz, save_npz
from robot_msgs.compare import (
    Comparison,
    Dataset,
    DatasetComparison,
    Recording,
    Run,
    RunComparison,
    compare_dataset,
    compare_run,
    comparison_key,
    default_pairs,
    split_key,
)
from robot_msgs.geometry import Transform
from robot_msgs.sensor import PointCloud
from robot_msgs.std import Header
from robot_msgs.trajectory import ImuTrajectory, JointTrajectory, WrenchTrajectory

T = np.linspace(0.0, 1.0, 51)
TARGET = JointTrajectory(t=T, joint_names=("j1", "j2"), positions=np.column_stack([T, -T]))
FT = WrenchTrajectory(t=T, forces=np.zeros((51, 3)), torques=np.zeros((51, 3)))


def joints(offset: float) -> JointTrajectory:
    return TARGET.replace(positions=TARGET.positions + offset)


def run(name: str = "r1", sim: float = 0.01, real: float = 0.03, **kwargs) -> Run:
    return Run(
        name=name,
        recordings={
            "target": Recording(streams={"joints": TARGET}),
            "sim": Recording(streams={"joints": joints(sim), "ft": FT}),
            "real": Recording(streams={"joints": joints(real), "ft": FT}),
        },
        **kwargs,
    )


def test_default_pairs_are_the_reference_to_each_then_each_to_every_later_one():
    assert default_pairs(("target", "sim", "real", "real2")) == (
        ("target", "sim"), ("target", "real"), ("target", "real2"),
        ("sim", "real"), ("sim", "real2"), ("real", "real2"),
    )  # fmt: skip
    assert run().resolved_pairs() == (("target", "sim"), ("target", "real"), ("sim", "real"))
    assert run().reference == "target"
    assert run(pairs=[("real", "sim")]).resolved_pairs() == (("real", "sim"),)


def test_compare_run_compares_every_shared_stream_and_records_every_skip():
    result = compare_run(run())
    assert list(result.comparisons) == [
        "target→sim/joints", "target→real/joints", "sim→real/joints", "sim→real/ft",
    ]  # fmt: skip
    assert result.get("target", "real", "joints").metrics["positions/j1"].mean == pytest.approx(
        -0.03
    )
    assert result.get("sim", "real", "joints").metrics["positions/j2"].rms == pytest.approx(0.02)
    assert dict(result.skipped) == {
        "target→sim/ft": "only 'sim' recorded 'ft'",
        "target→real/ft": "only 'real' recorded 'ft'",
    }
    with pytest.raises(KeyError, match="skipped: only 'sim'"):
        result.get("target", "sim", "ft")


def test_a_pair_is_compared_with_the_difference_of_its_offsets():
    late = run(real=0.0, time_offset_s={"sim": 0.1, "real": 0.3})
    assert late.offset("sim", "real") == pytest.approx(0.2)
    assert late.offset("target", "sim") == pytest.approx(0.1)
    result = compare_run(late)
    assert result.get("sim", "real", "joints").time_offset_s == pytest.approx(0.2)
    assert result.get("target", "real", "joints").time_offset_s == pytest.approx(0.3)


def test_streams_of_different_types_or_that_compare_refuses_are_skipped_with_why():
    imu = ImuTrajectory(t=T, orientations=np.tile([0, 0, 0, 1.0], (51, 1)),
                        angular_velocities=np.zeros((51, 3)), linear_accelerations=np.zeros((51, 3)))  # fmt: skip
    odd = Run(
        name="odd",
        recordings={
            "sim": Recording(streams={"a": FT, "b": TARGET}),
            "real": Recording(streams={"a": imu, "b": TARGET.select(["j1"])}),
        },
    )
    result = compare_run(odd)
    assert result.comparisons == {}
    assert result.skipped["sim→real/a"] == "different types: WrenchTrajectory vs ImuTrajectory"
    assert result.skipped["sim→real/b"].startswith("ValueError: joints differ")
    with pytest.raises(ValueError, match="joints differ"):
        compare_run(odd, errors="raise")


def test_rows_carry_the_run_pair_stream_metric_stats_and_tags():
    result = compare_run(run(tags={"speed": 0.5, "rig": "fb_v2"}))
    row = result.rows()[0]
    assert {key: row[key] for key in ("run", "a", "b", "stream", "metric")} == {
        "run": "r1", "a": "target", "b": "sim", "stream": "joints", "metric": "positions/j1",
    }  # fmt: skip
    assert row["rms"] == pytest.approx(0.01) and row["speed"] == 0.5 and row["rig"] == "fb_v2"


def test_a_dataset_rolls_up_and_worst_names_where_to_drill_in():
    dataset = Dataset(
        runs=(run("slow", real=0.03, tags={"speed": 0.5}), run("fast", real=0.08, tags={"speed": 1})),
        title="speed sweep",
    )
    result = compare_dataset(dataset)
    assert isinstance(result, DatasetComparison) and list(result.runs) == ["slow", "fast"]
    assert len(result.rows()) == 2 * len(compare_run(dataset.runs[0]).rows())
    assert result.worst("positions/j1") == ("fast", "target", "real", "joints", pytest.approx(0.08))
    assert result.worst("positions/j1", "mean")[4] == pytest.approx(-0.08)  # signed, by |value|
    assert {row["stream"] for row in result.skipped()} == {"ft"}
    with pytest.raises(KeyError):
        result.worst("no_such_metric")
    with pytest.raises(ValueError, match="ErrorStats field"):
        result.worst("positions/j1", "loudest")


def test_dataset_and_its_comparison_round_trip_through_json_and_npz(tmp_path):
    cloud = PointCloud(points=np.random.default_rng(1).normal(size=(40, 3)))
    dataset = Dataset(
        runs=(
            run(tags={"speed": 0.5, "payload_kg": 1, "ok": True, "rig": "fb"}),
            Run(
                name="scan",
                recordings={
                    "sim": Recording(streams={"cloud": cloud}, header=Header(meta={"engine": "mj"})),
                    "real": Recording(streams={"cloud": cloud.transformed(Transform())}),
                },
            ),
        ),
        title="mixed",
    )
    result = compare_dataset(dataset)
    for message in (dataset, result):
        decoded = type(message).from_dict(json.loads(json.dumps(message.to_dict())))
        assert decoded == message
        save_npz(message, tmp_path / "m.npz")
        assert load_npz(tmp_path / "m.npz") == message
    decoded = Dataset.from_dict(json.loads(json.dumps(dataset.to_dict())))
    assert list(decoded.runs[0].recordings) == ["target", "sim", "real"]  # order is the reference
    tags = decoded.runs[0].tags
    assert (type(tags["speed"]), type(tags["payload_kg"]), type(tags["ok"])) == (float, int, bool)
    assert isinstance(result.runs["scan"].get("sim", "real", "cloud").error, PointCloud)


def test_keys_split_back_and_run_refuses_names_that_would_not():
    assert split_key(comparison_key("sim", "real", "camera/color")) == ("sim", "real", "camera/color")
    for bad in ("sim/1", "a→b"):
        with pytest.raises(ValueError, match="contains"):
            Run(name="r", recordings={bad: Recording()})
    with pytest.raises(ValueError, match="names no recording"):
        run(pairs=[("target", "gazebo")])
    with pytest.raises(ValueError, match="with itself"):
        run(pairs=[("sim", "sim")])
    with pytest.raises(ValueError, match="time_offset_s names no recording"):
        run(time_offset_s={"gazebo": 0.1})
    with pytest.raises(TypeError, match="str, int, float or bool"):
        run(tags={"speed": [0.5]})
    with pytest.raises(ValueError, match="shadow"):
        run(tags={"rms": 1.0})
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(runs=(run(), run()))
    with pytest.raises(TypeError, match="robot_msgs message"):
        Recording(streams={"x": np.zeros(3)})
    with pytest.raises(TypeError, match="Comparison"):
        RunComparison(run="r", comparisons={"a→b/s": "not a comparison"})
    assert isinstance(compare_run(run()).comparisons["target→sim/joints"], Comparison)
