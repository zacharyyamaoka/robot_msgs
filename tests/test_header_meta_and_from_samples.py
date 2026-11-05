"""Two gaps found when bam_rack_twin adopted JointTrajectory (bam_ws b32c07a8).

Written before the code.

1. `Header.meta`: an immutable str -> str mapping for context ROS has no field for (an origin,
   units, provenance). It rides the header, so it survives slicing, `.at()`, `.resample()`, every
   serialisation, and is dropped on ROS export, where std_msgs/Header has only stamp and frame_id.
2. `Trajectory.from_samples(..., duplicate_times=...)`: the constructor keeps its strictly
   increasing invariant; recorded data with repeated or backwards stamps goes through an explicit
   cleaner that says exactly what it changed, in its return value and in the header's meta.
"""

import json
import pickle

import numpy as np
import pytest

from robot_msgs import allclose
from robot_msgs._core import load_npz, save_npz
from robot_msgs.std import Header
from robot_msgs.trajectory import (
    JointTrajectory,
    PoseTrajectory,
    TimeRepair,
    Trajectory,
    concatenate,
)

META = {"origin_rad": "-0.3737", "units": "rad, rad/s, N.m"}


def traj(meta=META, n=11):
    t = np.linspace(0.0, 1.0, n)
    return JointTrajectory(
        t=t,
        joint_names=("j",),
        positions=t.reshape(-1, 1),
        channels={"kp": np.full(n, 40.0)},
        header=Header(stamp=5.0, frame_id="board", meta=meta),
    )


# ── 1. Header.meta ───────────────────────────────────────────────────────────


def test_header_meta_defaults_empty_and_is_immutable():
    assert dict(Header().meta) == {}
    header = Header(stamp=1.0, frame_id="base", meta=META)
    assert dict(header.meta) == META
    with pytest.raises(TypeError):
        header.meta["units"] = "deg"  # type: ignore[index]
    source = dict(META)
    header = Header(meta=source)
    source["units"] = "deg"
    assert header.meta["units"] == "rad, rad/s, N.m"  # copied at construction


def test_header_meta_is_strings_only_and_never_coerced():
    with pytest.raises(ValueError, match="meta"):
        Header(meta={"origin_rad": -0.3737})  # a float would be silently str()-ed; refuse instead
    with pytest.raises(ValueError, match="meta"):
        Header(meta={1: "x"})


def test_header_meta_takes_part_in_equality_hash_and_repr():
    assert Header(meta=META) == Header(meta=dict(reversed(list(META.items()))))
    assert Header(meta=META) != Header()
    assert hash(Header(meta=META)) == hash(Header(meta=dict(META)))
    assert "origin_rad" in repr(Header(meta=META))
    assert "meta" not in repr(Header(frame_id="x"))
    assert Header(meta=META).time_shifted(1.0).meta == META


def test_meta_survives_slicing_at_resample_and_every_view():
    message = traj()
    assert message[2:5].header.meta == META
    assert message[[0, 3]].header.meta == META
    assert message.window(0.2, 0.6).header.meta == META
    assert message.resample(dt=0.25).header.meta == META
    assert message.at(0.123).header.meta == META  # a point carries its trajectory's header
    assert message[3].header == message.header
    assert all(point.header.meta == META for point in message)
    assert message.rebased().header.meta == META
    assert message.joint("j").header.meta == META
    assert message.with_derivatives().header.meta == META


def test_points_rebuild_the_trajectory_with_its_header():
    message = traj()
    again = JointTrajectory.from_points(message.to_points(), joint_names=("j",))
    assert again == message
    other = traj(meta={"origin_rad": "0.0"})
    with pytest.raises(ValueError, match="header"):
        JointTrajectory.from_points([message[0], other[1]], joint_names=("j",))


def test_meta_survives_dict_json_npz_and_pickle(tmp_path):
    message = traj()
    assert JointTrajectory.from_dict(json.loads(json.dumps(message.to_dict()))) == message
    save_npz(message, tmp_path / "m.npz")
    assert load_npz(tmp_path / "m.npz").header.meta == META
    assert pickle.loads(pickle.dumps(message)).header.meta == META


def test_concatenate_refuses_pieces_with_different_meta():
    with pytest.raises(ValueError, match="meta"):
        concatenate([traj(), traj(meta={"origin_rad": "0.1"}).time_shifted(2.0)])
    assert concatenate([traj(), traj().time_shifted(2.0)]).header.meta == META


def test_ros_export_drops_meta_and_keeps_everything_else():
    pytest.importorskip("rosbags")
    from robot_msgs.ros import from_ros, to_ros

    message = traj()
    ros = to_ros(message, drop_extras=True)
    assert not hasattr(ros.header, "meta")
    back = from_ros(ros)
    assert dict(back.header.meta) == {}
    assert back.header.frame_id == "board" and back.header.stamp == 5.0
    assert allclose(back, message.replace(channels={}, header=Header(stamp=5.0, frame_id="board")))


# ── 2. from_samples ─────────────────────────────────────────────────────────

STAMPS = [0.0, 0.1, 0.1, 0.2, 0.15, 0.3]  # one repeat (index 2), one backwards stamp (index 4)
VALUES = [0.0, 1.0, 3.0, 2.0, 9.0, 4.0]


def recorded(policy, **extra):
    return JointTrajectory.from_samples(
        t=STAMPS,
        joint_names=("j",),
        positions=np.reshape(VALUES, (-1, 1)),
        channels={"mode": np.array([1, 1, 1, 2, 2, 3]), "wall": np.arange(6.0)},
        header=Header(stamp=7.0, frame_id="board", meta=META),
        duplicate_times=policy,
        **extra,
    )


def test_the_constructor_still_refuses_repeated_stamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        JointTrajectory(t=STAMPS, joint_names=("j",), positions=np.reshape(VALUES, (-1, 1)))


def test_duplicate_times_is_required_and_checked():
    with pytest.raises(TypeError):
        JointTrajectory.from_samples(t=[0.0, 1.0], joint_names=("j",), positions=[[0.0], [1.0]])
    with pytest.raises(ValueError, match="duplicate_times"):
        recorded("keep_first")


def test_error_policy_raises_on_bad_stamps_and_reports_no_change_on_clean_ones():
    with pytest.raises(ValueError, match=r"t\[2\]"):
        recorded("error")
    clean, repair = JointTrajectory.from_samples(
        t=[0.0, 1.0], joint_names=("j",), positions=[[0.0], [1.0]], duplicate_times="error"
    )
    assert isinstance(repair, TimeRepair) and not repair.changed
    assert repair.dropped == () and repair.averaged == ()
    assert dict(clean.header.meta) == {}  # nothing changed, nothing stamped


def test_clean_samples_come_back_unchanged_under_every_policy():
    t = np.linspace(0.0, 1.0, 5)
    plain = JointTrajectory(t=t, joint_names=("j",), positions=t.reshape(-1, 1))
    for policy in ("error", "drop_later", "average"):
        message, repair = JointTrajectory.from_samples(
            t=t, joint_names=("j",), positions=t.reshape(-1, 1), duplicate_times=policy
        )
        assert message == plain and not repair.changed and repair.samples_in == 5


def test_drop_later_keeps_the_first_of_every_repeat_and_drops_backwards_stamps():
    message, repair = recorded("drop_later")
    assert message.t.tolist() == [0.0, 0.1, 0.2, 0.3]
    # the original values, untouched
    assert message.positions[:, 0].tolist() == [0.0, 1.0, 2.0, 4.0]
    assert message.channels["wall"].tolist() == [0.0, 1.0, 3.0, 5.0]
    assert message.channels["mode"].tolist() == [1, 1, 2, 3]
    assert repair.changed and repair.policy == "drop_later"
    assert repair.samples_in == 6 and repair.samples_out == 4
    assert repair.dropped == (2, 4)  # indices into the samples as given
    assert repair.dropped_t == (0.1, 0.15)
    assert repair.averaged == ()
    assert "2 of 6" in str(repair)


def test_the_repair_is_written_into_the_header_meta_beside_the_callers():
    message, repair = recorded("drop_later")
    meta = dict(message.header.meta)
    assert {k: meta[k] for k in META} == META  # the caller's entries are kept
    assert meta == {**META, **repair.as_meta()}
    assert meta["time_repair.policy"] == "drop_later"
    assert json.loads(meta["time_repair.dropped"]) == [2, 4]
    assert json.loads(meta["time_repair.dropped_t"]) == [0.1, 0.15]
    assert meta["time_repair.samples_in"] == "6"
    assert message.header.stamp == 7.0 and message.header.frame_id == "board"


def test_average_merges_equal_stamps_and_still_drops_backwards_ones():
    message, repair = JointTrajectory.from_samples(
        t=STAMPS,
        joint_names=("j",),
        positions=np.reshape(VALUES, (-1, 1)),
        channels={"wall": np.arange(6.0)},
        duplicate_times="average",
    )
    assert message.t.tolist() == [0.0, 0.1, 0.2, 0.3]
    assert message.positions[:, 0].tolist() == [0.0, 2.0, 2.0, 4.0]  # (1 + 3) / 2 at t = 0.1
    assert message.channels["wall"].tolist() == [0.0, 1.5, 3.0, 5.0]
    assert repair.averaged == ((1, 2),)
    assert repair.dropped == (4,)  # a backwards stamp has no partner to average with
    assert json.loads(message.header.meta["time_repair.averaged"]) == [[1, 2]]


def test_average_refuses_to_invent_a_value_for_a_held_field():
    with pytest.raises(ValueError, match="mode"):
        JointTrajectory.from_samples(
            t=[0.0, 1.0, 1.0],
            joint_names=("j",),
            positions=[[0.0], [1.0], [1.0]],
            channels={"mode": np.array([1, 2, 3])},
            duplicate_times="average",
        )
    message, _ = JointTrajectory.from_samples(  # equal held values are fine
        t=[0.0, 1.0, 1.0],
        joint_names=("j",),
        positions=[[0.0], [1.0], [3.0]],
        channels={"mode": np.array([1, 2, 2])},
        duplicate_times="average",
    )
    assert message.channels["mode"].tolist() == [1, 2]


def test_average_of_orientations_is_a_unit_quaternion_on_the_short_arc():
    q = np.array([0.0, 0.0, np.sin(0.1), np.cos(0.1)])
    message, repair = PoseTrajectory.from_samples(
        t=[0.0, 1.0, 1.0],
        positions=np.zeros((3, 3)),
        orientations=np.stack([[0, 0, 0, 1.0], q, -q]),  # q and -q are the same rotation
        duplicate_times="average",
    )
    assert repair.averaged == ((1, 2),)
    assert np.allclose(np.abs(message.orientations[1]), np.abs(q))


def test_from_samples_works_on_any_trajectory_and_checks_lengths():
    series, repair = Trajectory.from_samples(
        t=[0.0, 0.0, 1.0], channels={"temp": [20.0, 21.0, 22.0]}, duplicate_times="drop_later"
    )
    assert series.t.tolist() == [0.0, 1.0] and repair.dropped == (1,)
    with pytest.raises(ValueError, match="positions"):
        JointTrajectory.from_samples(
            t=[0.0, 0.0, 1.0], joint_names=("j",), positions=[[0.0]], duplicate_times="drop_later"
        )
