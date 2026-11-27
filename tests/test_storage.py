"""Files: `msg.save` / `Type.load`, `robot_msgs.save` / `load` bundles, and the new state types."""

from __future__ import annotations

import json

import numpy as np
import pytest
from catalog import EXAMPLES

import robot_msgs
from robot_msgs import Msg, load_npz
from robot_msgs.behavior import BTState
from robot_msgs.control import ServoState
from robot_msgs.robot import RobotConfig, RobotState
from robot_msgs.sensor import CameraSettings, JointState

NEW_TYPES = (BTState, RobotState, RobotConfig)
NEW = [example for example in EXAMPLES if type(example) in NEW_TYPES]


def bt_state(**changes) -> BTState:
    base = dict(
        tree_id="Bringup",
        node_status={"Bringup/Sequence": "RUNNING", "Bringup/Sequence/Home": "SUCCESS"},
        blackboard={"homed": True, "bins": ["paper"], "speed": {"max": 0.5}},
        tick=3,
        clock_s=1.5,
    )
    return BTState(**{**base, **changes})


def joints() -> JointState:
    return JointState(
        name=("j1", "j2", "j3"), position=[0.1, -1.2, 2.0], velocity=[0.0, np.nan, 0.25]
    )


def robot_state() -> RobotState:
    return RobotState(components={"bt": bt_state(), "joints": joints()})


# ── every new type through both encodings ────────────────────────────────────


@pytest.mark.parametrize("suffix", [".json", ".npz", ""])
@pytest.mark.parametrize("message", NEW, ids=[type(m).__name__ for m in NEW])
def test_new_types_round_trip_through_every_encoding(message, suffix, tmp_path):
    path = tmp_path / f"m{suffix}"
    written = message.save(path)
    assert written.suffix == (suffix or ".npz")
    assert type(message).load(path) == message
    assert Msg.load(path) == message


def test_every_catalog_message_round_trips_through_json_and_npz(tmp_path):
    for index, message in enumerate(EXAMPLES):
        for suffix in (".json", ".npz"):
            assert Msg.load(message.save(tmp_path / f"{index}{suffix}")) == message, message


def test_no_suffix_appends_npz_like_numpy_and_load_finds_it(tmp_path):
    state = robot_state()
    assert state.save(tmp_path / "arm") == tmp_path / "arm.npz"
    assert RobotState.load(tmp_path / "arm") == state
    assert state.save(tmp_path / "arm.v2") == tmp_path / "arm.v2.npz"  # unknown suffix: appended
    assert RobotState.load(str(tmp_path / "arm.v2")) == state


def test_json_is_human_readable_with_arrays_inline(tmp_path):
    path = robot_state().save(tmp_path / "arm.json")
    tree = json.loads(path.read_text())
    assert tree["_type"] == "robot/RobotState"
    assert tree["components"]["bt"]["blackboard"]["homed"] is True
    assert tree["components"]["joints"]["position"]["data"] == [0.1, -1.2, 2.0]
    assert path.read_text().count("\n") > 20  # indented, one value per line


def test_npz_is_save_npz_format_without_pickle(tmp_path):
    state = robot_state()
    path = state.save(tmp_path / "arm.npz")
    assert load_npz(path) == state
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)


def test_an_array_bearing_joint_state_inside_robot_state_keeps_its_arrays(tmp_path):
    state = robot_state()
    for suffix in (".json", ".npz"):
        loaded = RobotState.load(state.save(tmp_path / f"arm{suffix}"))
        position = loaded["joints"].position
        assert position.dtype == np.float64 and not position.flags.writeable
        np.testing.assert_array_equal(loaded["joints"].velocity, [0.0, np.nan, 0.25])


# ── type checks ──────────────────────────────────────────────────────────────


def test_loading_a_file_as_the_wrong_type_fails_clearly(tmp_path):
    path = bt_state().save(tmp_path / "bt.json")
    with pytest.raises(TypeError, match=r"holds a behavior/BTState, not a sensor/JointState"):
        JointState.load(path)
    with pytest.raises(TypeError, match="holds a bundle"):
        RobotState.load(robot_msgs.save([robot_state()], tmp_path / "list.npz"))
    with pytest.raises(FileNotFoundError, match="no saved message"):
        BTState.load(tmp_path / "missing")


# ── bundles ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_bundles_of_mixed_types_round_trip_with_their_container(suffix, tmp_path):
    servo = ServoState(name=("j1",), position=[0.5], fault=[0], seq=7)
    step = {"state": robot_state(), "action": servo, "history": [joints(), joints()],
            "pair": (bt_state(), CameraSettings(gain=3.0))}  # fmt: skip
    loaded = robot_msgs.load(robot_msgs.save(step, tmp_path / f"step{suffix}"))
    assert type(loaded) is dict and list(loaded) == list(step)
    assert type(loaded["history"]) is list and type(loaded["pair"]) is tuple
    assert loaded["state"] == step["state"] and loaded["action"] == servo
    assert loaded["history"] == step["history"] and loaded["pair"] == step["pair"]


def test_a_batch_of_same_type_messages_is_a_list(tmp_path):
    history = [joints().replace(position=[float(i), 0.0, 0.0]) for i in range(50)]
    loaded = robot_msgs.load(robot_msgs.save(history, tmp_path / "history"))
    assert loaded == history
    assert robot_msgs.load(robot_msgs.save(robot_state(), tmp_path / "one")) == robot_state()


def test_bundles_refuse_what_is_not_a_message(tmp_path):
    with pytest.raises(TypeError, match="got int"):
        robot_msgs.save([robot_state(), 3], tmp_path / "bad.npz")
    with pytest.raises(TypeError, match="keys are strings"):
        robot_msgs.save({1: robot_state()}, tmp_path / "bad.npz")
    assert not list(tmp_path.iterdir())  # nothing written


# ── atomic writes ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(suffix, tmp_path, monkeypatch):
    path = tmp_path / f"arm{suffix}"
    old = robot_state()
    old.save(path)
    before = path.read_bytes()

    import robot_msgs._storage as storage

    real_fsync = storage.os.fsync

    def power_cut(fd):  # the bytes are in the temporary file, the rename never happens
        raise OSError("simulated power cut")

    monkeypatch.setattr(storage.os, "fsync", power_cut)
    with pytest.raises(OSError, match="simulated power cut"):
        old.with_components(bt=bt_state(tick=99)).save(path)
    monkeypatch.setattr(storage.os, "fsync", real_fsync)

    assert path.read_bytes() == before
    assert RobotState.load(path) == old
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


# ── the new types themselves ─────────────────────────────────────────────────


def test_bt_state_freezes_its_blackboard_and_validates_statuses():
    state = bt_state()
    assert state.blackboard["bins"] == ("paper",)
    with pytest.raises(TypeError):
        state.blackboard["speed"]["max"] = 1.0  # type: ignore[index]
    assert state.status_of("Bringup/Sequence/Home") == "SUCCESS"
    assert state.running() == ("Bringup/Sequence",)
    with pytest.raises(ValueError, match="one of"):
        bt_state(node_status={"Bringup": "DONE"})
    with pytest.raises(ValueError, match="JSON values only"):
        bt_state(blackboard={"image": np.zeros(3)})
    with pytest.raises(ValueError, match="whole number"):
        bt_state(tick=-1)
    assert bt_state(blackboard={"n": np.int64(2)}).blackboard["n"] == 2
    # a blackboard value shaped like an encoded array stays a plain mapping through a file
    odd = bt_state(blackboard={"camera": {"dtype": "uint8", "data": [1]}})
    assert BTState.from_dict(json.loads(json.dumps(odd.to_dict()))) == odd


def test_robot_state_reads_by_name_and_updates_by_copy():
    state = robot_state()
    assert state["joints"] == joints() and "bt" in state and len(state) == 2
    moved = state.with_components(joints=joints().replace(position=[0.0, 0.0, 0.0]))
    assert moved["joints"].position.tolist() == [0.0, 0.0, 0.0]
    assert state["joints"] == joints()  # the original is untouched
    assert list(state.without("bt").components) == ["joints"]
    nested = RobotState(components={"cell": state, "camera": CameraSettings(gain=1.0)})
    assert nested["cell"]["bt"].tick == 3
    with pytest.raises(KeyError, match="no component 'arm'"):
        state["arm"]
    with pytest.raises(TypeError, match="must be a robot_msgs message"):
        RobotState(components={"bt": {"tick": 3}})
    config = RobotConfig(components={"camera": CameraSettings(gain=1.0)})
    assert config.with_components(camera=CameraSettings(gain=2.0))["camera"].gain == 2.0


def test_state_and_config_save_together_as_one_bundle(tmp_path):
    config = RobotConfig(components={"camera": CameraSettings(exposure_time=0.01)})
    loaded = robot_msgs.load(robot_msgs.save({"config": config, "state": robot_state()},
                                             tmp_path / "arm"))  # fmt: skip
    assert loaded == {"config": config, "state": robot_state()}
