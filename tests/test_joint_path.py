"""Joint paths: one joint-named message, addressed at any level of the cell by name prefix.

WHY (Zach, Daily Note 2026-11-28): "cell.right_arm.shoulder_joint.command() ... address at the different
levels you want" and "commands should be addressed by human readable name". The key test is
`test_a_command_to_right_arm_reaches_only_rights_servos`: two mounted copies of ONE arm table, left and right
on different buses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_msgs.control import (
    GroundTruth,
    JointTable,
    Servo,
    ServoCommand,
    ServoState,
    add_prefix,
    is_under,
    join_path,
    select,
    split_path,
    strip_prefix,
)
from robot_msgs.control.joint_path import matching, strip_path
from robot_msgs.std import Header

NAN = float("nan")


def allclose(actual, expected) -> bool:
    """Numbers equal within 1e-12, NaN where NaN (NaN means "board default" and must survive)."""
    return bool(np.allclose(np.asarray(actual, float), np.asarray(expected, float), atol=1e-12, equal_nan=True))


def arm() -> JointTable:
    """One arm description's binding: bare names, no bus yet (the mount's deployment sets it)."""
    return JointTable(
        joints={
            "shoulder": Servo(can_id=11),
            "elbow": Servo(can_id=12, sign=-1, offset_rad=0.30),
            "wrist": Servo(can_id=13, offset_rad=-0.25),
        },
    )


def cell() -> JointTable:
    return JointTable.combine(
        arm().replace(bus="fdcanusb:LEFT").mounted("left_arm"),
        arm().replace(bus="pi3hat:1").mounted("right_arm"),
    )


def cell_command() -> ServoCommand:
    return ServoCommand(
        name=("right_arm/wrist", "left_arm/shoulder", "right_arm/shoulder", "left_arm/elbow"),
        position=[0.5, NAN, -0.1, 0.2],
        velocity=[0.2, -0.4, 0.0, 1.0],
        feedforward_effort=[1.0, 2.0, NAN, 0.5],
        stiffness=[30.0, NAN, 0.0, 10.0],
        damping=[1.0, 1.0, NAN, 2.0],
        max_effort=[4.0, 0.0, NAN, 3.0],
        mode=("position", "brake", "stop", "position"),
        state_stamp=10.0,
        state_seq=7,
        header=Header(stamp=10.002, frame_id="cell"),
    )


def cell_state() -> ServoState:
    return ServoState(
        name=("left_arm/shoulder", "right_arm/elbow", "right_arm/shoulder"),
        position=[0.1, 1.2, NAN],
        velocity=[0.0, 0.3, -0.2],
        effort=[0.5, NAN, 1.0],
        temperature=[30.0, 31.0, 32.0],
        voltage=[24.0, 24.1, 23.9],
        fault=[0, 0, 39],
        seq=42,
        header=Header(stamp=5.0, frame_id="cell"),
    )


# --- path helpers -----------------------------------------------------------------------------------------


def test_join_and_split_round_trip():
    assert join_path("right_arm", "shoulder") == "right_arm/shoulder"
    assert join_path("cell/right_arm", "shoulder") == "cell/right_arm/shoulder"
    assert join_path("", "shoulder") == "shoulder"
    assert split_path("right_arm/shoulder") == ("right_arm", "shoulder")
    assert split_path("") == ()
    assert join_path(*split_path("a/b/c")) == "a/b/c"


@pytest.mark.parametrize("bad", ["a//b", "/a", "a/"])
def test_an_empty_segment_is_refused(bad):
    with pytest.raises(ValueError, match="empty segment"):
        split_path(bad)


def test_prefix_matching_respects_segments():
    assert is_under("arm/x", "arm")
    assert is_under("arm", "arm")
    assert is_under("arm/x/y", "arm/x")
    assert not is_under("arm2/x", "arm")
    assert not is_under("arm2", "arm")
    assert not is_under("arm", "arm/x")
    assert is_under("anything/at/all", "")
    assert matching(["arm/x", "arm2/x", "arm"], "arm") == ("arm/x", "arm")


def test_strip_path_needs_a_name_strictly_inside():
    assert strip_path("right_arm/shoulder", "right_arm") == "shoulder"
    assert strip_path("shoulder", "") == "shoulder"
    with pytest.raises(ValueError):
        strip_path("right_arm2/shoulder", "right_arm")
    with pytest.raises(ValueError):
        strip_path("right_arm", "right_arm")


# --- mount + combine ---------------------------------------------------------------------------------------


def test_mounted_prefixes_names_and_pins_the_bus():
    mounted = arm().replace(bus="can0").mounted("right_arm")
    assert mounted.names == ("right_arm/shoulder", "right_arm/elbow", "right_arm/wrist")
    assert mounted.bus is None
    assert mounted.address("right_arm/elbow") == ("can0", 12)
    assert mounted.joints["right_arm/elbow"].sign == -1.0
    assert mounted.joints["right_arm/elbow"].offset_rad == 0.30
    with pytest.raises(ValueError):
        arm().mounted("")


def test_mount_combine_round_trips_through_to_servos_and_to_joints():
    table = cell()
    assert len(table) == 6
    command = cell_command()
    servos = table.to_servos(command)
    assert servos.name == ("pi3hat:1#13", "fdcanusb:LEFT#11", "pi3hat:1#11", "fdcanusb:LEFT#12")
    back = table.to_joints(servos)
    assert back.name == command.name
    for field in ("position", "velocity", "feedforward_effort", "stiffness", "damping", "max_effort"):
        assert allclose(getattr(back, field), getattr(command, field)), field
    assert back.mode == command.mode
    state = cell_state()
    assert allclose(table.to_joints(table.to_servos(state)).position, state.position)


def test_subtree_gives_the_arms_own_table_back():
    right = cell().subtree("right_arm")
    assert right.names == arm().names
    assert right.address("shoulder") == ("pi3hat:1", 11)
    assert right.joints["elbow"].offset_rad == 0.30
    kept = cell().subtree("left_arm", strip=False)
    assert kept.names == ("left_arm/shoulder", "left_arm/elbow", "left_arm/wrist")
    with pytest.raises(KeyError):
        cell().subtree("right")  # segment-safe: no "right" level


def test_combine_refuses_a_duplicate_joint_name():
    with pytest.raises(ValueError, match="'right_arm/shoulder' is in tables 0 and 1"):
        JointTable.combine(
            arm().replace(bus="a").mounted("right_arm"), arm().replace(bus="b").mounted("right_arm")
        )


def test_combine_refuses_one_servo_claimed_twice():
    # Two arms mounted with the same bus: CAN 11 on "can0" would drive both shoulders.
    with pytest.raises(ValueError, match="share servo 'can0#11'"):
        JointTable.combine(
            arm().replace(bus="can0").mounted("left_arm"),
            arm().replace(bus="can0").mounted("right_arm"),
        )
    # And with no bus at all: both keys are the bare "11".
    with pytest.raises(ValueError, match="share servo '11'"):
        JointTable.combine(arm().mounted("left_arm"), arm().mounted("right_arm"))


# --- subtree select / strip / re-prefix ---------------------------------------------------------------------


def test_select_a_level_keeps_every_column_and_nan():
    command = cell_command()
    right = select(command, "right_arm")
    assert right.name == ("right_arm/wrist", "right_arm/shoulder")
    assert allclose(right.position, [0.5, -0.1])
    assert allclose(right.feedforward_effort, [1.0, NAN])
    assert allclose(right.stiffness, [30.0, 0.0])
    assert allclose(right.damping, [1.0, NAN])
    assert allclose(right.max_effort, [4.0, NAN])
    assert right.mode == ("position", "stop")
    assert right.state_stamp == 10.0 and right.state_seq == 7
    assert right.header.stamp == 10.002 and right.header.frame_id == "cell"


def test_select_a_set_and_one():
    command = cell_command()
    both = select(command, ["left_arm/shoulder", "right_arm/shoulder"])
    assert both.name == ("left_arm/shoulder", "right_arm/shoulder")  # message order, not argument order
    assert both.mode == ("brake", "stop")
    one = select(command, "left_arm/elbow")
    assert one.name == ("left_arm/elbow",)
    assert allclose(one.position, [0.2])
    assert select(command, "").name == command.name  # the root: all


def test_select_refuses_a_path_matching_nothing():
    with pytest.raises(KeyError, match="right"):
        select(cell_command(), "right")  # not a prefix of "right_arm/..."
    with pytest.raises(KeyError, match="left_arm/hand"):
        select(cell_command(), ["left_arm", "left_arm/hand"])


def test_select_state_keeps_measurements_and_seq():
    right = select(cell_state(), "right_arm")
    assert right.name == ("right_arm/elbow", "right_arm/shoulder")
    assert allclose(right.position, [1.2, NAN])
    assert allclose(right.effort, [NAN, 1.0])
    assert allclose(right.temperature, [31.0, 32.0])
    assert list(right.fault) == [0, 39] and right.fault.dtype == np.int64
    assert right.faulted and right.seq == 42 and right.header.stamp == 5.0


def test_a_states_simulator_truth_follows_the_same_view():
    truth = GroundTruth(
        name=("left_arm/shoulder", "right_arm/elbow", "right_arm/shoulder", "right_arm/wrist"),
        position=[0.0, 1.0, 2.0, 3.0],
        stop_effort=[0.0, NAN, 5.0, 0.0],
        source="mujoco",
        header=Header(stamp=5.0),
    )
    state = cell_state().replace(truth=truth)
    right = select(state, "right_arm", strip=True)
    assert right.truth.name == ("elbow", "shoulder", "wrist")  # truth's own rows under the level
    assert allclose(right.truth.position, [1.0, 2.0, 3.0])
    assert allclose(right.truth.stop_effort, [NAN, 5.0, 0.0])
    assert right.truth.source == "mujoco"
    assert add_prefix(right, "right_arm").truth.name[0] == "right_arm/elbow"
    one = select(state, "left_arm/shoulder")
    assert one.truth.name == ("left_arm/shoulder",)


def test_strip_and_re_prefix_preserve_values():
    command = cell_command()
    bare = select(command, "right_arm", strip=True)
    assert bare.name == ("wrist", "shoulder")
    again = add_prefix(bare, "right_arm")
    assert again.name == ("right_arm/wrist", "right_arm/shoulder")
    assert again == select(command, "right_arm")
    assert strip_prefix(again, "right_arm") == bare
    with pytest.raises(ValueError):
        strip_prefix(command, "right_arm")  # left_arm rows are not inside it
    with pytest.raises(ValueError, match="one path"):
        select(command, ["right_arm"], strip=True)


def test_select_refuses_other_types():
    with pytest.raises(TypeError):
        select(cell().subtree("right_arm"), "shoulder")  # type: ignore[arg-type]


# --- end to end -------------------------------------------------------------------------------------------------


def test_a_command_to_right_arm_reaches_only_rights_servos():
    """Two mounted copies of the same arm table, left on an fdcanusb and right on a pi3hat.

    An arm-level command written against the arm's bare names, mounted as `right_arm`, goes out only on the
    right arm's bus — and a cell-wide command viewed at `right_arm` reaches the same servos with the same values.
    """
    table = cell()
    arm_command = ServoCommand(
        name=("shoulder", "elbow"),
        position=[0.4, math.pi / 4],
        velocity=[0.0, 0.1],
        mode=("position", "position"),
        header=Header(stamp=1.0),
    )
    servos = table.to_servos(add_prefix(arm_command, "right_arm"))
    assert servos.name == ("pi3hat:1#11", "pi3hat:1#12")
    left_keys = {table.key(n) for n in table.names if is_under(n, "left_arm")}
    assert left_keys == {"fdcanusb:LEFT#11", "fdcanusb:LEFT#12", "fdcanusb:LEFT#13"}
    assert not left_keys & set(servos.name)
    # elbow: servo = sign * (joint - offset) = -(pi/4 - 0.30)
    assert allclose(servos.position, [0.4, -(math.pi / 4 - 0.30)])

    # The same arm table, mounted left, sends the same arm command to left's boards instead.
    left_servos = table.to_servos(add_prefix(arm_command, "left_arm"))
    assert left_servos.name == ("fdcanusb:LEFT#11", "fdcanusb:LEFT#12")
    assert allclose(left_servos.position, servos.position)

    # View a whole-cell command at the right_arm level: only right's servo keys come out.
    viewed = table.to_servos(select(cell_command(), "right_arm"))
    assert set(viewed.name) <= {"pi3hat:1#11", "pi3hat:1#12", "pi3hat:1#13"}
    assert viewed.name == ("pi3hat:1#13", "pi3hat:1#11")
