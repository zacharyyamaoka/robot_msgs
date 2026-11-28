"""JointTable: controllers address joints by name; only the table knows a CAN id or a transport.

WHY (Zach, 2026-11-27): "if I change from Can USB etc. I shouldn't need to change my code". The key test is
`test_renumbering_or_moving_a_servo_changes_nothing_joint_named`.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robot_msgs._core import allclose
from robot_msgs.control import (
    JointTable,
    Servo,
    ServoCommand,
    ServoState,
    parse_servo_key,
    servo_key,
)
from robot_msgs.std import Header

NAN = float("nan")


def table() -> JointTable:
    return JointTable(
        bus="fdcanusb:0607623A",
        joints={
            "j1": Servo(can_id=11),
            "j2": Servo(can_id=12, sign=-1, offset_rad=0.30),
            "c1": Servo(can_id=13, sign=+1, offset_rad=-0.25),
        },
    )


def command() -> ServoCommand:
    return ServoCommand(
        name=("c1", "j2", "j1"),  # not table order: the caller's order is kept
        position=[0.5, NAN, -0.1],
        velocity=[0.2, -0.4, 0.0],
        feedforward_effort=[1.0, 2.0, NAN],
        stiffness=[30.0, NAN, 0.0],
        damping=[1.0, 1.0, NAN],
        max_effort=[4.0, 0.0, NAN],
        mode=("position", "brake", "stop"),
        state_stamp=10.0,
        state_seq=7,
        header=Header(stamp=10.002, frame_id="arm"),
    )


def state() -> ServoState:
    return ServoState(
        name=("j1", "j2", "c1"),
        position=[0.1, 1.2, NAN],
        velocity=[0.0, 0.3, -0.2],
        effort=[0.5, -1.5, 0.25],
        temperature=[31.0, 30.0, 29.0],
        voltage=[24.0, 24.1, 24.0],
        fault=[0, 33, 0],
        seq=812,
        header=Header(stamp=10.0),
    )


def test_command_to_servos_applies_sign_and_offset_and_passes_the_rest():
    out = table().to_servos(command())
    assert out.name == ("fdcanusb:0607623A#13", "fdcanusb:0607623A#12", "fdcanusb:0607623A#11")
    # servo = sign * (joint - offset)
    np.testing.assert_allclose(out.position, [0.5 + 0.25, NAN, -0.1], equal_nan=True)
    np.testing.assert_allclose(out.velocity, [0.2, 0.4, 0.0])
    np.testing.assert_allclose(out.feedforward_effort, [1.0, -2.0, NAN], equal_nan=True)
    # gains and caps are magnitudes: never signed
    np.testing.assert_array_equal(out.stiffness, command().stiffness)
    np.testing.assert_array_equal(out.damping, command().damping)
    np.testing.assert_array_equal(out.max_effort, command().max_effort)
    assert out.mode == ("position", "brake", "stop")
    assert (out.state_stamp, out.state_seq, out.header) == (10.0, 7, command().header)


def test_state_to_joints_is_joint_equals_sign_servo_plus_offset():
    servo_side = ServoState(
        name=("fdcanusb:0607623A#12", "fdcanusb:0607623A#13"),
        position=[1.0, 2.0],
        velocity=[0.5, 0.5],
        effort=[1.0, 1.0],
        fault=[0, 2],
    )
    out = table().to_joints(servo_side)
    assert out.name == ("j2", "c1")
    np.testing.assert_allclose(out.position, [-1.0 + 0.30, 2.0 - 0.25])
    np.testing.assert_allclose(out.velocity, [-0.5, 0.5])
    np.testing.assert_allclose(out.effort, [-1.0, 1.0])
    np.testing.assert_array_equal(out.fault, [0, 2])


@pytest.mark.parametrize("message", [command(), state()], ids=["ServoCommand", "ServoState"])
def test_joint_to_servo_to_joint_is_identity(message):
    t = table()
    back = t.to_joints(t.to_servos(message))
    assert allclose(back, message)  # sign -1, offsets, NaN, mode, stamps all survive
    assert back.name == message.name


@pytest.mark.parametrize("message", [command(), state()], ids=["ServoCommand", "ServoState"])
@pytest.mark.parametrize(
    "rewire",
    [
        pytest.param({"j2": Servo(can_id=42, sign=-1, offset_rad=0.30)}, id="renumber-can-id"),
        pytest.param(
            {"c1": Servo(can_id=13, bus="pi3hat:2", offset_rad=-0.25)}, id="move-one-servo-bus"
        ),
    ],
)
def test_renumbering_or_moving_a_servo_changes_nothing_joint_named(message, rewire):
    """THE contract: a CAN id or transport change is a table edit; joint-named traffic is identical."""
    before = table()
    after = before.replace(joints={**before.joints, **rewire})
    assert (
        before.to_servos(message).name != after.to_servos(message).name
    )  # the bus side did change...
    for t in (before, after):
        assert allclose(
            t.to_joints(t.to_servos(message)), message
        )  # ...and the joint side never sees it
    # the board-frame values are the same too: only the address moved
    np.testing.assert_array_equal(
        before.to_servos(message).position, after.to_servos(message).position
    )


def test_swapping_the_whole_transport_changes_only_the_keys():
    usb, hat = table(), table().replace(bus="pi3hat:1")
    a, b = usb.to_servos(command()), hat.to_servos(command())
    assert b.name == ("pi3hat:1#13", "pi3hat:1#12", "pi3hat:1#11")
    assert allclose(a.replace(name=b.name), b)
    assert allclose(hat.to_joints(b), usb.to_joints(a))


def test_with_zero_replaces_offsets_from_the_homed_map():
    homed = table().with_zero({"j2": 1.25, "c1": 0.0})
    assert homed.joints["j2"].offset_rad == 1.25 and homed.joints["j2"].sign == -1.0
    assert homed.joints["c1"].offset_rad == 0.0
    assert homed.joints["j1"] == table().joints["j1"]
    assert table().joints["j2"].offset_rad == 0.30  # a new table; the original is unchanged
    out = homed.to_servos(ServoCommand(name=("j2",), position=[1.25]))
    assert out.position.tolist() == [0.0]  # the homed zero reads 0 on the board
    with pytest.raises(KeyError, match="w9"):
        table().with_zero({"w9": 0.0})


def test_addresses_and_keys_are_what_a_bus_driver_opens():
    t = table().replace(joints={**table().joints, "c1": Servo(can_id=13, bus="can0")})
    assert t.addresses() == {
        "fdcanusb:0607623A#11": ("fdcanusb:0607623A", 11),
        "fdcanusb:0607623A#12": ("fdcanusb:0607623A", 12),
        "can0#13": ("can0", 13),
    }
    assert t.joint_at("can0#13") == "c1"
    assert [parse_servo_key(k) for k in t.addresses()] == list(t.addresses().values())
    assert servo_key(None, 12) == "12" and parse_servo_key("12") == (None, 12)
    assert parse_servo_key(servo_key("fdcanusb:/dev/serial/by-id/usb-x:1", 7)) == (
        "fdcanusb:/dev/serial/by-id/usb-x:1",
        7,
    )
    assert JointTable(joints={"j": Servo(can_id=5)}).key("j") == "5"


def test_validation_errors():
    with pytest.raises(ValueError, match="share servo"):
        JointTable(bus="can0", joints={"a": Servo(can_id=1), "b": Servo(can_id=1)})
    with pytest.raises(
        ValueError, match="share servo"
    ):  # an explicit bus equal to the default is the same servo
        JointTable(bus="can0", joints={"a": Servo(can_id=1), "b": Servo(can_id=1, bus="can0")})
    JointTable(
        bus="can0", joints={"a": Servo(can_id=1), "b": Servo(can_id=1, bus="can1")}
    )  # other bus: fine
    with pytest.raises(ValueError, match="sign"):
        Servo(can_id=1, sign=0.5)
    with pytest.raises(ValueError, match="can_id"):
        Servo(can_id=-1)
    with pytest.raises(ValueError, match="can_id"):
        Servo(can_id=1.5)
    with pytest.raises(ValueError, match="offset_rad"):
        Servo(can_id=1, offset_rad=math.inf)
    with pytest.raises(ValueError, match="bus"):
        Servo(can_id=1, bus="can0#2")
    with pytest.raises(TypeError, match="Servo"):
        JointTable(joints={"a": 12})
    with pytest.raises(KeyError, match="w9"):
        table().to_servos(ServoCommand(name=("w9",), position=[0.0]))
    with pytest.raises(KeyError, match="can0#99"):
        table().to_joints(ServoState(name=("can0#99",), position=[0.0]))
    with pytest.raises(TypeError, match="ServoCommand or a ServoState"):
        table().to_servos(object())  # type: ignore[arg-type]


def test_codec_round_trip_keeps_order_and_lookups():
    t = table()
    back = JointTable.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t and hash(back) == hash(t)
    assert back.names == ("j1", "j2", "c1")
    assert back.joint_at("fdcanusb:0607623A#12") == "j2"
    assert allclose(back.to_servos(command()), t.to_servos(command()))
