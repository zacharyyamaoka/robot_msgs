"""HardwareInterface + GroundTruth: one loop on sim and real; the truth rides on the simulated state and is never on a real one.

WHY (Zach, Daily Note 2026-11-28): "We pass the command seq straight into the command seq controller and tick it;
that will send the ServoCommands to the HardwareInterface. From the arm you can read back the servo state ... in metadata here if
you're in simulator you should include the ground truth as well ... switch onto real it's kinda like the 1 line switch".
"""

from __future__ import annotations

import numpy as np
import pytest
from test_joint_command_controller import Plant  # the suite's PD-on-inertia joint plant

from robot_msgs._core import from_dict
from robot_msgs.control import (
    HardwareInterface,
    HardwareBackend,
    HardwareDeployment,
    BackendUnavailable,
    GroundTruth,
    JointCommandController,
    JointCommandSequence,
    JointTable,
    Servo,
    ServoCommand,
    ServoState,
    parse_servo_key,
)
from robot_msgs.control import helpers as h
from robot_msgs.std import Header

TABLE = JointTable(
    bus="sim",
    joints={
        "shoulder": Servo(can_id=1, sign=-1, offset_rad=0.25),
        "elbow": Servo(can_id=2),
    },
)
ENCODER_OFFSET = 0.02  # the board reads this much more than the truth: the gap truth exposes


class PlantBackend:
    """A servo-keyed backend over the suite's `Plant` (board frame = truth + offset), truth on every read."""

    def __init__(self, table: JointTable, *, simulated: bool = True, start=None) -> None:
        self.keys = tuple(table.addresses())
        self.plant = Plant(names=self.keys, q=0.0)
        for i in range(len(self.keys)):
            self.plant.q[i] = 0.0 if start is None else start[i]
            self.plant.offset[i] = ENCODER_OFFSET
        self.simulated = simulated
        self.period_s = 0.01
        self.writes: list[ServoCommand] = []

    def read(self) -> ServoState:
        state = self.plant.read()
        truth = GroundTruth(
            name=tuple(TABLE.joint_at(k) for k in self.keys),
            position=[
                TABLE.joints[TABLE.joint_at(k)].sign * q
                + TABLE.joints[TABLE.joint_at(k)].offset_rad
                for k, q in zip(self.keys, self.plant.q, strict=True)
            ],
            effort=self.plant.tau.copy(),
            source="test-plant",
            header=state.header,
        )
        return state.replace(truth=truth)

    def write(self, command, defines=None):
        assert all(parse_servo_key(name)[0] == "sim" for name in command.name), command.name
        self.writes.append(command)
        self.plant.step(command, dict(defines or {}))


def deployment(real=None) -> HardwareDeployment:
    return HardwareDeployment(
        name="two-joint test arm",
        joints=TABLE,
        backends={"sim": PlantBackend, **({"real": real} if real else {})},
    )


def run(arm: HardwareInterface, sequence: JointCommandSequence, seconds: float = 4.0) -> list[ServoState]:
    controller = JointCommandController(arm.joint_names)
    controller.submit(sequence)
    states = [arm.read()]
    for _ in range(int(seconds / arm.period_s)):
        tick = controller.tick(states[-1], arm.period_s)
        arm.write(tick.command, tick.defines)
        states.append(arm.read())
        if controller.status != "executing":
            break
    assert controller.status == "done", controller.status
    return states


SEQUENCE = JointCommandSequence(
    name="two moves",
    steps=[
        h.hold("shoulder", max_effort=5.0),
        h.move_to("elbow", 0.4, max_vel=1.0, max_effort=5.0),
        h.move_to("shoulder", -0.3, max_vel=1.0, max_effort=5.0),
    ],
)


def test_the_loop_runs_joint_named_and_the_backend_sees_only_servo_keys() -> None:
    arm = HardwareInterface(deployment(), backend="sim")
    states = run(arm, SEQUENCE)
    last = states[-1]
    assert last.name == ("shoulder", "elbow")
    assert last.position[0] == pytest.approx(-0.3, abs=0.01)
    assert last.position[1] == pytest.approx(0.4, abs=0.01)
    assert arm.backend.writes and all(c.name == ("sim#1", "sim#2") for c in arm.backend.writes)


def test_sim_carries_truth_joint_named_and_it_differs_from_the_measurement_by_the_encoder_offset() -> (
    None
):
    arm = HardwareInterface(deployment(), backend="sim")
    last = run(arm, SEQUENCE)[-1]
    assert isinstance(last.truth, GroundTruth) and last.truth.name == last.name
    assert last.truth.header.stamp == last.header.stamp
    # joint = sign * board + offset, board = truth_servo + ENCODER_OFFSET  ->  measured - true = sign * ENCODER_OFFSET
    gap = np.asarray(last.position) - np.asarray(last.truth.position)
    assert gap == pytest.approx([-ENCODER_OFFSET, ENCODER_OFFSET], abs=1e-12)


def test_a_real_backend_never_hands_truth_up_even_if_its_driver_sets_one() -> None:
    arm = HardwareInterface(deployment(real=lambda table: PlantBackend(table, simulated=False)), backend="real")
    states = run(arm, SEQUENCE)
    assert not arm.simulated
    assert all(state.truth is None for state in states)


def test_a_real_backend_that_cannot_open_refuses_cleanly_and_an_unknown_name_lists_the_choices() -> (
    None
):
    def refuse(table):
        raise BackendUnavailable("no bus here")

    with pytest.raises(BackendUnavailable, match="no bus here"):
        HardwareInterface(deployment(real=refuse), backend="real")
    with pytest.raises(KeyError, match=r"\['sim'\]"):
        HardwareInterface(deployment(), backend="real")


def test_defines_go_to_the_servo_frame() -> None:
    arm = HardwareInterface(deployment(), backend="sim")
    seen = {}
    arm.backend.write = lambda command, defines=None: seen.update(defines or {})
    arm.write(ServoCommand(name=("shoulder",), position=[0.0]), {"shoulder": 1.25, "elbow": -0.5})
    # servo = sign * (joint - offset)
    assert seen == {"sim#1": pytest.approx(-1.0), "sim#2": pytest.approx(-0.5)}


def test_servo_state_truth_is_optional_typed_and_round_trips() -> None:
    plain = ServoState(name=("a",), position=[0.1])
    assert plain.truth is None
    assert from_dict(plain.to_dict()) == plain
    truth = GroundTruth(
        name=("a",),
        position=[0.09],
        velocity=[0.0],
        effort=[1.5],
        stop_effort=[0.0],
        source="mujoco",
        header=Header(stamp=0.5),
    )
    state = plain.replace(truth=truth)
    back = from_dict(state.to_dict())
    assert back == state and back.truth.of("a") == pytest.approx(0.09)
    assert state != plain  # truth is part of the value
    with pytest.raises(TypeError, match="GroundTruth"):
        ServoState(name=("a",), position=[0.1], truth={"position": [0.0]})
    with pytest.raises(ValueError, match="2 values"):
        GroundTruth(name=("a",), position=[0.0, 1.0])
    # a dict written before ServoState had `truth` still loads
    old = plain.to_dict()
    old.pop("truth", None)
    assert from_dict(old) == plain


def test_arm_names_still_import():
    """Arm / ArmBackend / ArmDeployment are the pre-2026-11-28 names: the same objects, kept for old callers."""
    from robot_msgs.control import Arm, ArmBackend, ArmDeployment

    assert Arm is HardwareInterface and ArmBackend is HardwareBackend and ArmDeployment is HardwareDeployment


def test_truth_meta_carries_what_only_some_oracles_know():
    """`truth.meta`: the untyped rest of what an oracle knows; it round-trips and travels through select()."""
    from robot_msgs.control import select

    truth = GroundTruth(name=("a/j1", "b/j1"), position=[0.1, 0.2], source="mujoco",
                        meta={"contacts": 3, "object/cap.z": 0.012})
    assert isinstance(truth.meta["contacts"], float)
    assert truth.meta["contacts"] == 3
    assert from_dict(truth.to_dict(), GroundTruth).meta == truth.meta
    state = ServoState(name=("a/j1", "b/j1"), position=[0.1, 0.2], truth=truth)
    assert select(state, "a").truth.meta == truth.meta
    assert GroundTruth(name=("j",), position=[0.0]).meta == {}
