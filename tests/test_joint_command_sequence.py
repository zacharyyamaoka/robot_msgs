"""JointCommandSequence + JointCommandSequenceController on a toy joint with a hard stop.

The toy plant: a joint that follows its setpoint exactly while it has torque, cannot pass a stop,
and falls at a fixed speed when limp. Simple enough that every expected number is hand-derived.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_msgs.control import (
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandSequenceController,
    JointCommandStep,
    Reached,
    Settled,
    Stall,
)
from robot_msgs.sensor import JointState
from robot_msgs.std import Header

DT = 0.01


class ToyJoint:
    """Position follows the command's setpoint, clamped at `stop`; limp means falling at `fall`."""

    def __init__(self, position=0.0, stop=0.4, fall=0.0, floor=-math.inf):
        self.position, self.stop, self.fall, self.floor = position, stop, fall, floor
        self.velocity = 0.0
        self.offset = 0.0  # what define_position changes: reported = true + offset
        self.time = 0.0

    def state(self) -> JointState:
        return JointState(
            name=("j",),
            position=[self.position + self.offset],
            velocity=[self.velocity],
            effort=[0.0],
            header=Header(stamp=self.time),
        )

    def apply(self, command, defines) -> None:
        if "j" in defines:
            self.offset = defines["j"] - self.position
        before = self.position
        if command.max_effort[0] == 0.0:
            self.position = max(self.floor, self.position - self.fall * DT)
        else:
            self.position = min(self.stop, float(command.position[0]) - self.offset)
        self.velocity = (self.position - before) / DT
        self.time += DT


def run(sequence, joint, seconds=30.0):
    controller = JointCommandSequenceController(sequence)
    for _ in range(int(seconds / DT)):
        command, defines = controller.tick(joint.state(), DT)
        if controller.status != "running":
            return controller, command
        joint.apply(command, defines)
    return controller, command


def homing() -> JointCommandSequence:
    return JointCommandSequence(
        joint_names=("j",),
        steps=(
            JointCommandStep(
                name="settle",
                commands=(JointCommand.limp("j"),),
                done_when=(Elapsed(duration=0.5),),
                timeout=2.0,
            ),
            JointCommandStep(
                name="seek",
                commands=(JointCommand.move_by("j", 1.0, speed=0.5, max_effort=8.0),),
                done_when=(Stall(joint="j"),),
                timeout=5.0,
                define_position={"j": 2.0},
            ),
            JointCommandStep(
                name="home",
                commands=(JointCommand.to("j", 1.5, speed=0.5, max_effort=4.0),),
                done_when=(Reached(joints=("j",), tolerance=0.005),),
                timeout=5.0,
            ),
        ),
    )


def test_a_homing_routine_finds_the_stop_defines_it_and_goes_home():
    joint = ToyJoint(position=0.1, stop=0.4)
    controller, _ = run(homing(), joint)
    assert controller.status == "done", controller.message
    assert [e.outcome for e in controller.events] == ["Elapsed", "Stall j", "Reached"]
    # the stop is now 2.0 in the joint's frame, and home 1.5 is 0.5 back from it
    assert joint.position + joint.offset == pytest.approx(1.5, abs=1e-9)
    assert joint.position == pytest.approx(0.4 - 0.5, abs=1e-9)
    seek = controller.events[1]
    # ramp 0.5 rad/s from 0.1 hits the stop at 0.6 s, then lead > 0.06 and 0.2 s of dwell
    assert seek.finished - seek.started == pytest.approx(0.6 + 0.12 + 0.2, abs=0.03)


def test_done_keeps_commanding_the_last_step():
    joint = ToyJoint(position=0.1)
    controller, command = run(homing(), joint)
    command, defines = controller.tick(joint.state(), DT)
    assert controller.status == "done" and not defines
    assert command.position[0] == pytest.approx(1.5) and command.max_effort[0] == 4.0


def test_a_limp_joint_is_nan_position_and_no_torque_on_the_wire():
    controller = JointCommandSequenceController(homing())
    command, _ = controller.tick(ToyJoint().state(), DT)
    assert np.isnan(command.position[0]) and command.max_effort[0] == 0.0
    assert command.state_stamp == 0.0


def test_a_step_not_done_in_time_aborts_everything_limp():
    sequence = JointCommandSequence(
        joint_names=("j",),
        steps=(
            JointCommandStep(
                name="seek",
                commands=(JointCommand.move_by("j", 1.0, speed=0.5, max_effort=8.0),),
                done_when=(Stall(joint="j"),),
                timeout=1.0,
            ),
        ),
    )
    controller, command = run(sequence, ToyJoint(stop=10.0))
    assert controller.status == "aborted" and "within 1.0 s" in controller.message
    assert controller.events[-1].outcome == "timeout"
    assert np.isnan(command.position[0]) and command.max_effort[0] == 0.0


def test_the_guard_trips_on_a_drive_pushing_fast_but_not_on_a_limp_fall():
    def state(velocity, effort):
        return JointState(name=("j",), position=[0.0], velocity=[velocity], effort=[effort])

    powered = JointCommandSequence(
        joint_names=("j",),
        max_speed=1.0,
        steps=(
            JointCommandStep(
                name="hold",
                commands=(JointCommand.hold("j", max_effort=5.0),),
                done_when=(Elapsed(duration=10.0),),
                timeout=20.0,
            ),
        ),
    )
    braking = JointCommandSequenceController(powered)
    braking.tick(state(-2.0, +3.0), DT)  # gravity pulls it down, the drive pushes back
    assert braking.status == "running"
    runaway = JointCommandSequenceController(powered)
    runaway.tick(state(-2.0, -3.0), DT)
    assert runaway.status == "aborted" and runaway.events[-1].outcome == "guard"

    limp = JointCommandSequence(
        joint_names=("j",),
        max_speed=1.0,
        steps=(
            JointCommandStep(
                name="drop",
                commands=(JointCommand.limp("j"),),
                done_when=(Elapsed(duration=10.0),),
                timeout=20.0,
            ),
        ),
    )
    falling = JointCommandSequenceController(limp)
    falling.tick(state(-5.0, 0.0), DT)
    assert falling.status == "running"


def test_settled_waits_for_the_dwell():
    sequence = JointCommandSequence(
        joint_names=("j",),
        steps=(
            JointCommandStep(
                name="drop",
                commands=(JointCommand.limp("j"),),
                done_when=(Settled(joints=("j",), speed=0.01, dwell=0.3),),
                timeout=5.0,
            ),
        ),
    )
    joint = ToyJoint(position=0.5, fall=1.0, floor=0.0)  # lands at 0.5 s
    controller, _ = run(sequence, joint)
    assert controller.status == "done"
    assert controller.events[0].finished == pytest.approx(0.5 + 0.3, abs=0.03)


class SaggingJoint(ToyJoint):
    """A ToyJoint that rests `sag` past every setpoint (gravity carrying it beyond its target)."""

    def __init__(self, sag, **kwargs):
        super().__init__(**kwargs)
        self.sag = sag

    def apply(self, command, defines) -> None:
        before = self.position
        super().apply(command, defines)
        if command.max_effort[0] != 0.0:
            self.position -= self.sag
        self.velocity = (self.position - before) / DT


def test_two_conditions_of_one_kind_on_one_joint_keep_their_own_timers():
    # WHY (2026-11-12, the five-bar homing bench under gravity): a tight Reached that is never met must
    # not reset a looser Reached's dwell on the same joint every tick; each condition keeps its own.
    sequence = JointCommandSequence(
        joint_names=("j",),
        steps=(
            JointCommandStep(
                name="lower",
                commands=(JointCommand.move_by("j", -0.2, speed=0.5, max_effort=8.0),),
                done_when=(
                    Reached(joints=("j",), tolerance=0.06),
                    Reached(joints=("j",), tolerance=0.3, speed=0.02, dwell=0.5),
                ),
                timeout=5.0,
            ),
        ),
    )
    joint = SaggingJoint(sag=0.1, position=0.0)  # rests 0.1 beyond its target: the tight reach never holds
    controller, _ = run(sequence, joint)
    assert controller.status == "done"
    assert controller.events[0].outcome == "Reached"
    # the setpoint arrives at 0.4 s; the loose reach then needs its 0.5 s dwell (one tick of motion settling)
    assert controller.events[0].finished == pytest.approx(0.4 + 0.5, abs=0.03)


def test_a_hold_after_a_hold_keeps_its_setpoint_instead_of_relatching_the_sag():
    hold = JointCommand.hold("j", max_effort=5.0)
    sequence = JointCommandSequence(
        joint_names=("j",),
        steps=tuple(
            JointCommandStep(
                name=f"hold {i}", commands=(hold,), done_when=(Elapsed(duration=0.1),), timeout=1.0
            )
            for i in range(3)
        ),
    )
    controller = JointCommandSequenceController(sequence)
    sagged = 0.0
    for _ in range(40):
        state = JointState(name=("j",), position=[sagged], velocity=[0.0], effort=[1.0])
        command, _ = controller.tick(state, DT)
        sagged -= 0.01  # the joint sags under load every tick
    assert controller.status == "done"
    assert command.position[0] == 0.0  # still the first step's setpoint


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: JointCommand(joint="j", mode="fly"), "mode"),
        (lambda: JointCommand(joint="j", mode="hold"), "max_effort"),
        (lambda: JointCommand.move_by("j", 0.0, speed=1.0, max_effort=1.0), "distance"),
        (lambda: JointCommand.to("j", math.nan, speed=1.0, max_effort=1.0), "position"),
        (lambda: Stall(joint="j", dwell=0.0), "dwell"),
        (
            lambda: JointCommandStep(
                name="s",
                commands=(JointCommand.limp("a"),),
                done_when=(Stall(joint="b"),),
                timeout=1.0,
            ),
            "watches",
        ),
        (
            lambda: JointCommandStep(
                name="s",
                commands=(JointCommand.limp("a"), JointCommand.limp("a")),
                done_when=(Elapsed(duration=1.0),),
                timeout=1.0,
            ),
            "one command per joint",
        ),
        (
            lambda: JointCommandStep(
                name="s",
                commands=(JointCommand.limp("a"),),
                done_when=(Elapsed(duration=1.0),),
                timeout=1.0,
                define_position={"b": 0.0},
            ),
            "cannot define",
        ),
        (
            lambda: JointCommandSequence(joint_names=("a",), steps=homing().steps),
            "unknown joints",
        ),
    ],
)
def test_bad_routines_are_refused_at_construction(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_sequence_round_trips_through_json_with_its_mixed_conditions():
    import json

    sequence = homing()
    again = JointCommandSequence.from_dict(json.loads(json.dumps(sequence.to_dict())))
    assert again == sequence
    assert isinstance(again.steps[1].done_when[0], Stall)


# ── EffortStall: a stop is where the drive pushes past its load, not where it stops moving ────────


class GravityJoint(ToyJoint):
    """A ToyJoint under a load rising toward its stop - `load` (N.m, against the + direction) at the stop,
    in proportion short of it, like gravity on an arm tilting up - with a capped P drive of stiffness `kp`
    (N.m/rad) plus the command's feed-forward: it stops wherever cap + feed-forward cannot carry the load,
    and its effort reads what the drive applies."""

    def __init__(self, load=5.0, kp=400.0, **kwargs):
        super().__init__(**kwargs)
        self.load, self.kp, self.effort = load, kp, 0.0

    def state(self) -> JointState:
        state = super().state()
        return JointState(name=state.name, position=state.position, velocity=state.velocity,
                          effort=[self.effort], header=state.header)

    def apply(self, command, defines) -> None:
        if "j" in defines:
            self.offset = defines["j"] - self.position
        before = self.position
        cap = float(command.max_effort[0])
        target = float(command.position[0]) - self.offset
        drive = float(np.clip(self.kp * (target - self.position) + float(command.feedforward_effort[0]), -cap, cap))
        self.effort = drive
        if drive > self.load * max(self.position, 0.0) / self.stop:  # carries the load here: follows, never past the stop
            self.position = min(self.stop, max(self.position, min(target, self.position + 1.0 * DT)))
        self.velocity = (self.position - before) / DT
        self.time += DT


def effort_seek(cap: float, feedforward: float, condition) -> JointCommandSequence:
    return JointCommandSequence(
        joint_names=("j",),
        steps=(
            JointCommandStep(
                name="seek",
                commands=(JointCommand.move_by("j", 1.0, speed=0.5, max_effort=cap, feedforward_effort=feedforward),),
                done_when=(condition,),
                timeout=5.0,
                define_position={"j": 2.0},
            ),
        ),
    )


def test_an_effort_stall_is_the_stop_and_never_gravity():
    """The full arm's J1 (2026-11-14): with its feed-forward off and its cap under the load plus a margin, a seek stops
    short where its drive saturates on gravity. A lead rule reads that as the stop; EffortStall - effort past `effort`
    while still - does not: it ends on the real stop with a cap above the threshold, and times out (never defines) below."""
    from robot_msgs.control import EffortStall

    # cap 11 over a 5 N.m load, no feed-forward: it reaches the stop and presses there past 9 N.m
    joint = GravityJoint(position=0.1, stop=0.4, load=5.0)
    controller, _ = run(effort_seek(11.0, 0.0, EffortStall(joint="j", effort=9.0)), joint)
    assert controller.status == "done" and controller.events[-1].outcome == "EffortStall j"
    assert joint.position == pytest.approx(0.4)  # on the stop, where the drive pressed past 9 N.m
    # cap 6 against 7 N.m at the stop: it stops short where gravity reaches 6 and can never press 9 - it times out
    short = GravityJoint(position=0.1, stop=0.4, load=7.0)
    controller, _ = run(effort_seek(6.0, 0.0, EffortStall(joint="j", effort=9.0)), short)
    assert controller.status == "aborted" and controller.events[-1].outcome == "timeout" and short.position < 0.4
    # the lead rule on that same seek records the gravity stall as a stop - the trap EffortStall exists for
    trap = GravityJoint(position=0.1, stop=0.4, load=7.0)
    controller, _ = run(effort_seek(6.0, 0.0, Stall(joint="j")), trap)
    assert controller.status == "done" and controller.events[-1].outcome == "Stall j" and trap.position < 0.4


def test_an_effort_stall_round_trips_and_is_refused_without_a_positive_threshold():
    import json

    from robot_msgs.control import EffortStall

    sequence = effort_seek(11.0, 0.0, EffortStall(joint="j", effort=9.0, speed=0.02, dwell=0.3))
    again = JointCommandSequence.from_dict(json.loads(json.dumps(sequence.to_dict())))
    assert again == sequence and isinstance(again.steps[0].done_when[0], EffortStall)
    with pytest.raises(ValueError, match="effort"):
        EffortStall(joint="j", effort=0.0)
