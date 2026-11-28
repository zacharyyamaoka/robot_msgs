"""ControllerManager: one call switches or refuses by priority, each controller at its rate, stop is a manager call,
and a controller switched out with nothing in its place holds (never drops)."""

import math

import numpy as np
from test_joint_command_controller import Plant

from robot_msgs.control import (
    ControllerManager,
    JointCommandController,
    JointCommandSequence,
    ServoCommand,
)
from robot_msgs.control import helpers as h

NAMES = ("j", "k")


class Counter:
    """A JointController-shaped stub: holds `target` on its joints and counts its ticks."""

    def __init__(self, joints, target):
        self.joint_names, self.target, self.ticks, self.dts = tuple(joints), float(target), 0, []
        self.status = "running"

    def tick(self, state, dt):
        self.ticks += 1
        self.dts.append(dt)
        n = len(self.joint_names)
        return ServoCommand(
            name=self.joint_names, position=np.full(n, self.target), max_effort=np.full(n, 2.0)
        ), {}

    def replace_input(self, target):
        self.target = float(target)


def manager():
    m = ControllerManager(NAMES)
    m.add("jcc", JointCommandController(NAMES), joints=NAMES, priority=10)
    m.add("low", Counter(NAMES, 0.3), joints=NAMES, priority=1, rate_hz=25)
    return m


def drive(m, plant, seconds, calls=()):
    ticks, calls = [], sorted(calls, key=lambda c: c[0])
    for k in range(int(round(seconds / 0.01))):
        while calls and calls[0][0] <= plant.t + 1e-9:
            calls.pop(0)[1]()
        out = m.tick(plant.read(), 0.0 if k == 0 else 0.01)
        plant.step(out.command, out.defines)
        ticks.append(out)
    return ticks


def test_one_call_switches_in_and_a_lower_priority_goal_is_refused():
    m, plant = manager(), Plant(names=NAMES)
    goal = m.send_goal(
        "jcc", JointCommandSequence(steps=[h.move_to("j", 0.4, 1.0, 5.0), h.hold("k", 3.0)])
    )
    assert goal is not None and m.active() == ("jcc",)
    drive(m, plant, 0.3)
    assert m.send_goal("low", 0.0) is None and "refused" in m.last_refusal
    drive(m, plant, 2.0)
    assert m._slots["jcc"].controller.result(goal).status == "done"
    assert abs(plant.q[0] - 0.4) < math.radians(1.0)


def test_a_higher_priority_goal_takes_the_joints_in_the_same_tick():
    m, plant = manager(), Plant(names=NAMES)
    assert m.send_goal("low", 0.3) == "low"
    drive(m, plant, 0.2)
    m.send_goal(
        "jcc", JointCommandSequence(steps=[h.hold("j", 3.0), h.hold("k", 3.0), h.wait(1.0)])
    )
    out = drive(m, plant, 0.02)[-1]
    assert m.active() == ("jcc",) and out.owner == {"j": "jcc", "k": "jcc"}


def test_each_controller_ticks_at_its_own_rate():
    m, plant = manager(), Plant(names=NAMES)
    m.send_goal("low", 0.3)
    drive(m, plant, 1.0)
    low = m._slots["low"].controller
    assert 24 <= low.ticks <= 26  # 25 Hz on a 100 Hz host
    assert all(abs(dt - 0.04) < 1e-6 for dt in low.dts[1:])


def test_stop_is_a_manager_call_and_wins_next_tick():
    m, plant = manager(), Plant(names=NAMES)
    m.send_goal("low", 0.3)
    ticks = drive(m, plant, 0.5, calls=[(0.3, m.stop)])
    assert ticks[29].modes == ("position", "position") and ticks[30].modes == ("stop", "stop")
    m.send_goal("low", 0.1)  # a new goal re-arms the joints it takes
    assert drive(m, plant, 0.05)[-1].modes == ("position", "position")


def test_a_controller_switched_out_alone_leaves_its_joints_holding():
    m, plant = manager(), Plant(names=NAMES)
    m.send_goal("low", 0.3)
    drive(m, plant, 1.0)
    m.switch(deactivate=["low"])
    out = drive(m, plant, 0.05)[-1]
    assert out.owner == {"j": "hold", "k": "hold"}
    assert abs(out.command.position[0] - 0.3) < 0.02 and out.command.max_effort[0] == 2.0


def test_manual_switch_ignores_priority_and_never_commanded_joints_are_limp():
    m, plant = ControllerManager(NAMES), Plant(names=NAMES)
    m.add("a", Counter(("j",), 0.2), joints=("j",), priority=5)
    m.add("b", Counter(("j",), -0.2), joints=("j",), priority=1)
    m.switch(activate=["a"])
    out = drive(m, plant, 0.05)[-1]
    assert out.owner == {"j": "a", "k": "limp"} and np.isnan(out.command.position[1])
    m.switch(activate=["b"])
    assert drive(m, plant, 0.05)[-1].owner["j"] == "b"
