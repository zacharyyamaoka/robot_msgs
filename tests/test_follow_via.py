"""Follow by composition: a JointCommandController hands a trajectory to an EXTERNAL trajectory controller
(`helpers.follow(trajectory, via="jtc")`) instead of nesting one, and the ControllerManager runs the two side by
side (Zach, 2026-11-27: "I 100% want to chain controllers via composition not nesting").

The JTC here is a minimal robot_controllers-shaped one (`status` / `message`, `replace_input`, `tick`) on the
same one-joint numpy plant as the JCC tests; robot_controllers' own JointTrajectoryController is checked against
this seam in that repo's suite.
"""

import math

import numpy as np
import pytest
from test_joint_command_controller import Plant, run

from robot_msgs.control import (
    ControllerManager,
    JointCommand,
    JointCommandController,
    JointCommandSequence,
    ServoCommand,
)
from robot_msgs.control import helpers as h
from robot_msgs.trajectory import JointTrajectory

NAMES = ("j", "k")


def ramp(joint="j", to=0.5, seconds=1.0):
    t = np.linspace(0.0, seconds, 11)
    return JointTrajectory(t=t, joint_names=(joint,), positions=(to * t / seconds)[:, None])


class MiniJTC:
    """Time-driven: samples the trajectory at its clock; `abort_at` (s) aborts with a message, holding."""

    def __init__(self, joints, abort_at=None):
        self.joint_names, self.abort_at = tuple(joints), abort_at
        self.status, self.message, self.trajectory, self.clock = "done", "", None, 0.0
        self.inputs = 0

    def replace_input(self, trajectory):
        self.trajectory, self.clock, self.status, self.message = trajectory, -1.0, "running", ""
        self.inputs += 1

    def tick(self, state, dt):
        traj = self.trajectory
        self.clock = 0.0 if self.clock < 0 else self.clock + dt
        if self.abort_at is not None and self.clock >= self.abort_at and self.status == "running":
            self.status, self.message = "aborted", "j was 9.00 deg off the path"
        position = traj.joint_state_at(min(self.clock, traj.duration), extrapolate="clamp").position
        if self.status == "running" and self.clock + 1e-9 >= traj.duration:
            self.status = "done"
        n = len(traj.joint_names)
        return ServoCommand(
            name=traj.joint_names, position=position, velocity=np.zeros(n), max_effort=np.full(n, 4.0)
        ), {}


def recipe(via="jtc", then=True):
    steps = [h.hold("k", 3.0), h.follow(ramp(), 4.0, via=via)]
    if then:
        steps.append(h.move_to("j", 0.2, 1.0, 5.0))
    return JointCommandSequence(name="follow then move", steps=steps)


# ── the JCC alone: a follow request out, the JTC's status in ─────────────────────────────────────────────────


def drive_by_hand(jcc, jtc, plant, seconds, abort_after=None):
    """What the manager does, by hand: route the request, write the followed joints from the JTC."""
    ticks, seen = [], set()
    for k in range(int(round(seconds / 0.01))):
        out = jcc.tick(plant.read(), 0.0 if k == 0 else 0.01)
        command = out.command
        for request in out.follows:
            if request.id not in seen:
                seen.add(request.id)
                jtc.replace_input(request.trajectory)
            sub, _ = jtc.tick(plant.read().joint_state(), 0.0 if jtc.clock < 0 else 0.01)
            i = command.name.index("j")
            position = command.position.copy()
            cap = command.max_effort.copy()
            position[i], cap[i] = sub.position[0], sub.max_effort[0]
            command = command.replace(position=position, max_effort=cap)
            jcc.follow_status(request.id, jtc.status, jtc.message)
        plant.step(command, out.defines)
        ticks.append(out)
    return ticks


def test_a_follow_via_jtc_lends_its_joints_and_completes_when_the_jtc_reports_done():
    jcc, jtc, plant = JointCommandController(NAMES), MiniJTC(("j",)), Plant(names=NAMES)
    goal = jcc.submit(recipe())
    ticks = drive_by_hand(jcc, jtc, plant, 3.0)
    following = [t for t in ticks if t.follows]
    assert following and all(t.followed_joints == ("j",) for t in following)
    request = following[0].follows[0]
    assert request.via == "jtc" and request.goal == goal and request.trajectory.joint_names == ("j",)
    # WHY: while it follows, the JCC writes NO target on j (the JTC does), but keeps writing k
    for t in following:
        assert math.isnan(t.command.position[0]) and t.command.max_effort[0] == 0.0
        assert t.command.max_effort[1] == 3.0
        assert "follow via jtc" in t.log["intent"]["j"]
    assert jcc.result(goal).status == "done"
    assert not ticks[-1].follows  # the move_to after it took j back
    assert abs(plant.q[0] - 0.2) < math.radians(1.0)
    assert jtc.inputs == 1  # the trajectory went out once, not once per tick


def test_a_follow_via_jtc_fails_with_the_jtcs_reason_when_it_aborts():
    jcc, jtc, plant = JointCommandController(NAMES), MiniJTC(("j",), abort_at=0.4), Plant(names=NAMES)
    goal = jcc.submit(recipe())
    ticks = drive_by_hand(jcc, jtc, plant, 1.5)
    result = jcc.result(goal)
    assert result.status == "aborted"
    assert result.reason == "follow via jtc aborted: j was 9.00 deg off the path"
    assert any(e.kind == "follow" for e in result.events)
    assert jcc.status == "fault" and not ticks[-1].follows
    assert np.isfinite(ticks[-1].command.position[0])  # hold where it is, never limp


def test_a_jtc_that_never_reports_times_the_follow_out():
    jcc, plant = JointCommandController(NAMES), Plant(names=NAMES)
    goal = jcc.submit(JointCommandSequence(steps=[h.follow(ramp(), 4.0, via="jtc", timeout=0.3)]))
    run(jcc, plant, 0.6)
    assert jcc.result(goal).status == "aborted" and "timeout" in jcc.result(goal).reason


def test_via_is_a_follow_only_field_and_the_nested_default_is_unchanged():
    with pytest.raises(ValueError, match="via"):
        JointCommand(joint="j", mode="hold", max_effort=1.0, via="jtc")
    jcc, plant = JointCommandController(NAMES), Plant(names=NAMES)
    goal = jcc.submit(JointCommandSequence(steps=[h.follow(ramp(), 4.0)]))
    ticks, _ = run(jcc, plant, 2.0)
    assert jcc.result(goal).status == "done" and not any(t.follows for t in ticks)
    assert abs(plant.q[0] - 0.5) < math.radians(1.0)


# ── the manager: JCC and JTC as peers ────────────────────────────────────────────────────────────────────────


def manager(jtc, *, jtc_rate=None):
    m = ControllerManager(NAMES)
    m.add("jcc", JointCommandController(NAMES), joints=NAMES, priority=10)
    m.add("jtc", jtc, joints=("j",), priority=5, rate_hz=jtc_rate)
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


@pytest.mark.parametrize("jtc_rate", [None, 50.0])
def test_manager_routes_the_follow_to_the_jtc_side_by_side_and_feeds_its_status_back(jtc_rate):
    jtc, plant = MiniJTC(("j",)), Plant(names=NAMES)
    m = manager(jtc, jtc_rate=jtc_rate)
    goal = m.send_goal("jcc", recipe(then=False))
    ticks = drive(m, plant, 0.6)
    assert jtc.inputs == 1 and m.active() == ("jcc", "jtc")
    assert ticks[-1].owner == {"j": "jtc", "k": "jcc"}
    assert m.owners == {"j": "jcc", "k": "jcc"}  # the joints stay the JCC's for priorities
    # the wire on j is the JTC's command (cap 4.0), not the JCC's no-target
    assert np.isfinite(ticks[-1].command.position[0]) and ticks[-1].command.max_effort[0] == 4.0
    assert abs(ticks[-1].command.position[0] - 0.3) < 0.05
    drive(m, plant, 1.5)
    jcc = m._slots["jcc"].controller
    assert jcc.result(goal).status == "done"
    assert abs(plant.q[0] - 0.5) < math.radians(1.0)


def test_manager_follow_then_the_jcc_takes_the_joint_back():
    jtc, plant = MiniJTC(("j",)), Plant(names=NAMES)
    m = manager(jtc)
    goal = m.send_goal("jcc", recipe())
    ticks = drive(m, plant, 3.0)
    assert m._slots["jcc"].controller.result(goal).status == "done"
    assert ticks[-1].owner == {"j": "jcc", "k": "jcc"} and m.active() == ("jcc",)
    assert abs(plant.q[0] - 0.2) < math.radians(1.0)


def test_manager_abort_reaches_the_jcc_result():
    jtc, plant = MiniJTC(("j",), abort_at=0.3), Plant(names=NAMES)
    m = manager(jtc)
    goal = m.send_goal("jcc", recipe())
    ticks = drive(m, plant, 1.0)
    result = m._slots["jcc"].controller.result(goal)
    assert result.status == "aborted" and "j was 9.00 deg off the path" in result.reason
    assert ticks[-1].owner["j"] == "jcc"  # the JCC holds j again


def test_manager_stop_during_an_external_follow_wins_next_tick():
    jtc, plant = MiniJTC(("j",)), Plant(names=NAMES)
    m = manager(jtc)
    goal = m.send_goal("jcc", recipe())
    ticks = drive(m, plant, 0.8, calls=[(0.4, m.stop)])
    assert ticks[39].owner["j"] == "jtc" and ticks[40].modes == ("stop", "stop")
    assert all(t.modes == ("stop", "stop") for t in ticks[40:])
    assert m._slots["jcc"].controller.result(goal).status == "stopped"
    assert m.active() == ("jcc",)  # the lend ended with the follow


def test_manager_refuses_a_follow_via_an_unknown_controller_into_the_jcc_result():
    plant = Plant(names=NAMES)
    m = ControllerManager(NAMES)
    m.add("jcc", JointCommandController(NAMES), joints=NAMES, priority=10)
    goal = m.send_goal("jcc", recipe(via="nope"))
    drive(m, plant, 0.2)
    result = m._slots["jcc"].controller.result(goal)
    assert result.status == "aborted" and "no controller 'nope'" in result.reason


def test_a_higher_priority_goal_taking_the_joint_ends_the_lend_and_fails_the_follow():
    jtc, plant = MiniJTC(("j",)), Plant(names=NAMES)
    m = manager(jtc)
    m.add("top", MiniJTC(("j",)), joints=("j",), priority=20)
    goal = m.send_goal("jcc", recipe())
    drive(m, plant, 0.3)
    assert m.send_goal("top", ramp(to=-0.2)) == "top"
    ticks = drive(m, plant, 0.1)
    assert ticks[-1].owner["j"] == "top" and "jtc" not in m.active()
    result = m._slots["jcc"].controller.result(goal)
    assert result.status == "aborted" and "taken from jcc" in result.reason
