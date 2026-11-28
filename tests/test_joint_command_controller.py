"""JointCommandController + the helper functions, on a one-joint plant with a hard stop (numpy only).

The full acceptance lives in the Joint Command Controller track (ruler v2: 45 cases on the UR5e and the
1-DOF pendulum in MuJoCo, plant truth as the oracle; ruler v1: 28 on the planar arm). These pin the
library's own promises: helpers are functions that return primitives, hold-don't-drop, stop wins,
find_hardstop's mean / spread / refusal, the watchdog clear and the fault slack.
"""

import math

import numpy as np
import pytest

from robot_msgs.control import (
    PRIMITIVES,
    Command,
    Condition,
    Delay,
    InCollision,
    JointCommand,
    JointCommandController,
    JointCommandSequence,
    JointCommandSequenceController,
    JointCommandStep,
    Reached,
    ServoCommand,
    ServoState,
    Stall,
)
from robot_msgs.control import helpers as h
from robot_msgs.std import Header


class Plant:
    """Joints with a board PD (kp, kd, the cap), unit-ish inertia, and optional hard stops (rad)."""

    def __init__(self, names=("j",), q=0.0, stops=None, kp=60.0, kd=2.0, inertia=0.05):
        self.names = tuple(names)
        self.q = np.full(len(self.names), float(q))
        self.v = np.zeros(len(self.names))
        self.offset = np.zeros(
            len(self.names)
        )  # board frame = truth + offset (a define changes it)
        self.stops = dict(stops or {})
        self.kp, self.kd, self.inertia = kp, kd, inertia
        self.t = 0.0
        self.tau = np.zeros(len(self.names))

    def read(self):
        return ServoState(
            name=self.names,
            position=self.q + self.offset,
            velocity=self.v.copy(),
            effort=self.tau.copy(),
            header=Header(stamp=self.t),
        )

    def step(self, command: ServoCommand, defines, dt=0.01, substeps=20):
        for name, value in defines.items():
            i = self.names.index(name)
            self.offset[i] = value - self.q[i]
        h_ = dt / substeps
        for _ in range(substeps):
            tau = np.zeros(len(self.names))
            for i, name in enumerate(self.names):
                mode = command.mode_of(name)
                if mode in ("stop", "brake"):
                    tau[i] = -0.5 * self.v[i] if mode == "brake" else 0.0
                    continue
                kp = (
                    self.kp
                    if command.stiffness is None or np.isnan(command.stiffness[i])
                    else command.stiffness[i]
                )
                kd = (
                    self.kd
                    if command.damping is None or np.isnan(command.damping[i])
                    else command.damping[i]
                )
                sp = command.position[i] - self.offset[i]
                pd = 0.0 if np.isnan(sp) else kp * (sp - self.q[i])
                ff = 0.0 if command.feedforward_effort is None else command.feedforward_effort[i]
                cap = (
                    math.inf
                    if command.max_effort is None or np.isnan(command.max_effort[i])
                    else command.max_effort[i]
                )
                tau[i] = float(np.clip(pd - kd * self.v[i] + ff, -cap, cap))
            self.tau = tau
            self.v += tau / self.inertia * h_
            self.q += self.v * h_
            for name, (low, high) in self.stops.items():
                i = self.names.index(name)
                if self.q[i] > high:
                    self.q[i], self.v[i] = high, min(self.v[i], 0.0)
                if self.q[i] < low:
                    self.q[i], self.v[i] = low, max(self.v[i], 0.0)
        self.t += dt


def run(controller, plant, seconds, *, board_mode=None, calls=()):
    """Tick for `seconds`; `calls` are (t, fn) run before that tick. Returns every tick and every define."""
    ticks, defined, calls = [], {}, sorted(calls, key=lambda c: c[0])
    for k in range(int(round(seconds / 0.01))):
        while calls and calls[0][0] <= plant.t + 1e-9:
            calls.pop(0)[1]()
        out = controller.tick(
            plant.read(),
            0.0 if k == 0 else 0.01,
            board_mode=board_mode(plant.t) if board_mode else None,
        )
        defined.update(out.defines)
        plant.step(out.command, out.defines)
        ticks.append(out)
    return ticks, defined


# ── the DSL is functions that return primitives ──────────────────────────────────────────────────────────────


def test_every_helper_is_a_function_that_returns_only_primitives():
    parts = [
        h.limp("j"),
        h.stop("j"),
        h.stop_all(),
        h.brake("j"),
        h.hold("j", 3.0),
        h.velocity("j", 0.2, 3.0),
        h.effort("j", 0.5),
        h.define("j", 0.1),
        h.set_joint_home("j", 0.0),
        h.move_to("j", 0.3, 1.0, 3.0),
        h.move_by("j", -0.2, 1.0, 3.0),
        h.wait(0.5),
        h.wait_until(InCollision("j"), timeout=2.0),
        h.find_hardstop("j", +1, 0.3, 2.0, 0.5, repeats=3, define=0.4),
        h.find_hardstop_and_hold("j", -1, 0.3, 2.0, timeout=3.0),
        h.verify_hard_stop("j", 3, +1, 0.3, 2.0, 0.5, expected=0.4),
        h.at_once(h.move_to("j", 0.1, 1.0, 3.0), h.move_to("k", 0.2, 1.0, 3.0)),
    ]
    for part in parts:
        assert isinstance(part, list) and part
        assert all(isinstance(p, PRIMITIVES) for p in part), part


def test_find_hardstop_repeats_is_n_recorded_conditions_and_one_reduce():
    part = h.find_hardstop("j", +1, 0.3, 2.0, 0.5, repeats=5, define=0.4)
    recorded = [p for p in part if isinstance(p, Condition) and p.record is not None]
    reduces = [p for p in part if isinstance(p, Command) and p.reduce is not None]
    assert len(recorded) == 5 and len(reduces) == 1 and part[-1] is reduces[0]
    assert part[1].fail[0].reason.startswith("no contact on j within max_travel")


def test_a_recipe_composes_helper_calls_and_round_trips():
    recipe = JointCommandSequence(
        name="home j then park",
        steps=[
            h.hold("k", 3.0),
            h.find_hardstop("j", +1, 0.3, 2.0, 0.5, repeats=2, define=0.4),
            h.move_to("j", 0.1, 1.0, 3.0),
            h.stop_all(),
        ],
    )
    assert recipe.joints == ("k", "j")
    assert all(isinstance(p, PRIMITIVES) for p in recipe.steps)
    assert JointCommandSequence.from_dict(recipe.to_dict()) == recipe


def test_at_once_refuses_a_sequential_helper():
    with pytest.raises(ValueError, match="at_once"):
        h.at_once(h.find_hardstop("j", +1, 0.3, 2.0, 0.5))


# ── the controller ───────────────────────────────────────────────────────────────────────────────────────────


def test_status_goes_idle_executing_done_and_a_move_arrives():
    plant, jcc = Plant(), JointCommandController(("j",))
    assert jcc.status == "idle"
    goal = jcc.submit(JointCommandSequence(steps=[h.move_to("j", 0.4, 1.0, 5.0)]), client="test")
    ticks, _ = run(jcc, plant, 0.05)
    assert jcc.status == "executing"
    run(jcc, plant, 2.0)
    assert jcc.result(goal).status == "done" and jcc.status == "done"
    assert abs(plant.q[0] - 0.4) < math.radians(1.0)
    assert ticks[0].modes == ("position",) and set(ticks[0].log) >= {
        "goal",
        "step",
        "intent",
        "pending",
    }


@pytest.mark.parametrize("repeats", [1, 5])
def test_find_hardstop_defines_so_the_mean_of_its_touches_reads_define(repeats):
    plant, jcc = Plant(stops={"j": (-3.0, 0.3)}), JointCommandController(("j",))
    plant.offset[:] = 0.25  # the board frame is wrong by 0.25 rad
    goal = jcc.submit(
        JointCommandSequence(
            steps=[h.find_hardstop("j", +1, 0.3, 3.0, 0.8, repeats=repeats, define=0.3)]
        ),
        client="test",
    )
    _, defined = run(jcc, plant, 12.0)
    result = jcc.result(goal)
    assert result.status == "done", result.reason
    report = result.hardstops["j"]
    assert len(report.touches) == repeats and report.spread <= 0.001 and report.defined == 0.3
    assert "j" in defined
    # define 0.3 at the true stop (0.3): the board frame now matches the truth within 0.1°
    assert abs(plant.offset[0]) < math.radians(0.1)


def test_a_stop_that_moves_is_refused_no_define_and_it_holds():
    plant, jcc = Plant(stops={"j": (-3.0, 0.3)}), JointCommandController(("j",))
    goal = jcc.submit(
        JointCommandSequence(
            steps=[h.find_hardstop("j", +1, 0.3, 3.0, 0.8, repeats=3, define=0.3)]
        ),
        client="test",
    )
    run(jcc, plant, 3.0)  # the first touch (0.8 rad at 0.3 rad/s reaches the stop by ~1 s)
    plant.stops["j"] = (-3.0, 0.31)  # then the stop moves 0.57°
    ticks, defined = run(jcc, plant, 9.0)
    result = jcc.result(goal)
    assert result.status == "aborted" and "spread" in result.reason
    assert result.hardstops["j"].refused and result.hardstops["j"].defined is None and not defined
    last = ticks[-1].command
    assert (
        last.mode_of("j") == "position" and np.isfinite(last.position[0]) and last.max_effort[0] > 0
    )  # holds
    assert jcc.status == "fault"


def test_no_contact_within_max_travel_aborts_without_a_define_and_holds():
    plant, jcc = Plant(), JointCommandController(("j",))
    goal = jcc.submit(
        JointCommandSequence(steps=[h.find_hardstop("j", +1, 0.5, 3.0, 0.3, define=1.0)]),
        client="t",
    )
    ticks, defined = run(jcc, plant, 4.0)
    assert (
        jcc.result(goal).status == "aborted"
        and "no contact" in jcc.result(goal).reason
        and not defined
    )
    assert np.isfinite(ticks[-1].command.position[0]) and ticks[-1].command.max_effort[0] > 0


def test_a_timeout_holds_where_the_arm_is_never_limp():
    plant, jcc = Plant(stops={"j": (-3.0, 0.2)}), JointCommandController(("j",))
    goal = jcc.submit(
        JointCommandSequence(steps=[h.move_to("j", 0.6, 1.0, 2.0, timeout=1.0)]), client="t"
    )
    ticks, _ = run(jcc, plant, 2.0)
    assert jcc.result(goal).status == "aborted" and "timeout" in jcc.result(goal).reason
    assert (
        abs(ticks[-1].command.position[0] - 0.2) < 0.02 and ticks[-1].command.max_effort[0] == 2.0
    )


def test_stop_wins_on_the_next_tick_and_preempts():
    plant, jcc = Plant(names=("j", "k")), JointCommandController(("j", "k"))
    goal = jcc.submit(
        JointCommandSequence(steps=[h.hold("j", 3.0), h.hold("k", 3.0), h.wait(5.0)]), client="a"
    )
    ticks, _ = run(jcc, plant, 0.5, calls=[(0.3, lambda: jcc.stop(client="c"))])
    k = 30
    assert ticks[k - 1].modes == ("position", "position") and ticks[k].modes == ("stop", "stop")
    assert jcc.result(goal).status == "stopped"


def test_stop_all_names_every_joint_and_brake_is_a_mode():
    plant, jcc = Plant(names=("j", "k")), JointCommandController(("j", "k"))
    jcc.submit(JointCommandSequence(steps=[h.brake("j"), h.stop_all(), h.brake("k")]), client="a")
    ticks, _ = run(jcc, plant, 0.05)
    assert ticks[-1].modes == ("stop", "brake")


def test_effort_sends_zero_gains_on_its_joint_and_board_defaults_elsewhere():
    plant, jcc = Plant(names=("j", "k")), JointCommandController(("j", "k"))
    jcc.submit(
        JointCommandSequence(steps=[h.effort("j", 0.4), h.hold("k", 3.0), h.wait(0.5)]), client="a"
    )
    ticks, _ = run(jcc, plant, 0.1)
    command = ticks[-1].command
    assert (
        command.stiffness[0] == 0.0 and command.damping[0] == 0.0 and np.isnan(command.position[0])
    )
    assert np.isnan(command.stiffness[1]) and np.isnan(command.damping[1])  # board default
    assert command.feedforward_effort[0] == 0.4


def test_send_is_refused_on_a_claimed_joint_and_kept_fed_on_a_free_one():
    plant, jcc = Plant(names=("j", "k")), JointCommandController(("j", "k"))
    jcc.submit(JointCommandSequence(steps=[h.hold("j", 3.0), h.wait(5.0)]), client="a")
    run(jcc, plant, 0.02)
    assert not jcc.send(ServoCommand(name=("j",), position=[0.0], max_effort=[1.0]), client="b")
    assert jcc.send(
        ServoCommand(name=("k",), position=[0.1]), client="b"
    )  # no cap: board default (0.8)
    ticks, _ = run(jcc, plant, 0.1)
    assert ticks[-1].command.position[1] == 0.1 and np.isnan(ticks[-1].command.max_effort[1])


def test_a_watchdog_latch_gets_one_stop_then_the_move_resumes_from_where_the_arm_is():
    plant, jcc = Plant(), JointCommandController(("j",))
    goal = jcc.submit(JointCommandSequence(steps=[h.move_to("j", 0.5, 1.0, 5.0)]), client="a")
    ticks, _ = run(
        jcc, plant, 3.0, board_mode=lambda t: {"j": "timeout"} if 0.30 <= t < 0.32 else {}
    )
    stops = [k for k, tick in enumerate(ticks) if tick.modes == ("stop",)]
    assert stops and len(stops) <= 2 and jcc.result(goal).status == "done"


def test_a_board_fault_faults_the_goal_and_slacks_that_joint():
    plant, jcc = Plant(names=("j", "k")), JointCommandController(("j", "k"), slack={"k": "brake"})
    goal = jcc.submit(
        JointCommandSequence(steps=[h.move_to("j", 0.5, 0.5, 5.0), h.hold("k", 3.0)]), client="a"
    )
    ticks, _ = run(jcc, plant, 1.0, board_mode=lambda t: {"k": "fault"} if t >= 0.5 else {})
    assert jcc.result(goal).status == "faulted" and jcc.status == "fault"
    assert ticks[-1].modes[1] == "stop" and ticks[-1].modes[0] == "position"


def test_until_in_collision_ends_a_velocity_seek_on_the_stop():
    plant, jcc = Plant(stops={"j": (-3.0, 0.2)}), JointCommandController(("j",))
    goal = jcc.submit(
        JointCommandSequence(
            steps=[h.velocity("j", 0.4, 3.0), h.wait_until(InCollision("j"), timeout=3.0)]
        ),
        client="a",
    )
    run(jcc, plant, 3.0)
    assert jcc.result(goal).status == "done" and abs(plant.q[0] - 0.2) < 1e-6


def test_a_preempt_starts_from_the_old_setpoint():
    plant, jcc = Plant(), JointCommandController(("j",))
    jcc.submit(JointCommandSequence(steps=[h.move_to("j", 1.0, 1.0, 5.0)]), client="a")
    run(jcc, plant, 0.4)
    before = jcc.tick(plant.read(), 0.01).command.position[0]
    jcc.submit(JointCommandSequence(steps=[h.move_to("j", -0.5, 1.0, 5.0)]), client="b")
    after = jcc.tick(plant.read(), 0.01).command.position[0]
    assert abs(after - before) < 0.02


# ── 0.3 steps still run, on both controllers ────────────────────────────────────────────────────────────────


def step_sequence():
    return JointCommandSequence(
        joint_names=("j", "k"),
        steps=(
            JointCommandStep(
                name="seek",
                commands=(JointCommand.move_by("j", 1.0, speed=0.3, max_effort=3.0),),
                done_when=(Stall(joint="j"),),
                timeout=6.0,
                define_position={"j": 0.3},
            ),
            JointCommandStep(
                name="park",
                commands=(JointCommand.to("j", 0.0, speed=1.0, max_effort=3.0),),
                done_when=(Reached(joints=("j",), tolerance=0.02),),
                timeout=3.0,
            ),
        ),
    )


def test_a_step_sequence_runs_on_the_new_controller_with_its_define():
    plant, jcc = (
        Plant(names=("j", "k"), stops={"j": (-3.0, 0.3)}),
        JointCommandController(("j", "k")),
    )
    goal = jcc.submit(step_sequence(), client="a")
    ticks, defined = run(jcc, plant, 8.0)
    assert jcc.result(goal).status == "done", jcc.result(goal).reason
    assert "j" in defined and abs(plant.q[0] - 0.0) < 0.03
    assert np.isnan(ticks[5].command.position[1])  # the step names only j: k is limp


def test_the_old_controller_refuses_primitives_and_new_modes():
    with pytest.raises(ValueError, match="JointCommandController"):
        JointCommandSequenceController(JointCommandSequence(steps=[h.move_to("j", 0.1, 1.0, 3.0)]))
    with pytest.raises(ValueError, match="stop"):
        JointCommandSequenceController(
            JointCommandSequence(
                joint_names=("j",),
                steps=(
                    JointCommandStep(
                        name="s",
                        commands=(JointCommand.stop("j"),),
                        done_when=(Reached(joints=("j",), tolerance=0.1),),
                        timeout=1.0,
                    ),
                ),
            )
        )


def test_a_bare_primitive_list_is_a_sequence():
    sequence = JointCommandSequence(
        steps=[Command((JointCommand.hold("j", max_effort=2.0),)), Delay(0.3)]
    )
    plant, jcc = Plant(), JointCommandController(("j",))
    goal = jcc.submit(sequence, client="a")
    run(jcc, plant, 0.5)
    assert jcc.result(goal).status == "done"
