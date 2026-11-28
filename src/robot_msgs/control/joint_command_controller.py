"""JointCommandController: the one writer to a set of servos. It ticks a `JointCommandSequence`'s primitives.

Clients ask it to run a command or a sequence; it checks each step against the measured state every tick
until it is done or says why not (Zach, 2026-11-27: "what is even the point of a controller? well it kinda
like adds that feedback loop check"). Five calls:

- `submit(sequence, client=...) -> goal id`: run it. A new sequence on joints another one holds preempts that
  one, starting from its setpoints so nothing jumps.
- `stop(client=..., joints=None)`: the board's stop on these joints (all when None) from the next tick. Any
  client may call it and it always wins.
- `send(command, client=...) -> bool`: fire and forget, re-sent every tick until replaced; False when a named
  joint belongs to a running sequence.
- `tick(state, dt, board_mode=None) -> JointCommandTick`: one host tick: the measured `ServoState` in; the
  `ServoCommand` (with its per-servo `mode`), the positions defined this tick and the command-log row out.
- `result(goal_id) -> JointCommandResult | None`: how a sequence ended.

`status` goes `idle` -> `executing` -> `done` | `fault`. A sequence that ends badly (a timeout, a failed
Condition, a refused define, an aborting guard, a preempt with nothing after it) HOLDS its powered joints
where they are; it never goes limp. Brake or stop only for a board fault, a lost bus (`board_mode`
`"timeout"`: one stop clears the watchdog latch, then the setpoints resume from the measured pose) or an
explicit stop (PEP 0029 addendum 2; "the robot should just try to hold its current position always",
2026-11-13). Guards log and continue unless the sequence says `guard_policy="abort"` (2026-11-13).

Built from the Joint Command Controller track's S011-primitives (ruler v2: 45/45 dev and holdout on the UR5e
and the 1-DOF pendulum) and S010 (ruler v1 28/28); its ramp, Reached and Stall numbers are robot_msgs'
`JointCommandSequenceController`'s, kept number for number. `JointCommandSequenceController` stays as it was.

Follow: a `follower(trajectory, max_effort)` factory makes the sub-controller for a trajectory (anything with
`tick(state, dt) -> (ServoCommand, defines)` and `status`, e.g. robot_controllers' JointTrajectoryController);
the default samples the trajectory (`SampledFollower`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from robot_msgs.control.joint_command import (
    ALL_JOINTS,
    EffortStall,
    Elapsed,
    JointCommand,
    JointCommandSequence,
    Reached,
    Settled,
    Stall,
)
from robot_msgs.control.primitives import (
    Command,
    Condition,
    Delay,
    FollowEnded,
    InCollision,
    TravelEnd,
)
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.std import Header

STALL = Stall(
    joint="_"
)  # the hand-rolled contact rule: lead 0.06 rad, speed 0.01 rad/s, dwell 0.2 s
TIMEOUT_FACTOR = 1.5
TIMEOUT_MARGIN_S = 1.0
# WHY 3 ticks before a second stop: one stop clears a watchdog latch; a latch read within the round trip is the
# same one (S003)
WATCHDOG_STOP_COOLDOWN_TICKS = 3

GoalStatus = Literal["done", "aborted", "preempted", "stopped", "rejected", "faulted"]
ControllerStatus = Literal["idle", "executing", "done", "fault"]


class Detector(Protocol):
    """A contact detector behind `is_in_collision(joint)`: one verdict per joint."""

    def contact(self, state: Any, command: ServoCommand | None, t: float) -> Mapping[str, bool]: ...


class Follower(Protocol):
    """A trajectory sub-controller: `status` is `running` until its trajectory has played out (`done`)."""

    status: str

    def tick(self, state: Any, dt: float) -> tuple[ServoCommand, Mapping[str, float]]: ...


@dataclass(frozen=True)
class Event:
    """Something the command log should show: `kind` is `condition`, `timeout`, `guard`, `fault`, ..."""

    t: float
    kind: str
    joint: str | None
    message: str


@dataclass(frozen=True)
class HardstopReport:
    """One hard-stop find, board frame BEFORE any define it made: every touch (rad), their mean and spread
    (max - min); `defined` the value it defined (None if none), `refused` why it refused (empty if it did not)."""

    joint: str
    touches: tuple[float, ...]
    mean: float
    spread: float
    defined: float | None = None
    refused: str = ""


@dataclass(frozen=True)
class JointCommandResult:
    """How a sequence ended: `status` (`GoalStatus`), why, what happened, and every hard stop it found."""

    status: str
    reason: str = ""
    events: tuple[Event, ...] = ()
    hardstops: Mapping[str, HardstopReport] = field(default_factory=dict)


@dataclass(frozen=True)
class JointCommandTick:
    """One tick's output: the wire `command` (its `mode` per servo), `defines` (joint -> value declared this
    tick) and the command viewer's `log` row (`goal`, `step`, `intent` per joint, `pending` conditions)."""

    command: ServoCommand
    defines: Mapping[str, float]
    log: Mapping[str, Any]

    @property
    def modes(self) -> tuple[str, ...]:
        return self.command.mode or ("position",) * len(self.command.name)


class SampledFollower:
    """The default follower: the trajectory's sampled position, velocity and effort at the follow's clock."""

    def __init__(self, trajectory: Any, max_effort: float | None = None) -> None:
        self.trajectory = trajectory
        self.status = "running"
        self.clock_s = 0.0
        self._cap = np.full(
            trajectory.n_joints, math.nan if max_effort is None else float(max_effort)
        )

    def tick(self, state: Any, dt: float) -> tuple[ServoCommand, Mapping[str, float]]:
        self.clock_s += dt
        point = self.trajectory.joint_state_at(
            self.trajectory.start_time + self.clock_s, extrapolate="clamp"
        )
        ended = self.clock_s + 1e-9 >= self.trajectory.duration
        if ended:
            self.status = "done"
        velocity = np.zeros(len(point.name)) if ended or point.velocity is None else point.velocity
        return ServoCommand(
            name=point.name,
            position=point.position,
            velocity=velocity,
            feedforward_effort=point.effort,
            max_effort=self._cap,
        ), {}


# ── one joint's drive (what its last JointCommand set) ─────────────────────────────────────────────────────────


class _Drive:
    """`mode`: limp, stop, brake, hold, ramp, velocity, effort, follow."""

    __slots__ = (
        "mode",
        "setpoint",
        "target",
        "speed",
        "velocity",
        "effort",
        "max_effort",
        "feedforward",
        "follow",
        "frozen",
        "started",
        "direction",
    )

    def __init__(
        self,
        mode: str,
        setpoint: float = math.nan,
        target: float = math.nan,
        speed: float = 0.0,
        velocity: float = 0.0,
        effort: float = 0.0,
        max_effort: float = math.nan,
        feedforward: float = 0.0,
        follow: _Follow | None = None,
    ) -> None:
        self.mode, self.setpoint, self.target, self.speed = mode, setpoint, target, speed
        self.velocity, self.effort, self.max_effort, self.feedforward = (
            velocity,
            effort,
            max_effort,
            feedforward,
        )
        self.follow = follow
        self.frozen = False
        self.started = False
        # WHY fixed when the ramp is made: once the setpoint lands on its target, "toward the target" has no sign,
        # and the lead rule read a seek pressing on a lower stop as leading the wrong way (ruler v2 holdout P11)
        self.direction = (
            1 if (velocity >= 0.0 if mode == "velocity" else target >= setpoint) else -1
        )

    @property
    def settled(self) -> bool:
        return self.mode == "hold" or (self.mode == "ramp" and self.setpoint == self.target)

    def advance(self, dt: float) -> None:
        """The stock ramp: the setpoint moves speed x dt toward the target, landing on it exactly."""
        if self.mode == "ramp":
            remaining = self.target - self.setpoint
            step = self.speed * dt
            self.setpoint = (
                self.target
                if abs(remaining) <= step
                else self.setpoint + math.copysign(step, remaining)
            )
        elif self.mode == "velocity" and not self.frozen:
            if self.started:
                self.setpoint += self.velocity * dt
            self.started = True
            self.target = self.setpoint

    def shift(self, by: float) -> None:
        self.setpoint += by
        self.target += by

    def wire(self, joint: str) -> tuple[float, float, float, float, float, float]:
        """(position, velocity, feedforward, max_effort, stiffness, damping); NaN gain/cap = board default."""
        nan = math.nan
        if self.mode == "follow":
            return self.follow.wire(joint)
        if self.mode == "effort":
            cap = abs(self.effort) if math.isnan(self.max_effort) else self.max_effort
            return (nan, 0.0, self.effort, cap, 0.0, 0.0)
        if self.mode in ("hold", "ramp"):
            return (self.setpoint, 0.0, self.feedforward, self.max_effort, nan, nan)
        if self.mode == "velocity":
            return (
                self.setpoint,
                0.0 if self.frozen else self.velocity,
                self.feedforward,
                self.max_effort,
                nan,
                nan,
            )
        return (nan, 0.0, 0.0, 0.0, nan, nan)  # limp (stop / brake carry no setpoint)

    def text(self) -> str:
        if self.mode == "ramp":
            return f"ramp to {self.target:+.4f} at {self.speed:g} rad/s (setpoint {self.setpoint:+.4f})"
        if self.mode == "hold":
            return f"hold {self.setpoint:+.4f}"
        if self.mode == "velocity":
            return f"velocity {self.velocity:+.3f} rad/s" + (
                " (done: holding)" if self.frozen else ""
            )
        if self.mode == "effort":
            return f"effort {self.effort:+.3f} N·m"
        if self.mode == "follow":
            return self.follow.text()
        return self.mode


class _Follow:
    """One trajectory followed by every joint of a Command that names it."""

    def __init__(
        self,
        trajectory: Any,
        joints: tuple[str, ...],
        max_effort: float | None,
        make: Callable[[Any, float | None], Follower],
    ) -> None:
        if set(trajectory.joint_names) != set(joints):
            trajectory = trajectory.select(joints)
        self.cap = max_effort
        self.trajectory = trajectory
        self.follower = make(trajectory, max_effort)
        self.column = {joint: k for k, joint in enumerate(trajectory.joint_names)}
        self.last = {joint: float(trajectory.positions[-1][k]) for joint, k in self.column.items()}
        self.clock_s = 0.0
        self.command: ServoCommand | None = None

    @property
    def ended(self) -> bool:
        return self.follower.status == "done"

    @property
    def remaining_s(self) -> float:
        return max(0.0, float(self.trajectory.duration) - self.clock_s)

    def advance(self, joint_state: Any, dt: float) -> None:
        self.clock_s += dt
        self.command, _ = self.follower.tick(joint_state, dt)

    def wire(self, joint: str) -> tuple[float, float, float, float, float, float]:
        command = self.command
        k = command.name.index(joint)
        cap = math.nan if command.max_effort is None else float(command.max_effort[k])
        if math.isinf(cap):
            cap = (
                math.nan
            )  # WHY: an uncapped follow goes out as the board's own cap, not +inf (0.8 barrier)
        feedforward = (
            0.0 if command.feedforward_effort is None else float(command.feedforward_effort[k])
        )
        velocity = 0.0 if command.velocity is None else float(command.velocity[k])
        return (float(command.position[k]), velocity, feedforward, cap, math.nan, math.nan)

    def text(self) -> str:
        return f"follow ({self.follower.status}, t={self.clock_s:.2f} of {float(self.trajectory.duration):.2f} s)"


class _JointView:
    """The part of a JointState a follower reads, over a ServoState without copying it."""

    __slots__ = ("name", "position", "velocity", "effort", "header", "_index")

    def __init__(self, state: Any, index: Mapping[str, int]) -> None:
        self.name, self.position, self.velocity, self.effort = (
            state.name,
            state.position,
            state.velocity,
            state.effort,
        )
        self.header = state.header
        self._index = index

    def index_of(self, name: str) -> int:
        return self._index[name]

    def position_of(self, name: str) -> float:
        return float(self.position[self._index[name]])


# ── one sequence, executed primitive by primitive ──────────────────────────────────────────────────────────────


class _Run:
    """One submitted sequence: a program counter over its primitives and a drive per joint."""

    def __init__(
        self,
        run_id: int,
        client: str,
        sequence: JointCommandSequence,
        joints: tuple[str, ...],
        everyone: tuple[str, ...],
        previous: Mapping[str, _Drive],
        has_detector: bool,
        follower: Callable[[Any, float | None], Follower],
    ) -> None:
        self.id = run_id
        self.client = client
        self.goal = sequence
        self.program = sequence.primitives
        self.joints = joints
        self.everyone = everyone
        self.drives: dict[str, _Drive] = {}
        self._previous = dict(previous)
        self.has_detector = has_detector
        self.follower = follower
        self.pc = 0
        self.time = 0.0
        self.ticks = 0
        self.status = "running"
        self.reason = ""
        self.events: list[Event] = []
        self.records: dict[str, list[float]] = {}
        self.hardstops: dict[str, HardstopReport] = {}
        self.contacts: Mapping[str, bool] = {}
        self.efforts: Mapping[str, float] = {}
        self._wait_start: float | None = None
        self._wait_timeout = math.inf
        self._timers: dict[Any, float] = {}
        self._anchors: dict[Any, float] = {}
        self._guarded: set[str] = set()

    @property
    def current(self) -> Any:
        return self.program[self.pc] if self.pc < len(self.program) else None

    @property
    def step_name(self) -> str:
        primitive = (
            self.current
            if self.current is not None
            else (self.program[-1] if self.program else None)
        )
        return getattr(primitive, "name", "")

    def waits_on_detector(self) -> bool:
        current = self.current
        return (
            self.status == "running"
            and isinstance(current, Condition)
            and any(isinstance(p, InCollision) for p in current.until)
        )

    def mode_of(self, joint: str) -> str | None:
        drive = self.drives.get(joint)
        return drive.mode if drive is not None and drive.mode in ("stop", "brake") else None

    def tick(
        self, joint_state: Any, dt: float, measured: dict[str, float], velocity: Mapping[str, float]
    ) -> dict[str, float]:
        # WHY dt 0 on the first tick: the sequence starts now (robot_controllers' TrajectoryClock rule)
        dt = 0.0 if self.ticks == 0 else dt
        self.ticks += 1
        self.time += dt
        for follow in {
            id(d.follow): d.follow for d in self.drives.values() if d.mode == "follow"
        }.values():
            follow.advance(joint_state, dt)
        for drive in self.drives.values():
            drive.advance(dt)
        defines: dict[str, float] = {}
        first = True
        while self.status == "running" and self.pc < len(self.program):
            primitive = self.program[self.pc]
            if isinstance(primitive, Command):
                self._apply(primitive, joint_state, measured, defines)
                self.pc += 1
                continue
            if self._wait_start is None:
                self._begin_wait(primitive)
                step_dt = 0.0
            else:
                step_dt = dt if first else 0.0
            first = False
            if isinstance(primitive, Delay):
                if self.time - self._wait_start + 1e-9 >= primitive.seconds:
                    self._end_wait()
                    continue
                break
            outcome = self._evaluate(primitive, measured, velocity, step_dt)
            if outcome == "done":
                if primitive.record is not None:
                    self.records.setdefault(primitive.record.tag, []).append(
                        measured[primitive.record.joint]
                    )
                self.events.append(Event(self.time, "condition", None, f"{primitive.name}: done"))
                self._end_wait()
                continue
            if outcome is not None:
                self.abort(outcome)
                break
            if self.time - self._wait_start + 1e-9 >= self._wait_timeout:
                self.abort(
                    f"timeout: step {primitive.name!r} was not done within {self._wait_timeout:.2f} s "
                    f"(waiting on {', '.join(type(p).__name__ for p in primitive.until)})"
                )
                self.events.append(Event(self.time, "timeout", None, primitive.name))
            break
        if self.status == "running" and self.pc >= len(self.program):
            self.status, self.reason = "done", f"finished all {len(self.program)} primitives"
        if self.status != "running":
            for drive in self.drives.values():
                drive.frozen = True
        return defines

    def abort(self, reason: str) -> None:
        if self.status == "running":
            self.status, self.reason = "aborted", reason

    def _begin_wait(self, primitive: Delay | Condition) -> None:
        self._wait_start = self.time
        self._timers.clear()
        self._anchors.clear()
        if isinstance(primitive, Delay):
            self._wait_timeout = math.inf
        elif primitive.timeout is not None:
            self._wait_timeout = primitive.timeout
        else:
            self._wait_timeout = TIMEOUT_FACTOR * self._expected(primitive) + TIMEOUT_MARGIN_S

    def _end_wait(self) -> None:
        self.pc += 1
        self._wait_start = None

    def _expected(self, condition: Condition) -> float:
        """How long the condition should take: the longest remaining ramp, a trajectory's rest, an Elapsed."""
        longest = 0.0
        for predicate in condition.until:
            if isinstance(predicate, Elapsed):
                longest = max(longest, predicate.duration)
            elif isinstance(predicate, FollowEnded):
                for drive in self.drives.values():
                    if drive.mode == "follow":
                        longest = max(longest, drive.follow.remaining_s)
            for joint in predicate.watched():
                drive = self.drives.get(joint)
                if drive is not None and drive.mode == "ramp" and drive.speed > 0.0:
                    longest = max(longest, abs(drive.target - drive.setpoint) / drive.speed)
        return longest

    # ── Commands ───────────────────────────────────────────────────────────────────────────────────────────────

    def _start_of(self, joint: str, measured: Mapping[str, float]) -> float:
        """Where a new setpoint starts: the previous setpoint if that drive was settled, else the measured position.

        WHY: a hold or move after a hold must not jump by the sag (bam_homing H3). WHY "within the Stall lead of
        it": a ramp that ended pressing on a stop is settled on paper, but its setpoint sits past the stop; a
        back-off or a rest from there would never leave the stop."""
        drive = self.drives.get(joint)
        inherited = drive is None and joint in self._previous
        drive = drive or self._previous.get(joint)
        # WHY a moving ramp counts when inherited: a preempt must start from the old setpoint, not the measured
        # pose that lags it (S009's A01 rule; robot_msgs' own test caught a 2.2° jump at 1 rad/s)
        usable = drive is not None and (
            drive.settled or (inherited and drive.mode in ("ramp", "velocity"))
        )
        if (
            usable
            and math.isfinite(drive.setpoint)
            and abs(drive.setpoint - measured[joint]) < STALL.lead
        ):
            return drive.setpoint
        return measured[joint]

    def _apply(
        self,
        command: Command,
        joint_state: Any,
        measured: dict[str, float],
        defines: dict[str, float],
    ) -> None:
        if command.reduce is not None:
            self._reduce(command, measured, defines)
            if self.status != "running":
                return
        follows: dict[int, list[JointCommand]] = {}
        for joint_command in command.commands:
            if joint_command.mode == "follow":
                follows.setdefault(id(joint_command.trajectory), []).append(joint_command)
                continue
            targets = self.everyone if joint_command.joint == ALL_JOINTS else (joint_command.joint,)
            for joint in targets:
                self._apply_one(joint, joint_command, measured, defines)
        for group in follows.values():
            joints = tuple(c.joint for c in group)
            caps = {c.max_effort for c in group}
            follow = _Follow(
                group[0].trajectory, joints, caps.pop() if len(caps) == 1 else None, self.follower
            )
            follow.advance(joint_state, 0.0)
            for joint in joints:
                self.drives[joint] = _Drive("follow", follow=follow)

    def _apply_one(
        self,
        joint: str,
        command: JointCommand,
        measured: dict[str, float],
        defines: dict[str, float],
    ) -> None:
        mode = command.mode
        ff = float(command.feedforward_effort)
        cap = math.nan if command.max_effort is None else float(command.max_effort)
        if mode in ("limp", "stop", "brake"):
            self.drives[joint] = _Drive(mode)
        elif mode == "hold":
            here = self._start_of(joint, measured)
            self.drives[joint] = _Drive("hold", here, here, max_effort=cap, feedforward=ff)
        elif mode in ("position", "move_by"):
            start = self._start_of(joint, measured)
            target = (
                float(command.position) if mode == "position" else start + float(command.distance)
            )
            self.drives[joint] = _Drive(
                "ramp", start, target, float(command.speed), max_effort=cap, feedforward=ff
            )
        elif mode == "velocity":
            start = self._start_of(joint, measured)
            drive = _Drive(
                "velocity",
                start,
                start,
                velocity=float(command.velocity),
                max_effort=cap,
                feedforward=ff,
            )
            drive.started = True
            self.drives[joint] = drive
        elif mode == "effort":
            self.drives[joint] = _Drive("effort", effort=ff, max_effort=cap)
        elif mode == "define":
            self._define(joint, float(command.position), measured, defines, cap)
        else:
            raise ValueError(f"JointCommandController: no JointCommand mode {mode!r}")

    def _define(
        self,
        joint: str,
        value: float,
        measured: dict[str, float],
        defines: dict[str, float],
        cap: float = math.nan,
    ) -> None:
        shift = value - measured[joint]
        defines[joint] = value
        measured[joint] = value
        drive = self.drives.get(joint)
        if drive is None:
            previous = self._previous.get(joint)
            if (
                previous is not None
                and previous.mode in ("hold", "ramp")
                and math.isfinite(previous.setpoint)
            ):
                drive = _Drive(
                    "hold",
                    previous.setpoint,
                    previous.setpoint,
                    max_effort=previous.max_effort,
                    feedforward=previous.feedforward,
                )
            else:
                here = value - shift
                drive = _Drive("hold", here, here, max_effort=cap)
            self.drives[joint] = drive
        if drive.mode in ("hold", "ramp", "velocity"):
            drive.shift(shift)
        prefix = f"touch:{joint}:"
        self.records = {
            tag: [p + shift for p in values] if tag.startswith(prefix) else values
            for tag, values in self.records.items()
        }

    def _reduce(
        self, command: Command, measured: dict[str, float], defines: dict[str, float]
    ) -> None:
        reduce = command.reduce
        joint = reduce.joint
        touches = tuple(self.records.get(reduce.tag, ()))
        if not touches:
            self.abort(f"no touch recorded on {joint}")
            return
        mean = float(np.mean(touches))
        spread = float(max(touches) - min(touches))
        why = ""
        if (reduce.define is not None or reduce.verify) and spread > reduce.tolerance + 1e-12:
            why = (
                f"refused: the spread of {joint}'s {len(touches)} touches is {math.degrees(spread):.4f}° "
                f"({spread:.5f} rad), over the {reduce.tolerance:g} rad tolerance"
            )
        elif reduce.expected is not None and abs(mean - reduce.expected) > reduce.tolerance + 1e-12:
            why = (
                f"refused: {joint}'s stop reads {mean:+.5f} rad, {math.degrees(mean - reduce.expected):+.4f}° from "
                f"the expected {reduce.expected:+.5f} rad (tolerance {reduce.tolerance:g} rad)"
            )
        if why:
            self.hardstops[joint] = HardstopReport(joint, touches, mean, spread, None, why)
            self.abort(why)
            return
        report = HardstopReport(joint, touches, mean, spread)
        if reduce.define is not None:
            # the joint rests at its last touch; define so the MEAN of the touches reads `define`
            self._define(joint, reduce.define + (measured[joint] - mean), measured, defines)
            report = HardstopReport(joint, touches, mean, spread, reduce.define)
        self.hardstops[joint] = report

    # ── Conditions ─────────────────────────────────────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        condition: Condition,
        measured: Mapping[str, float],
        velocity: Mapping[str, float],
        dt: float,
    ) -> str | None:
        """ "done", a fail reason, or None (keep waiting)."""
        held = [
            self._holds(index, p, measured, velocity, dt) for index, p in enumerate(condition.until)
        ]
        if all(held) if condition.mode == "all" else any(held):
            return "done"
        for index, fail in enumerate(condition.fail):
            if self._holds(("fail", index), fail.predicate, measured, velocity, dt):
                return fail.reason
        return None

    def _dwelt(self, key: Any, ok: bool, dt: float, dwell: float) -> bool:
        self._timers[key] = self._timers.get(key, 0.0) + dt if ok else 0.0
        return self._timers[key] + 1e-9 >= dwell

    def _holds(
        self,
        key: Any,
        predicate: Any,
        measured: Mapping[str, float],
        velocity: Mapping[str, float],
        dt: float,
    ) -> bool:
        if isinstance(predicate, Elapsed):
            return self.time - self._wait_start + 1e-9 >= predicate.duration
        if isinstance(predicate, Reached):
            ok = True
            for joint in predicate.joints:
                drive = self.drives.get(joint)
                here = (
                    drive is not None
                    and drive.setpoint == drive.target
                    and abs(measured[joint] - drive.target) <= predicate.tolerance
                    and abs(velocity[joint]) <= predicate.speed
                )
                ok = self._dwelt((key, joint), here, dt, predicate.dwell) and ok
            return ok
        if isinstance(predicate, Settled):
            ok = True
            for joint in predicate.joints:
                ok = (
                    self._dwelt(
                        (key, joint), abs(velocity[joint]) <= predicate.speed, dt, predicate.dwell
                    )
                    and ok
                )
            return ok
        if isinstance(predicate, InCollision):
            if self.has_detector:
                touching = bool(self.contacts.get(predicate.joint, False))
                return self._dwelt(
                    key, self._still(key, measured[predicate.joint], touching), dt, STALL.dwell
                )
            if predicate.effort is not None:
                return self._effort_stall(
                    key, predicate.joint, predicate.effort, STALL.speed, STALL.dwell, velocity, dt
                )
            return self._lead_stall(key, predicate.joint, STALL, measured, velocity, dt)
        if isinstance(predicate, Stall):
            return self._lead_stall(key, predicate.joint, predicate, measured, velocity, dt)
        if isinstance(predicate, EffortStall):
            return self._effort_stall(
                key,
                predicate.joint,
                predicate.effort,
                predicate.speed,
                predicate.dwell,
                velocity,
                dt,
            )
        if isinstance(predicate, TravelEnd):
            drive = self.drives.get(predicate.joint)
            at_end = drive is not None and drive.mode == "ramp" and drive.setpoint == drive.target
            return self._dwelt(
                key, self._still(key, measured[predicate.joint], at_end), dt, STALL.dwell
            )
        if isinstance(predicate, FollowEnded):
            follows = {
                id(d.follow): d.follow for d in self.drives.values() if d.mode == "follow"
            }.values()
            return bool(follows) and all(
                f.ended
                and all(abs(measured[j] - f.last[j]) <= predicate.tolerance for j in f.column)
                for f in follows
            )
        raise ValueError(f"JointCommandController: no predicate {type(predicate).__name__}")

    def _still(self, key: Any, here: float, ok: bool) -> bool:
        """Moved less than the Stall speed x dwell since the dwell began (WHY: a per-tick speed reads encoder
        counts; one count reads as a 0.25 rad/s spike, S007)."""
        anchor = self._anchors.get(key)
        still = anchor is not None and abs(here - anchor) <= STALL.speed * STALL.dwell
        if not (ok and still):
            self._anchors[key] = here
        return ok and still

    def _lead_stall(
        self,
        key: Any,
        joint: str,
        rule: Stall,
        measured: Mapping[str, float],
        velocity: Mapping[str, float],
        dt: float,
    ) -> bool:
        drive = self.drives.get(joint)
        if drive is None or drive.mode not in ("ramp", "velocity"):
            return False
        lead = drive.direction * (drive.setpoint - measured[joint])
        return self._dwelt(
            key, lead > rule.lead and abs(velocity[joint]) < rule.speed, dt, rule.dwell
        )

    def _effort_stall(
        self,
        key: Any,
        joint: str,
        threshold: float,
        speed: float,
        dwell: float,
        velocity: Mapping[str, float],
        dt: float,
    ) -> bool:
        drive = self.drives.get(joint)
        effort = self.efforts.get(joint)
        if drive is None or effort is None or drive.mode not in ("ramp", "velocity"):
            return False
        pushing = drive.direction * effort > threshold
        return self._dwelt(key, pushing and abs(velocity[joint]) < speed, dt, dwell)

    # ── guard, ending ──────────────────────────────────────────────────────────────────────────────────────────

    def guard(self, joint_state: Any) -> None:
        limit = self.goal.max_speed
        if limit is None or self.status != "running" or joint_state.velocity is None:
            return
        velocity, effort = joint_state.velocity, joint_state.effort
        for joint, drive in self.drives.items():
            if drive.mode in ("limp", "stop", "brake"):
                continue
            i = joint_state.index_of(joint)
            speed = float(velocity[i])
            # WHY only a drive pushing the way it runs (five-bar 3° proof, 2026-11-10): gravity may move the arm
            # on purpose with a drive braking it; a runaway is a drive doing positive work
            pushing = effort is None or float(effort[i]) * speed > 0.0
            if abs(speed) <= limit or not pushing:
                continue
            message = f"{joint} was driven faster than {limit:g} rad/s ({speed:+.2f}, its drive pushing that way)"
            if self.goal.guard_policy == "abort":
                self.abort(f"guard: {message}")
                self.events.append(Event(self.time, "guard", joint, message))
                return
            if joint not in self._guarded:  # WHY once per joint: a log, not a log line every tick
                self._guarded.add(joint)
                self.events.append(Event(self.time, "guard", joint, message + "; continuing"))

    def result(self, status: str | None = None, reason: str | None = None) -> JointCommandResult:
        return JointCommandResult(
            status or self.status,
            self.reason if reason is None else reason,
            tuple(self.events),
            dict(self.hardstops),
        )

    def ending(self, joint: str, measured: float, why: str) -> Any:
        """What `joint` does once this sequence stops owning it badly: hold where it is if it was powered."""
        drive = self.drives.get(joint)
        if drive is None:
            return _Latched("limp", why)
        if drive.mode in ("stop", "brake", "limp"):
            return _Latched(drive.mode, why)
        if drive.mode == "effort":
            return _HeldAt(measured, abs(drive.effort), 0.0, why)
        if drive.mode == "follow":
            return _HeldAt(
                measured, math.nan if drive.follow.cap is None else drive.follow.cap, 0.0, why
            )
        return _HeldAt(measured, drive.max_effort, drive.feedforward, why)

    def pending_text(self) -> tuple[str, ...]:
        current = self.current
        if isinstance(current, Condition):
            return tuple(f"goal {self.id}: {type(p).__name__}" for p in current.until)
        if isinstance(current, Delay):
            return (f"goal {self.id}: Delay {current.seconds:g} s",)
        return ()


# ── per-joint writers other than a sequence ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _HeldAt:
    """A latched hold, latched once and never re-read (bam_homing H3)."""

    position: float
    max_effort: float
    feedforward: float
    why: str

    def as_drive(self) -> _Drive:
        return _Drive(
            "hold",
            self.position,
            self.position,
            max_effort=self.max_effort,
            feedforward=self.feedforward,
        )


@dataclass(frozen=True)
class _Sent:
    client: str
    position: float
    velocity: float
    feedforward: float
    max_effort: float
    stiffness: float
    damping: float
    mode: str


@dataclass(frozen=True)
class _FaultSlack:
    mode: str
    why: str


@dataclass(frozen=True)
class _Latched:
    mode: str
    why: str


class JointCommandController:
    """The one writer to `joint_names`; see the module docstring.

    - `slack` (joint -> `stop` | `brake`): what a faulted joint goes to (default `stop`; brake on a wrist or a
      claw that must not drop, PEP 0047). A board that reports a fault itself always gets `stop`.
    - `detector`: what `is_in_collision(joint)` reads when given (else the hand-rolled Stall / EffortStall rule).
    - `follower(trajectory, max_effort)`: the trajectory sub-controller for follow commands.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...] | list[str],
        *,
        slack: Mapping[str, str] | None = None,
        detector: Detector | None = None,
        follower: Callable[[Any, float | None], Follower] = SampledFollower,
    ) -> None:
        self.joint_names = tuple(str(j) for j in joint_names)
        self._detector = detector
        self._follower = follower
        self._last_command: ServoCommand | None = None
        self._slack = {joint: "stop" for joint in self.joint_names}
        for joint, mode in dict(slack or {}).items():
            if joint in self._slack:
                if mode not in ("stop", "brake"):
                    raise ValueError(f"slack {joint!r}: {mode!r} is not stop or brake")
                self._slack[joint] = mode
        self._writer: dict[str, Any] = {name: None for name in self.joint_names}
        self._running: list[_Run] = []
        self._results: dict[int, JointCommandResult] = {}
        self._last_goal: int | None = None
        self._next_id = 0
        self._time = 0.0
        self._ticks = 0
        self._clearing: dict[str, int] = {}
        self._pending_end: list[tuple[_Run, frozenset[str]]] = []
        self._index: dict[str, int] | None = None
        self._index_names: tuple[str, ...] = ()

    # ── the five calls ─────────────────────────────────────────────────────────────────────────────────────────

    @property
    def status(self) -> ControllerStatus:
        """`idle` (nothing submitted, or the last goal was stopped / preempted / rejected), `executing`,
        `done` (the last goal finished), `fault` (it aborted or a board faulted: the joints hold or are slack)."""
        if self._running:
            return "executing"
        if self._last_goal is None:
            return "idle"
        last = self._results.get(self._last_goal)
        if last is None:
            return "executing"
        return {"done": "done", "aborted": "fault", "faulted": "fault"}.get(last.status, "idle")

    def submit(self, sequence: JointCommandSequence, *, client: str = "") -> int:
        run_id = self._next_id
        self._next_id += 1
        self._last_goal = run_id
        try:
            if not isinstance(sequence, JointCommandSequence):
                raise TypeError(
                    f"submit takes a JointCommandSequence, got {type(sequence).__name__}"
                )
            joints = self._claims(sequence)
            unknown = sorted(set(joints) - set(self.joint_names))
            if unknown:
                raise ValueError(f"unknown joints {unknown}")
        except (ValueError, TypeError) as error:
            self._results[run_id] = JointCommandResult("rejected", str(error))
            return run_id
        previous: dict[str, _Drive] = {}
        for joint in joints:
            writer = self._writer.get(joint)
            if isinstance(writer, _Run) and joint in writer.drives:
                previous[joint] = writer.drives[
                    joint
                ]  # WHY: a preempting sequence starts from the old setpoint
            elif isinstance(writer, _HeldAt):
                previous[joint] = writer.as_drive()
        run = _Run(
            run_id,
            client,
            sequence,
            joints,
            self.joint_names,
            previous,
            self._detector is not None,
            self._follower,
        )
        taken = frozenset(joints)
        for old in [old for old in self._running if taken & set(old.joints)]:
            self._end(old, "preempted", f"preempted by goal {run_id} from {client}", taken)
        self._running.append(run)
        for joint in joints:
            self._writer[joint] = run
        return run_id

    def stop(self, *, client: str = "", joints: tuple[str, ...] | None = None) -> None:
        named = tuple(self.joint_names if joints is None else joints)
        for run in [run for run in self._running if set(named) & set(run.joints)]:
            self._end(run, "stopped", f"stop from {client} on {', '.join(named)}", frozenset(named))
        for joint in named:
            if joint in self._writer:
                self._writer[joint] = _Latched("stop", f"stop from {client}")

    def send(self, command: ServoCommand, *, client: str = "") -> bool:
        claimed = {joint for run in self._running for joint in run.joints}
        names = tuple(command.name)
        if any(joint in claimed or joint not in self._writer for joint in names):
            return False

        def value(values: Any, i: int, default: float) -> float:
            return default if values is None else float(values[i])

        for i, joint in enumerate(names):
            # WHY NaN for an absent field: the board default (0.8), so a send without a cap is legal
            self._writer[joint] = _Sent(
                client,
                float(command.position[i]),
                value(command.velocity, i, math.nan),
                value(command.feedforward_effort, i, math.nan),
                value(command.max_effort, i, math.nan),
                value(command.stiffness, i, math.nan),
                value(command.damping, i, math.nan),
                command.mode_of(joint),
            )
        return True

    def result(self, goal_id: int) -> JointCommandResult | None:
        return self._results.get(goal_id)

    def tick(
        self, state: Any, dt: float, *, board_mode: Mapping[str, Any] | None = None
    ) -> JointCommandTick:
        """`state`: a `ServoState` (or JointState) for at least these joints. `board_mode` (joint -> `"timeout"`
        for a watchdog-latched board, `"fault"` for a faulted one; anything else is ignored); a non-zero
        `state.fault` code is a fault too."""
        board_mode = board_mode or {}
        self._time += dt
        self._ticks += 1
        clearing = self._watchdog_latched(board_mode)
        measured = dict(zip(state.name, (float(v) for v in state.position), strict=True))
        if state.velocity is None:
            raise ValueError("JointCommandController needs measured velocities")
        velocity = dict(zip(state.name, (float(v) for v in state.velocity), strict=True))
        for run, taken in self._pending_end:
            self._release(run, measured, f"goal {run.id} {self._results[run.id].status}", taken)
        self._pending_end.clear()
        self._react_to_faults(state, board_mode, measured)

        defines: dict[str, float] = {}
        runs = list({id(w): w for w in self._writer.values() if isinstance(w, _Run)}.values())
        if runs:
            if self._index is None or self._index_names != state.name:
                self._index_names = state.name
                self._index = {name: i for i, name in enumerate(state.name)}
            joint_state = _JointView(state, self._index)
            contacts: Mapping[str, bool] = {}
            if self._detector is not None and any(run.waits_on_detector() for run in runs):
                contacts = dict(self._detector.contact(state, self._last_command, self._time))
            efforts = (
                {}
                if state.effort is None
                else dict(zip(state.name, (float(v) for v in state.effort), strict=True))
            )
            for run in runs:
                run.contacts = contacts
                run.efforts = efforts
                if clearing:
                    self._relatch(run, measured, clearing)
                view = dict(measured)
                defines.update(run.tick(joint_state, dt, view, velocity))
                run.guard(joint_state)
                self._settle(run, measured)
        command, log = self._wire(state, board_mode, clearing)
        self._last_command = command
        return JointCommandTick(command=command, defines=defines, log=log)

    # ── internals ──────────────────────────────────────────────────────────────────────────────────────────────

    def _claims(self, sequence: JointCommandSequence) -> tuple[str, ...]:
        """The joints a sequence claims while it runs: every joint it commands or watches (ALL_JOINTS = all)."""
        names = sequence.joints
        for primitive in sequence.primitives:
            if isinstance(primitive, Command) and ALL_JOINTS in primitive.joints:
                return self.joint_names
        return names

    def _wire(
        self, state: Any, board_mode: Mapping[str, Any], clearing: Mapping[str, str]
    ) -> tuple[ServoCommand, dict[str, Any]]:
        n = len(self.joint_names)
        nan = math.nan
        position = np.full(n, nan)
        command_velocity = np.zeros(n)
        max_effort = np.zeros(n)
        feedforward = np.zeros(n)
        stiffness = np.full(n, nan)
        damping = np.full(n, nan)
        modes: list[str] = []
        intents: dict[str, str] = {}
        for i, joint in enumerate(self.joint_names):
            writer = self._writer[joint]
            mode = "position"
            if isinstance(writer, _Run):
                drive = writer.drives.get(joint)
                if drive is None:
                    intents[joint] = f"goal {writer.id}: not commanded yet (limp)"
                else:
                    (
                        position[i],
                        command_velocity[i],
                        feedforward[i],
                        max_effort[i],
                        stiffness[i],
                        damping[i],
                    ) = drive.wire(joint)
                    mode = writer.mode_of(joint) or "position"
                    intents[joint] = f"goal {writer.id}: {drive.text()}"
            elif isinstance(writer, _Sent):
                position[i], command_velocity[i] = writer.position, writer.velocity
                feedforward[i], max_effort[i] = writer.feedforward, writer.max_effort
                stiffness[i], damping[i] = writer.stiffness, writer.damping
                mode = writer.mode
                intents[joint] = f"send from {writer.client}"
            elif isinstance(writer, _HeldAt):
                position[i], max_effort[i], feedforward[i] = (
                    writer.position,
                    writer.max_effort,
                    writer.feedforward,
                )
                intents[joint] = f"hold ({writer.why})"
            elif isinstance(writer, _Latched):
                mode = "position" if writer.mode == "limp" else writer.mode
                intents[joint] = f"{writer.mode} ({writer.why})"
            elif isinstance(writer, _FaultSlack):
                mode = "stop" if board_mode.get(joint) == "fault" else writer.mode
                intents[joint] = f"{mode} ({writer.why})"
            else:
                intents[joint] = "limp (no writer)"
            if joint in clearing and mode != "stop":
                mode = "stop"
                intents[joint] += "; stop to clear the watchdog latch"
            modes.append(mode)
        explicit = ~np.isnan(stiffness) | ~np.isnan(damping)
        header = getattr(state, "header", None) or Header()
        command = ServoCommand(
            name=self.joint_names,
            position=position,
            velocity=command_velocity,
            feedforward_effort=feedforward,
            stiffness=stiffness if explicit.any() else None,
            damping=damping if explicit.any() else None,
            max_effort=max_effort,
            state_stamp=header.stamp,
            mode=tuple(modes),
            header=Header(stamp=header.stamp, frame_id=header.frame_id),
        )
        running = self._running[-1] if self._running else None
        log = {
            "goal": None if running is None else running.id,
            "step": None if running is None else running.step_name,
            "goals": {run.id: run.step_name for run in self._running},
            "status": self.status,
            "intent": intents,
            "pending": tuple(text for run in self._running for text in run.pending_text()),
        }
        return command, log

    def _react_to_faults(
        self, state: Any, board_mode: Mapping[str, Any], measured: Mapping[str, float]
    ) -> None:
        codes = getattr(state, "fault", None)
        for i, joint in enumerate(state.name):
            code = 0 if codes is None else int(codes[i])
            if board_mode.get(joint) != "fault" and code == 0:
                continue
            writer = self._writer.get(joint, None)
            if isinstance(writer, _FaultSlack) or joint not in self._writer:
                continue
            why = f"{joint} board fault (mode {board_mode.get(joint)!r}, fault code {code})"
            if isinstance(writer, _Run) and writer.id not in self._results:
                self._results[writer.id] = writer.result("faulted", why)
                self._drop_running(writer)
                self._release(writer, measured, f"goal {writer.id} faulted", frozenset({joint}))
            self._writer[joint] = _FaultSlack(self._slack[joint], why)

    def _settle(self, run: _Run, measured: Mapping[str, float]) -> None:
        if run.status == "running" or run.id in self._results:
            return
        self._results[run.id] = run.result()
        self._drop_running(run)
        if run.status == "aborted":
            self._release(run, measured, f"goal {run.id} aborted", frozenset())
        # done: it keeps commanding its last drives until another writer takes the joints

    def _end(self, run: _Run, status: str, reason: str, taken: frozenset[str]) -> None:
        if run.id not in self._results:
            self._results[run.id] = run.result(status, reason)
        run.status = status
        self._drop_running(run)
        self._pending_end.append((run, taken))

    def _drop_running(self, run: _Run) -> None:
        if run in self._running:
            self._running.remove(run)

    def _release(
        self, run: _Run, measured: Mapping[str, float], why: str, taken: frozenset[str]
    ) -> None:
        for joint in run.joints:
            if joint not in taken and self._writer.get(joint) is run:
                self._writer[joint] = run.ending(joint, measured[joint], why)

    def _watchdog_latched(self, board_mode: Mapping[str, Any]) -> dict[str, str]:
        clearing: dict[str, str] = {}
        for name in self.joint_names:
            writer = self._writer[name]
            if board_mode.get(name) != "timeout" or (
                isinstance(writer, _Latched) and writer.mode == "stop"
            ):
                continue
            sent = self._clearing.get(name)
            if sent is not None and self._ticks - sent < WATCHDOG_STOP_COOLDOWN_TICKS:
                continue
            self._clearing[name] = self._ticks
            clearing[name] = "stop"
        return clearing

    @staticmethod
    def _relatch(run: _Run, measured: Mapping[str, float], joints: Mapping[str, str]) -> None:
        """WHY: a ramp kept running while the board ignored it (watchdog latch); resume it from where the arm is."""
        for name in joints:
            drive = run.drives.get(name)
            if (
                drive is not None
                and drive.mode in ("ramp", "velocity", "hold")
                and name in measured
            ):
                drive.setpoint = measured[name]
                if drive.mode != "ramp":
                    drive.target = measured[name]
