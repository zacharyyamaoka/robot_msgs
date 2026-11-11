"""JointCommandSequenceController: run a `JointCommandSequence`, one tick at a time.

Tickable like the joint trajectory controller: `tick(state, dt)` takes the measured `JointState`
and returns the `ServoCommand` to send plus the positions to define this tick. It holds no clock
and talks to no hardware, so the same controller runs on a twin and on the real drives.

`status` is `running`, `done` (the last step finished; it keeps commanding that step) or `aborted`
(a timeout or the speed guard; every joint limp from then on, `message` says why).

On the wire, a limp joint is `position` NaN with `max_effort` 0: no position target and no torque.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from robot_msgs.control.joint_command import (
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    Settled,
    Stall,
)
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.sensor import JointState
from robot_msgs.std import Header

Status = Literal["running", "done", "aborted"]


class JointController(Protocol):
    """What one controller that commands a set of joints offers. One controller drives at a time."""

    status: Status

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, Mapping[str, float]]: ...


@dataclass(frozen=True)
class StepEvent:
    """How a step ended: `outcome` is the condition that held (`"Stall j2"`, `"Reached"`,
    `"Settled"`, `"Elapsed"`), or `"timeout"` / `"guard"` when it aborted the sequence."""

    index: int
    name: str
    outcome: str
    started: float
    finished: float


@dataclass
class _Track:
    command: JointCommand
    setpoint: float = math.nan
    target: float = math.nan
    direction: int = 0
    lead_time: float = 0.0
    slow_time: float = 0.0
    reached_time: float = 0.0


@dataclass
class JointCommandSequenceController:
    """Runs `sequence`; see the module docstring. `time` is seconds since the first tick."""

    sequence: JointCommandSequence
    status: Status = "running"
    message: str = ""
    step_index: int = 0
    time: float = 0.0
    events: list[StepEvent] = field(default_factory=list)
    _tracks: dict[str, _Track] = field(default_factory=dict, repr=False)
    _step_started: float = 0.0
    _started: bool = False

    @property
    def step(self) -> JointCommandStep:
        return self.sequence.steps[min(self.step_index, len(self.sequence.steps) - 1)]

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        """Advance by `dt` seconds on `state`; return the command and the positions defined."""
        names = self.sequence.joint_names
        positions = {name: state.position_of(name) for name in names}
        if state.velocity is None:
            raise ValueError("JointCommandSequenceController needs measured velocities")
        velocities = {name: float(state.velocity[state.index_of(name)]) for name in names}
        efforts = (
            {name: float(state.effort[state.index_of(name)]) for name in names}
            if state.effort is not None
            else {}
        )
        defines: dict[str, float] = {}
        if self.status == "aborted":
            return self._limp(state), defines
        if not self._started:
            self._begin(positions)
        self.time += dt
        if self.status == "running":
            elapsed = self.time - self._step_started
            # WHY only a drive pushing the way it runs (five-bar 3° proof, 2026-11-10): homing lets
            # gravity move the arm on purpose, at up to 4.5 rad/s with a drive braking it. A guard
            # on speed alone aborted 4 of 2552 cells that home without it; a runaway is a drive
            # doing positive work, effort and velocity the same sign.
            powered = [c.joint for c in self.step.commands if c.mode != "limp"]
            fast = [
                name
                for name in powered
                if abs(velocities[name]) > self.sequence.max_speed
                and efforts.get(name, velocities[name]) * velocities[name] > 0.0
            ]
            if fast:
                self._abort(
                    "guard",
                    f"{fast[0]} was driven faster than {self.sequence.max_speed} "
                    f"({velocities[fast[0]]:+.2f}, its drive pushing that way) "
                    f"in step {self.step.name!r}",
                )
                return self._limp(state), defines
            outcome = self._done_by(positions, velocities, dt, elapsed)
            if outcome is not None:
                step = self.step
                self.events.append(
                    StepEvent(self.step_index, step.name, outcome, self._step_started, self.time)
                )
                for joint, value in step.define_position.items():
                    shift = value - positions[joint]
                    defines[joint] = value
                    positions[joint] = value
                    track = self._tracks.get(joint)
                    if track is not None:  # what is still commanded moves with the new frame
                        track.setpoint += shift
                        track.target += shift
                if self.step_index + 1 < len(self.sequence.steps):
                    self.step_index += 1
                    self._begin(positions)
                else:
                    self.status = "done"
                    self.message = f"finished all {len(self.sequence.steps)} steps"
            elif elapsed >= self.step.timeout:
                self._abort(
                    "timeout", f"step {self.step.name!r} was not done within {self.step.timeout} s"
                )
                return self._limp(state), defines
        return self._command(state, dt), defines

    # ── internals ────────────────────────────────────────────────────────────

    def _begin(self, positions: Mapping[str, float]) -> None:
        self._started = True
        self._step_started = self.time
        previous = self._tracks
        self._tracks = {}
        for command in self.step.commands:
            here = positions[command.joint]
            track = _Track(command=command, setpoint=here, target=here)
            if command.mode == "hold":
                # WHY a hold after a hold keeps its setpoint: re-latching at the measured (sagged)
                # pose every step walks the arm down one sag per step (bam_homing H3, item 51).
                old = previous.get(command.joint)
                if old is not None and old.command.mode == "hold":
                    track.setpoint = track.target = old.setpoint
            elif command.mode == "move_by":
                track.target = here + float(command.distance)  # type: ignore[arg-type]
            elif command.mode == "position":
                track.target = float(command.position)  # type: ignore[arg-type]
            if track.target != track.setpoint:
                track.direction = 1 if track.target > track.setpoint else -1
            self._tracks[command.joint] = track

    def _command(self, state: JointState, dt: float) -> ServoCommand:
        names = self.sequence.joint_names
        position = np.full(len(names), np.nan)
        max_effort = np.zeros(len(names))
        feedforward = np.zeros(len(names))
        for index, name in enumerate(names):
            track = self._tracks.get(name)
            if track is None or track.command.mode == "limp":
                continue
            command = track.command
            if command.mode in ("move_by", "position"):
                remaining = track.target - track.setpoint
                step = float(command.speed) * dt  # type: ignore[arg-type]
                track.setpoint = (
                    track.target
                    if abs(remaining) <= step
                    else track.setpoint + math.copysign(step, remaining)
                )
            position[index] = track.setpoint
            max_effort[index] = float(command.max_effort)  # type: ignore[arg-type]
            feedforward[index] = command.feedforward_effort
        return self._servo_command(state, position, max_effort, feedforward)

    def _limp(self, state: JointState) -> ServoCommand:
        size = len(self.sequence.joint_names)
        return self._servo_command(state, np.full(size, np.nan), np.zeros(size), np.zeros(size))

    def _servo_command(self, state, position, max_effort, feedforward) -> ServoCommand:
        # WHY velocity 0, not the ramp rate: the five-bar proof (1084/1084 cells) ran pure position
        # setpoints; a velocity target would change what was proved.
        return ServoCommand(
            name=self.sequence.joint_names,
            position=position,
            velocity=np.zeros(len(position)),
            feedforward_effort=feedforward,
            max_effort=max_effort,
            state_stamp=state.header.stamp,
            header=Header(stamp=state.header.stamp, frame_id=state.header.frame_id),
        )

    def _done_by(self, positions, velocities, dt: float, elapsed: float) -> str | None:
        for condition in self.step.done_when:
            if isinstance(condition, Elapsed):
                if elapsed + 1e-9 >= condition.duration:
                    return "Elapsed"
            elif isinstance(condition, Stall):
                track = self._tracks[condition.joint]
                lead = (track.direction or 1) * (track.setpoint - positions[condition.joint])
                slow = abs(velocities[condition.joint]) < condition.speed
                track.lead_time = track.lead_time + dt if (lead > condition.lead and slow) else 0.0
                if track.lead_time + 1e-9 >= condition.dwell:
                    return f"Stall {condition.joint}"
            elif isinstance(condition, Reached):
                ok = True
                for name in condition.joints:
                    track = self._tracks[name]
                    here = (
                        track.setpoint == track.target
                        and abs(positions[name] - track.target) <= condition.tolerance
                        and abs(velocities[name]) <= condition.speed
                    )
                    track.reached_time = track.reached_time + dt if here else 0.0
                    ok = ok and track.reached_time + 1e-9 >= condition.dwell
                if ok:
                    return "Reached"
            elif isinstance(condition, Settled):
                ok = True
                for name in condition.joints:
                    track = self._tracks[name]
                    slow = abs(velocities[name]) <= condition.speed
                    track.slow_time = track.slow_time + dt if slow else 0.0
                    ok = ok and track.slow_time + 1e-9 >= condition.dwell
                if ok:
                    return "Settled"
        return None

    def _abort(self, outcome: str, message: str) -> None:
        self.events.append(
            StepEvent(self.step_index, self.step.name, outcome, self._step_started, self.time)
        )
        self.status, self.message = "aborted", message
