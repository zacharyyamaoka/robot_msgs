"""JointCommand, its done-when conditions, and JointCommandSequence: a routine described as data.

A sequence is an ordered list of steps. Each step gives every joint it names one `JointCommand`
(limp, hold, move by, move to), says when the step is done (any of its `done_when`), and aborts the
whole sequence if it is not done within `timeout`. When a step finishes, the joints in its
`define_position` are told "your position is now this" (a homing reference, moteus's
`set_output_exact`). `JointCommandSequenceController` runs one.

WHY a sequence of commands with completion conditions, not a trajectory (Zach, 2026-11-10, the
five-bar blind homing): homing drives into contacts and waits for stalls, so when a step ends is
measured, not planned. The description (this data) stays apart from the execution (the
controller), the "story teller" pattern of bam_controllers.

Units are SI: rad (or m) for positions and distances, rad/s for speeds, N·m (or N) for efforts,
s for durations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from robot_msgs._core import FrozenDict, Msg, set_field

Mode = Literal["limp", "hold", "move_by", "position"]
MODES: tuple[str, ...] = ("limp", "hold", "move_by", "position")


def _positive(owner: str, name: str, value: float | None) -> float:
    if value is None or not (math.isfinite(float(value)) and float(value) > 0.0):
        raise ValueError(f"{owner}.{name} must be finite and positive, got {value!r}")
    return float(value)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointCommand(Msg):
    """What one joint does during one step.

    - `limp`: no torque.
    - `hold`: hold where the step starts (a hold after a hold keeps the earlier setpoint).
    - `move_by`: ramp the setpoint by `distance` from where the step starts, at `speed`.
    - `position`: ramp the setpoint to `position`, at `speed` (needs a defined position).

    Every mode except `limp` needs `max_effort` (the drive's output cap, N·m or N) and may add a
    constant `feedforward_effort` (e.g. the static gravity torque at a waypoint).
    """

    joint: str
    mode: Mode
    position: float | None = None
    distance: float | None = None
    speed: float | None = None
    max_effort: float | None = None
    feedforward_effort: float = 0.0

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        if self.mode not in MODES:
            raise ValueError(f"JointCommand {self.joint}: mode {self.mode!r} is not one of {MODES}")
        for name in ("position", "distance", "speed", "max_effort"):
            value = getattr(self, name)
            if value is not None:
                set_field(self, name, float(value))
        set_field(self, "feedforward_effort", float(self.feedforward_effort))
        if self.mode == "limp":
            return
        owner = f"JointCommand {self.joint}"
        _positive(owner, "max_effort", self.max_effort)
        if not math.isfinite(self.feedforward_effort):
            raise ValueError(f"{owner}: feedforward_effort must be finite")
        if self.mode in ("move_by", "position"):
            _positive(owner, "speed", self.speed)
        if self.mode == "move_by" and (
            self.distance is None or not math.isfinite(self.distance) or self.distance == 0.0
        ):
            raise ValueError(f"{owner}: move_by needs a finite, non-zero distance")
        if self.mode == "position" and (self.position is None or not math.isfinite(self.position)):
            raise ValueError(f"{owner}: position mode needs a finite position")

    @classmethod
    def limp(cls, joint: str) -> JointCommand:
        """No torque on `joint`."""
        return cls(joint=joint, mode="limp")

    @classmethod
    def hold(
        cls, joint: str, *, max_effort: float, feedforward_effort: float = 0.0
    ) -> JointCommand:
        """Hold `joint` where the step starts, under `max_effort`."""
        return cls(
            joint=joint, mode="hold", max_effort=max_effort, feedforward_effort=feedforward_effort
        )

    @classmethod
    def move_by(
        cls,
        joint: str,
        distance: float,
        *,
        speed: float,
        max_effort: float,
        feedforward_effort: float = 0.0,
    ) -> JointCommand:
        """Ramp `joint` by `distance` from where the step starts, at `speed`, under `max_effort`."""
        return cls(
            joint=joint,
            mode="move_by",
            distance=distance,
            speed=speed,
            max_effort=max_effort,
            feedforward_effort=feedforward_effort,
        )

    @classmethod
    def to(
        cls,
        joint: str,
        position: float,
        *,
        speed: float,
        max_effort: float,
        feedforward_effort: float = 0.0,
    ) -> JointCommand:
        """Ramp `joint` to `position`, at `speed`, under `max_effort`."""
        return cls(
            joint=joint,
            mode="position",
            position=position,
            speed=speed,
            max_effort=max_effort,
            feedforward_effort=feedforward_effort,
        )


# ── done-when conditions ─────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Elapsed(Msg):
    """Done when the step has run `duration` seconds."""

    duration: float

    def __post_init__(self) -> None:
        set_field(self, "duration", float(self.duration))
        if not (math.isfinite(self.duration) and self.duration >= 0.0):
            raise ValueError("Elapsed.duration must be finite and >= 0")

    def watched(self) -> tuple[str, ...]:
        """The joints this condition reads (none)."""
        return ()


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Reached(Msg):
    """Done when every listed joint's setpoint is at its target and the joint is within
    `tolerance` of it, slower than `speed`, for `dwell` seconds."""

    joints: tuple[str, ...]
    tolerance: float
    speed: float = 0.01
    dwell: float = 0.05

    def __post_init__(self) -> None:
        set_field(self, "joints", tuple(str(j) for j in self.joints))
        for name in ("tolerance", "speed", "dwell"):
            set_field(self, name, _positive("Reached", name, getattr(self, name)))

    def watched(self) -> tuple[str, ...]:
        return self.joints


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Settled(Msg):
    """Done when every listed joint has moved slower than `speed` for `dwell` seconds."""

    joints: tuple[str, ...]
    speed: float
    dwell: float

    def __post_init__(self) -> None:
        set_field(self, "joints", tuple(str(j) for j in self.joints))
        for name in ("speed", "dwell"):
            set_field(self, name, _positive("Settled", name, getattr(self, name)))

    def watched(self) -> tuple[str, ...]:
        return self.joints


# WHY these defaults (the five-bar blind homing, 550/550 twin cells, 2026-11-10): 0.06 rad of lead
# is well past a PD's sag under homing caps, so a joint still following its setpoint never reads as
# stalled; 0.01 rad/s is far under the slowest seek (0.15 rad/s); 0.2 s rides out stick-slip.
@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Stall(Msg):
    """Done when `joint`'s setpoint leads it by more than `lead` (in the direction of motion)
    while it moves slower than `speed`, for `dwell` seconds: something stopped it."""

    joint: str
    lead: float = 0.06
    speed: float = 0.01
    dwell: float = 0.2

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        for name in ("lead", "speed", "dwell"):
            set_field(self, name, _positive("Stall", name, getattr(self, name)))

    def watched(self) -> tuple[str, ...]:
        return (self.joint,)


# WHY an effort threshold beside the lead rule (the full arm's J1, 2026-11-14): a seek against gravity whose
# feed-forward is off, or whose cap sits under its load plus friction, stops short where its drive saturates on
# gravity - and a lead rule reads that as the stop (the rack recorded one at -88 deg, 2 deg short, at 7 N.m with
# no feed-forward). Pressing on a stop is the one place the drive's effort climbs past the load plus a margin while
# the joint stands still; a cap under `effort` then never defines a stop - the step times out instead.
@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class EffortStall(Msg):
    """Done when `joint` moves slower than `speed` while its measured effort pushes the way its setpoint runs
    with more than `effort` (N·m or N), for `dwell` seconds: it is pressing something. Size `effort` above the
    largest load the seek carries plus a margin, and under its cap. A state without effort never satisfies it."""

    joint: str
    effort: float
    speed: float = 0.01
    dwell: float = 0.2

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        for name in ("effort", "speed", "dwell"):
            set_field(self, name, _positive("EffortStall", name, getattr(self, name)))

    def watched(self) -> tuple[str, ...]:
        return (self.joint,)


DoneWhen = Elapsed | Reached | Settled | Stall | EffortStall
"""A step's completion condition."""


# ── steps and the sequence ───────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointCommandStep(Msg):
    """One step: a `JointCommand` per joint it names, done when ANY of `done_when` holds, the whole
    sequence aborted if not done within `timeout` (s). Joints the step does not name go limp.

    `define_position` (joint → rad or m) is applied when the step finishes: each listed joint's
    current position is declared to be that value (a homing reference).
    """

    name: str
    commands: tuple[JointCommand, ...]
    done_when: tuple[DoneWhen, ...]
    timeout: float
    define_position: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_field(self, "name", str(self.name))
        set_field(self, "commands", tuple(self.commands))
        set_field(self, "done_when", tuple(self.done_when))
        set_field(
            self,
            "define_position",
            FrozenDict({str(k): float(v) for k, v in dict(self.define_position).items()}),
        )
        if not self.commands or not self.done_when:
            raise ValueError(f"step {self.name!r}: a step needs commands and a done_when")
        joints = [command.joint for command in self.commands]
        if len(joints) != len(set(joints)):
            raise ValueError(f"step {self.name!r}: one command per joint, got {joints}")
        set_field(self, "timeout", _positive(f"step {self.name!r}", "timeout", self.timeout))
        for condition in self.done_when:
            unknown = set(condition.watched()) - set(joints)
            if unknown:
                raise ValueError(
                    f"step {self.name!r}: its done_when watches {sorted(unknown)}, "
                    "which it gives no command"
                )
        for joint, value in self.define_position.items():
            if joint not in joints or not math.isfinite(value):
                raise ValueError(f"step {self.name!r}: cannot define {joint} as {value!r}")

    def command(self, joint: str) -> JointCommand:
        for command in self.commands:
            if command.joint == joint:
                return command
        raise KeyError(f"step {self.name!r} has no command for {joint!r}")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointCommandSequence(Msg):
    """The steps in order for joints `joint_names`, plus one abort guard for every step.

    `max_speed` (rad/s or m/s): the sequence aborts when a POWERED joint is driven faster than
    this (its effort and velocity the same sign). A limp joint, or one its drive is braking, is
    moving under gravity, which a homing routine may use on purpose, and may go faster.
    """

    joint_names: tuple[str, ...]
    steps: tuple[JointCommandStep, ...]
    name: str = ""
    max_speed: float = 3.0

    def __post_init__(self) -> None:
        set_field(self, "joint_names", tuple(str(j) for j in self.joint_names))
        set_field(self, "steps", tuple(self.steps))
        set_field(self, "name", str(self.name))
        if not self.steps:
            raise ValueError("JointCommandSequence needs at least one step")
        known = set(self.joint_names)
        if len(known) != len(self.joint_names):
            raise ValueError(f"JointCommandSequence.joint_names has duplicates: {self.joint_names}")
        for step in self.steps:
            unknown = {command.joint for command in step.commands} - known
            if unknown:
                raise ValueError(f"step {step.name!r}: unknown joints {sorted(unknown)}")
        set_field(self, "max_speed", _positive("JointCommandSequence", "max_speed", self.max_speed))

    def __len__(self) -> int:
        return len(self.steps)
