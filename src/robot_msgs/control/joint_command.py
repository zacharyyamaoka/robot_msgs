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

Since 0.8 a sequence may also be written as the three primitive step kinds Zach designed in Dec 2025
(`Command`, `Delay`, `Condition`, in `robot_msgs.control.primitives`), built by the helper FUNCTIONS in
`robot_msgs.control.helpers` and run by `JointCommandController`. A `JointCommandStep` is itself sugar for
primitives (`JointCommandSequence.primitives`). `JointCommand` grew the modes a trajectory cannot carry:
`velocity`, `effort`, `stop`, `brake`, `define`, `follow`.

Units are SI: rad (or m) for positions and distances, rad/s for speeds, N·m (or N) for efforts,
s for durations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from robot_msgs._core import FrozenDict, Msg, set_field

if TYPE_CHECKING:
    from robot_msgs.trajectory import JointTrajectory

# WHY Any for `trajectory` below, not JointTrajectory: robot_msgs.trajectory imports control, and
# from_dict reads the type hints at run time; a message under Any decodes by its "_type" key
Mode = Literal[
    "limp", "hold", "move_by", "position", "velocity", "effort", "stop", "brake", "define", "follow"
]
MODES: tuple[str, ...] = (
    "limp", "hold", "move_by", "position", "velocity", "effort", "stop", "brake", "define", "follow"
)
SEQUENCE_CONTROLLER_MODES: tuple[str, ...] = ("limp", "hold", "move_by", "position")
"""The modes `JointCommandSequenceController` (0.3-0.7) runs; the rest need `JointCommandController`."""
ALL_JOINTS = "*"
"""`JointCommand(joint=ALL_JOINTS, mode="stop")`: every joint the controller drives (limp / stop / brake only)."""


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

    `hold`, `move_by` and `position` need `max_effort` (the drive's output cap, N·m or N) and may add a
    constant `feedforward_effort` (e.g. the static gravity torque at a waypoint).

    Since 0.8 (run by `JointCommandController` only):

    - `velocity`: a host-ramped setpoint leading at `velocity` (rad/s) under `max_effort`, no end.
    - `effort`: pure torque `feedforward_effort` (no position target, zero gains); cap `max_effort`
      (None = |effort|).
    - `stop` / `brake`: the board's stop (driver off; clears a fault and a watchdog latch) or brake
      (windings shorted). `joint` may be `ALL_JOINTS`, as may `limp`'s.
    - `define`: "your position is now `position`" (moteus set_output_exact); the joint keeps what it
      was doing, re-framed. With nothing commanded it holds under `max_effort` (None = board cap).
    - `follow`: play `trajectory` (t = 0 when the Command runs) on this joint under `max_effort`
      (None = board cap). Joints of one Command that share a trajectory follow it together.
    """

    joint: str
    mode: Mode
    position: float | None = None
    distance: float | None = None
    speed: float | None = None
    max_effort: float | None = None
    feedforward_effort: float = 0.0
    velocity: float | None = None
    trajectory: Any = None  # a JointTrajectory (Any: robot_msgs.trajectory imports control)

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        if self.mode not in MODES:
            raise ValueError(f"JointCommand {self.joint}: mode {self.mode!r} is not one of {MODES}")
        for name in ("position", "distance", "speed", "max_effort", "velocity"):
            value = getattr(self, name)
            if value is not None:
                set_field(self, name, float(value))
        set_field(self, "feedforward_effort", float(self.feedforward_effort))
        owner = f"JointCommand {self.joint}"
        if self.joint == ALL_JOINTS and self.mode not in ("limp", "stop", "brake"):
            raise ValueError(f"{owner}: only limp, stop and brake may name every joint")
        if self.mode in ("limp", "stop", "brake"):
            return
        if not math.isfinite(self.feedforward_effort):
            raise ValueError(f"{owner}: feedforward_effort must be finite")
        if self.mode in ("effort", "define", "follow"):
            if self.max_effort is not None:
                _positive(owner, "max_effort", self.max_effort)
            if self.mode == "define" and (self.position is None or not math.isfinite(self.position)):
                raise ValueError(f"{owner}: define needs a finite position")
            if self.mode == "follow":
                names = getattr(self.trajectory, "joint_names", None)
                if names is None or self.joint not in names:
                    raise ValueError(f"{owner}: follow needs a JointTrajectory that names {self.joint!r}")
            return
        _positive(owner, "max_effort", self.max_effort)
        if self.mode == "velocity":
            if self.velocity is None or not math.isfinite(self.velocity):
                raise ValueError(f"{owner}: velocity mode needs a finite velocity")
            return
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

    @classmethod
    def at_velocity(
        cls, joint: str, velocity: float, *, max_effort: float, feedforward_effort: float = 0.0
    ) -> JointCommand:
        """Seek `joint` at `velocity` (a host ramp, no end), under `max_effort`."""
        return cls(
            joint=joint,
            mode="velocity",
            velocity=velocity,
            max_effort=max_effort,
            feedforward_effort=feedforward_effort,
        )

    @classmethod
    def effort(cls, joint: str, effort: float, *, max_effort: float | None = None) -> JointCommand:
        """Pure torque `effort` on `joint` (zero gains, no position target)."""
        return cls(joint=joint, mode="effort", feedforward_effort=effort, max_effort=max_effort)

    @classmethod
    def stop(cls, joint: str = ALL_JOINTS) -> JointCommand:
        """The board's stop on `joint` (default: every joint)."""
        return cls(joint=joint, mode="stop")

    @classmethod
    def brake(cls, joint: str = ALL_JOINTS) -> JointCommand:
        """The board's brake on `joint` (default: every joint)."""
        return cls(joint=joint, mode="brake")

    @classmethod
    def define(cls, joint: str, position: float, *, max_effort: float | None = None) -> JointCommand:
        """'Your position is now `position`' (set_output_exact)."""
        return cls(joint=joint, mode="define", position=position, max_effort=max_effort)

    @classmethod
    def follow(
        cls, joint: str, trajectory: JointTrajectory, *, max_effort: float | None = None
    ) -> JointCommand:
        """Play `trajectory`'s column for `joint`."""
        return cls(joint=joint, mode="follow", trajectory=trajectory, max_effort=max_effort)


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
    """A recipe read top to bottom: `steps` are `JointCommandStep`s or (0.8) the primitives `Command`,
    `Delay`, `Condition`; lists inside `steps` are flattened, so helper calls compose directly:
    `JointCommandSequence(steps=[find_hardstop(...), move_to(...)])`.

    `joint_names`: the joints it drives; `()` (0.8) = every joint it names. A `JointCommandStep`'s
    unnamed joints go limp, so a sequence of steps needs `joint_names`.

    `max_speed` (rad/s or m/s, None = no guard): a POWERED joint driven faster than this (its effort
    and velocity the same sign) trips the guard. A limp joint, or one its drive is braking, is moving
    under gravity, which a homing routine may use on purpose, and may go faster. `guard_policy`
    (`JointCommandController`): `log` records it and keeps going (Zach's 2026-11-13 ruling), `abort`
    ends the sequence and holds. `JointCommandSequenceController` always aborts.
    """

    joint_names: tuple[str, ...] = ()
    steps: tuple[Any, ...]
    name: str = ""
    max_speed: float | None = 3.0
    guard_policy: Literal["log", "abort"] = "log"

    def __post_init__(self) -> None:
        from robot_msgs.control.primitives import PRIMITIVES

        set_field(self, "joint_names", tuple(str(j) for j in self.joint_names))
        steps = tuple(_flatten(self.steps))
        set_field(self, "steps", steps)
        set_field(self, "name", str(self.name))
        if not steps:
            raise ValueError("JointCommandSequence needs at least one step")
        for step in steps:
            if not isinstance(step, (JointCommandStep, *PRIMITIVES)):
                raise TypeError(
                    f"JointCommandSequence: {type(step).__name__} is not a JointCommandStep, "
                    "Command, Delay or Condition"
                )
        known = set(self.joint_names)
        if len(known) != len(self.joint_names):
            raise ValueError(f"JointCommandSequence.joint_names has duplicates: {self.joint_names}")
        if any(isinstance(step, JointCommandStep) for step in steps) and not self.joint_names:
            raise ValueError("a JointCommandSequence of JointCommandSteps needs joint_names")
        if self.joint_names:
            unknown = set(self._named()) - known
            if unknown:
                raise ValueError(f"JointCommandSequence {self.name!r}: unknown joints {sorted(unknown)}")
        if self.max_speed is not None:
            set_field(self, "max_speed", _positive("JointCommandSequence", "max_speed", self.max_speed))
        if self.guard_policy not in ("log", "abort"):
            raise ValueError(f"guard_policy is log or abort, got {self.guard_policy!r}")

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def primitives(self) -> tuple[Any, ...]:
        """The recipe as `Command` / `Delay` / `Condition` only: what `JointCommandController` runs.

        A `JointCommandStep` becomes a Command (its commands, limp for the joints it does not name), a
        Condition (any of its `done_when`, its `timeout`) and, with `define_position`, a define Command."""
        from robot_msgs.control.primitives import Command, Condition

        out: list[Any] = []
        for step in self.steps:
            if not isinstance(step, JointCommandStep):
                out.append(step)
                continue
            named = {command.joint for command in step.commands}
            limp = tuple(JointCommand.limp(j) for j in self.joint_names if j not in named)
            out.append(Command(step.commands + limp, name=step.name))
            out.append(Condition(step.done_when, mode="any", timeout=step.timeout, name=step.name))
            if step.define_position:
                defines = tuple(JointCommand.define(j, v) for j, v in step.define_position.items())
                out.append(Command(defines, name=step.name))
        return tuple(out)

    @property
    def joints(self) -> tuple[str, ...]:
        """Every joint it commands or watches, first-named order (`ALL_JOINTS` left out)."""
        return self.joint_names or self._named()

    def _named(self) -> tuple[str, ...]:
        from robot_msgs.control.primitives import Command, Condition

        seen: dict[str, None] = {}
        for primitive in self.primitives:
            if isinstance(primitive, Command):
                names: tuple[str, ...] = primitive.joints
            elif isinstance(primitive, Condition):
                names = primitive.watched()
            else:
                names = ()
            for joint in names:
                if joint != ALL_JOINTS:
                    seen.setdefault(joint, None)
        return tuple(seen)


def _flatten(items: Any) -> Any:
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item
