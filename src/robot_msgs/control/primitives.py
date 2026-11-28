"""The three primitive step kinds a `JointCommandController` runs: `Command`, `Delay`, `Condition`.

Zach's Dec 2025 MoteusSeqController design, settled 2026-11-27 as the backend of the joint command API:

- `Command([JointCommand, ...])` sets each named joint's mode and setpoint from now on. It never waits.
- `Delay(seconds)` waits; every joint keeps its last Command.
- `Condition(until, timeout=..., fail=...)` waits until its predicates hold. Not done within `timeout` ends the
  sequence aborted (the reason names the timeout); a `fail` predicate that holds first ends it aborted with
  that predicate's reason ("no contact before max travel").

A `JointCommandSequence` is a list of these, read top to bottom like a recipe. The helper FUNCTIONS in
`robot_msgs.control.helpers` return lists of them (`find_hardstop`, `move_to`, ...), so a recipe needs no
intermediate classes (Zach, 2026-11-27: "I want all of the helpers to be functions ... that return a list of
the primitives").

Two data hooks keep a repeated hard-stop find inside three kinds (ruler v2 of the Joint Command Controller
track found it needs them, so they are first-class): `Condition(record=Record(joint, tag))` records the
joint's measured position when the Condition holds (a touch), and `Command(reduce=Reduce(...))` turns the
recorded touches into a `HardstopReport` (mean, spread) and defines the joint, or refuses and aborts.

Contact is hand-rolled behind ONE call, `InCollision(joint)` = `is_in_collision(joint)` (Zach, 2026-11-27
:1339, "for v1 I prefer to just do hand rolled as we have done so far, while collision detection is
unstable"): the `Stall` lead rule, or `EffortStall` when `effort` is given, or the controller's `detector`.

Units are SI (rad, rad/s, N·m, s); degrees only in text a human reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from robot_msgs._core import Msg, set_field
from robot_msgs.control.joint_command import (
    ALL_JOINTS,
    EffortStall,
    Elapsed,
    JointCommand,
    Reached,
    Settled,
    Stall,
)

# WHY 1 mrad: sim2real's repeat_checked_seek refuses a stop that does not repeat within 1 mrad
HARDSTOP_TOLERANCE_RAD = 0.001
# WHY 0.035 rad (2°): the Joint Command Controller spec's "back off (say 2°)"
HARDSTOP_BACKOFF_RAD = 0.035


def _positive(owner: str, name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{owner}.{name} must be finite and positive, got {value!r}")
    return value


# ── the predicates added in 0.8 (Elapsed, Reached, Settled, Stall, EffortStall are joint_command's) ──────────


@dataclass(frozen=True, eq=False, repr=False)
class InCollision(Msg):
    """`is_in_collision(joint)`: the joint is pressing on something and at rest against it.

    Behind the call (v1, hand-rolled): the controller's `detector` when it has one; else the `Stall` lead
    rule (the setpoint leads by more than 0.06 rad while the joint moves slower than 0.01 rad/s for 0.2 s),
    or the `EffortStall` rule with this `effort` threshold (N·m) when it is set."""

    joint: str
    effort: float | None = None

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        if self.effort is not None:
            set_field(self, "effort", _positive("InCollision", "effort", self.effort))

    def watched(self) -> tuple[str, ...]:
        return (self.joint,)


@dataclass(frozen=True, eq=False, repr=False)
class TravelEnd(Msg):
    """`joint`'s ramp reached its target and the joint settled there with no contact: a seek that ran out of
    travel. Meant as a `fail` predicate ("no contact before max travel")."""

    joint: str

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))

    def watched(self) -> tuple[str, ...]:
        return (self.joint,)


@dataclass(frozen=True, eq=False, repr=False)
class FollowEnded(Msg):
    """Every trajectory followed so far has played to its end and each of its joints is within `tolerance`
    (rad) of its last point."""

    tolerance: float = math.radians(1.0)

    def __post_init__(self) -> None:
        set_field(self, "tolerance", _positive("FollowEnded", "tolerance", self.tolerance))

    def watched(self) -> tuple[str, ...]:
        return ()


Predicate = (
    Elapsed | Reached | Settled | Stall | EffortStall | InCollision | TravelEnd | FollowEnded
)
PREDICATES: tuple[type, ...] = (
    Elapsed,
    Reached,
    Settled,
    Stall,
    EffortStall,
    InCollision,
    TravelEnd,
    FollowEnded,
)


# ── the two data hooks ─────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False)
class Record(Msg):
    """When the Condition carrying it holds, record `joint`'s measured position under `tag` (a touch)."""

    joint: str
    tag: str

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        set_field(self, "tag", str(self.tag))


@dataclass(frozen=True, eq=False, repr=False)
class Reduce(Msg):
    """At the Command carrying it, before its joint commands: `joint`'s touches recorded under `tag` become a
    `HardstopReport` (every touch, mean, spread = max - min).

    - `define` set: a spread within `tolerance` defines the joint so the MEAN of the touches reads `define`
      (the joint rests at its last touch); a wider spread refuses, and the sequence ends aborted naming it.
    - `verify` (with or without `define`): a spread past `tolerance` refuses; with `expected` set, a mean
      more than `tolerance` from it refuses too (a hard stop that is not where the frame says).
    WHY refuse on spread: an average hides a stop that moves, and the spread is what catches it."""

    joint: str
    tag: str
    define: float | None = None
    tolerance: float = HARDSTOP_TOLERANCE_RAD
    verify: bool = False
    expected: float | None = None

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        set_field(self, "tag", str(self.tag))
        for name in ("define", "expected"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Reduce.{name} must be finite, got {value!r}")
                set_field(self, name, value)
        set_field(self, "tolerance", _positive("Reduce", "tolerance", self.tolerance))
        set_field(self, "verify", bool(self.verify))


@dataclass(frozen=True, eq=False, repr=False)
class FailWhen(Msg):
    """A Condition's failure: when `predicate` holds before the Condition is done, the sequence ends aborted
    with `reason`."""

    predicate: Predicate
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, PREDICATES):
            raise TypeError(f"FailWhen: {type(self.predicate).__name__} is not a predicate")
        set_field(self, "reason", str(self.reason))


# ── the three primitives ───────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False)
class Command(Msg):
    """Set each joint in `commands` (one `JointCommand` per joint) from now on. Never waits.

    `reduce`, if set, runs first (a repeated hard-stop find's last Command). `name` is the recipe step it
    came from (the command log's `step`)."""

    commands: tuple[JointCommand, ...]
    reduce: Reduce | None = None
    name: str = ""

    def __post_init__(self) -> None:
        commands = tuple(self.commands)
        set_field(self, "commands", commands)
        set_field(self, "name", str(self.name))
        for command in commands:
            if not isinstance(command, JointCommand):
                raise TypeError(
                    f"Command {self.name!r}: {type(command).__name__} is not a JointCommand"
                )
        joints = [command.joint for command in commands]
        if not joints and self.reduce is None:
            raise ValueError(f"Command {self.name!r} commands no joint")
        if len(joints) != len(set(joints)):
            raise ValueError(f"Command {self.name!r}: one JointCommand per joint, got {joints}")
        if ALL_JOINTS in joints and len(joints) > 1:
            raise ValueError(
                f"Command {self.name!r}: {ALL_JOINTS!r} (every joint) must stand alone"
            )

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(command.joint for command in self.commands)


@dataclass(frozen=True, eq=False, repr=False)
class Delay(Msg):
    """Wait `seconds`. Every joint keeps its last Command."""

    seconds: float
    name: str = ""

    def __post_init__(self) -> None:
        set_field(self, "seconds", float(self.seconds))
        set_field(self, "name", str(self.name))
        if not (math.isfinite(self.seconds) and self.seconds >= 0.0):
            raise ValueError(f"Delay.seconds must be finite and >= 0, got {self.seconds!r}")


@dataclass(frozen=True, eq=False, repr=False)
class Condition(Msg):
    """Wait until the `until` predicates hold (`mode` `all` or `any`).

    - `fail`: `FailWhen`s (or `(predicate, reason)` pairs); one that holds first ends the sequence aborted.
    - `timeout` (s): not done by then ends it aborted, reason naming the timeout. `None` derives one when
      the Condition starts: 1.5 x (the longest remaining ramp ÷ its speed, a trajectory's rest, an
      `Elapsed`) + 1 s.
    - `record`: when it holds, record a joint's measured position (a touch, for `Reduce`)."""

    until: tuple[Predicate, ...]
    mode: Literal["all", "any"] = "all"
    fail: tuple[FailWhen, ...] = ()
    timeout: float | None = None
    record: Record | None = None
    name: str = ""

    def __post_init__(self) -> None:
        until = tuple(self.until) if isinstance(self.until, (list, tuple)) else (self.until,)
        set_field(self, "until", until)
        fails = tuple(f if isinstance(f, FailWhen) else FailWhen(f[0], f[1]) for f in self.fail)
        set_field(self, "fail", fails)
        set_field(self, "name", str(self.name))
        if not until:
            raise ValueError(f"Condition {self.name!r} waits on no predicate")
        for predicate in until:
            if not isinstance(predicate, PREDICATES):
                raise TypeError(
                    f"Condition {self.name!r}: {type(predicate).__name__} is not a predicate"
                )
        if self.mode not in ("all", "any"):
            raise ValueError(f"Condition.mode is all or any, got {self.mode!r}")
        if self.timeout is not None:
            set_field(self, "timeout", _positive("Condition", "timeout", self.timeout))

    def watched(self) -> tuple[str, ...]:
        """The joints its `until` predicates read."""
        return tuple(joint for predicate in self.until for joint in predicate.watched())


Primitive = Command | Delay | Condition
PRIMITIVES: tuple[type, ...] = (Command, Delay, Condition)
