"""The recipe DSL: plain FUNCTIONS that return a list of primitives (`Command`, `Delay`, `Condition`).

    from robot_msgs.control import JointCommandSequence, JointCommandController
    from robot_msgs.control.helpers import find_hardstop_and_hold, hold, move_to, stop_all

    recipe = JointCommandSequence(name="home elbow, then park", steps=[
        hold("shoulder", max_effort=12.0),
        find_hardstop_and_hold("elbow", direction=-1, max_vel=0.3, max_effort=3.0, max_travel=1.2,
                               repeats=3, define=-2.356),
        move_to("elbow", -1.0, max_vel=1.0, max_effort=8.0),
        stop_all(),
    ])

WHY functions, never classes (Zach, 2026-11-27): "I want all of the helpers to be functions ... that return a
list of the primitives ... that way we never need to make all of these random intermediate classes and the
function still gives us that nice readable API." The data stays three primitive kinds, so logs, viewers and
other languages read one format; a helper is only a way to write it.

A helper that ends by itself (`move_to`, `find_hardstop`, `follow`) carries its own Condition; one that never
ends (`hold`, `velocity`, `effort`) is a bare Command, so pair it with `wait` / `wait_until`. `at_once` runs
several of them together and waits for all.

Units are SI (rad, rad/s, N·m, s).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from robot_msgs.control.joint_command import ALL_JOINTS, JointCommand, Reached
from robot_msgs.control.primitives import (
    HARDSTOP_BACKOFF_RAD,
    HARDSTOP_TOLERANCE_RAD,
    Command,
    Condition,
    Delay,
    FailWhen,
    FollowEnded,
    InCollision,
    Predicate,
    Primitive,
    Record,
    Reduce,
    TravelEnd,
)

ONE_DEGREE = math.radians(1.0)
# WHY a re-touch travel of 4 backoffs: it starts 1 backoff off the stop, so the setpoint leads past it by 3
# (0.105 rad at the default). The hand-rolled lead rule needs 0.06 rad of lead held for 0.2 s before the ramp's
# end-of-travel rule (TravelEnd, also 0.2 s) can say "no contact"; at 3 backoffs the two raced (ruler v2 holdout)
RETOUCH_TRAVEL_BACKOFFS = 4.0
# WHY 0.75 of the backoff: the back-off only has to leave the stop; CAN 11's dry friction (1.12 N·m on a
# 63.7 N·m/rad loop) parks it up to 1.0° short of a target, more than a quarter of a 2° backoff
BACKOFF_REACHED_FRACTION = 0.75


# ── one-Command helpers ────────────────────────────────────────────────────────────────────────────────────────


def limp(*joints: str, name: str = "") -> list[Primitive]:
    """No torque on `joints` (position mode, no target)."""
    return [Command(tuple(JointCommand.limp(j) for j in joints), name=name or "limp")]


def stop(*joints: str, name: str = "") -> list[Primitive]:
    """The board's stop on `joints`: driver off; clears a fault and a watchdog latch."""
    return [Command(tuple(JointCommand.stop(j) for j in joints), name=name or "stop")]


def stop_all(name: str = "") -> list[Primitive]:
    """The board's stop on every joint the controller drives."""
    return [Command((JointCommand.stop(ALL_JOINTS),), name=name or "stop all")]


def brake(*joints: str, name: str = "") -> list[Primitive]:
    """The board's brake on `joints`: windings shorted, passive damping."""
    return [Command(tuple(JointCommand.brake(j) for j in joints), name=name or "brake")]


def hold(
    joint: str, max_effort: float, feedforward_effort: float = 0.0, *, name: str = ""
) -> list[Primitive]:
    """Hold `joint` where it is (its last setpoint when it was already settled there) under `max_effort`.
    Never ends: pair it with `wait` / `wait_until`, or let the next Commands run while it holds."""
    command = JointCommand.hold(joint, max_effort=max_effort, feedforward_effort=feedforward_effort)
    return [Command((command,), name=name or f"hold {joint}")]


def velocity(
    joint: str, vel: float, max_effort: float, feedforward_effort: float = 0.0, *, name: str = ""
) -> list[Primitive]:
    """Seek `joint` at `vel` rad/s (a host ramp) under `max_effort`. Never ends: pair it with `wait_until`."""
    command = JointCommand.at_velocity(
        joint, vel, max_effort=max_effort, feedforward_effort=feedforward_effort
    )
    return [Command((command,), name=name or f"velocity {joint}")]


def effort(
    joint: str, torque: float, max_effort: float | None = None, *, name: str = ""
) -> list[Primitive]:
    """Pure torque `torque` on `joint` (zero gains, no position target). Never ends."""
    return [
        Command(
            (JointCommand.effort(joint, torque, max_effort=max_effort),),
            name=name or f"effort {joint}",
        )
    ]


def define(
    joint: str, position: float, max_effort: float | None = None, *, name: str = ""
) -> list[Primitive]:
    """'Your position is now `position`' (moteus set_output_exact). No motion; the joint keeps what it was doing."""
    return [
        Command(
            (JointCommand.define(joint, position, max_effort=max_effort),),
            name=name or f"define {joint}",
        )
    ]


def set_joint_home(joint: str, position: float, *, name: str = "") -> list[Primitive]:
    """Declare where `joint` is now (its home reference). The same as `define`."""
    return define(joint, position, name=name or f"set {joint} home")


# ── helpers that end by themselves ─────────────────────────────────────────────────────────────────────────────


def move_to(
    joint: str,
    position: float,
    max_vel: float,
    max_effort: float,
    feedforward_effort: float = 0.0,
    *,
    tolerance: float = ONE_DEGREE,
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Ramp `joint` to `position` at `max_vel`; done when Reached within `tolerance`."""
    name = name or f"move {joint} to {position:+.4f}"
    command = JointCommand.to(
        joint, position, speed=max_vel, max_effort=max_effort, feedforward_effort=feedforward_effort
    )
    return [
        Command((command,), name=name),
        Condition((Reached(joints=(joint,), tolerance=tolerance),), timeout=timeout, name=name),
    ]


def move_by(
    joint: str,
    distance: float,
    max_vel: float,
    max_effort: float,
    feedforward_effort: float = 0.0,
    *,
    tolerance: float = ONE_DEGREE,
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Ramp `joint` by `distance` from where it starts at `max_vel`; done when Reached within `tolerance`."""
    name = name or f"move {joint} by {distance:+.4f}"
    command = JointCommand.move_by(
        joint, distance, speed=max_vel, max_effort=max_effort, feedforward_effort=feedforward_effort
    )
    return [
        Command((command,), name=name),
        Condition((Reached(joints=(joint,), tolerance=tolerance),), timeout=timeout, name=name),
    ]


def follow(
    trajectory: Any,
    max_effort: float | None = None,
    *,
    tolerance: float = ONE_DEGREE,
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Play `trajectory` (a `JointTrajectory`; t = 0 now) on every joint it names, under `max_effort` (None =
    board caps); done at its end with every joint within `tolerance` of its last point."""
    name = name or "follow"
    commands = tuple(
        JointCommand.follow(j, trajectory, max_effort=max_effort) for j in trajectory.joint_names
    )
    return [
        Command(commands, name=name),
        Condition((FollowEnded(tolerance),), timeout=timeout, name=name),
    ]


def wait(seconds: float, *, name: str = "") -> list[Primitive]:
    """Wait `seconds`; every joint keeps its last Command."""
    return [Delay(seconds, name=name or f"wait {seconds:g} s")]


def wait_until(
    *predicates: Predicate,
    mode: str = "any",
    timeout: float | None = None,
    fail: Sequence[FailWhen] = (),
    name: str = "",
) -> list[Primitive]:
    """Wait until any (`mode="all"`: every) of `predicates` holds, e.g. `wait_until(InCollision("elbow"))`."""
    return [
        Condition(
            tuple(predicates),
            mode=mode,
            fail=tuple(fail),
            timeout=timeout,
            name=name or "wait until",
        )
    ]


def at_once(
    *parts: Sequence[Primitive],
    until: Sequence[Predicate] = (),
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Run helpers together: every part's Commands now, then ONE Condition waiting for all of their endings
    (and every `until` predicate). Each part is Commands plus at most one closing Condition (not `find_hardstop`,
    whose touches are sequential: put it after an `at_once`, or inline). A part's own timeout is dropped for
    `timeout` (None = derived)."""
    commands: list[Primitive] = []
    waits: list[Predicate] = list(until)
    for part in parts:
        part = list(part)
        closing = part[-1] if part and isinstance(part[-1], Condition) else None
        body = part[:-1] if closing is not None else part
        if any(not isinstance(p, Command) for p in body) or (
            closing is not None and (closing.fail or closing.record)
        ):
            raise ValueError(
                "at_once takes helpers made of Commands and one plain closing Condition"
            )
        commands += [p if not name else Command(p.commands, p.reduce, name) for p in body]
        if closing is not None:
            waits += list(closing.until)
    if waits:
        commands.append(
            Condition(tuple(waits), mode="all", timeout=timeout, name=name or "at once")
        )
    return commands


# ── the hard stop ──────────────────────────────────────────────────────────────────────────────────────────────


def find_hardstop(
    joint: str,
    direction: int,
    max_vel: float,
    max_effort: float,
    max_travel: float | None = None,
    *,
    timeout: float | None = None,
    repeats: int = 1,
    backoff: float = HARDSTOP_BACKOFF_RAD,
    probe_speed: float | None = None,
    define: float | None = None,
    tolerance: float = HARDSTOP_TOLERANCE_RAD,
    feedforward_effort: float = 0.0,
    verify: bool = False,
    expected: float | None = None,
    contact_effort: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Seek `joint` in `direction` (+1 / -1) at `max_vel` under `max_effort` until `is_in_collision(joint)`,
    then hold resting against the stop.

    - `max_travel` (rad) bounds the seek: no contact by then aborts, "no contact ... within max_travel", with
      no define. Without it the seek is a velocity command and `timeout` (s) bounds it. Give at least one.
    - `repeats` > 1: back off `backoff` and re-touch at `probe_speed` (None = max_vel / 2), same side and
      speed (a touch from the other side measures backlash, not the stop). Every touch is recorded.
    - The last Command reduces the touches to a `HardstopReport` (mean, spread). With `define` set, a spread
      within `tolerance` (1 mrad) defines the joint so the mean reads `define`; a wider spread refuses and
      aborts naming it. `verify` refuses a wide spread with no define; `expected` also checks the mean.
    - `contact_effort` (N·m): read contact by the EffortStall rule instead of the Stall lead rule (a seek
      against gravity whose drive saturates short of the stop).
    """
    if direction not in (1, -1):
        raise ValueError(f"find_hardstop {joint}: direction must be +1 or -1, got {direction!r}")
    if max_travel is None and timeout is None:
        raise ValueError(
            f"find_hardstop {joint}: give max_travel or timeout, or the seek may run forever"
        )
    if int(repeats) != repeats or repeats < 1:
        raise ValueError(
            f"find_hardstop {joint}: repeats must be a whole number >= 1, got {repeats!r}"
        )
    probe = max_vel / 2.0 if probe_speed is None else float(probe_speed)
    name = name or f"find hardstop {joint}"
    tag = f"touch:{joint}:{name}"
    ramp = dict(max_effort=max_effort, feedforward_effort=feedforward_effort)
    touched = InCollision(joint, effort=contact_effort)

    def one(command: JointCommand) -> Command:
        return Command((command,), name=name)

    if max_travel is not None:
        seek = JointCommand.move_by(joint, direction * float(max_travel), speed=max_vel, **ramp)
        no_contact = (
            FailWhen(
                TravelEnd(joint), f"no contact on {joint} within max_travel {max_travel:g} rad"
            ),
        )
    else:
        seek = JointCommand.at_velocity(joint, direction * float(max_vel), **ramp)
        no_contact = ()
    out: list[Primitive] = [
        one(seek),
        Condition(
            (touched,), fail=no_contact, timeout=timeout, record=Record(joint, tag), name=name
        ),
    ]
    for touch in range(2, int(repeats) + 1):
        out += [
            one(JointCommand.move_by(joint, -direction * backoff, speed=probe, **ramp)),
            Condition(
                (Reached(joints=(joint,), tolerance=BACKOFF_REACHED_FRACTION * backoff),),
                timeout=timeout,
                name=name,
            ),
            one(
                JointCommand.move_by(
                    joint, direction * RETOUCH_TRAVEL_BACKOFFS * backoff, speed=probe, **ramp
                )
            ),
            Condition(
                (touched,),
                timeout=timeout,
                record=Record(joint, tag),
                name=name,
                fail=(
                    FailWhen(
                        TravelEnd(joint),
                        f"no contact on {joint} at re-touch {touch}: the stop moved away by more than "
                        f"{2 * backoff:g} rad",
                    ),
                ),
            ),
        ]
    # WHY a hold at the last touch, not the seek's setpoint: the seek's setpoint is up to max_travel past the
    # stop at the cap; a finished find rests against the stop (the track's S007 rule)
    reduce = Reduce(
        joint, tag, define=define, tolerance=tolerance, verify=verify, expected=expected
    )
    out.append(Command((JointCommand.hold(joint, **ramp),), reduce=reduce, name=name))
    return out


def find_hardstop_and_hold(
    joint: str,
    direction: int,
    max_vel: float,
    max_effort: float,
    max_travel: float | None = None,
    **options: Any,
) -> list[Primitive]:
    """The Aug 3 2026 recipe's name for `find_hardstop` (which already ends holding at the stop)."""
    return find_hardstop(joint, direction, max_vel, max_effort, max_travel, **options)


def verify_hard_stop(
    joint: str,
    n: int,
    direction: int,
    max_vel: float,
    max_effort: float,
    max_travel: float | None = None,
    *,
    expected: float | None = None,
    tolerance: float = HARDSTOP_TOLERANCE_RAD,
    **options: Any,
) -> list[Primitive]:
    """Touch `joint`'s stop `n` times and refuse (abort) when the touches spread past `tolerance`, or their
    mean is more than `tolerance` from `expected` (the stop where the current frame says it is). No define."""
    return find_hardstop(
        joint,
        direction,
        max_vel,
        max_effort,
        max_travel,
        repeats=n,
        verify=True,
        expected=expected,
        tolerance=tolerance,
        name=options.pop("name", f"verify {joint} stop"),
        **options,
    )
