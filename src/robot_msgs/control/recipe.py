"""Short ways to WRITE a recipe: one vocabulary of short keywords over the functions in `helpers`.

    from robot_msgs.control import JointCommandSequence
    from robot_msgs.control.recipe import ALL, Group, Joint, deg, script, step, wait

    J1 = Joint("shoulder", max_tau=12, ff=5)          # a joint's defaults, declared once
    J2 = Joint("elbow", vel=0.5, max_tau=3)
    Arm = Group(J1, J2, name="Arm")

    recipe = JointCommandSequence(name="home the elbow, then park", steps=[
        J1.hold(),
        J2.find_hardstop(dir=-1, travel=1.0, define=deg(-135)),
        J2.move_by(deg(20), max_tau=8), wait(0.3),
        Arm.stop(),
    ])

WHY (Zach, 2026-11-28): `find_hardstop_and_hold("elbow", -1, 0.5, 3.0, 1.0, define=-2.35619)` is bare numbers,
"I don't know the function signature type thing. pos=20, vel=... max_tau=... you can use short hand names, but
very helpful IMO. Just so its readable", and, after moteus tview's `A>d pos` / `11,12>d pos` / `11>d pos`:
"All, a set of, or one".

Nothing here is a second implementation: every entry point resolves its joints and its short keywords, then calls
the helper of the same name in `helpers`, so a recipe written here is the same primitives as one written there.
Only `call` differs: it is the step as written in the style used (`elbow.move_by(0.349066, max_tau=8)`,
`stop(Arm)`, `Arm[ALL].stop()`, or the text line itself).

The same step, five ways:

    move_by("elbow", deg(20), vel=0.5, max_tau=8)     # a verb over a name
    J2.move_by(deg(20), max_tau=8)                    # a Joint that carries its defaults
    move_by(J2, deg(20), max_tau=8)                   # a verb over a Joint, a Group, or several
    Arm["elbow"].move_by(deg(20), max_tau=8)          # tview's addressing: Arm[ALL], Arm[J1, J2], Arm["elbow"]
    script("elbow> move_by 20deg max_tau=8", joints=Arm)   # tview-style text, one step per line

The short names (`VOCABULARY`; `signature("move_by")` lists a verb's own):

    pos      target position, rad                        by       distance from where it starts, rad
    vel      speed, rad/s                                max_tau  effort cap, N·m
    ff       constant feedforward torque, N·m            tau      pure torque, N·m
    tol      tolerance, rad                              timeout  abort when not done by then, s
    dir      seek direction, +1 or -1                    travel   the most a seek may travel, rad
    repeats  touches of a hard stop                      define   what the stop reads afterwards, rad

A keyword left out takes the joint's default (`Joint(..., vel=, max_tau=, ff=)`), else the helper's own. One that
has neither raises, naming the joint and the keyword.

Units are SI (rad, rad/s, N·m, s); `deg(20)` (and `20deg` in a text line) converts at the call site.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, ParamSpec

from robot_msgs.control import helpers
from robot_msgs.control.helpers import (
    _arg_text,
    at_once,
    follow,
    outline,
    step,
    stop_all,
    wait,
    wait_until,
)
from robot_msgs.control.joint_command import ALL_JOINTS
from robot_msgs.control.primitives import Command, Primitive

__all__ = [
    "ALL",
    "Group",
    "Joint",
    "VERBS",
    "VOCABULARY",
    "at_once",
    "brake",
    "define",
    "deg",
    "effort",
    "find_hardstop",
    "follow",
    "hold",
    "limp",
    "move_by",
    "move_to",
    "outline",
    "script",
    "signature",
    "step",
    "stop",
    "stop_all",
    "velocity",
    "verify_hardstop",
    "wait",
    "wait_until",
]

deg = math.radians
"""`deg(20)`: 20 degrees as the radians the code runs on."""

ALL = ALL_JOINTS
"""Every joint: `Arm[ALL]` is every joint of that group (tview's `A>`); `stop(ALL)` every joint the controller drives."""

VOCABULARY: dict[str, str] = {
    "pos": "target position, rad",
    "by": "distance from where the joint starts, rad (signed)",
    "vel": "speed, rad/s",
    "max_tau": "effort cap, N·m",
    "ff": "constant feedforward torque, N·m",
    "tau": "pure torque, N·m (signed)",
    "tol": "tolerance, rad",
    "timeout": "abort when not done by then, s",
    "dir": "seek direction, +1 or -1",
    "travel": "the most a seek may travel, rad",
    "repeats": "touches of a hard stop",
    "define": "what the joint reads at the stop afterwards, rad",
    "backoff": "how far to back off between touches, rad",
    "probe_vel": "re-touch speed, rad/s",
    "verify": "refuse a stop whose touches spread past tol",
    "expected": "where the stop should read now, rad",
    "contact_tau": "read contact by effort past this, N·m",
}
"""Every short keyword and what it means, with its unit. One vocabulary for the functions, the methods and the text."""

# the short names that may be written in degrees in a text line (`20deg`, `vel=30deg` = 30 °/s)
_ANGLES = frozenset(
    {"pos", "by", "vel", "tol", "travel", "define", "backoff", "probe_vel", "expected"}
)
# the defaults a Joint declares once
_JOINT_DEFAULTS = ("vel", "max_tau", "ff")

_MOVE = {
    "vel": "max_vel",
    "max_tau": "max_effort",
    "ff": "feedforward_effort",
    "tol": "tolerance",
    "timeout": "timeout",
}
_SEEK = {
    "dir": "direction",
    "vel": "max_vel",
    "max_tau": "max_effort",
    "travel": "max_travel",
    "timeout": "timeout",
}
_TOUCH = {
    "tol": "tolerance",
    "ff": "feedforward_effort",
    "backoff": "backoff",
    "probe_vel": "probe_speed",
    "contact_tau": "contact_effort",
}


@dataclass(frozen=True)
class _Verb:
    """How one verb maps onto its helper."""

    helper: Callable[..., list[Primitive]]
    # short keyword -> the helper's own keyword, in the order a step reads
    names: Mapping[str, str] = dataclasses.field(default_factory=dict)
    # the one value a step may give without its keyword: `move_by(J2, deg(20))`, `elbow> move_by 20deg`
    value: str | None = None
    needs: tuple[str, ...] = ()
    # on several joints: "broadcast" = the helper takes them all (one Command); "one command" = each joint's
    # Command merged into one; "at once" = `helpers.at_once`; "one joint" = refuse
    several: str = "one command"


VERBS: dict[str, _Verb] = {
    "limp": _Verb(helpers.limp, several="broadcast"),
    "stop": _Verb(helpers.stop, several="broadcast"),
    "brake": _Verb(helpers.brake, several="broadcast"),
    "hold": _Verb(
        helpers.hold, {"max_tau": "max_effort", "ff": "feedforward_effort"}, needs=("max_tau",)
    ),
    "move_to": _Verb(
        helpers.move_to,
        {"pos": "position", **_MOVE},
        value="pos",
        needs=("pos", "vel", "max_tau"),
        several="at once",
    ),
    "move_by": _Verb(
        helpers.move_by,
        {"by": "distance", **_MOVE},
        value="by",
        needs=("by", "vel", "max_tau"),
        several="at once",
    ),
    "velocity": _Verb(
        helpers.velocity,
        {"vel": "vel", "max_tau": "max_effort", "ff": "feedforward_effort"},
        value="vel",
        needs=("vel", "max_tau"),
    ),
    "effort": _Verb(
        helpers.effort, {"tau": "torque", "max_tau": "max_effort"}, value="tau", needs=("tau",)
    ),
    "define": _Verb(
        helpers.define, {"pos": "position", "max_tau": "max_effort"}, value="pos", needs=("pos",)
    ),
    "find_hardstop": _Verb(
        helpers.find_hardstop,
        {**_SEEK, "repeats": "repeats", "define": "define", **_TOUCH}
        | {"verify": "verify", "expected": "expected"},
        needs=("dir", "vel", "max_tau"),
        several="one joint",
    ),
    "verify_hardstop": _Verb(
        helpers.verify_hard_stop,
        {"repeats": "n", **_SEEK, "expected": "expected", **_TOUCH},
        needs=("repeats", "dir", "vel", "max_tau"),
        several="one joint",
    ),
}
"""The verbs: each is a function here (`move_by(J2, ...)`), a method on `Joint` and `Group` (`J2.move_by(...)`)
and a word in a `script` line (`elbow> move_by ...`), and each expands through the helper of the same name."""


def signature(verb: str) -> list[tuple[str, str, str]]:
    """A verb's keywords as rows of (short name, what it means with its unit, default), in the order a step reads.

    `signature("move_by")[0]` is `("by", "distance from where the joint starts, rad (signed)", "required")`."""
    spec = _verb(verb)
    # verify_hard_stop hands its other options on to find_hardstop, so their defaults are read there
    parameters = {
        **inspect.signature(helpers.find_hardstop).parameters,
        **inspect.signature(spec.helper).parameters,
    }
    rows = []
    for short, long in spec.names.items():
        own = parameters[long].default
        if short in _JOINT_DEFAULTS and short != spec.value:
            rest = "required" if short in spec.needs else _arg_text(own)
            default = f"the Joint's {short}, else {rest}"
        elif short in spec.needs:
            default = "required"
        else:
            default = _arg_text(own)
        rows.append((short, VOCABULARY[short], default))
    return rows


def _verb(verb: str) -> _Verb:
    if verb not in VERBS:
        raise ValueError(f"no verb `{verb}`; the verbs are {', '.join(VERBS)}")
    return VERBS[verb]


# ── a joint, and a group of them ───────────────────────────────────────────────────────────────────────────────


class _Steps:
    """The verbs as methods, shared by `Joint` (one joint) and `Group` (all, a set, or one).

    Each returns the list of primitives the helper of the same name returns, with `call` = the step as written
    (`elbow.move_by(0.349066)`, `Arm.stop()`). On a group: `limp` / `stop` / `brake` / `hold` / `velocity` /
    `effort` / `define` are ONE Command with one JointCommand per joint; `move_to` / `move_by` run the joints
    together and wait for all (`helpers.at_once`); a hard-stop find takes one joint."""

    # WHY locals(): each method's body is "this verb, these joints, exactly the keywords of its signature".
    # `_expand` reads the short names out of it, so a keyword is spelled once, in the signature a reader sees.

    def limp(self, *, name: str = "") -> list[Primitive]:
        """No torque."""
        return _expand("limp", self, locals())

    def stop(self, *, name: str = "") -> list[Primitive]:
        """The board's stop: driver off; clears a fault and a watchdog latch."""
        return _expand("stop", self, locals())

    def brake(self, *, name: str = "") -> list[Primitive]:
        """The board's brake: windings shorted, passive damping."""
        return _expand("brake", self, locals())

    def hold(
        self, *, max_tau: float | None = None, ff: float | None = None, name: str = ""
    ) -> list[Primitive]:
        """Hold where it is under `max_tau` (N·m), plus a constant `ff` (N·m). Never ends: pair with `wait`."""
        return _expand("hold", self, locals())

    def move_to(
        self,
        pos: float | None = None,
        *,
        vel: float | None = None,
        max_tau: float | None = None,
        ff: float | None = None,
        tol: float | None = None,
        timeout: float | None = None,
        name: str = "",
    ) -> list[Primitive]:
        """Ramp to `pos` (rad) at `vel` (rad/s) under `max_tau` (N·m); done within `tol` (rad, default 1°)."""
        return _expand("move_to", self, locals())

    def move_by(
        self,
        by: float | None = None,
        *,
        vel: float | None = None,
        max_tau: float | None = None,
        ff: float | None = None,
        tol: float | None = None,
        timeout: float | None = None,
        name: str = "",
    ) -> list[Primitive]:
        """Ramp `by` (rad, signed) from where it starts at `vel` (rad/s) under `max_tau` (N·m); done within `tol`."""
        return _expand("move_by", self, locals())

    def velocity(
        self,
        vel: float | None = None,
        *,
        max_tau: float | None = None,
        ff: float | None = None,
        name: str = "",
    ) -> list[Primitive]:
        """Seek at `vel` (rad/s, signed) under `max_tau` (N·m). Never ends: pair with `wait_until`."""
        return _expand("velocity", self, locals())

    def effort(
        self, tau: float | None = None, *, max_tau: float | None = None, name: str = ""
    ) -> list[Primitive]:
        """Pure torque `tau` (N·m, signed; zero gains, no position target), capped at `max_tau`. Never ends."""
        return _expand("effort", self, locals())

    def define(
        self, pos: float | None = None, *, max_tau: float | None = None, name: str = ""
    ) -> list[Primitive]:
        """'Your position is now `pos`' (rad; moteus set_output_exact). No motion."""
        return _expand("define", self, locals())

    def find_hardstop(
        self,
        *,
        dir: int | None = None,  # noqa: A002
        vel: float | None = None,
        max_tau: float | None = None,
        travel: float | None = None,
        timeout: float | None = None,
        repeats: int | None = None,
        define: float | None = None,
        tol: float | None = None,
        ff: float | None = None,
        backoff: float | None = None,
        probe_vel: float | None = None,
        verify: bool | None = None,
        expected: float | None = None,
        contact_tau: float | None = None,
        name: str = "",
    ) -> list[Primitive]:
        """Seek in `dir` (+1 / -1) at `vel` (rad/s) under `max_tau` (N·m) until contact, then hold on the stop.
        `travel` (rad) or `timeout` (s) bounds the seek; `repeats` touches; `define` (rad) is what the stop
        reads afterwards (refused when the touches spread past `tol`, 1 mrad)."""
        return _expand("find_hardstop", self, locals())

    def verify_hardstop(
        self,
        *,
        repeats: int | None = None,
        dir: int | None = None,  # noqa: A002
        vel: float | None = None,
        max_tau: float | None = None,
        travel: float | None = None,
        timeout: float | None = None,
        expected: float | None = None,
        tol: float | None = None,
        ff: float | None = None,
        backoff: float | None = None,
        probe_vel: float | None = None,
        contact_tau: float | None = None,
        name: str = "",
    ) -> list[Primitive]:
        """Touch the stop `repeats` times; refuse when the touches spread past `tol` (1 mrad) or their mean is
        more than `tol` from `expected` (rad). No define."""
        return _expand("verify_hardstop", self, locals())


@dataclass(frozen=True)
class Joint(_Steps):
    """A joint and the defaults its steps take, declared once: `J2 = Joint("elbow", vel=0.5, max_tau=3)`.

    Then a step is short, `J2.move_by(deg(20))`, and a keyword on the step overrides the default,
    `J2.move_by(deg(20), max_tau=8)`. `vel` (rad/s), `max_tau` (N·m effort cap), `ff` (N·m constant
    feedforward); `id` is what a `Group` also addresses it by (its CAN id, `Arm[11]`, `11> stop`)."""

    name: str
    _: KW_ONLY
    vel: float | None = None
    max_tau: float | None = None
    ff: float = 0.0
    id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Joint needs a name, got {self.name!r}")
        if self.id is not None and (isinstance(self.id, bool) or not isinstance(self.id, int)):
            raise ValueError(f"Joint {self.name}: id is a whole number, got {self.id!r}")

    def __str__(self) -> str:
        return self.name


class Group(_Steps):
    """Joints addressed together: `Arm = Group(J1, J2, name="Arm")`, then `Arm.stop()`.

    Address all, a set, or one, as moteus tview's `A>`, `11,12>`, `11>` do:

        Arm[ALL]  (or Arm.all)          every joint in the group
        Arm[J1, J2]                     a set: by Joint, by name `Arm["shoulder", "elbow"]`, by id `Arm[11, 12]`
        Arm["elbow"]                    one

    A selection is itself a Group, so every verb works on it and its `call` reads as written
    (`Arm["elbow"].move_by(0.349066)`). Members are `Joint`s, names (a joint with no defaults) or other Groups."""

    def __init__(self, *joints: Joint | str | Group, name: str = "Arm") -> None:
        members = _joints(joints)
        if not members:
            raise ValueError(f"Group {name!r} needs at least one joint")
        ids = [joint.id for joint in members if joint.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Group {name!r}: two joints share an id: {ids}")
        self.joints: tuple[Joint, ...] = members
        self.name = str(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def all(self) -> Group:
        """Every joint in the group (the same as `[ALL]`)."""
        return Group(*self.joints, name=f"{self.name}.all")

    def __getitem__(self, key: Any) -> Group:
        keys = key if isinstance(key, tuple) else (key,)
        text = f"{self.name}[{', '.join(_written(k) for k in keys)}]"
        if any(isinstance(k, str) and k == ALL for k in keys):
            if len(keys) > 1:
                raise LookupError(f"{text}: ALL stands alone (it already is every joint)")
            return Group(*self.joints, name=text)
        return Group(*(self._find(k) for k in keys), name=text)

    def _find(self, key: Any) -> Joint:
        by_id = isinstance(key, int) and not isinstance(key, bool)
        wanted = key.name if isinstance(key, Joint) else key
        for joint in self.joints:
            if (joint.id == wanted) if by_id else (joint.name == wanted):
                return joint
        holds = ", ".join(j.name if j.id is None else f"{j.name} (id {j.id})" for j in self.joints)
        what = f"id {key}" if by_id else f"joint {_written(key)}"
        raise LookupError(f"{self.name} has no {what}; it holds {holds}")

    def __iter__(self):
        return iter(self.joints)

    def __len__(self) -> int:
        return len(self.joints)

    def __repr__(self) -> str:
        return f"Group({', '.join(self.names)}, name={self.name!r})"


def _joints(targets: Any) -> tuple[Joint, ...]:
    """The joints a target names, first-named order: a name, a `Joint`, a `Group`, or several of them."""
    found: dict[str, Joint] = {}

    def add(target: Any) -> None:
        if isinstance(target, Joint):
            found.setdefault(target.name, target)
        elif isinstance(target, Group):
            for joint in target.joints:
                found.setdefault(joint.name, joint)
        elif isinstance(target, str):
            found.setdefault(target, Joint(target))
        elif isinstance(target, (list, tuple)):
            for item in target:
                add(item)
        elif isinstance(target, int) and not isinstance(target, bool):
            raise TypeError(f"an id ({target}) is addressed through its Group: Arm[{target}]")
        else:
            raise TypeError(
                f"{type(target).__name__} is not a joint: give a name, a Joint or a Group"
            )

    add(targets)
    return tuple(found.values())


# ── every style ends here ──────────────────────────────────────────────────────────────────────────────────────


def _expand(verb: str, targets: Any, given: Mapping[str, Any]) -> list[Primitive]:
    """The primitives for `verb` on `targets`: the helper's, called with the short keywords under its own names."""
    spec = _verb(verb)
    joints = _joints(targets)
    if not joints:
        raise ValueError(f"{verb}: no joint to command")
    name = given.get("name") or ""
    if spec.several == "broadcast":
        return spec.helper(*(joint.name for joint in joints), name=name)
    if spec.several == "one joint" and len(joints) > 1:
        listed = ", ".join(joint.name for joint in joints)
        raise ValueError(
            f"{verb} takes one joint (its touches run one after another), got {listed}: write one step per joint"
        )
    parts = [_one(verb, spec, joint, given) for joint in joints]
    if len(parts) == 1:
        return parts[0]
    if spec.several == "at once":
        return helpers.at_once(*parts, timeout=given.get("timeout"), name=name)
    # "one command": each part is the helper's single Command; a group sends them as ONE Command with one
    # JointCommand per joint (as `helpers.stop("a", "b")` does)
    commands = tuple(c for part in parts for primitive in part for c in primitive.commands)
    return [Command(commands, name=name or verb)]


def _one(verb: str, spec: _Verb, joint: Joint, given: Mapping[str, Any]) -> list[Primitive]:
    defaults = {short: getattr(joint, short) for short in _JOINT_DEFAULTS}
    kwargs: dict[str, Any] = {}
    for short, long in spec.names.items():
        value = given.get(short)
        if value is None and short != spec.value:
            value = defaults.get(short)
        if value is None:
            if short in spec.needs:
                hint = (
                    f', or declare it once: Joint("{joint.name}", {short}=...)'
                    if short in defaults and short != spec.value
                    else ""
                )
                raise ValueError(
                    f"{verb} {joint.name}: needs {short}= ({VOCABULARY[short]}). Give it on the step{hint}"
                )
            continue
        kwargs[long] = value
    if "travel" in spec.names and given.get("travel") is None and given.get("timeout") is None:
        raise ValueError(
            f"{verb} {joint.name}: give travel= (rad) or timeout= (s), or the seek may run forever"
        )
    if given.get("name"):
        kwargs["name"] = given["name"]
    return spec.helper(joint.name, **kwargs)


def _written(value: Any) -> str:
    """One argument as it reads on a recipe line: a Joint or a Group by its name, the rest as `helpers` writes it."""
    if isinstance(value, Joint):
        return value.name
    if isinstance(value, Group):
        return value.name
    if isinstance(value, str) and value == ALL:
        return "ALL"
    if isinstance(value, (list, tuple)) and any(isinstance(v, (Joint, Group)) for v in value):
        return f"[{', '.join(_written(v) for v in value)}]"
    return _arg_text(value)


def _stamp(primitives: Iterable[Primitive], call: str) -> list[Primitive]:
    return [dataclasses.replace(primitive, call=call) for primitive in primitives]


P = ParamSpec("P")


def _as_written(fn: Callable[P, list[Primitive]]) -> Callable[P, list[Primitive]]:
    """Stamp `call` = the step as it was written: `hold("shoulder", max_tau=12)`, `stop(Arm)`, and for a
    method `elbow.move_by(0.349066)`. Positional stays positional, keywords stay keywords; `name=` is left out
    (it is the step name, shown beside the call)."""
    method = next(iter(inspect.signature(fn).parameters), "") == "self"

    @functools.wraps(fn)
    def verb(*args: P.args, **kwargs: P.kwargs) -> list[Primitive]:
        primitives = fn(*args, **kwargs)
        try:
            shown = [_written(value) for value in (args[1:] if method else args)]
            shown += [
                f"{key}={_written(value)}"
                for key, value in kwargs.items()
                if key != "name" and value is not None
            ]
            owner = f"{_written(args[0])}." if method else ""
            call = f"{owner}{fn.__name__}({', '.join(shown)})"
        except Exception:  # noqa: BLE001
            # WHY swallow: the call text is for reading and must never stop the recipe (as helpers._recipe)
            call = f"{fn.__name__}(...)"
        return _stamp(primitives, call)

    return verb


for _name in VERBS:
    setattr(_Steps, _name, _as_written(getattr(_Steps, _name)))
del _name


# ── verbs over targets: a name, a Joint, a Group, or several ───────────────────────────────────────────────────


@_as_written
def limp(*targets: Joint | Group | str, name: str = "") -> list[Primitive]:
    """No torque on every target: `limp(Arm)`, `limp(J1, J2)`, `limp("elbow")`. One Command."""
    return _expand("limp", targets, locals())


@_as_written
def stop(*targets: Joint | Group | str, name: str = "") -> list[Primitive]:
    """The board's stop (driver off) on every target: `stop(Arm)`, `stop(J1, J2)`, `stop("elbow")`. One Command."""
    return _expand("stop", targets, locals())


@_as_written
def brake(*targets: Joint | Group | str, name: str = "") -> list[Primitive]:
    """The board's brake (windings shorted) on every target. One Command."""
    return _expand("brake", targets, locals())


@_as_written
def hold(
    *targets: Joint | Group | str,
    max_tau: float | None = None,
    ff: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Hold every target where it is under `max_tau` (N·m), plus a constant `ff` (N·m): `hold(J1)`,
    `hold("shoulder", max_tau=12, ff=5)`, `hold(Arm)`. One Command. Never ends: pair with `wait`."""
    return _expand("hold", targets, locals())


@_as_written
def move_to(
    target: Any,
    pos: float | None = None,
    *,
    vel: float | None = None,
    max_tau: float | None = None,
    ff: float | None = None,
    tol: float | None = None,
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Ramp `target` to `pos` (rad) at `vel` (rad/s) under `max_tau` (N·m); done within `tol` (rad, default 1°).
    A Group (or a list of targets) moves together and waits for all."""
    return _expand("move_to", target, locals())


@_as_written
def move_by(
    target: Any,
    by: float | None = None,
    *,
    vel: float | None = None,
    max_tau: float | None = None,
    ff: float | None = None,
    tol: float | None = None,
    timeout: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Ramp `target` `by` (rad, signed) from where it starts at `vel` (rad/s) under `max_tau` (N·m); done
    within `tol` (rad, default 1°). A Group (or a list of targets) moves together and waits for all."""
    return _expand("move_by", target, locals())


@_as_written
def velocity(
    target: Any,
    vel: float | None = None,
    *,
    max_tau: float | None = None,
    ff: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Seek `target` at `vel` (rad/s, signed) under `max_tau` (N·m). Never ends: pair with `wait_until`."""
    return _expand("velocity", target, locals())


@_as_written
def effort(
    target: Any, tau: float | None = None, *, max_tau: float | None = None, name: str = ""
) -> list[Primitive]:
    """Pure torque `tau` (N·m, signed) on `target`, capped at `max_tau`. Never ends."""
    return _expand("effort", target, locals())


@_as_written
def define(
    target: Any, pos: float | None = None, *, max_tau: float | None = None, name: str = ""
) -> list[Primitive]:
    """'`target`, your position is now `pos`' (rad; moteus set_output_exact). No motion."""
    return _expand("define", target, locals())


@_as_written
def find_hardstop(
    target: Any,
    *,
    dir: int | None = None,  # noqa: A002
    vel: float | None = None,
    max_tau: float | None = None,
    travel: float | None = None,
    timeout: float | None = None,
    repeats: int | None = None,
    define: float | None = None,
    tol: float | None = None,
    ff: float | None = None,
    backoff: float | None = None,
    probe_vel: float | None = None,
    verify: bool | None = None,
    expected: float | None = None,
    contact_tau: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Seek `target` (one joint) in `dir` (+1 / -1) at `vel` (rad/s) under `max_tau` (N·m) until contact, then
    hold on the stop. `travel` (rad) or `timeout` (s) bounds the seek; `repeats` touches; `define` (rad) is
    what the stop reads afterwards (refused when the touches spread past `tol`, 1 mrad)."""
    return _expand("find_hardstop", target, locals())


@_as_written
def verify_hardstop(
    target: Any,
    *,
    repeats: int | None = None,
    dir: int | None = None,  # noqa: A002
    vel: float | None = None,
    max_tau: float | None = None,
    travel: float | None = None,
    timeout: float | None = None,
    expected: float | None = None,
    tol: float | None = None,
    ff: float | None = None,
    backoff: float | None = None,
    probe_vel: float | None = None,
    contact_tau: float | None = None,
    name: str = "",
) -> list[Primitive]:
    """Touch `target`'s stop `repeats` times; refuse when the touches spread past `tol` (1 mrad) or their mean
    is more than `tol` from `expected` (rad). No define."""
    return _expand("verify_hardstop", target, locals())


# ── tview-style text: one step per line ────────────────────────────────────────────────────────────────────────

_COMMENT = re.compile(r"(^|\s)#.*$")
_WHOLE = re.compile(r"[+-]?\d+")


def script(text: str, joints: Group | Sequence[Joint | str] | None = None) -> list[Primitive]:
    """A recipe as text, one step per line, after moteus tview (`A>d stop`, `11,12>d pos ...`, `11>d pos ...`):

        Park 20° off the stop:                  # a line ending in ":" labels the indented steps under it
            elbow> move_by 20deg vel=0.5 max_tau=8
            wait 0.3
        A> stop

    A step is `<targets>> <verb> [value] [key=value ...]`. Targets: `A` (all of `joints`), `a,b` (a set), `a`
    (one), by name or by `Joint.id`. Verbs and keywords are the ones of this module (`signature(verb)`); the
    first value may drop its keyword (`move_by 20deg` = `by=20deg`). Numbers are SI; a `deg` suffix on an angle
    converts (`-135deg`, `vel=30deg` = 30 °/s). `wait <seconds>` takes no target. `#` starts a comment.

    `joints` (a `Group`, or names / `Joint`s) is who `A` and the ids are, and where a step's defaults come
    from; without it a target is a bare name and `A` is every joint the controller drives (limp, stop, brake).
    Every primitive's `call` is its line as written. An error names the line number and the line."""
    group = joints if isinstance(joints, Group) or joints is None else Group(*joints, name="A")
    out: list[Primitive] = []
    label: tuple[int, str, int] | None = None  # (indent, words, line number) of the open label
    stretch: list[Primitive] = []

    def close() -> None:
        nonlocal label, stretch
        if label is None:
            return
        if not stretch:
            raise ValueError(
                f'script line {label[2]}: the label "{label[1]}" has no steps: indent its steps under it'
            )
        out.extend(step(label[1], stretch))
        label, stretch = None, []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = _COMMENT.sub("", raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        line = line.strip()
        try:
            if line.endswith(":"):
                if label is not None and indent > label[0]:
                    raise ValueError("a label inside a labelled stretch: un-indent it")
                close()
                if not line[:-1].strip():
                    raise ValueError("a label needs words before its `:`")
                label = (indent, line[:-1].strip(), number)
                continue
            if label is not None and indent <= label[0]:
                close()
            primitives = _stamp(_line(line, group), line)
        except (ValueError, LookupError, TypeError) as error:
            if str(error).startswith("script line "):
                raise
            raise ValueError(f"script line {number}: {error}\n    {line}") from None
        (stretch if label is not None else out).extend(primitives)
    close()
    if not out:
        raise ValueError("script: no steps")
    return out


def _line(line: str, group: Group | None) -> list[Primitive]:
    targets, arrow, rest = line.partition(">")
    if not arrow:
        targets, rest = "", line
    words = rest.split()
    if not words:
        raise ValueError("no verb after `>`")
    verb, words = words[0], words[1:]
    if verb == "wait":
        if arrow:
            raise ValueError("wait takes no target: write `wait 0.3`")
        if len(words) != 1 or "=" in words[0]:
            raise ValueError("wait takes the seconds: write `wait 0.3`")
        return helpers.wait(_number("timeout", words[0]))
    if verb not in VERBS:
        raise ValueError(f"no verb `{verb}`; the verbs are {', '.join(VERBS)}, wait")
    spec = VERBS[verb]
    if not arrow:
        raise ValueError(
            f"{verb} needs a target before `>`: `A> {verb}`, `elbow> {verb}` or `shoulder,elbow> {verb}`"
        )
    given: dict[str, Any] = {}
    for index, word in enumerate(words):
        key, equals, value = word.partition("=")
        if not equals:
            if index or spec.value is None:
                takes = ", ".join(spec.names) or "no keywords"
                raise ValueError(f"`{word}` needs its keyword ({verb} takes {takes})")
            key, value = spec.value, word
        if key not in spec.names:
            takes = ", ".join(spec.names) or "none"
            raise ValueError(f"{verb} has no keyword `{key}`; it takes {takes}")
        if key in given:
            raise ValueError(f"`{key}` is given twice")
        given[key] = _number(key, value)
    return _expand(verb, _targets(targets, group), given)


def _targets(text: str, group: Group | None) -> Any:
    names = [name.strip() for name in text.split(",")]
    if not all(names):
        raise ValueError("a target is missing: write `A>`, `elbow>` or `shoulder,elbow>`")
    if "A" in names:
        if len(names) > 1:
            raise ValueError("`A` (all) stands alone")
        return ALL if group is None else group
    if group is not None:
        return [group._find(int(name) if _WHOLE.fullmatch(name) else name) for name in names]
    for name in names:
        if _WHOLE.fullmatch(name):
            raise ValueError(f"an id ({name}) needs joints=: script(text, joints=Group(...))")
    return names


def _number(key: str, text: str) -> Any:
    if key == "verify" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    degrees = text.endswith("deg")
    digits = text[:-3] if degrees else text
    if degrees and key not in _ANGLES:
        raise ValueError(f"`{key}` is not an angle ({VOCABULARY[key]}); drop the `deg`")
    try:
        value: Any = int(digits) if _WHOLE.fullmatch(digits) and not degrees else float(digits)
    except ValueError:
        raise ValueError(f"`{text}` is not a number: write 0.5, -1 or 20deg") from None
    return math.radians(value) if degrees else value
