"""ControllerManager: one owner of the bus, several controllers, one writer per joint.

ros2_control's shape, but one call (Zach, 2026-11-27: "How do we kinda manage multiple controllers that may want
to send?"; the Joint Command Controller spec's "a manager that switches, not clients that ask"):

    manager = ControllerManager(joint_names)
    manager.add("jcc", JointCommandController(joint_names), joints=joint_names, priority=10)
    manager.add("jtc", JointTrajectoryController(...), joints=arm_joints, priority=5, rate_hz=250)
    manager.send_goal("jcc", recipe)             # switches jcc in for the joints it needs, or refuses
    out = manager.tick(servo_state, dt)          # each active controller at its own rate, merged per joint
    manager.stop()                               # the board's stop on every joint next tick; always wins

- `send_goal(name, goal, priority=None)`: the controller named takes the joints the goal needs (its registered
  joints when the goal names none). A switch that would take a joint from an ACTIVE controller of higher priority
  is refused (returns None, `last_refusal` says why); otherwise the overlapping controllers are deactivated in the
  same tick (ros2_control's STRICT switch) and the goal goes to the controller (`submit` for a
  JointCommandController, `replace_input` for a trajectory controller).
- `switch(activate=..., deactivate=...)`: manual switching stays possible, priorities ignored.
- `tick(state, dt)`: a controller with `rate_hz` is ticked when its period has elapsed (its last command held in
  between); joints no controller owns hold where they were when their controller let go (hold, don't drop), or are
  limp if nothing ever commanded them. A manager `stop()` overrides every controller until a new goal or switch.
- `stop(joints=None)`: a manager call, not a controller.
- A JointCommandController goal with `helpers.follow(trajectory, via="jtc")`: the JCC lends those joints out
  (`JointCommandTick.follows`); the manager gives the trajectory to the controller registered as `jtc`
  (`replace_input`), writes those joints from the JTC while the follow lasts, and reports the JTC's `status` /
  `message` back to the JCC every tick (`follow_status`), so the JCC's Follow step completes, fails or times
  out on it. The joints stay the JCC's for priorities (a goal that takes them from the JCC ends the lend), and
  a manager `stop()` still wins next tick. WHY peers, not nesting (Zach, 2026-11-27): "I 100% want to chain
  controllers via composition not nesting... you get full access to the JTC but it's just side by side."

Controllers are anything with `tick(state, dt)` returning a `JointCommandTick` or a `(ServoCommand, defines)` pair
(robot_msgs' `JointController` protocol). A JointCommandController gets the `ServoState` (it reads faults and
`board_mode`); any other controller gets `state.joint_state()`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs.control.joint_command_controller import JointCommandController, JointCommandTick
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.std import Header


@dataclass
class _Slot:
    name: str
    controller: Any
    joints: tuple[str, ...]
    priority: float
    rate_hz: float | None
    active: bool = False
    owned: set[str] = field(default_factory=set)
    since: float = math.inf
    last: ServoCommand | None = None
    last_defines: Mapping[str, float] = field(default_factory=dict)
    lent: set[str] = field(default_factory=set)  # joints a JCC lent this controller for a follow


@dataclass
class _Route:
    """One JCC follow routed to trajectory controller `via` on `joints`; `refusal` set = reported aborted."""

    via: str
    joints: tuple[str, ...]
    refusal: str = ""


@dataclass(frozen=True)
class ManagerTick:
    """One tick: the merged wire `command` (per-servo `mode`), `defines` declared this tick, `owner` per joint
    (a controller name, `stop`, `hold` or `limp`) and each active controller's own log row."""

    command: ServoCommand
    defines: Mapping[str, float]
    owner: Mapping[str, str]
    logs: Mapping[str, Any]

    @property
    def modes(self) -> tuple[str, ...]:
        return self.command.mode or ("position",) * len(self.command.name)


class ControllerManager:
    """See the module docstring."""

    def __init__(self, joint_names: Iterable[str]) -> None:
        self.joint_names = tuple(str(j) for j in joint_names)
        self._slots: dict[str, _Slot] = {}
        self._owner: dict[str, str | None] = dict.fromkeys(self.joint_names)
        self._stopped: set[str] = set()
        self._held: dict[
            str, tuple[float, float]
        ] = {}  # joint -> (position, cap) latched when its owner let go
        self._released: set[str] = set()
        self._lent: dict[str, tuple[str, str]] = {}  # joint -> (lending JCC, trajectory controller)
        self._routes: dict[tuple[str, int], _Route] = {}  # (JCC name, FollowRequest id) -> route
        self.last_refusal = ""

    # ── registry and switching ─────────────────────────────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        controller: Any,
        *,
        joints: Iterable[str],
        priority: float = 0.0,
        rate_hz: float | None = None,
    ) -> None:
        joints = tuple(str(j) for j in joints)
        unknown = sorted(set(joints) - set(self.joint_names))
        if unknown:
            raise ValueError(f"controller {name!r}: unknown joints {unknown}")
        if name in self._slots:
            raise ValueError(f"controller {name!r} is already added")
        if rate_hz is not None and not (math.isfinite(rate_hz) and rate_hz > 0.0):
            raise ValueError(f"controller {name!r}: rate_hz must be positive, got {rate_hz!r}")
        self._slots[name] = _Slot(name, controller, joints, float(priority), rate_hz)

    @property
    def owners(self) -> dict[str, str | None]:
        return dict(self._owner)

    def active(self) -> tuple[str, ...]:
        return tuple(name for name, slot in self._slots.items() if slot.active)

    def switch(
        self,
        *,
        activate: Iterable[str] = (),
        deactivate: Iterable[str] = (),
        joints: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        """Manual switch: deactivate, then activate (each on its registered joints, or `joints[name]`)."""
        for name in deactivate:
            self._deactivate(self._slot(name))
        for name in activate:
            slot = self._slot(name)
            wanted = tuple((joints or {}).get(name, slot.joints))
            for other in {self._owner[j] for j in wanted} - {None, name}:
                self._take(self._slots[other], wanted)
            self._activate(slot, wanted)

    def send_goal(
        self, name: str, goal: Any, *, priority: float | None = None, client: str = ""
    ) -> Any:
        """Give `goal` to controller `name`, switching it in for the joints it needs; None when refused."""
        slot = self._slot(name)
        wanted = self._joints_of(slot, goal)
        priority = slot.priority if priority is None else float(priority)
        for joint in wanted:
            other = self._owner.get(joint)
            if other is not None and other != name and self._slots[other].priority > priority:
                self.last_refusal = (
                    f"{name} (priority {priority:g}) refused: {joint} belongs to {other} "
                    f"(priority {self._slots[other].priority:g})"
                )
                return None
        for other in {self._owner[j] for j in wanted} - {None, name}:
            self._take(self._slots[other], wanted)
        self._activate(slot, wanted)
        self.last_refusal = ""
        controller = slot.controller
        if hasattr(controller, "submit"):
            return controller.submit(goal, client=client or name)
        if hasattr(controller, "replace_input"):
            controller.replace_input(goal)
            return name
        raise TypeError(f"controller {name!r} takes no goals (no submit or replace_input)")

    def stop(self, joints: Iterable[str] | None = None) -> None:
        """The board's stop on these joints (all when None) from the next tick, over every controller."""
        named = self.joint_names if joints is None else tuple(joints)
        self._stopped |= set(named)
        for slot in self._slots.values():
            if slot.active and hasattr(slot.controller, "stop") and set(named) & slot.owned:
                slot.controller.stop(
                    client="manager", joints=tuple(set(named) & set(slot.controller.joint_names))
                )

    # ── the tick ───────────────────────────────────────────────────────────────────────────────────────────────

    def tick(
        self, state: Any, dt: float, *, board_mode: Mapping[str, Any] | None = None
    ) -> ManagerTick:
        measured = dict(zip(state.name, (float(v) for v in state.position), strict=True))
        for joint in self._released:
            if joint in self._held and math.isnan(self._held[joint][0]):
                self._held[joint] = (measured[joint], self._held[joint][1])
        self._released.clear()
        logs: dict[str, Any] = {}
        defines: dict[str, float] = {}
        joint_state = state.joint_state() if hasattr(state, "joint_state") else state
        # WHY JCCs first: a follow they lend out this tick reaches its trajectory controller in the same tick
        slots = sorted(
            self._slots.values(), key=lambda s: not isinstance(s.controller, JointCommandController)
        )
        for slot in slots:
            if not slot.active:
                continue
            if isinstance(slot.controller, JointCommandController):
                self._report_follows(slot)
            slot.since += dt
            period = 0.0 if slot.rate_hz is None else 1.0 / slot.rate_hz
            if slot.last is not None and slot.since + 1e-9 < period:
                continue
            step = slot.since if math.isfinite(slot.since) else 0.0
            slot.since = 0.0
            if isinstance(slot.controller, JointCommandController):
                out = slot.controller.tick(state, step, board_mode=board_mode)
            else:
                out = slot.controller.tick(joint_state, step)
            if isinstance(out, JointCommandTick):
                slot.last, slot.last_defines = out.command, out.defines
                logs[slot.name] = out.log
                if isinstance(slot.controller, JointCommandController):
                    self._route_follows(slot, out.follows)
            else:
                slot.last, slot.last_defines = out[0], out[1]
            defines.update({j: v for j, v in slot.last_defines.items() if j in slot.owned})
        return self._merge(state, defines, logs)

    def _merge(self, state: Any, defines: dict[str, float], logs: dict[str, Any]) -> ManagerTick:
        n = len(self.joint_names)
        nan = math.nan
        position, velocity = np.full(n, nan), np.zeros(n)
        feedforward, max_effort = np.zeros(n), np.zeros(n)
        stiffness, damping = np.full(n, nan), np.full(n, nan)
        modes: list[str] = []
        owner: dict[str, str] = {}
        for i, joint in enumerate(self.joint_names):
            name = self._owner[joint]
            slot = self._slots.get(name) if name is not None else None
            if joint in self._stopped:
                modes.append("stop")
                owner[joint] = "stop"
                continue
            lent = self._lent.get(joint)
            if lent is not None and name == lent[0]:
                slot, name = self._slots[lent[1]], lent[1]  # the JTC writes it, side by side with the JCC
            if slot is not None and slot.last is not None and joint in slot.last.name:
                c, k = slot.last, slot.last.name.index(joint)
                position[i] = c.position[k]
                velocity[i] = 0.0 if c.velocity is None else c.velocity[k]
                feedforward[i] = 0.0 if c.feedforward_effort is None else c.feedforward_effort[k]
                max_effort[i] = nan if c.max_effort is None else c.max_effort[k]
                stiffness[i] = nan if c.stiffness is None else c.stiffness[k]
                damping[i] = nan if c.damping is None else c.damping[k]
                modes.append(c.mode_of(joint))
                owner[joint] = name
            elif joint in self._held and not math.isnan(self._held[joint][0]):
                position[i], max_effort[i] = self._held[joint]
                modes.append("position")
                owner[joint] = "hold"
            else:
                modes.append("position")  # never commanded: limp (no target, no torque)
                owner[joint] = "limp"
        explicit = ~np.isnan(stiffness) | ~np.isnan(damping)
        header = getattr(state, "header", None) or Header()
        command = ServoCommand(
            name=self.joint_names,
            position=position,
            velocity=velocity,
            feedforward_effort=feedforward,
            stiffness=stiffness if explicit.any() else None,
            damping=damping if explicit.any() else None,
            max_effort=max_effort,
            state_stamp=header.stamp,
            mode=tuple(modes),
            header=Header(stamp=header.stamp, frame_id=header.frame_id),
        )
        return ManagerTick(command=command, defines=defines, owner=owner, logs=logs)

    # ── internals ──────────────────────────────────────────────────────────────────────────────────────────────

    def _slot(self, name: str) -> _Slot:
        if name not in self._slots:
            raise KeyError(f"no controller {name!r}; add() it first")
        return self._slots[name]

    def _joints_of(self, slot: _Slot, goal: Any) -> tuple[str, ...]:
        names = tuple(getattr(goal, "joints", ()) or getattr(goal, "joint_names", ()) or ())
        if (
            not names
            or isinstance(slot.controller, JointCommandController)
            and not set(names) <= set(slot.joints)
        ):
            return slot.joints
        return tuple(j for j in names if j in slot.joints) or slot.joints

    def _activate(self, slot: _Slot, joints: tuple[str, ...]) -> None:
        if not slot.active:
            slot.since, slot.last = math.inf, None
        slot.active = True
        slot.owned |= set(joints)
        for joint in joints:
            self._owner[joint] = slot.name
            self._stopped.discard(joint)
            self._held.pop(joint, None)

    def _take(self, slot: _Slot, joints: Iterable[str]) -> None:
        """Another controller takes `joints` from `slot` (the rest of slot's joints stay with it)."""
        joints = set(joints)
        self._unlend(joints, f"taken from {slot.name}")
        slot.owned -= joints
        slot.lent -= joints
        if not slot.owned and not slot.lent:
            slot.active = False

    # ── follows lent by a JointCommandController to a trajectory controller ───────────────────────────────────

    def _report_follows(self, slot: _Slot) -> None:
        for (jcc, request_id), route in self._routes.items():
            if jcc != slot.name:
                continue
            if route.refusal:
                slot.controller.follow_status(request_id, "aborted", route.refusal)
                continue
            # WHY "running" when it has no status: a controller that never says done times the step out
            via = self._slots[route.via].controller
            status = str(getattr(via, "status", "running"))
            slot.controller.follow_status(request_id, status, str(getattr(via, "message", "")))

    def _route_follows(self, slot: _Slot, follows: tuple[Any, ...]) -> None:
        live = {request.id: request for request in follows}
        for key in [key for key in self._routes if key[0] == slot.name and key[1] not in live]:
            self._unroute(key)
        for request in follows:
            key = (slot.name, request.id)
            route = self._routes.get(key)
            if route is not None:
                gone = set(route.joints) - set(request.joints)
                if gone:
                    self._unlend(gone, "")
                    route.joints = tuple(j for j in route.joints if j not in gone)
                continue
            self._routes[key] = self._route(slot, request)

    def _route(self, slot: _Slot, request: Any) -> _Route:
        target = self._slots.get(request.via)
        joints = tuple(j for j in request.joints if self._owner.get(j) == slot.name)
        refusal = ""
        if target is None:
            refusal = f"no controller {request.via!r} in the manager to follow via"
        elif isinstance(target.controller, JointCommandController) or not hasattr(
            target.controller, "replace_input"
        ):
            refusal = f"{request.via} takes no trajectory (no replace_input)"
        elif not set(request.joints) <= set(target.joints):
            refusal = f"{request.via} does not drive {sorted(set(request.joints) - set(target.joints))}"
        elif not joints:
            refusal = f"{slot.name} owns none of {list(request.joints)} in the manager"
        if refusal:
            return _Route(request.via, (), refusal)
        try:
            target.controller.replace_input(request.trajectory)
        except Exception as error:  # e.g. robot_controllers' TapeFeeder refusing while it runs
            return _Route(request.via, (), f"{request.via} refused the trajectory: {error}")
        mine = set(joints)
        if target.owned & mine:  # the JTC held some of them itself: they are the JCC's now
            target.owned -= mine
        target.active = True
        target.lent |= mine
        target.since, target.last = math.inf, None  # WHY: tick it this tick, on the new trajectory
        for joint in joints:
            self._lent[joint] = (slot.name, request.via)
        return _Route(request.via, joints)

    def _unroute(self, key: tuple[str, int]) -> None:
        route = self._routes.pop(key)
        self._unlend(set(route.joints), "")

    def _unlend(self, joints: set[str], why: str) -> None:
        for joint in sorted(joints):
            lent = self._lent.pop(joint, None)
            if lent is None:
                continue
            target = self._slots[lent[1]]
            target.lent.discard(joint)
            if not target.owned and not target.lent:
                target.active = False
            if why:
                for (jcc, _), route in self._routes.items():
                    if jcc == lent[0] and joint in route.joints and not route.refusal:
                        route.refusal = f"{joint} {why} by another goal"

    def _deactivate(self, slot: _Slot) -> None:
        # WHY latch a hold, not limp: a controller switched out with nothing in its place must not drop the arm
        # (hold, don't drop: PEP 0029 addendum 2); the position is read on the next tick
        self._unlend(set(slot.lent), f"switched out of {slot.name}")
        self._unlend({j for j, (jcc, _) in self._lent.items() if jcc == slot.name}, "")
        for joint in slot.owned:
            if self._owner.get(joint) == slot.name:
                self._owner[joint] = None
                cap = math.nan
                if (
                    slot.last is not None
                    and joint in slot.last.name
                    and slot.last.max_effort is not None
                ):
                    cap = float(slot.last.max_effort[slot.last.name.index(joint)])
                self._held[joint] = (math.nan, cap)
                self._released.add(joint)
        slot.owned.clear()
        slot.active = False
