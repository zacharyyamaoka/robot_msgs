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
        for slot in self._slots.values():
            if not slot.active:
                continue
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
        slot.owned -= set(joints)
        if not slot.owned:
            slot.active = False

    def _deactivate(self, slot: _Slot) -> None:
        # WHY latch a hold, not limp: a controller switched out with nothing in its place must not drop the arm
        # (hold, don't drop: PEP 0029 addendum 2); the position is read on the next tick
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
