"""Arm — the thin box a script talks to: joint-named `ServoCommand` in, joint-named `ServoState` out, on sim or real.

    arm = Arm(UR5E, backend="sim")          # the one-line switch: backend="real" on the rig
    controller.submit(sequence)
    state = arm.read()
    while controller.status == "executing":
        tick = controller.tick(state, arm.period_s)
        arm.write(tick.command, tick.defines)
        state = arm.read()
    state.truth                             # GroundTruth on sim, None on real

WHY no middleware (Zach, Daily Note 2026-11-28): "We pass the command seq straight into the command seq controller
and tick it; that will send the ServoCommands to the Arm. From the arm you can read back the servo state and you
are good!" The Arm is the `JointTable` (name <-> servo key, sign, offset; robot_sdk docs/joint_controllers_arm.html
P1) in front of a backend; it owns no thread, no clock and no port. like_keras can wrap it as a component later;
a script needs nothing but this.

- An `ArmDeployment` names one robot: its `JointTable` and, per backend name (``"sim"``, ``"real"``), a factory
  `(table, **options) -> ArmBackend`. robot_msgs never imports a simulator or a bus driver: the deployment's owner passes them
  in (the layer rule), so a UR5e deployment lives beside its plant, not here.
- An `ArmBackend` speaks servo keys (``"<bus>#<can_id>"``, `JointTable.addresses()`): `read()` a `ServoState`,
  `write(command, defines)`; `period_s` is its loop period and `simulated` says whether it may carry truth.
- The Arm maps both ways through the table and enforces one rule: a backend that is not `simulated` never hands a
  `GroundTruth` upward (`ServoState.truth` is `None` on real, always), so a test that asserts on truth cannot pass on
  real by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from robot_msgs.control.joint_table import JointTable
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.control.servo_state import ServoState


@runtime_checkable
class ArmBackend(Protocol):
    """What drives the servos behind an Arm: a MuJoCo plant, a moteus bus, a vendor SDK. Servo-keyed both ways."""

    simulated: bool
    period_s: float

    def read(self) -> ServoState: ...

    def write(self, command: ServoCommand, defines: Mapping[str, float] | None = None) -> Any: ...


class BackendUnavailable(RuntimeError):
    """A deployment's backend cannot be opened here (no hardware, no driver, refused on purpose)."""


@dataclass(frozen=True)
class ArmDeployment:
    """One robot: `name`, its `JointTable`, and per backend name a factory `(table, **options) -> ArmBackend`."""

    name: str
    joints: JointTable
    backends: Mapping[str, Callable[..., ArmBackend]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.joints, JointTable):
            raise TypeError(
                f"ArmDeployment.joints must be a JointTable, got {type(self.joints).__name__}"
            )
        object.__setattr__(self, "backends", dict(self.backends))


class Arm:
    """A deployment opened on one backend: `read()` / `write()` joint-named; see the module docstring.

    `Arm(deployment, backend="sim", **options)` builds the backend from the deployment (`options` go to its factory:
    the simulated world, a seed; give every backend the same ones so the switch stays one word);
    `Arm(table, backend=<an ArmBackend>)` takes one directly (a test's loopback, a backend you built yourself).
    """

    def __init__(
        self,
        deployment: ArmDeployment | JointTable,
        backend: str | ArmBackend = "sim",
        **options: Any,
    ) -> None:
        if isinstance(deployment, ArmDeployment):
            table = deployment.joints
            self.name = deployment.name
        elif isinstance(deployment, JointTable):
            table = deployment
            self.name = ""
        else:
            raise TypeError(
                f"Arm takes an ArmDeployment or a JointTable, got {type(deployment).__name__}"
            )
        if isinstance(backend, str):
            if not isinstance(deployment, ArmDeployment):
                raise TypeError(
                    "a backend named by a string needs an ArmDeployment (a JointTable has no backends)"
                )
            factories = deployment.backends
            if backend not in factories:
                raise KeyError(
                    f"deployment {deployment.name!r} has no backend {backend!r}; it has {sorted(factories)}"
                )
            self.backend_name = backend
            built = factories[backend](table, **options)
        else:
            if options:
                raise TypeError(
                    f"options {sorted(options)} go to a named backend's factory, not to a built backend"
                )
            self.backend_name = type(backend).__name__
            built = backend
        if not isinstance(built, ArmBackend):
            raise TypeError(
                f"backend {self.backend_name!r} is not an ArmBackend (read, write, period_s, simulated)"
            )
        self.joints = table
        self.backend = built

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.joints.names

    @property
    def period_s(self) -> float:
        return float(self.backend.period_s)

    @property
    def simulated(self) -> bool:
        return bool(self.backend.simulated)

    def read(self) -> ServoState:
        """The backend's newest sample, joint-named and in the joint frame; `truth` only when simulated."""
        state = self.joints.to_joints(self.backend.read())
        if not self.simulated and state.truth is not None:
            # WHY strip, not raise: the guarantee is about what a caller sees, and a real driver that copies a field
            # through must not stop the arm. The test suite pins that it is stripped.
            state = state.replace(truth=None)
        return state

    def write(self, command: ServoCommand, defines: Mapping[str, float] | None = None) -> Any:
        """Send a joint-named `command` (and `defines`, joint -> position this tick) to the servos it names."""
        servo_defines = None
        if defines:
            servo_defines = {}
            for joint, value in defines.items():
                servo = self.joints.joints[joint]
                servo_defines[self.joints.key(joint)] = servo.sign * (
                    float(value) - servo.offset_rad
                )
        return self.backend.write(self.joints.to_servos(command), servo_defines)

    def close(self) -> None:
        """Close the backend when it has a `close()`."""
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Arm:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Arm({self.name or 'table'!r}, backend={self.backend_name!r}, joints={list(self.joint_names)})"


__all__ = ["Arm", "ArmBackend", "ArmDeployment", "BackendUnavailable"]
