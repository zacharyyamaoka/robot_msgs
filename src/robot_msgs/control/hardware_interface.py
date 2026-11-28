"""HardwareInterface — the thin box a script talks to: joint-named `ServoCommand` in, joint-named `ServoState` out, on sim or real.

    hw = HardwareInterface(UR5E, backend="sim")           # the one-line switch: backend="real" on the rig
    controller.submit(sequence)
    state = hw.read()
    while controller.status == "executing":
        tick = controller.tick(state, hw.period_s)
        hw.write(tick.command, tick.defines)
        state = hw.read()
    state.truth                             # GroundTruth on sim, None on real

WHY no middleware (Zach, Daily Note 2026-11-28): "We pass the command seq straight into the command seq controller
and tick it; that will send the ServoCommands to the HardwareInterface. From the arm you can read back the servo state and you
are good!" The HardwareInterface is the `JointTable` (name <-> servo key, sign, offset; robot_sdk docs/joint_controllers_arm.html
P1) in front of a backend; it owns no thread, no clock and no port. like_keras can wrap it as a component later;
a script needs nothing but this.

- An `HardwareDeployment` names one robot: its `JointTable` and, per backend name (``"sim"``, ``"real"``), a factory
  `(table, **options) -> HardwareBackend`. robot_msgs never imports a simulator or a bus driver: the deployment's owner passes them
  in (the layer rule), so a UR5e deployment lives beside its plant, not here.
- An `HardwareBackend` speaks servo keys (``"<bus>#<can_id>"``, `JointTable.addresses()`): `read()` a `ServoState`,
  `write(command, defines)`; `period_s` is its loop period and `simulated` says whether it may carry truth.
- Connecting to one concrete deployment (Zach, Daily Note 2026-11-28: "you get like a HardwareInterface to a Particular
  Deployment"): ``HardwareInterface.from_robot_description(rd, MujocoConfig(), config=robot_config, parameters=twin_values)``.
  The `BackendConfig` (a frozen dataclass per backend: how a run happens) builds the backend from the robot's description `rd`
  (nominal values), its `RobotConfig` (this copy's `JointTable` and transports) and `parameters` (a twin's calibrated values,
  or `None` for the nominal ones). robot_msgs owns only that protocol; the simulators and bus drivers that implement it live
  downstream (sim2real, a rack driver), so the dependency arrow still points from them to here.
- The HardwareInterface maps both ways through the table and enforces one rule: a backend that is not `simulated` never hands a
  `GroundTruth` upward (`ServoState.truth` is `None` on real, always), so a test that asserts on truth cannot pass on
  real by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from robot_msgs.control.joint_table import JointTable
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.control.servo_state import ServoState

if TYPE_CHECKING:
    from robot_msgs.robot import RobotConfig

# WHY HardwareInterface and not Arm (Zach, Daily Note 2026-11-28): "I want the type though to not be Arm, but instead be
# HardwareInterface. Arm is just like a semantic meaning, don't mix into the back end implementation... Arm is not gonna
# scale to Hand, Cell, etc." The same type fronts an arm, a hand, a rack station or a whole cell (a combined JointTable).


@runtime_checkable
class HardwareBackend(Protocol):
    """What drives the servos behind an HardwareInterface: a MuJoCo plant, a moteus bus, a vendor SDK. Servo-keyed both ways."""

    simulated: bool
    period_s: float

    def read(self) -> ServoState: ...

    def write(self, command: ServoCommand, defines: Mapping[str, float] | None = None) -> Any: ...


@runtime_checkable
class BackendConfig(Protocol):
    """How a run happens on one backend, as a frozen dataclass (``MujocoConfig(physics_dt_s=...)``, ``RealConfig(adapter=...)``):
    `build(rd, config, parameters)` returns the `HardwareBackend` for robot description `rd`, robot copy `config` (a `RobotConfig`)
    and `parameters` (calibrated values, or `None` for the description's nominal ones).

    WHY one config type per backend, the config selecting it (Zach, Daily Note 2026-11-28, pick P2): a timestep means nothing to a
    real board and an adapter serial nothing to MuJoCo, so each backend's knobs stay its own, and switching backend is switching
    one value. WHY `parameters` beside `rd` and not written into it (the same note): the description holds the nominal values
    every copy shares, the twin the calibrated values of this copy; a backend composes the two, never edits the description."""

    def build(self, rd: Any, config: RobotConfig, parameters: Any = None) -> HardwareBackend: ...


class BackendUnavailable(RuntimeError):
    """A deployment's backend cannot be opened here (no hardware, no driver, refused on purpose)."""


@dataclass(frozen=True)
class HardwareDeployment:
    """One robot: `name`, its `JointTable`, and per backend name a factory `(table, **options) -> HardwareBackend`."""

    name: str
    joints: JointTable
    backends: Mapping[str, Callable[..., HardwareBackend]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.joints, JointTable):
            raise TypeError(
                f"HardwareDeployment.joints must be a JointTable, got {type(self.joints).__name__}"
            )
        object.__setattr__(self, "backends", dict(self.backends))


class HardwareInterface:
    """A deployment opened on one backend: `read()` / `write()` joint-named; see the module docstring.

    `HardwareInterface(deployment, backend="sim", **options)` builds the backend from the deployment (`options` go to its factory:
    the simulated world, a seed; give every backend the same ones so the switch stays one word);
    `HardwareInterface(table, backend=<an HardwareBackend>)` takes one directly (a test's loopback, a backend you built yourself).
    """

    def __init__(
        self,
        deployment: HardwareDeployment | JointTable,
        backend: str | HardwareBackend = "sim",
        **options: Any,
    ) -> None:
        if isinstance(deployment, HardwareDeployment):
            table = deployment.joints
            self.name = deployment.name
        elif isinstance(deployment, JointTable):
            table = deployment
            self.name = ""
        else:
            raise TypeError(
                f"HardwareInterface takes an HardwareDeployment or a JointTable, got {type(deployment).__name__}"
            )
        if isinstance(backend, str):
            if not isinstance(deployment, HardwareDeployment):
                raise TypeError(
                    "a backend named by a string needs an HardwareDeployment (a JointTable has no backends)"
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
        if not isinstance(built, HardwareBackend):
            raise TypeError(
                f"backend {self.backend_name!r} is not an HardwareBackend (read, write, period_s, simulated)"
            )
        self.joints = table
        self.backend = built
        # what from_robot_description connected it to; None when built from a deployment or a table directly
        self.description: Any = None
        self.backend_config: BackendConfig | None = None
        self.config: RobotConfig | None = None
        self.parameters: Any = None

    @classmethod
    def from_robot_description(
        cls,
        rd: Any,
        backend: BackendConfig,
        *,
        config: RobotConfig | None = None,
        parameters: Any = None,
    ) -> HardwareInterface:
        """A HardwareInterface to one concrete deployment: robot `rd`, copy `config`, on the backend `backend` selects.

            hw = HardwareInterface.from_robot_description(rd, MujocoConfig(), config=PENDULUM_CAN11, parameters=fitted.parameters)

        `config` None takes ``rd.robot_config`` (an attribute, or a method called with no arguments) when rd carries one, and
        refuses otherwise: a connection is to one copy, never to "a robot" in general. `parameters` None means the description's
        nominal values. The interface keeps all four (`.description`, `.backend_config`, `.config`, `.parameters`), so what it is
        connected to is always readable from it.

        WHY a classmethod mirroring sim2real's ``RobotTwin.from_robot_description(rd)`` (Zach, Daily Note 2026-11-28: "sim2real
        first connects to it with a nominal robottwin then [calibrates], as sim2real lives on top of this"): generate, connect,
        calibrate. robot_msgs imports no description package and no backend; it reads `rd` only through the backend config.
        """
        if not isinstance(backend, BackendConfig):
            raise TypeError(
                f"from_robot_description takes a BackendConfig (a config with build(rd, config, parameters)), got {type(backend).__name__}"
            )
        if config is None:
            found = getattr(rd, "robot_config", None)
            config = found() if callable(found) else found
            if config is None:
                raise TypeError(
                    f"from_robot_description needs config=<RobotConfig>: {type(rd).__name__} carries no robot_config, and a "
                    "HardwareInterface connects to one robot copy (its JointTable and transports), never to a robot in general"
                )
        from robot_msgs.robot import RobotConfig  # noqa: PLC0415  WHY lazy: robot.RobotConfig reads control's JointTable

        if not isinstance(config, RobotConfig):
            raise TypeError(f"config must be a robot_msgs.robot.RobotConfig, got {type(config).__name__}")
        built = backend.build(rd, config, parameters)
        if not isinstance(built, HardwareBackend):
            close = getattr(built, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"{type(backend).__name__}.build returned {type(built).__name__}, not an HardwareBackend (read, write, period_s, simulated)"
            )
        hw = cls(config.joints, backend=built)
        hw.name, hw.backend_name = config.name, type(backend).__name__
        hw.description, hw.backend_config, hw.config, hw.parameters = rd, backend, config, parameters
        return hw

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

    def __enter__(self) -> HardwareInterface:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"HardwareInterface({self.name or 'table'!r}, backend={self.backend_name!r}, joints={list(self.joint_names)})"


__all__ = ["HardwareInterface", "HardwareBackend", "HardwareDeployment", "BackendConfig", "BackendUnavailable"]
