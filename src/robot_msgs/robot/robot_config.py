"""RobotConfig — how the robot is set up, as each component's own configuration message."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.robot._components import Components
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class RobotConfig(Components, Msg):
    """Component name → that component's configuration message, e.g.
    ``RobotConfig(components={"limits": JointLimits(...), "camera": CameraSettings(...)})``.

    Same shape and methods as `RobotState`: ``config["camera"]``, ``with_components``, ``without``.

    WHY separate from `RobotState`, saved together only when you bundle them: keras keeps config
    (``get_config`` / ``from_config``, ``config.json``) apart from state (``model.weights.h5``)
    and ``model.save`` zips both into one ``.keras`` file; torch keeps ``state_dict()`` apart from
    the code and arguments that build the model, and Hugging Face's ``save_pretrained`` writes
    ``config.json`` beside the weights. Both: two things, one artifact on demand. Here:
    ``robot_msgs.save({"config": config, "state": state}, "arm.npz")`` (Zach asked, 2026-11-26).

    One robot copy's hardware configuration rides here too (2026-11-28): `name` (this copy, ``"pendulum_can11"``), component
    ``"joints"`` (its `JointTable`: joint -> bus and CAN id, sign, offset, homed zero) and component ``"transports"`` (its
    `Transports`: bus key -> device). WHY here (Zach, Daily Note 2026-11-28, the model he agreed: the RobotDescription holds the
    nominal values every copy shares, a sim2real RobotTwin the calibrated ones unique to a copy, and the RobotConfig the
    configured ones unique to a copy; "I think the joints table is just like part of the RobotConfig maybe? idk" - the default
    taken). `HardwareInterface.from_robot_description(rd, <BackendConfig>, config=<RobotConfig>)` connects to the copy it names.
    A transport for a bus no joint is on is refused (a stale entry).
    """

    components: Mapping[str, Msg] = field(default_factory=FrozenDict)
    header: Header = field(default_factory=Header)
    name: str = ""

    def __post_init__(self) -> None:
        self._freeze_components()
        set_field(self, "name", str(self.name))
        if "transports" in self.components and "joints" in self.components:
            unused = sorted(set(self.transports.buses) - set(self.buses))
            if unused:
                raise ValueError(f"RobotConfig {self.name!r}: transports for {unused}, which no joint's servo is on "
                                 f"(buses {list(self.buses)})")

    @property
    def joints(self):
        """Component ``"joints"``, the copy's `JointTable` (a KeyError naming the components when it has none)."""
        from robot_msgs.control.joint_table import JointTable  # noqa: PLC0415  WHY lazy: control imports nothing of robot

        table = self["joints"]
        if not isinstance(table, JointTable):
            raise TypeError(f"RobotConfig component 'joints' must be a JointTable, got {type(table).__name__}")
        return table

    @property
    def transports(self):
        """Component ``"transports"`` (`Transports`); an empty one when the config names none (a sim-only config)."""
        from robot_msgs.control.transports import Transports  # noqa: PLC0415

        found = self.components.get("transports")
        if found is None:
            return Transports()
        if not isinstance(found, Transports):
            raise TypeError(f"RobotConfig component 'transports' must be a Transports, got {type(found).__name__}")
        return found

    @property
    def buses(self) -> tuple[str, ...]:
        """The bus keys the joints' servos are on, in joint order."""
        return tuple(dict.fromkeys(bus for bus, _ in self.joints.addresses().values() if bus is not None))

    def transport(self, bus: str) -> str:
        """The device `bus` is reached through (KeyError when the config names none)."""
        return self.transports.device(bus)

    def with_zero(self, homed: Mapping[str, float]) -> RobotConfig:
        """This config with its table's homed zero (`JointTable.with_zero`): runtime state after each power-up's homing."""
        return self.with_components(joints=self.joints.with_zero(homed))

    @classmethod
    def for_hardware(cls, name: str, joints, transports: Mapping[str, str] | None = None, **components: Msg) -> RobotConfig:
        """``RobotConfig.for_hardware("pendulum_can11", table, {"fdcanusb:272813EF": "/dev/serial/by-id/..."})``."""
        from robot_msgs.control.transports import Transports  # noqa: PLC0415

        parts: dict[str, Msg] = {"joints": joints, **components}
        if transports:
            parts["transports"] = Transports(buses=transports)
        return cls(name=name, components=parts)
