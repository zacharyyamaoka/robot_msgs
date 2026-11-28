"""JointTable — which servo drives each named joint, and how its board's frame maps to the joint's.

The thin "Arm" box between controllers and the bus reads one of these. Above it everything is joint-named
(`ServoCommand.name == ("j2", "c1", ...)`); below it the bus addresses servos by `(bus, can_id)`. The table
is the only place a CAN id or a transport key lives, so renumbering a board or moving it from an fdcanusb to a
pi3hat is a table edit and nothing else changes.

WHY (Zach, Daily Note 2026-11-27): "our commands right now we are putting in Can ID information thats silly.
we should be addressing by joint name and then the arm can basically like add in the can info and stuff
after... if I change from Can USB etc. I shouldn't need to change my code... what if you want to apply like a
joint offset or a zero... 1 light weight thing above". Design P1 in robot_sdk/docs/joint_controllers_arm.html.

Per servo it does only the 1:1 things: name <-> address, and ``joint = sign * servo + offset_rad``. No gear
ratio: the boards apply it themselves (moteus ``rotor_to_output_ratio``; bam_fivebar's boards run 0.1), so a
host-side ratio would apply it twice. The five-bar coupling is not here either: it stays in the robot model.

The servo side is a plain `ServoCommand` / `ServoState` whose `name[i]` is the servo's key ``"<bus>#<can_id>"``
(``"12"`` when no bus is set). WHY that and not a ``dict[(bus, can_id), row]``: it stays a message (log, codec,
compare, `mode`, NaN = board default all unchanged), one per tick across every bus, in the caller's order. A
bus driver builds its controllers once from `JointTable.addresses()` (key -> `(bus, can_id)`, as
``moteus.Controller(id=can_id, transport=transports[bus])``) and then walks `name` each tick; the inverse
lookup that bam_rack_twin's ``by_can`` dict hand-builds today is `JointTable.joint_at(key)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.control.joint_path import join_path, matching, split_path, strip_path
from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.control.servo_state import ServoState

KEY_SEPARATOR = "#"
"""Between the bus and the CAN id in a servo key: ``"fdcanusb:0607623A#12"``. A device path may hold ``/``
and ``:``, never ``#``; the CAN id is the digits after the last one."""


def servo_key(bus: str | None, can_id: int) -> str:
    """The servo-side name for `(bus, can_id)`: ``"<bus>#<can_id>"``, or ``"<can_id>"`` with no bus."""
    return str(int(can_id)) if bus is None else f"{bus}{KEY_SEPARATOR}{int(can_id)}"


def parse_servo_key(key: str) -> tuple[str | None, int]:
    """`servo_key`'s inverse: ``"can0#12"`` -> ``("can0", 12)``, ``"12"`` -> ``(None, 12)``."""
    bus, sep, can_id = str(key).rpartition(KEY_SEPARATOR)
    return (bus if sep else None), int(can_id)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Servo(Msg):
    """One joint's servo: where it is on the bus, and ``joint = sign * servo + offset_rad``.

    - `can_id`: the board's id on its bus.
    - `bus`: a transport key (``"fdcanusb:<serial>"``, ``"can0"``, ``"pi3hat:1"``); `None` means the table's
      `JointTable.bus`.
    - `sign`: +1 or -1, which way the board's output turns the joint. Velocity and effort map by it too.
    - `offset_rad`: the joint's position (rad, or m) when the board reads 0. After homing it is runtime state:
      `JointTable.with_zero` writes it.
    """

    can_id: int
    bus: str | None = None
    sign: float = 1.0
    offset_rad: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.can_id, bool) or int(self.can_id) != self.can_id or int(self.can_id) < 0:
            raise ValueError(f"Servo.can_id must be a non-negative integer, got {self.can_id!r}")
        set_field(self, "can_id", int(self.can_id))
        if self.bus is not None:
            bus = str(self.bus)
            if not bus or KEY_SEPARATOR in bus:
                raise ValueError(
                    f"Servo.bus must be non-empty and free of {KEY_SEPARATOR!r}, got {bus!r}"
                )
            set_field(self, "bus", bus)
        sign = float(self.sign)
        if sign not in (-1.0, 1.0):
            raise ValueError(f"Servo.sign is +1 or -1, got {self.sign!r}")
        set_field(self, "sign", sign)
        offset = float(self.offset_rad)
        if not np.isfinite(offset):
            raise ValueError(f"Servo.offset_rad must be finite, got {self.offset_rad!r}")
        set_field(self, "offset_rad", offset)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointTable(Msg):
    """Joint name -> `Servo`, in joint order; the only place a CAN id or transport key lives.

    `bus` is the default transport key for servos that name none. Joint names are unique (a mapping) and so
    are servo addresses `(bus, can_id)`. Convert with `to_servos` (joint-named command or state -> servo
    keys) and `to_joints` (the inverse); `with_zero` returns the table with homing's offsets.
    """

    joints: Mapping[str, Servo]
    bus: str | None = None

    def __post_init__(self) -> None:
        joints = FrozenDict((str(name), servo) for name, servo in dict(self.joints).items())
        for name, servo in joints.items():
            if not name:
                raise ValueError("JointTable: a joint name must be non-empty")
            if not isinstance(servo, Servo):
                raise TypeError(
                    f"JointTable.joints[{name!r}] must be a Servo, got {type(servo).__name__}"
                )
        set_field(self, "joints", joints)
        if self.bus is not None:
            bus = str(self.bus)
            if not bus or KEY_SEPARATOR in bus:
                raise ValueError(
                    f"JointTable.bus must be non-empty and free of {KEY_SEPARATOR!r}, got {bus!r}"
                )
            set_field(self, "bus", bus)
        seen: dict[str, str] = {}
        for name in joints:
            key = self.key(name)
            if key in seen:
                raise ValueError(
                    f"JointTable: joints {seen[key]!r} and {name!r} share servo {key!r}"
                )
            seen[key] = name
        object.__setattr__(self, "_by_key", seen)

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> tuple[str, ...]:
        """The joint names, in table order."""
        return tuple(self.joints)

    def address(self, joint: str) -> tuple[str | None, int]:
        """Joint `joint`'s servo as `(bus, can_id)`, the servo's own bus or else the table's."""
        servo = self._servo(joint)
        return (servo.bus if servo.bus is not None else self.bus), servo.can_id

    def key(self, joint: str) -> str:
        """Joint `joint`'s servo-side name, ``"<bus>#<can_id>"`` (see `servo_key`)."""
        return servo_key(*self.address(joint))

    def addresses(self) -> dict[str, tuple[str | None, int]]:
        """Servo key -> `(bus, can_id)`, in joint order: what a bus driver opens its controllers from."""
        return {self.key(name): self.address(name) for name in self.joints}

    def joint_at(self, key: str) -> str:
        """The joint the servo with key `key` drives."""
        try:
            return self._by_key[str(key)]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(
                f"JointTable: no joint on servo {key!r}; servos: {list(self._by_key)}"
            ) from None  # type: ignore[attr-defined]

    def with_zero(self, homed: Mapping[str, float]) -> JointTable:
        """This table with `offset_rad` replaced by homing's, per joint (``joint = sign * servo + homed[j]``).

        The homed zero is runtime state, written after every power-up's homing; joints not in `homed` keep
        theirs. An unknown joint name raises, so a stale homed map can't silently miss a joint.
        """
        unknown = sorted(set(homed) - set(self.joints))
        if unknown:
            raise KeyError(f"JointTable.with_zero: {unknown} not in the table {list(self.joints)}")
        return self.replace(
            joints={
                name: (servo.replace(offset_rad=float(homed[name])) if name in homed else servo)
                for name, servo in self.joints.items()
            }
        )

    def mounted(self, prefix: str) -> JointTable:
        """This table mounted into a cell under `prefix`: ``shoulder`` -> ``right_arm/shoulder``.

        Servos are unchanged, and each one is pinned to its resolved bus (its own, else this table's), so the
        mounted table still says where every board is once `combine` drops the per-table default.
        WHY (Zach, Daily Note 2026-11-28: "cell.right_arm.shoulder_joint.command() ... address at the
        different levels you want"): an arm's own table names joints bare, and the cell adds the prefix, so
        one arm description serves as left or right. See `robot_msgs.control.joint_path`.
        """
        if not split_path(prefix):
            raise ValueError("JointTable.mounted needs a non-empty prefix")
        return JointTable(
            joints={join_path(prefix, name): self._resolved(name) for name in self.joints}
        )

    def subtree(self, prefix: str, *, strip: bool = True) -> JointTable:
        """The joints under `prefix` (segment-safe), by default with the prefix removed: `mounted`'s inverse.

        ``cell.subtree("right_arm")`` is the right arm's own table again (bare names, each servo on its
        resolved bus). `strip=False` keeps the cell names. A prefix with no joint under it raises.
        """
        names = matching(self.joints, prefix)
        rename = (lambda n: strip_path(n, prefix)) if strip else (lambda n: n)
        return JointTable(joints={rename(name): self._resolved(name) for name in names})

    @classmethod
    def combine(cls, *tables: JointTable) -> JointTable:
        """Several (mounted) tables as one cell table, in argument order.

        Refuses a joint name in two tables and two joints on one servo `(bus, can_id)` — the same board
        claimed twice, e.g. two arms mounted with neither bus set. Each servo keeps its resolved bus.
        """
        joints: dict[str, Servo] = {}
        owner: dict[str, int] = {}
        for index, table in enumerate(tables):
            if not isinstance(table, JointTable):
                raise TypeError(f"JointTable.combine takes JointTables, got {type(table).__name__}")
            for name in table.joints:
                if name in joints:
                    raise ValueError(
                        f"JointTable.combine: joint {name!r} is in tables {owner[name]} and {index}"
                    )
                joints[name] = table._resolved(name)
                owner[name] = index
        # The constructor refuses a shared (bus, can_id) and names both joints.
        return cls(joints=joints)

    def _resolved(self, joint: str) -> Servo:
        servo = self._servo(joint)
        return servo if servo.bus is not None or self.bus is None else servo.replace(bus=self.bus)

    def to_servos(self, message: ServoCommand | ServoState) -> ServoCommand | ServoState:
        """A joint-named command or state -> the same message named by servo key, in the board's frame.

        ``servo = sign * (joint - offset)`` for position, ``sign * x`` for velocity and effort; gains, caps,
        `mode`, stamps and NaN pass through. Names a subset of joints in any order; the order is kept.
        """
        _check_type(message)
        rows = [self._servo(name) for name in message.name]
        sign, offset = _sign_offset(rows)
        keys = tuple(self.key(name) for name in message.name)
        return _mapped(message, keys, sign, lambda p: sign * (p - offset))

    def to_joints(self, message: ServoCommand | ServoState) -> ServoCommand | ServoState:
        """`to_servos`' inverse: a servo-keyed command or state -> joint-named, ``joint = sign * servo + offset``."""
        _check_type(message)
        names = tuple(self.joint_at(key) for key in message.name)
        sign, offset = _sign_offset([self.joints[name] for name in names])
        return _mapped(message, names, sign, lambda p: sign * p + offset)

    def _servo(self, joint: str) -> Servo:
        try:
            return self.joints[joint]
        except KeyError:
            raise KeyError(f"JointTable: no joint {joint!r}; joints: {list(self.joints)}") from None


def _check_type(message: Any) -> None:
    if not isinstance(message, (ServoCommand, ServoState)):
        raise TypeError(
            f"JointTable maps a ServoCommand or a ServoState, got {type(message).__name__}"
        )


def _sign_offset(rows: list[Servo]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([servo.sign for servo in rows], dtype=np.float64),
        np.array([servo.offset_rad for servo in rows], dtype=np.float64),
    )


def _signed(values: np.ndarray | None, sign: np.ndarray) -> np.ndarray | None:
    return None if values is None else sign * values


def _mapped(message: Any, names: tuple[str, ...], sign: np.ndarray, position: Any) -> Any:
    if isinstance(message, ServoCommand):
        return message.replace(
            name=names,
            position=position(message.position),
            velocity=_signed(message.velocity, sign),
            feedforward_effort=_signed(message.feedforward_effort, sign),
        )
    if isinstance(message, ServoState):
        return message.replace(
            name=names,
            position=position(message.position),
            velocity=_signed(message.velocity, sign),
            effort=_signed(message.effort, sign),
        )
    raise AssertionError("unreachable: _check_type ran first")


__all__ = ["KEY_SEPARATOR", "JointTable", "Servo", "parse_servo_key", "servo_key"]
