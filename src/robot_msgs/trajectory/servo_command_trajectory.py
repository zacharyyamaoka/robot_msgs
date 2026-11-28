"""ServoCommandTrajectory — a log of `ServoCommand`s: what a controller sent its servos, over time.

Struct-of-arrays like `JointTrajectory`: `positions` is one ``(N, J)`` array, one column a servo.
Sim and real controllers record it alike, so what each one *asked for* can be compared too.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array
from robot_msgs.control import ServoCommand
from robot_msgs.control.servo_command import SERVO_MODES
from robot_msgs.std import Header
from robot_msgs.trajectory._joint_columns import (
    check_joint_columns,
    column_order,
    joint_index,
    select_joint_columns,
)
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)

_FIELDS = (  # (field on the trajectory, field on ServoCommand)
    ("positions", "position"),
    ("velocities", "velocity"),
    ("feedforward_effort", "feedforward_effort"),
    ("stiffness", "stiffness"),
    ("damping", "damping"),
    ("max_effort", "max_effort"),
)
MODE_CODES: dict[str, int] = {mode: code for code, mode in enumerate(SERVO_MODES)}
"""`mode` column code per `SERVO_MODES` entry: 0 `position`, 1 `stop`, 2 `brake`."""
LATENCY = "latency"
"""The channel `from_servo_commands` writes: seconds from the state sample to the command."""


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoCommandTrajectoryPoint(TrajectoryPoint):
    """One sample of a `ServoCommandTrajectory`: every servo's targets at `t`."""

    positions: Array | None = None
    velocities: Array | None = None
    feedforward_effort: Array | None = None
    stiffness: Array | None = None
    damping: Array | None = None
    max_effort: Array | None = None
    mode: Array | None = field(default=None, metadata={"dtype": np.int64})


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoCommandTrajectory(Trajectory):
    """Targets sent to servos `joint_names` over time; every per-sample array is ``(N, J)``.

    Units as `ServoCommand` (SI; NaN in `positions` = no position target). The float channel
    ``latency`` (s) is each command's sensor-to-command latency (``stamp - state_stamp``), NaN for
    an open-loop command; `from_servo_commands` writes it whenever any command names its state.
    WHY a channel, not a field: it is a property of the loop that sent the command, not a target.

    `mode` ``(N, J)`` int is each servo's `ServoCommand.mode` as its `MODE_CODES` code (0 position,
    1 stop, 2 brake), held between samples (a mode is never interpolated); None = every servo in
    position. WHY a column: without it a recorded stop or brake came back as a position command.
    """

    joint_names: tuple[str, ...]
    positions: Array = sample()
    velocities: Array | None = sample(default=None)
    feedforward_effort: Array | None = sample(default=None)
    stiffness: Array | None = sample(default=None)
    damping: Array | None = sample(default=None)
    max_effort: Array | None = sample(default=None)
    mode: Array | None = sample(interp="hold", default=None, dtype=np.int64)

    point_type: ClassVar[type[TrajectoryPoint]] = ServoCommandTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        check_joint_columns(self)
        if self.mode is not None and ((self.mode < 0) | (self.mode >= len(SERVO_MODES))).any():
            raise ValueError(f"ServoCommandTrajectory.mode codes must index SERVO_MODES {SERVO_MODES}")

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        return joint_index(self, name)

    def select(self, names: Sequence[str]) -> ServoCommandTrajectory:
        """Only servos `names`, in that order. Channels are per-sample, so they are kept whole."""
        return select_joint_columns(self, names)

    def command_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> ServoCommand:
        """The command at `time` as a `ServoCommand` stamped on the header's clock; its
        `state_stamp` comes back from the ``latency`` channel when that is set and finite."""
        point = self.at(time, extrapolate=extrapolate)
        stamp = self.header.stamp + time
        latency = point.channels.get(LATENCY)
        state_stamp = None if latency is None or not np.isfinite(latency) else stamp - latency
        return ServoCommand(
            name=self.joint_names,
            position=point.positions,
            velocity=point.velocities,
            feedforward_effort=point.feedforward_effort,
            stiffness=point.stiffness,
            damping=point.damping,
            max_effort=point.max_effort,
            state_stamp=state_stamp,
            mode=None if point.mode is None else tuple(SERVO_MODES[int(c)] for c in point.mode),
            header=Header(stamp=stamp, frame_id=self.header.frame_id),
        )

    @classmethod
    def from_servo_commands(
        cls, commands: Sequence[ServoCommand], **static: object
    ) -> ServoCommandTrajectory:
        """A recorded stream from a log of ServoCommands: stamps become `t`, the first one the
        header's stamp, every command in the first one's servo order (by name), the ``latency``
        channel when any command names the state it came from, and `mode` when any command sets it."""
        if not commands:
            raise ValueError("from_servo_commands needs at least one ServoCommand")
        names = commands[0].name
        orders = [column_order(c.name, names, "ServoCommand") for c in commands]
        stamps = np.array([c.header.stamp for c in commands], dtype=np.float64)

        def stacked(attribute: str) -> np.ndarray | None:
            values = [getattr(c, attribute) for c in commands]
            if all(v is None for v in values):
                return None
            if any(v is None for v in values):
                raise ValueError(f"ServoCommand.{attribute} is set on some commands but not all")
            return np.stack([v[order] for v, order in zip(values, orders, strict=True)])

        channels = {}
        if any(c.latency is not None for c in commands):
            channels[LATENCY] = np.array(
                [np.nan if c.latency is None else c.latency for c in commands]
            )
        header = Header(stamp=float(stamps[0]), frame_id=commands[0].header.frame_id)
        fields = {name: stacked(attribute) for name, attribute in _FIELDS}
        if any(c.mode is not None for c in commands):
            # WHY None counts as all position here (not "set on some but not all"): that is what it means
            fields["mode"] = np.array(
                [[MODE_CODES[c.mode_of(name)] for name in names] for c in commands], dtype=np.int64
            )
        return cls(
            t=stamps - stamps[0],
            joint_names=names,
            channels=static.pop("channels", channels),  # type: ignore[arg-type]
            header=static.pop("header", header),  # type: ignore[arg-type]
            **fields,  # type: ignore[arg-type]
            **static,  # type: ignore[arg-type]
        )
