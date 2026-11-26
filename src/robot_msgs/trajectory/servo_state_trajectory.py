"""ServoStateTrajectory — a log of `ServoState`s: what a set of servos reported, over time.

Struct-of-arrays like `JointTrajectory`: `positions` is one ``(N, J)`` array, one column a servo.
It is what a driver records on sim and on real, so the two can be compared stream by stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array
from robot_msgs.control import ServoState
from robot_msgs.std import Header
from robot_msgs.trajectory._joint_columns import (
    check_joint_columns,
    column_order,
    joint_index,
    select_joint_columns,
)
from robot_msgs.trajectory.joint_trajectory import JointTrajectory
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)

_FIELDS = (  # (field on the trajectory, field on ServoState)
    ("positions", "position"),
    ("velocities", "velocity"),
    ("effort", "effort"),
    ("temperature", "temperature"),
    ("voltage", "voltage"),
    ("fault", "fault"),
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoStateTrajectoryPoint(TrajectoryPoint):
    """One sample of a `ServoStateTrajectory`: every servo's reading at `t`."""

    positions: Array | None = None
    velocities: Array | None = None
    effort: Array | None = None
    temperature: Array | None = None
    voltage: Array | None = None
    fault: Array | None = field(default=None, metadata={"dtype": np.int64})


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoStateTrajectory(Trajectory):
    """What servos `joint_names` reported over time; every per-sample array is ``(N, J)``.

    Units as `ServoState`: `positions` rad or m, `velocities`, `effort` N·m or N, `temperature` °C,
    `voltage` V, `fault` the board's integer code (0 = no fault). `fault` is held between samples
    (a code is never interpolated); every other field is linear. `from_servo_states` keeps each
    sample's `seq` as the int channel ``seq``, so a dropped or repeated sample stays visible.
    """

    joint_names: tuple[str, ...]
    positions: Array = sample()
    velocities: Array | None = sample(default=None)
    effort: Array | None = sample(default=None)
    temperature: Array | None = sample(default=None)
    voltage: Array | None = sample(default=None)
    fault: Array | None = sample(interp="hold", default=None, dtype=np.int64)

    point_type: ClassVar[type[TrajectoryPoint]] = ServoStateTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        check_joint_columns(self)

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        return joint_index(self, name)

    def select(self, names: Sequence[str]) -> ServoStateTrajectory:
        """Only servos `names`, in that order. Channels are per-sample, so they are kept whole."""
        return select_joint_columns(self, names)

    def joint_trajectory(self) -> JointTrajectory:
        """The kinematic part (positions, velocities, effort), as the stack reads joints."""
        return JointTrajectory(
            t=self.t,
            joint_names=self.joint_names,
            positions=self.positions,
            velocities=self.velocities,
            effort=self.effort,
            channels=self.channels,
            header=self.header,
        )

    def state_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> ServoState:
        """The reading at `time` as a `ServoState` stamped on the header's clock."""
        point = self.at(time, extrapolate=extrapolate)
        seq = point.channels.get("seq")
        return ServoState(
            name=self.joint_names,
            position=point.positions,
            velocity=point.velocities,
            effort=point.effort,
            temperature=point.temperature,
            voltage=point.voltage,
            fault=point.fault,
            seq=0 if seq is None else int(seq),
            header=Header(stamp=self.header.stamp + time, frame_id=self.header.frame_id),
        )

    @classmethod
    def from_servo_states(
        cls, states: Sequence[ServoState], **static: object
    ) -> ServoStateTrajectory:
        """A recorded stream from a log of ServoStates: stamps become `t`, the first one the
        header's stamp, and every state is put in the first one's servo order (by name)."""
        if not states:
            raise ValueError("from_servo_states needs at least one ServoState")
        names = states[0].name
        orders = [column_order(s.name, names, "ServoState") for s in states]
        stamps = np.array([s.header.stamp for s in states], dtype=np.float64)

        def stacked(attribute: str) -> np.ndarray | None:
            values = [getattr(s, attribute) for s in states]
            if all(v is None for v in values):
                return None
            if any(v is None for v in values):
                raise ValueError(f"ServoState.{attribute} is set on some states but not all")
            return np.stack([v[order] for v, order in zip(values, orders, strict=True)])

        header = Header(stamp=float(stamps[0]), frame_id=states[0].header.frame_id)
        fields = {name: stacked(attribute) for name, attribute in _FIELDS}
        channels = {"seq": np.array([s.seq for s in states], dtype=np.int64)}
        return cls(
            t=stamps - stamps[0],
            joint_names=names,
            channels=static.pop("channels", channels),  # type: ignore[arg-type]
            header=static.pop("header", header),  # type: ignore[arg-type]
            **fields,  # type: ignore[arg-type]
            **static,  # type: ignore[arg-type]
        )
