"""JointTrajectory — joint positions (and optionally velocities, accelerations, effort) over time.

Mirrors trajectory_msgs/JointTrajectory, stored struct-of-arrays: `positions` is one
``(N, J)`` array instead of N point messages. Iterate it to get the ROS-style points back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array, adopt, set_field
from robot_msgs.sensor import JointState
from robot_msgs.std import Header
from robot_msgs.trajectory.joint_limit_violation import joint_limit_violations
from robot_msgs.trajectory.joint_limits import JointLimits
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
    sample_fields,
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointTrajectoryPoint(TrajectoryPoint):
    """trajectory_msgs/JointTrajectoryPoint; `t` is ROS's `time_from_start`."""

    positions: Array | None = None
    velocities: Array | None = None
    accelerations: Array | None = None
    effort: Array | None = None


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointTrajectory(Trajectory):
    """Joint-space motion keyed by joint name. Every per-sample array is ``(N, J)``.

    Layer anything else you need to send along (gains, feed-forward torque, modes) as
    `channels`, and per-joint bounds as `limits`.
    """

    joint_names: tuple[str, ...]
    positions: Array = sample()
    velocities: Array | None = sample(default=None)
    accelerations: Array | None = sample(default=None)
    effort: Array | None = sample(default=None)
    limits: JointLimits | None = None

    point_type: ClassVar[type[TrajectoryPoint]] = JointTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        names = tuple(str(name) for name in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError(f"JointTrajectory.joint_names must be unique, got {names}")
        set_field(self, "joint_names", names)
        for f in sample_fields(type(self)):
            values = getattr(self, f.name)
            if values is not None and (values.ndim != 2 or values.shape[1] != len(names)):
                raise ValueError(
                    f"JointTrajectory.{f.name} has shape {values.shape}, "
                    f"expected ({len(self)}, {len(names)}) for joints {names}"
                )
        if self.limits is not None and self.limits.joint_names != names:
            missing = sorted(set(names) - set(self.limits.joint_names))
            if missing:
                raise ValueError(f"JointTrajectory.limits has no entry for joints {missing}")
            set_field(self, "limits", self.limits.select(names))

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError:
            raise KeyError(f"no joint {name!r}; joints are {self.joint_names}") from None

    def select(self, names: Sequence[str]) -> JointTrajectory:
        """Only `names`, in that order. Channels are per-sample, so they are kept whole."""
        columns = [self.joint_index(name) for name in names]
        changes = {
            f.name: adopt(getattr(self, f.name)[:, columns])
            for f in sample_fields(type(self))
            if getattr(self, f.name) is not None
        }
        limits = None if self.limits is None else self.limits.select(names)
        return self.replace(joint_names=tuple(names), limits=limits, **changes)

    def joint(self, name: str) -> JointTrajectory:
        """One joint, still a JointTrajectory (so plotting, resampling etc. all work)."""
        return self.select([name])

    def with_derivatives(self, *, overwrite: bool = False) -> JointTrajectory:
        """Fill missing velocities/accelerations by finite differences on the real time grid."""
        if len(self) < 2:
            raise ValueError("derivatives need at least two samples")
        edge_order = 2 if len(self) >= 3 else 1
        velocities = self.velocities
        if velocities is None or overwrite:
            velocities = np.gradient(self.positions, self.t, axis=0, edge_order=edge_order)
        accelerations = self.accelerations
        if accelerations is None or overwrite:
            accelerations = np.gradient(velocities, self.t, axis=0, edge_order=edge_order)
        return self.replace(velocities=velocities, accelerations=accelerations)

    def time_scaled(self, factor: float) -> JointTrajectory:
        """The same path, `factor` times slower (``factor < 1`` is faster).

        Velocities scale by 1/factor and accelerations by 1/factor². WHY effort is dropped: the
        torque a retimed motion needs is not the old torque scaled, it needs the dynamics.
        """
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        start = self.t[0] if len(self) else 0.0
        return self.replace(
            t=start + (self.t - start) * factor,
            velocities=None if self.velocities is None else self.velocities / factor,
            accelerations=None if self.accelerations is None else self.accelerations / factor**2,
            effort=None,
        )

    def joint_state_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> JointState:
        """The commanded state at `time` as a sensor JointState on the header's clock."""
        point = self.at(time, extrapolate=extrapolate)
        return JointState(
            name=self.joint_names,
            position=point.positions,
            velocity=point.velocities,
            effort=point.effort,
            header=Header(stamp=self.header.stamp + time, frame_id=self.header.frame_id),
        )

    @classmethod
    def from_joint_states(cls, states: Sequence[JointState], **static: object) -> JointTrajectory:
        """A recorded trajectory from a log of JointStates (stamps become `t`)."""
        if not states:
            raise ValueError("from_joint_states needs at least one JointState")
        names = states[0].name
        states = [s if s.name == names else s.select(names) for s in states]
        stamps = np.array([s.header.stamp for s in states])

        def stacked(attribute: str) -> np.ndarray | None:
            values = [getattr(s, attribute) for s in states]
            if all(v is None for v in values):
                return None
            if any(v is None for v in values):
                raise ValueError(f"JointState.{attribute} is set on some states but not all")
            return np.stack(values)

        header = Header(stamp=float(stamps[0]), frame_id=states[0].header.frame_id)
        return cls(
            t=stamps - stamps[0],
            joint_names=names,
            positions=stacked("position"),
            velocities=stacked("velocity"),
            effort=stacked("effort"),
            header=static.pop("header", header),  # type: ignore[arg-type]
            **static,  # type: ignore[arg-type]
        )

    def limit_violations(self, limits: JointLimits | None = None) -> tuple[str, ...]:
        """Human-readable worst violation per joint and bound; empty when within limits.

        The text of `joint_limit_violations(self, limits)`, whose records carry the numbers and
        each position bound's kind (one rule, two views). Jerk is checked from finite differences
        of `accelerations` when both are present.
        """
        return tuple(str(violation) for violation in joint_limit_violations(self, limits))
