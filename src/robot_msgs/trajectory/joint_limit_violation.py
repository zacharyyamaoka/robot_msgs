"""JointLimitViolation — a joint past one of its bounds in a run, found by `joint_limit_violations`.

WHY a record and not only a sentence (2026-11-11): runs are judged after the fact (Zach: "check the
position read out and make sure it stayed within the limits the entire time"), and a verdict needs
the joint, the side, the kind of bound and the numbers. `JointTrajectory.limit_violations()` prints
the same records, so the sentence and the record can never disagree (one rule, two views).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.trajectory.joint_limits import POSITION_KINDS, JointLimits

if TYPE_CHECKING:
    from robot_msgs.trajectory.joint_trajectory import JointTrajectory

QUANTITIES = ("position", "velocity", "acceleration", "jerk", "effort")
SIDES = ("min", "max")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointLimitViolation(Msg):
    """One joint past one bound of its `JointLimits` during a run, over every sample past it.

    - `quantity` is ``position``, ``velocity``, ``acceleration``, ``jerk`` or ``effort``. `side` is
      ``min`` or ``max``; every quantity but position is a magnitude (``|value| > limit``), so its
      side is ``max``.
    - `kind` is the position bound's ``physical`` or ``software`` (`JointLimits.position_kind`);
      ``None`` when it is not stated, and for every other quantity.
    - `limit` is the bound. `value` is the sample furthest past it, as recorded (signed), `excess`
      how far past it is (> 0, in the quantity's unit) and `t` its time (s).
    - `t_first` and `t_last` are the first and the last sample past the bound, and `samples` how
      many were. Nothing is interpolated between samples: the record says what was sampled.
    """

    joint: str
    quantity: str
    side: str
    kind: str | None = None
    limit: float
    value: float
    excess: float
    t: float
    t_first: float
    t_last: float
    samples: int

    def __post_init__(self) -> None:
        set_field(self, "joint", str(self.joint))
        if self.quantity not in QUANTITIES:
            raise ValueError(f"JointLimitViolation.quantity {self.quantity!r} not in {QUANTITIES}")
        if self.side not in SIDES:
            raise ValueError(f"JointLimitViolation.side {self.side!r} not in {SIDES}")
        if self.quantity != "position" and self.side != "max":
            raise ValueError(f"{self.quantity} is a magnitude: its only side is 'max'")
        if self.kind is not None:
            if self.quantity != "position":
                raise ValueError("only a position bound has a kind")
            if str(self.kind) not in POSITION_KINDS:
                raise ValueError(f"JointLimitViolation.kind {self.kind!r} not in {POSITION_KINDS}")
            set_field(self, "kind", str(self.kind))
        for name in ("limit", "value", "excess", "t", "t_first", "t_last"):
            set_field(self, name, float(getattr(self, name)))
        set_field(self, "samples", int(self.samples))
        if not self.excess > 0.0:
            raise ValueError("a violation is past its bound: excess must be > 0")
        if self.samples < 1:
            raise ValueError("a violation has at least one sample past its bound")
        if not self.t_first <= self.t <= self.t_last:
            raise ValueError("the worst sample must lie between the first and the last one")

    def __str__(self) -> str:
        """One line: ``elbow: |velocity| reaches 0.99 > 0.5 at t=2 s`` (the worst sample)."""
        magnitude = self.quantity != "position"
        label = f"|{self.quantity}|" if magnitude else self.quantity
        relation = "<" if self.side == "min" else ">"
        shown = abs(self.value) if magnitude else self.value
        return (
            f"{self.joint}: {label} reaches {shown:.4g} {relation} {self.limit:.4g} "
            f"at t={self.t:.4g} s"
        )


def joint_limit_violations(
    trajectory: JointTrajectory,
    limits: JointLimits | None = None,
    *,
    quantities: Sequence[str] = QUANTITIES,
) -> tuple[JointLimitViolation, ...]:
    """Every bound of `limits` (default: the trajectory's own) the run goes past, one record each.

    One record per joint, quantity and side, however many times the run crossed that bound. Order:
    by quantity (`QUANTITIES`, a position's ``min`` before its ``max``), then by the trajectory's
    joint order, the order `JointTrajectory.limit_violations()` has always printed.

    - A sample exactly on a bound is inside it; a NaN sample is never past one.
    - Jerk, the one derived quantity, is the finite difference of `accelerations` over `t` (both
      needed, and at least two samples).
    - `quantities` picks what is checked: ``("position",)`` judges a measured run's joint positions
      alone, whatever else it carries.
    """
    unknown = sorted(set(quantities) - set(QUANTITIES))
    if unknown:
        raise ValueError(f"unknown quantities {unknown}; choose from {QUANTITIES}")
    limits = trajectory.limits if limits is None else limits.select(trajectory.joint_names)
    if limits is None:
        raise ValueError("no limits given and none attached to this trajectory")
    t = trajectory.t
    jerk = None
    wants_jerk = "jerk" in quantities and limits.max_jerk is not None
    if wants_jerk and trajectory.accelerations is not None and len(t) >= 2:
        jerk = np.gradient(trajectory.accelerations, t, axis=0)
    checks = (
        ("position", "min", trajectory.positions, limits.min_position),
        ("position", "max", trajectory.positions, limits.max_position),
        ("velocity", "max", trajectory.velocities, limits.max_velocity),
        ("acceleration", "max", trajectory.accelerations, limits.max_acceleration),
        ("jerk", "max", jerk, limits.max_jerk),
        ("effort", "max", trajectory.effort, limits.max_effort),
    )
    found: list[JointLimitViolation] = []
    for quantity, side, values, bound in checks:
        if quantity not in quantities or values is None or bound is None:
            continue
        if quantity == "position":
            excess = bound - values if side == "min" else values - bound
        else:
            excess = np.abs(values) - bound
        with np.errstate(invalid="ignore"):
            past = excess > 0.0  # NaN compares False: a NaN sample is never past a bound
        for column, name in enumerate(trajectory.joint_names):
            rows = np.flatnonzero(past[:, column])
            if rows.size == 0:
                continue
            worst = int(rows[np.argmax(excess[rows, column])])
            kind = None
            if quantity == "position" and limits.position_kind is not None:
                kind = limits.position_kind[column]
            found.append(
                JointLimitViolation(
                    joint=name,
                    quantity=quantity,
                    side=side,
                    kind=kind,
                    limit=bound[column],
                    value=values[worst, column],
                    excess=excess[worst, column],
                    t=t[worst],
                    t_first=t[rows[0]],
                    t_last=t[rows[-1]],
                    samples=rows.size,
                )
            )
    return tuple(found)
