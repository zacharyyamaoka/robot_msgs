"""JointLimits — per-joint bounds a trajectory must respect, named like Ruckig's inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from robot_msgs._core import Array, Msg, optional_array, set_field

BOUNDS = (
    "min_position",
    "max_position",
    "max_velocity",
    "max_acceleration",
    "max_jerk",
    "max_effort",
)

PHYSICAL = "physical"
SOFTWARE = "software"
POSITION_KINDS = (PHYSICAL, SOFTWARE)
PositionKind = Literal["physical", "software"]


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointLimits(Msg):
    """One value per joint for each bound that is known; `None` means unbounded.

    WHY Ruckig's names (`max_velocity`, `max_acceleration`, `max_jerk`): the limits feed
    straight into the trajectory generator, and RobotDescription can carry the same value.
    `max_*` bounds are magnitudes (|velocity| <= max_velocity); `inf` is allowed.

    `position_kind[i]` says what joint i's position bounds are. WHY (Zach, 2026-11-11):
    "Physical limits actually prevent further rotation. software limits don't."

    - ``"physical"``: a hard stop. Nothing gets past it, so a simulator enforces it (MuJoCo: a
      limited joint ``range``).
    - ``"software"``: a bound the plan and the controller must keep and the hardware does not
      enforce. A simulator lets the joint through, and the run is checked against it afterwards
      (`joint_limit_violations`).
    - ``None``: not stated, as every limit written before the concept existed. A simulator keeps
      its own default (MuJoCo enforces a ``range``).

    The field is ``None`` when no joint's kind is stated. One kind covers both of a joint's
    position bounds: MuJoCo's ``limited`` is per joint, and the URDF and the simulator must hold
    the same two numbers.
    """

    joint_names: tuple[str, ...]
    min_position: Array | None = None
    max_position: Array | None = None
    max_velocity: Array | None = None
    max_acceleration: Array | None = None
    max_jerk: Array | None = None
    max_effort: Array | None = None
    position_kind: tuple[PositionKind | None, ...] | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError(f"JointLimits.joint_names must be unique, got {names}")
        set_field(self, "joint_names", names)
        for bound in BOUNDS:
            values = optional_array(getattr(self, bound))
            if values is None:
                continue
            if values.shape != (len(names),):
                raise ValueError(
                    f"JointLimits.{bound} has shape {values.shape}, expected ({len(names)},)"
                )
            if np.any(np.isnan(values)):
                raise ValueError(f"JointLimits.{bound} contains NaN; use inf for unbounded")
            if bound.startswith("max_") and bound != "max_position" and np.any(values < 0):
                raise ValueError(f"JointLimits.{bound} is a magnitude and must be >= 0")
            set_field(self, bound, values)
        if self.min_position is not None and self.max_position is not None:
            if np.any(self.min_position > self.max_position):
                raise ValueError("JointLimits.min_position exceeds max_position")
        if self.position_kind is not None:
            # WHY str(): a description may hand in a dataclass_values StrValue carrying its source;
            # the message keeps the plain word (the evidence stays in the description).
            kinds = tuple(None if kind is None else str(kind) for kind in self.position_kind)
            if len(kinds) != len(names):
                raise ValueError(
                    f"JointLimits.position_kind has {len(kinds)} entries, joint_names has "
                    f"{len(names)}"
                )
            unknown = sorted({kind for kind in kinds if kind is not None} - set(POSITION_KINDS))
            if unknown:
                raise ValueError(
                    f"JointLimits.position_kind {unknown} is not one of {POSITION_KINDS} or None"
                )
            # WHY all-None -> None: "no kind stated" has one spelling, so equality and repr agree.
            set_field(self, "position_kind", None if all(k is None for k in kinds) else kinds)

    def select(self, names: Sequence[str]) -> JointLimits:
        """These limits for `names`, in that order."""
        index = [self.joint_names.index(name) for name in names]
        bounds = {b: None if getattr(self, b) is None else getattr(self, b)[index] for b in BOUNDS}
        kinds = None if self.position_kind is None else tuple(self.position_kind[i] for i in index)
        return JointLimits(joint_names=tuple(names), position_kind=kinds, **bounds)
