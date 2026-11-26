"""What every joint-keyed stream shares: every per-sample array is ``(N, J)``, one column a joint.

Private: `ServoStateTrajectory` and `ServoCommandTrajectory` validate and select their columns
through these, with the same rules and messages as `JointTrajectory`'s own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from robot_msgs._core import freeze_new, set_field
from robot_msgs.trajectory.trajectory import Trajectory, sample_fields


def check_joint_columns(trajectory: Trajectory) -> tuple[str, ...]:
    """Normalise `joint_names` and check every set per-sample field is ``(N, len(joint_names))``."""
    kind = type(trajectory).__name__
    names = tuple(str(name) for name in trajectory.joint_names)  # type: ignore[attr-defined]
    if len(set(names)) != len(names):
        raise ValueError(f"{kind}.joint_names must be unique, got {names}")
    set_field(trajectory, "joint_names", names)
    for f in sample_fields(type(trajectory)):
        values = getattr(trajectory, f.name)
        if values is not None and (values.ndim != 2 or values.shape[1] != len(names)):
            raise ValueError(
                f"{kind}.{f.name} has shape {values.shape}, "
                f"expected ({len(trajectory)}, {len(names)}) for joints {names}"
            )
    return names


def joint_index(trajectory: Any, name: str) -> int:
    try:
        return trajectory.joint_names.index(name)
    except ValueError:
        raise KeyError(f"no joint {name!r}; joints are {trajectory.joint_names}") from None


def select_joint_columns(trajectory: Any, names: Sequence[str], **changes: Any) -> Any:
    """`trajectory` with only `names`, in that order; channels are per-sample, so kept whole."""
    columns = [joint_index(trajectory, name) for name in names]
    selected = {
        f.name: freeze_new(getattr(trajectory, f.name)[:, columns])
        for f in sample_fields(type(trajectory))
        if getattr(trajectory, f.name) is not None
    }
    return trajectory.replace(joint_names=tuple(names), **selected, **changes)


def column_order(names: Sequence[str], wanted: Sequence[str], kind: str) -> list[int]:
    """Indices that put a message's `names` into the order `wanted` (raises naming the gap)."""
    missing = [name for name in wanted if name not in names]
    if missing or len(names) != len(wanted):
        raise ValueError(f"{kind} names {tuple(names)} differ from the first one's {tuple(wanted)}")
    return [list(names).index(name) for name in wanted]
