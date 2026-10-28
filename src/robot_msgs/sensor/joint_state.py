"""JointState — the measured state of a set of named joints. Mirrors sensor_msgs/JointState."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointState(Msg):
    """`position[i]` (rad or m), `velocity[i]` and `effort[i]` belong to joint `name[i]`.

    `velocity` and `effort` are `None` when not measured (ROS sends empty arrays).
    """

    name: tuple[str, ...]
    position: Array
    velocity: Array | None = None
    effort: Array | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.name)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"JointState.name has duplicate names: {duplicates}")
        set_field(self, "name", names)
        set_field(self, "position", frozen_array(self.position))
        set_field(self, "velocity", optional_array(self.velocity))
        set_field(self, "effort", optional_array(self.effort))
        for field_name in ("position", "velocity", "effort"):
            values = getattr(self, field_name)
            if values is None:
                continue
            if values.ndim != 1:
                raise ValueError(f"JointState.{field_name} must be 1-D, got shape {values.shape}")
            if len(values) != len(names):
                raise ValueError(
                    f"JointState.{field_name} has {len(values)} values, name has {len(names)}"
                )

    def __len__(self) -> int:
        return len(self.name)

    def index_of(self, name: str) -> int:
        try:
            return self.name.index(name)
        except ValueError:
            raise KeyError(f"JointState has no joint named {name!r}; joints: {self.name}") from None

    def position_of(self, name: str) -> float:
        return float(self.position[self.index_of(name)])

    def as_dict(self) -> dict[str, float]:
        """Joint name → position."""
        return {n: float(p) for n, p in zip(self.name, self.position, strict=True)}

    def select(self, names: Sequence[str]) -> JointState:
        """The named joints, in the order given."""
        order = [self.index_of(n) for n in names]
        return self.replace(
            name=tuple(names),
            position=self.position[order],
            velocity=None if self.velocity is None else self.velocity[order],
            effort=None if self.effort is None else self.effort[order],
        )
