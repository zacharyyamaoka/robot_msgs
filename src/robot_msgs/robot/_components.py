"""What `RobotState` and `RobotConfig` share: named component messages, updated by copy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from robot_msgs._core import FrozenDict, Msg, set_field

C = TypeVar("C", bound="Components")


class Components:
    """Mixin (not a `Msg`, so it registers no type) over a `components: Mapping[str, Msg]` field."""

    __slots__ = ()
    components: Mapping[str, Msg]

    def _freeze_components(self) -> None:
        label = f"{type(self).__name__}.components"
        frozen = {}
        for name, value in self.components.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label} keys are non-empty strings; got {name!r}")
            if not isinstance(value, Msg):
                raise TypeError(
                    f"{label}[{name!r}] must be a robot_msgs message, got {type(value).__name__}"
                )
            frozen[name] = value
        set_field(self, "components", FrozenDict(frozen))

    def __getitem__(self, name: str) -> Any:
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} has no component {name!r}; components: "
                f"{tuple(self.components)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self.components

    def __len__(self) -> int:
        return len(self.components)

    def with_components(self: C, components: Mapping[str, Msg] | None = None, **named: Msg) -> C:
        """A copy with these components added or replaced (the rest kept, in order)."""
        merged = {**self.components, **(components or {}), **named}
        return self.replace(components=merged)  # type: ignore[attr-defined]

    def without(self: C, *names: str) -> C:
        """A copy without the named components."""
        for name in names:
            self[name]  # a KeyError naming the missing one
        kept = {key: value for key, value in self.components.items() if key not in names}
        return self.replace(components=kept)  # type: ignore[attr-defined]
