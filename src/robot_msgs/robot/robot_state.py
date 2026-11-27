"""RobotState — the whole robot's state at one instant, as each component's own state message."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg
from robot_msgs.robot._components import Components
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class RobotState(Components, Msg):
    """Component name → that component's state message, e.g.
    ``RobotState(components={"bt": BTState(...), "joints": JointState(...)})``.

    ``state["joints"]`` reads one; ``state.with_components(joints=measured)`` and
    ``state.without("camera")`` return updated copies. A component can itself be a `RobotState`
    (an arm inside a cell), so the state composes as deep as the system does.

    WHY one composite, not one big class with a field per part (Zach, 2026-11-26: "the state
    vector has the entire state for all of the components. each component itself exposes its
    state as a separate thing"): the MDP's state is the sum of its parts, each part keeps its own
    type, and adding a component needs no new field here.
    Configuration is the separate `RobotConfig` (see its WHY for the keras / torch split).
    """

    components: Mapping[str, Msg] = field(default_factory=FrozenDict)
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        self._freeze_components()
