"""BTState — a behaviour tree's state at one instant: node statuses, blackboard, tick, clock."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.std import Header

# BT.CPP's NodeStatus (SKIPPED aside), spelled as robot_bt's trace writes it.
IDLE, RUNNING, SUCCESS, FAILURE = "IDLE", "RUNNING", "SUCCESS", "FAILURE"
STATUSES = (IDLE, RUNNING, SUCCESS, FAILURE)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class BTState(Msg):
    """Everything a behaviour tree needs to carry on from where it was.

    - `tree_id`: the tree that was ticked (its root ``<BehaviorTree ID>``).
    - `node_status`: node path → ``"IDLE" | "RUNNING" | "SUCCESS" | "FAILURE"``. Paths are
      robot_bt's (``"Bringup/Sequence/Home"``, ``#2`` for a repeated sibling), in tree order.
    - `blackboard`: key → a JSON value (None, bool, int, float, str, and lists / dicts of them).
      Nested dicts become read-only mappings and lists become tuples, so the state stays frozen.
    - `tick`: ticks of the root so far. `clock_s`: the tree's clock (s), sim or wall.

    WHY its own type (Zach, 2026-11-26: "a behaviour tree should have a BTState just like Joints
    and JointState"): a tree is a component, and every component exposes its state as a message
    that `RobotState` can hold beside the others and `save` can write.
    WHY the blackboard holds JSON values only: it is the part people read when a run goes wrong;
    arrays (an image, a joint state) belong in their own message, as another `RobotState` entry.
    """

    tree_id: str
    node_status: Mapping[str, str] = field(default_factory=FrozenDict)
    # WHY `object`, not Any: the decoder reads an Any-typed mapping with a "dtype" key as an
    # array; a blackboard value is only ever plain JSON.
    blackboard: Mapping[str, object] = field(default_factory=FrozenDict)
    tick: int = 0
    clock_s: float = 0.0
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        set_field(self, "tree_id", str(self.tree_id))
        statuses = {}
        for path, status in self.node_status.items():
            if not isinstance(path, str) or status not in STATUSES:
                raise ValueError(
                    f"BTState.node_status maps a node path to one of {STATUSES}; "
                    f"got {path!r}: {status!r}"
                )
            statuses[path] = status
        set_field(self, "node_status", FrozenDict(statuses))
        set_field(self, "blackboard", _frozen_json(self.blackboard, "BTState.blackboard"))
        if isinstance(self.tick, bool) or int(self.tick) != self.tick or self.tick < 0:
            raise ValueError(f"BTState.tick counts ticks, so a whole number ≥ 0; got {self.tick!r}")
        set_field(self, "tick", int(self.tick))
        clock = float(self.clock_s)
        if not math.isfinite(clock):
            raise ValueError(f"BTState.clock_s must be finite, got {clock}")
        set_field(self, "clock_s", clock)

    def status_of(self, path: str) -> str:
        """The status of the node at `path`."""
        try:
            return self.node_status[path]
        except KeyError:
            raise KeyError(
                f"BTState has no node at {path!r}; paths: {tuple(self.node_status)}"
            ) from None

    def running(self) -> tuple[str, ...]:
        """Paths of the nodes that are RUNNING, in tree order: where the tree is right now."""
        return tuple(path for path, status in self.node_status.items() if status == RUNNING)


def _frozen_json(value: Any, label: str) -> Any:
    """`value` as immutable JSON: mappings → FrozenDict (str keys), lists → tuples."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys are strings; got {key!r}")
            out[key] = _frozen_json(item, f"{label}[{key!r}]")
        return FrozenDict(out)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, np.generic) and not isinstance(value, np.ndarray):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"{label} holds JSON values only (None, bool, int, float, str, lists, dicts); got "
        f"{type(value).__name__}. Put arrays and messages in their own RobotState entry."
    )
