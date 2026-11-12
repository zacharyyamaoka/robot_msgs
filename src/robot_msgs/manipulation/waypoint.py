"""Waypoint — one named tool pose in a WaypointAction, with what the gripper and path do there."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import Pose, Vector3
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.manipulation._common import non_negative, positive

WaypointLabel = Literal[
    "start",
    "home",
    "approach",
    "pre_grasp",
    "grasp",
    "close",
    "lift",
    "carry",
    "pre_place",
    "place",
    "release",
    "retreat",
    "ready",
]
"""What a waypoint MEANS, for the tools that key on it (a cycle ends at `ready`, a moving target
is synchronised at `close`). Its `name` stays any string.

| label | the tool is | bam_traj_gen's pick-and-place name (2026-11-11) |
|---|---|---|
| `start` | where the plan starts (the current pose) | `start` |
| `home` | at the robot's rest pose | — |
| `approach` | high above the pick | `above_pick` |
| `pre_grasp` | just above the grasp, jaws open | `pre_grasp` |
| `grasp` | at the grasp pose, jaws still open | `start_grasp` |
| `close` | at the grasp, jaws closed on the object | `closed_grasp` |
| `lift` | lifting the object clear | `lift` |
| `carry` | high above the place, object held | `above_place` |
| `pre_place` | just above the place | `pre_place` |
| `place` | at the place pose, object still held | `start_place` |
| `release` | at the place, jaws open | `open_grasp` |
| `retreat` | backing off the placed object | `retreat` |
| `ready` | clear of the cell: the arm can take its next job | — |
"""

LABELS: tuple[str, ...] = WaypointLabel.__args__  # type: ignore[attr-defined]


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Waypoint(Msg):
    """A tool pose to pass through, in the `WaypointAction`'s frame, plus what happens there.

    - `name`: any label, unique within its action; `label`: what it means (`WaypointLabel`).
    - `gripper_width` (m): the jaw gap to have reached here; `None` keeps the previous one.
      `gripper_max_effort` (N) caps the jaws on the way (a gentle close); `None` = default.
    - `duration` (s): time to get here from the previous waypoint; `None` lets the generator
      choose from the limits. `dwell` (s): time to wait here before moving on.
    - `blend_radius` (m): round this corner with an arc of that radius (0 = stop at it; the last
      waypoint is always 0). `interpolation_points` / `blend_points`: extra Cartesian points on
      the straight segment ending here / on this corner's arc.
    - `target_velocity` (m/s, the action's frame): the waypoint rides on a moving target (a
      conveyor); the generator solves for when it gets there. `sync_at_target_time`: keep the
      offset the target had at the sync waypoint, without chasing it further (a lift after the
      close).
    """

    name: str
    pose: Pose
    label: WaypointLabel | None = None
    gripper_width: float | None = None
    gripper_max_effort: float | None = None
    duration: float | None = None
    dwell: float = 0.0
    blend_radius: float = 0.0
    interpolation_points: int = 0
    blend_points: int = 0
    target_velocity: Vector3 | None = None
    sync_at_target_time: bool = False

    def __post_init__(self) -> None:
        set_field(self, "name", str(self.name))
        if not self.name:
            raise ValueError("Waypoint.name cannot be empty")
        expect_type(f"Waypoint {self.name!r}.pose", self.pose, Pose)
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"Waypoint {self.name!r}: label {self.label!r} is not one of {LABELS}")
        owner = f"Waypoint {self.name!r}"
        set_field(self, "gripper_width", non_negative(owner, "gripper_width", self.gripper_width))
        if self.gripper_max_effort is not None:
            set_field(
                self,
                "gripper_max_effort",
                positive(owner, "gripper_max_effort", self.gripper_max_effort),
            )
        if self.duration is not None:
            set_field(self, "duration", positive(owner, "duration", self.duration))
        set_field(self, "dwell", non_negative(owner, "dwell", self.dwell, allow_none=False))
        set_field(
            self,
            "blend_radius",
            non_negative(owner, "blend_radius", self.blend_radius, allow_none=False),
        )
        for name in ("interpolation_points", "blend_points"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{owner}.{name} must be a whole number >= 0, got {value!r}")
            set_field(self, name, int(value))
        if self.target_velocity is not None:
            expect_type(f"{owner}.target_velocity", self.target_velocity, Vector3)
            v = self.target_velocity
            if not all(math.isfinite(c) for c in (v.x, v.y, v.z)):
                raise ValueError(f"{owner}.target_velocity must be finite")
        set_field(self, "sync_at_target_time", bool(self.sync_at_target_time))
