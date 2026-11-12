"""WaypointAction — a Cartesian motion request: named waypoints the tool frame passes through.

WHY one frame for every waypoint (Zach, bam_msgs waypoint_action.py, v3): "all points should be
for the same frame ID no?" The frame is `header.frame_id`, once, so a mixed-frame plan cannot be
written. WHY `start_state` beside the poses (same note, after MoveIt's MotionPlanRequest): you
always plan from the current state, and joint velocity at the start is not a pose.

This is the request the trajectory generator takes in. What it computes (IK solutions, blend
bookkeeping, the timed trajectory) stays in the generator and comes out as a `JointTrajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import PoseStamped
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.manipulation.cartesian_limits import CartesianLimits
from robot_msgs.manipulation.waypoint import LABELS, Waypoint, WaypointLabel
from robot_msgs.sensor import JointState
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class WaypointAction(Msg):
    """The waypoints in order, all in `header.frame_id`, for the tool frame `tool_frame`.

    - `start_state`: the arm + hand state the plan starts from; `None` = the generator reads it.
    - `arm_joints` / `hand_joints`: which joints carry it out.
    - `cartesian_limits`, `arm_limits`, `hand_limits`: limits for the whole action (a waypoint
      cannot loosen them).
    - `sync_target`: the name of the waypoint a moving target is caught at, when any waypoint
      has a `target_velocity`; `None` = the one labelled `close`, if there is exactly one.
    """

    waypoints: tuple[Waypoint, ...]
    tool_frame: str = "tcp"
    start_state: JointState | None = None
    arm_joints: tuple[str, ...] = ()
    hand_joints: tuple[str, ...] = ()
    cartesian_limits: CartesianLimits | None = None
    arm_limits: JointLimits | None = None
    hand_limits: JointLimits | None = None
    sync_target: str | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        set_field(self, "waypoints", tuple(self.waypoints))
        set_field(self, "tool_frame", str(self.tool_frame))
        set_field(self, "arm_joints", tuple(str(j) for j in self.arm_joints))
        set_field(self, "hand_joints", tuple(str(j) for j in self.hand_joints))
        expect_type("WaypointAction.header", self.header, Header)
        if not self.waypoints:
            raise ValueError("WaypointAction needs at least one waypoint")
        for waypoint in self.waypoints:
            expect_type("WaypointAction.waypoints[]", waypoint, Waypoint)
        names = [w.name for w in self.waypoints]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"WaypointAction has duplicate waypoint names: {duplicates}")
        if self.waypoints[-1].blend_radius != 0.0:
            raise ValueError(
                f"the last waypoint {names[-1]!r} cannot be blended (blend_radius must be 0)"
            )
        joints = self.arm_joints + self.hand_joints
        if len(set(joints)) != len(joints):
            raise ValueError(f"WaypointAction names a joint twice: {joints}")
        for side, limits, expected in (
            ("arm", self.arm_limits, self.arm_joints),
            ("hand", self.hand_limits, self.hand_joints),
        ):
            if limits is not None and expected and limits.joint_names != expected:
                raise ValueError(
                    f"{side}_limits is for {limits.joint_names}, {side}_joints is {expected}"
                )
        if self.sync_target is not None:
            set_field(self, "sync_target", str(self.sync_target))
            if self.sync_target not in names:
                raise ValueError(f"sync_target {self.sync_target!r} is not a waypoint: {names}")

    def __len__(self) -> int:
        return len(self.waypoints)

    def __getitem__(self, key: int | str) -> Waypoint:
        """A waypoint by position or by name."""
        if isinstance(key, str):
            return self.waypoints[self.index_of(key)]
        return self.waypoints[key]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.waypoints)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no waypoint named {name!r}; waypoints: {self.names}") from None

    def labelled(self, label: WaypointLabel) -> tuple[Waypoint, ...]:
        """Every waypoint carrying `label`, in order."""
        if label not in LABELS:
            raise ValueError(f"{label!r} is not one of {LABELS}")
        return tuple(w for w in self.waypoints if w.label == label)

    def pose_stamped(self, key: int | str) -> PoseStamped:
        """A waypoint's pose with the action's frame and stamp."""
        return PoseStamped(self.header, self[key].pose)

    @property
    def is_sync_required(self) -> bool:
        """True when any waypoint rides on a moving target."""
        return any(w.target_velocity is not None for w in self.waypoints)

    def sync_target_index(self) -> int | None:
        """Where a moving target is caught: `sync_target`, else the single `close` waypoint."""
        if self.sync_target is not None:
            return self.index_of(self.sync_target)
        closes = [i for i, w in enumerate(self.waypoints) if w.label == "close"]
        return closes[0] if len(closes) == 1 else None
