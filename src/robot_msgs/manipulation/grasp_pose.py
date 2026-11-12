"""GraspPose — one grasp a grasp detector proposes: a gripper pose plus the jaw widths."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import Pose, PoseStamped
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.manipulation._common import non_negative, positive
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class GraspPose(Msg):
    """Where the gripper's tool frame goes to grasp, and how wide its jaws are.

    - `pose`: the tool frame at the grasp, in `header.frame_id`.
    - `width` (m): the jaw gap to close to (0 = fully shut on nothing).
    - `pre_grasp_width` (m): the gap to open to before the approach; `None` lets the planner pick.
      Never smaller than `width`.
    - `score`: the detector's own quality, higher is better; `None` when it has none.
    - `object_id`: what is being grasped (a detection id or class), `""` when unknown.

    WHY a message, not a plan (Zach, 2026-11-11): "the grasping networks can output just a Grasp
    Pose ... if they want to start doing like multiple points to define like approach etc etc
    that is when the WaypointAction is good." A pick recipe (bam_traj_gen's
    `make_pick_and_place`) turns one of these into a `WaypointAction`.
    """

    pose: Pose
    width: float
    pre_grasp_width: float | None = None
    score: float | None = None
    object_id: str = ""
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        expect_type("GraspPose.pose", self.pose, Pose)
        expect_type("GraspPose.header", self.header, Header)
        set_field(self, "width", non_negative("GraspPose", "width", self.width, allow_none=False))
        if self.pre_grasp_width is not None:
            pre = positive("GraspPose", "pre_grasp_width", self.pre_grasp_width)
            if pre < self.width:
                raise ValueError(
                    f"GraspPose.pre_grasp_width {pre} is smaller than width {self.width}"
                )
            set_field(self, "pre_grasp_width", pre)
        if self.score is not None:
            score = float(self.score)
            if not math.isfinite(score):
                raise ValueError("GraspPose.score must be finite")
            set_field(self, "score", score)
        set_field(self, "object_id", str(self.object_id))

    @property
    def pose_stamped(self) -> PoseStamped:
        """The grasp pose with its frame and stamp."""
        return PoseStamped(self.header, self.pose)
