"""PoseStamped — a Pose plus the time and frame it is expressed in. Mirrors geometry_msgs."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Msg
from robot_msgs.geometry.pose import Pose
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False)
class PoseStamped(Msg):
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)

    def __post_init__(self) -> None:
        expect_type("PoseStamped.header", self.header, Header)
        expect_type("PoseStamped.pose", self.pose, Pose)

    @classmethod
    def from_xyz_rpy(
        cls,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        frame_id: str = "",
        stamp: float = 0.0,
    ) -> PoseStamped:
        return cls(Header(stamp, frame_id), Pose.from_xyz_rpy(x, y, z, roll, pitch, yaw))

    def shifted(
        self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, *, local: bool = False
    ) -> PoseStamped:
        """`Pose.shifted`, keeping the header."""
        return self.replace(pose=self.pose.shifted(dx, dy, dz, local=local))

    def rotated(
        self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, *, local: bool = False
    ) -> PoseStamped:
        """`Pose.rotated`, keeping the header."""
        return self.replace(pose=self.pose.rotated(roll, pitch, yaw, local=local))
