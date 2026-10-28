"""TransformStamped — a Transform between two named frames at a time. Mirrors geometry_msgs."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.transform import Transform
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False)
class TransformStamped(Msg):
    """``header.frame_id`` is the parent frame; `transform` maps `child_frame_id` into it."""

    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    transform: Transform = field(default_factory=Transform)

    def __post_init__(self) -> None:
        expect_type("TransformStamped.header", self.header, Header)
        expect_type("TransformStamped.transform", self.transform, Transform)
        set_field(self, "child_frame_id", str(self.child_frame_id))

    def inverse(self) -> TransformStamped:
        """The same relation seen from the other side: parent and child swap, stamp is kept."""
        return TransformStamped(
            self.header.replace(frame_id=self.child_frame_id),
            self.header.frame_id,
            self.transform.inverse(),
        )
