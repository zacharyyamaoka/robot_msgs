"""ObjectHypothesisWithPose — a class guess plus where the object is.

Mirrors vision_msgs/ObjectHypothesisWithPose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Array, Msg, optional_array, set_field
from robot_msgs.geometry import Pose
from robot_msgs.vision.object_hypothesis import ObjectHypothesis


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ObjectHypothesisWithPose(Msg):
    """`covariance` is the (6, 6) pose covariance ordered x, y, z, roll, pitch, yaw, or `None`."""

    hypothesis: ObjectHypothesis
    pose: Pose = field(default_factory=Pose)
    covariance: Array | None = None

    def __post_init__(self) -> None:
        covariance = optional_array(self.covariance)
        if covariance is not None and covariance.shape != (6, 6):
            raise ValueError(
                f"ObjectHypothesisWithPose.covariance must be (6, 6), got {covariance.shape}"
            )
        set_field(self, "covariance", covariance)
