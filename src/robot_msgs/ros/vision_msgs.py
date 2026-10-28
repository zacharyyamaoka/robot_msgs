"""The vision_msgs definitions the Jazzy typestore lacks, registered from the official .msg files.

rosbags' `Stores.ROS2_JAZZY` ships no vision_msgs. These are the definitions of
https://github.com/ros-perception/vision_msgs at tag 4.1.1 (commit e77fa2d9), the version of
`ros-jazzy-vision-msgs`, copied from `vision_msgs/msg/<Name>.msg` at that tag. Only the comment
lines were dropped; the field lines are unchanged, so the wire format and type hashes match.
"""

from __future__ import annotations

from typing import Any

from rosbags.typesys import get_types_from_msg

UPSTREAM = "https://github.com/ros-perception/vision_msgs (tag 4.1.1, commit e77fa2d9)"

MSG_DEFINITIONS: dict[str, str] = {
    "vision_msgs/msg/Point2D": "float64 x\nfloat64 y\n",
    "vision_msgs/msg/Pose2D": "vision_msgs/Point2D position\nfloat64 theta\n",
    "vision_msgs/msg/BoundingBox2D": (
        "vision_msgs/Pose2D center\nfloat64 size_x\nfloat64 size_y\n"
    ),
    "vision_msgs/msg/BoundingBox3D": "geometry_msgs/Pose center\ngeometry_msgs/Vector3 size\n",
    "vision_msgs/msg/ObjectHypothesis": "string class_id\nfloat64 score\n",
    "vision_msgs/msg/ObjectHypothesisWithPose": (
        "ObjectHypothesis hypothesis\ngeometry_msgs/PoseWithCovariance pose\n"
    ),
    "vision_msgs/msg/Detection2D": (
        "std_msgs/Header header\nObjectHypothesisWithPose[] results\n"
        "BoundingBox2D bbox\nstring id\n"
    ),
    "vision_msgs/msg/Detection2DArray": "std_msgs/Header header\nDetection2D[] detections\n",
    "vision_msgs/msg/Detection3D": (
        "std_msgs/Header header\nObjectHypothesisWithPose[] results\n"
        "BoundingBox3D bbox\nstring id\n"
    ),
    "vision_msgs/msg/Detection3DArray": "std_msgs/Header header\nDetection3D[] detections\n",
}


def definitions() -> dict[str, Any]:
    """All vision_msgs types, parsed, ready for `Typestore.register`."""
    types: dict[str, Any] = {}
    for name, text in MSG_DEFINITIONS.items():
        types.update(get_types_from_msg(text, name))
    return types
