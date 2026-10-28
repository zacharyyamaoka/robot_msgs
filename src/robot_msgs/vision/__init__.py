"""vision — what perception reports about the world (vision_msgs)."""

from robot_msgs.vision.bounding_box2d import BoundingBox2D
from robot_msgs.vision.bounding_box3d import BoundingBox3D
from robot_msgs.vision.detection2d import Detection2D
from robot_msgs.vision.detection2d_array import Detection2DArray
from robot_msgs.vision.detection3d import Detection3D
from robot_msgs.vision.detection3d_array import Detection3DArray
from robot_msgs.vision.object_hypothesis import ObjectHypothesis
from robot_msgs.vision.object_hypothesis_with_pose import ObjectHypothesisWithPose

__all__ = [
    "BoundingBox2D",
    "BoundingBox3D",
    "Detection2D",
    "Detection2DArray",
    "Detection3D",
    "Detection3DArray",
    "ObjectHypothesis",
    "ObjectHypothesisWithPose",
]
