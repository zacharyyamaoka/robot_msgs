"""fiducial — fiducial markers (ArUco, AprilTag, ChArUco): what a detector measures, where the
markers are mounted, and what a tracker estimates.

- Measured, per image: `FiducialArray` of `FiducialDetection` (id, dictionary, 4 corner px) and
  `FiducialPose` (``T_camera_marker`` with both planar PnP solutions).
- Described: `FiducialMount` (a marker on a parent frame: robot link, board, world),
  `FiducialMap` (all of them), `FiducialBoard` (a grid / ChArUco board → its mounts).
- Estimated: `FiducialTrackArray` of `FiducialTrack` (filtered pose + covariance, observed or
  predicted).

Conventions: corners top-left, top-right, bottom-right, bottom-left as printed (OpenCV); the
marker frame is OpenCV's IPPE_SQUARE frame (`marker_object_points`); a marker's key is
``(dictionary, id)`` with OpenCV's dictionary names; lengths in m, pixels in px; every 6×6
covariance is local-tangent (`robot_msgs.fiducial.covariance`).
"""

from robot_msgs.fiducial.covariance import local_to_parent_covariance, parent_to_local_covariance
from robot_msgs.fiducial.fiducial_array import FiducialArray
from robot_msgs.fiducial.fiducial_board import FiducialBoard
from robot_msgs.fiducial.fiducial_detection import CORNER_ORDER, FiducialDetection
from robot_msgs.fiducial.fiducial_map import FiducialMap
from robot_msgs.fiducial.fiducial_mount import FiducialMount, marker_object_points
from robot_msgs.fiducial.fiducial_pose import FiducialPose
from robot_msgs.fiducial.fiducial_track import FiducialTrack
from robot_msgs.fiducial.fiducial_track_array import FiducialTrackArray

__all__ = [
    "CORNER_ORDER",
    "FiducialArray",
    "FiducialBoard",
    "FiducialDetection",
    "FiducialMap",
    "FiducialMount",
    "FiducialPose",
    "FiducialTrack",
    "FiducialTrackArray",
    "local_to_parent_covariance",
    "marker_object_points",
    "parent_to_local_covariance",
]
