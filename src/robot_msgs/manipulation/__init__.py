"""manipulation — what a grasp detector hands the trajectory generator, and what it takes in.

A grasp detector emits a `GraspPose` (a tool pose plus jaw widths). A pick recipe, or a detector
that wants its own approach, emits a `WaypointAction`: named `Waypoint`s the tool frame passes
through, all in one frame, with what the gripper and path do at each. The generator turns that
into a `JointTrajectory`. moveit_msgs is the ROS analogue (`Grasp`, `MotionPlanRequest`).
"""

from robot_msgs.manipulation.cartesian_limits import CartesianLimits
from robot_msgs.manipulation.grasp_pose import GraspPose
from robot_msgs.manipulation.waypoint import LABELS, Waypoint, WaypointLabel

# isort: split
from robot_msgs.manipulation.waypoint_action import WaypointAction

__all__ = ["LABELS", "CartesianLimits", "GraspPose", "Waypoint", "WaypointAction", "WaypointLabel"]
