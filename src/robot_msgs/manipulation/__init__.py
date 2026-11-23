"""manipulation — what a grasp detector hands the trajectory generator, and what it takes in.

A grasp detector emits a `GraspPose` (a tool pose plus jaw widths). A pick recipe, or a detector
that wants its own approach, emits a `WaypointAction`: named `Waypoint`s the tool frame passes
through, all in one frame, with what the gripper and path do at each. The generator turns that
into a `JointTrajectory`. moveit_msgs is the ROS analogue (`Grasp`, `MotionPlanRequest`).

A robot's saved positions are `NamedPose`s ("ready", "home"): MoveIt's SRDF `<group_state>`, looked
up by name (`find_named_pose`) so no consumer copies the numbers.

Which body pairs a collision checker may skip is an `AllowedCollisionMatrix` (moveit_msgs'
message plus the SRDF `<disable_collisions>` reason per pair).
"""

from robot_msgs.manipulation.allowed_collision_matrix import (
    REASONS,
    AllowedCollisionMatrix,
    allowed_collision_matrix_from_srdf,
    allowed_collision_matrix_to_srdf,
)
from robot_msgs.manipulation.cartesian_limits import CartesianLimits
from robot_msgs.manipulation.grasp_pose import GraspPose
from robot_msgs.manipulation.named_pose import (
    NAMED_POSES_SCHEMA,
    InverseKinematics,
    NamedPose,
    check_named_poses,
    find_named_pose,
    named_poses_from_json,
    named_poses_from_srdf,
    named_poses_to_json,
    named_poses_to_srdf,
    nearest_solution,
)
from robot_msgs.manipulation.waypoint import LABELS, Waypoint, WaypointLabel

# isort: split
from robot_msgs.manipulation.waypoint_action import WaypointAction

__all__ = [
    "LABELS",
    "NAMED_POSES_SCHEMA",
    "REASONS",
    "AllowedCollisionMatrix",
    "CartesianLimits",
    "GraspPose",
    "InverseKinematics",
    "NamedPose",
    "Waypoint",
    "WaypointAction",
    "WaypointLabel",
    "allowed_collision_matrix_from_srdf",
    "allowed_collision_matrix_to_srdf",
    "check_named_poses",
    "find_named_pose",
    "named_poses_from_json",
    "named_poses_from_srdf",
    "named_poses_to_json",
    "named_poses_to_srdf",
    "nearest_solution",
]
