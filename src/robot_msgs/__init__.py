"""robot_msgs — standard robot message types as frozen Python dataclasses, with helpers.

Import types from their domain, like ROS packages::

    from robot_msgs.geometry import Pose, Quaternion
    from robot_msgs.sensor import JointState, Image
    from robot_msgs.trajectory import JointTrajectory
    from robot_msgs.control import ServoState, ServoCommand
    from robot_msgs.manipulation import GraspPose, WaypointAction
    from robot_msgs.fiducial import FiducialArray, FiducialMap
    from robot_msgs.std import LogRecord, DiagnosticRecord, Metrics
    from robot_msgs.behavior import BTState
    from robot_msgs.robot import RobotState, RobotConfig

Files: ``state.save("arm")`` / ``RobotState.load("arm")`` for one message,
``robot_msgs.save({"state": s, "action": a}, "step.npz")`` / ``robot_msgs.load(...)`` for many.
`robot_msgs.adopt(array)` hands a numpy array over without a copy (a view of a received buffer).
"""

from robot_msgs import (
    behavior,
    compare,
    control,
    coverage,
    fiducial,
    geometry,
    manipulation,
    robot,
    sensor,
    shape,
    space,
    std,
    trajectory,
    vision,
)
from robot_msgs._core import (
    Msg,
    adopt,
    allclose,
    from_dict,
    load_npz,
    message_types,
    save_npz,
    to_dict,
)
from robot_msgs._storage import load, save

__version__ = "0.8.0"

__all__ = [
    "behavior",
    "compare",
    "control",
    "coverage",
    "fiducial",
    "Msg",
    "adopt",
    "allclose",
    "from_dict",
    "geometry",
    "manipulation",
    "load",
    "load_npz",
    "message_types",
    "robot",
    "save",
    "save_npz",
    "sensor",
    "shape",
    "space",
    "std",
    "to_dict",
    "trajectory",
    "vision",
]
