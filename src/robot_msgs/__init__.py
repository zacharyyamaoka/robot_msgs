"""robot_msgs — standard robot message types as frozen Python dataclasses, with helpers.

Import types from their domain, like ROS packages::

    from robot_msgs.geometry import Pose, Quaternion
    from robot_msgs.sensor import JointState, Image
    from robot_msgs.trajectory import JointTrajectory
"""

from robot_msgs import compare, coverage, geometry, sensor, shape, space, std, trajectory, vision
from robot_msgs._core import Msg, allclose, from_dict, load_npz, message_types, save_npz, to_dict

__version__ = "0.2.0"

__all__ = [
    "compare",
    "coverage",
    "Msg",
    "allclose",
    "from_dict",
    "geometry",
    "load_npz",
    "message_types",
    "save_npz",
    "sensor",
    "shape",
    "space",
    "std",
    "to_dict",
    "trajectory",
    "vision",
]
