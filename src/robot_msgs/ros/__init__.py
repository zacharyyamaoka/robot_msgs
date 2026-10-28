"""robot_msgs <-> ROS 2 messages, with no ROS install: the rosbags typestore does the encoding.

ros_msg = to_ros(JointState(...))        # a rosbags dataclass, ready for serialize_cdr
msg = from_ros(ros_msg)                  # back to a robot_msgs value
write_bag("run1", [("/joint_states", msg)])      # an mcap rosbag2 Foxglove opens directly
for topic, msg in read_bag("run1"): ...
"""

from __future__ import annotations

try:
    import rosbags  # noqa: F401
except ImportError as error:
    raise ImportError("robot_msgs.ros needs rosbags: pip install 'robot-msgs[ros]'") from error

from robot_msgs.ros.bag import read_bag, write_bag
from robot_msgs.ros.convert import from_ros, ros_type, to_ros, typestore

__all__ = ["from_ros", "read_bag", "ros_type", "to_ros", "typestore", "write_bag"]
