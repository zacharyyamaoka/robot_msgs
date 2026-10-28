"""The UR5e pick as a ROS 2 bag: robot_msgs values in, standard ROS 2 Jazzy messages out.

uv run --directory /home/bam/robot_msgs --extra ros python examples/ros_export.py OUT_BAG_DIR
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from ur5e_pick import minimum_jerk, tool_trajectory, waypoint_trajectory

from robot_msgs import allclose
from robot_msgs.ros import read_bag, ros_type, to_ros, write_bag


def main(bag: Path) -> None:
    joints = minimum_jerk(waypoint_trajectory())
    tool = tool_trajectory(joints)

    ros_traj = to_ros(joints, drop_extras=True)  # ROS has no slot for the gripper channel or limits
    print(
        ros_type(joints),
        "with",
        len(ros_traj.points),
        "points;",
        "point[250].time_from_start =",
        ros_traj.points[250].time_from_start,
    )
    try:
        to_ros(joints)
    except ValueError as refusal:
        print("without drop_extras:", refusal)

    plain_joints = joints.replace(channels={}, limits=None)
    plain_tool = tool.replace(linear_velocities=None, angular_velocities=None)
    states = [joints.joint_state_at(t) for t in joints.t[::5]]  # a 25 Hz /joint_states stream
    write_bag(
        bag,
        [("/joint_states", state) for state in states]
        + [("/arm/trajectory", plain_joints), ("/tool0/path", plain_tool)],
    )

    read_back = list(read_bag(bag))
    print("bag", bag.name, dict(Counter(topic for topic, _ in read_back)))
    trajectory = next(msg for topic, msg in read_back if topic == "/arm/trajectory")
    print("trajectory survives the round trip:", allclose(trajectory, plain_joints, atol=1e-9))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
