"""A UR5e pick move as robot_msgs values.

Named waypoints → minimum-jerk JointTrajectory (with limits and a gripper channel) → the
tool0 PoseTrajectory through UR5e forward kinematics. Run it to see the views:

    uv run --directory /home/bam/robot_msgs python examples/ur5e_pick.py
"""

from __future__ import annotations

import numpy as np

from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory

UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Universal Robots' published UR5e DH parameters (metres, radians).
UR5E_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
UR5E_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
UR5E_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

UR5E_LIMITS = JointLimits(
    joint_names=UR5E_JOINTS,
    min_position=[-2 * np.pi] * 6,
    max_position=[2 * np.pi] * 6,
    max_velocity=[np.pi] * 6,
    max_acceleration=[8.0] * 6,
    max_jerk=[120.0] * 6,
    max_effort=[150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
)

WAYPOINTS = {  # name: (time s, joint angles rad, gripper closed)
    "home": (0.0, [0.0, -1.571, 1.571, -1.571, -1.571, 0.0], 0),
    "pregrasp": (1.6, [0.6, -1.20, 1.85, -2.22, -1.571, 0.6], 0),
    "grasp": (2.4, [0.6, -1.02, 1.93, -2.48, -1.571, 0.6], 1),
    "lift": (3.2, [0.6, -1.30, 1.70, -1.97, -1.571, 0.6], 1),
    "place": (5.0, [-0.9, -1.25, 1.75, -2.07, -1.571, -0.9], 0),
}


def ur5e_forward_kinematics(positions: np.ndarray) -> np.ndarray:
    """``(N, 6)`` joint angles → ``(N, 4, 4)`` base_link ← tool0 transforms (standard DH)."""
    count = positions.shape[0]
    transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    for joint in range(6):
        theta = positions[:, joint]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_a, sin_a = np.cos(UR5E_ALPHA[joint]), np.sin(UR5E_ALPHA[joint])
        link = np.zeros((count, 4, 4))
        link[:, 0, 0], link[:, 0, 1], link[:, 0, 2] = cos_t, -sin_t * cos_a, sin_t * sin_a
        link[:, 1, 0], link[:, 1, 1], link[:, 1, 2] = sin_t, cos_t * cos_a, -cos_t * sin_a
        link[:, 2, 1], link[:, 2, 2] = sin_a, cos_a
        link[:, 0, 3] = UR5E_A[joint] * cos_t
        link[:, 1, 3] = UR5E_A[joint] * sin_t
        link[:, 2, 3] = UR5E_D[joint]
        link[:, 3, 3] = 1.0
        transform = transform @ link
    return transform


def waypoint_trajectory() -> JointTrajectory:
    """The waypoints themselves: a sparse trajectory whose `name` channel labels each sample."""
    times, angles, gripper = zip(*WAYPOINTS.values(), strict=True)
    return JointTrajectory(
        t=times,
        joint_names=UR5E_JOINTS,
        positions=angles,
        channels={"name": np.array(list(WAYPOINTS)), "gripper": np.array(gripper)},
    )


def minimum_jerk(waypoints: JointTrajectory, dt: float = 0.008) -> JointTrajectory:
    """Rest-to-rest quintic segments between waypoints, sampled on a fixed controller grid."""
    times = np.arange(0.0, waypoints.end_time + dt / 2, dt)
    segment = np.clip(np.searchsorted(waypoints.t, times, side="right") - 1, 0, len(waypoints) - 2)
    start, end = waypoints.t[segment], waypoints.t[segment + 1]
    span = (end - start)[:, None]
    s = ((times - start) / (end - start))[:, None]
    q0, q1 = waypoints.positions[segment], waypoints.positions[segment + 1]
    shape = 10 * s**3 - 15 * s**4 + 6 * s**5
    shape_d = (30 * s**2 - 60 * s**3 + 30 * s**4) / span
    shape_dd = (60 * s - 180 * s**2 + 120 * s**3) / span**2
    gripper = waypoints.resample(times).channels["gripper"]
    return JointTrajectory(
        t=times,
        joint_names=waypoints.joint_names,
        positions=q0 + (q1 - q0) * shape,
        velocities=(q1 - q0) * shape_d,
        accelerations=(q1 - q0) * shape_dd,
        limits=UR5E_LIMITS,
        channels={"gripper": gripper},
        header=Header(stamp=0.0, frame_id="base_link"),
    )


def tool_trajectory(joints: JointTrajectory) -> PoseTrajectory:
    """Where tool0 goes: FK over the whole (N, 6) batch view in one call, no per-step loop."""
    return PoseTrajectory.from_matrices(
        joints.t,
        ur5e_forward_kinematics(joints.positions),
        child_frame_id="tool0",
        header=joints.header,
    ).with_derivatives()


def main() -> None:
    waypoints = waypoint_trajectory()
    joints = minimum_jerk(waypoints)
    tool = tool_trajectory(joints)

    print(joints)
    print(tool)
    print("batch view   joints.positions.shape =", joints.positions.shape)
    print("per step     joints[250] =", joints[250])
    print("by time      joints.at(2.0).positions =", np.round(joints.at(2.0).positions, 4))
    print("one joint    joints.joint('elbow_joint') =", joints.joint("elbow_joint"))
    print("by label     waypoints.at(2.0)['name'] =", waypoints.at(2.0)["name"])
    print("gripper      closed from t =", joints.t[np.argmax(joints.channels["gripper"] > 0)], "s")
    peak_speed = np.linalg.norm(tool.linear_velocities, axis=1).max()
    print(f"tool path    {tool.path_length():.3f} m, peak speed {peak_speed:.3f} m/s")
    print("limits       as planned:", joints.limit_violations() or "within limits")
    print("limits       2x faster: ")
    for problem in joints.time_scaled(0.5).limit_violations():
        print("               ", problem)


if __name__ == "__main__":
    main()
