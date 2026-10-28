"""One representative instance of every message type. The contract tests require full coverage,
so adding a type without adding it here fails the suite."""

import numpy as np

from robot_msgs.geometry import (
    Accel,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Twist,
    Vector3,
    Wrench,
)
from robot_msgs.sensor import CameraInfo, Image, Imu, JointState, PointCloud
from robot_msgs.std import Header
from robot_msgs.trajectory import (
    JointLimits,
    JointTrajectory,
    JointTrajectoryPoint,
    PoseTrajectory,
    PoseTrajectoryPoint,
    Trajectory,
    TrajectoryPoint,
)
from robot_msgs.vision import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

RNG = np.random.default_rng(3)
HEADER = Header(stamp=1_761_600_000.25, frame_id="camera_color_optical_frame")
POSE = Pose.from_xyz_rpy(0.4, -0.1, 0.25, 0.0, np.pi / 2, 0.3)
NAMES = ("j1", "j2", "j3")
JOINT_TRAJ = JointTrajectory(
    t=np.linspace(0, 1, 11),
    joint_names=NAMES,
    positions=RNG.normal(size=(11, 3)),
    velocities=RNG.normal(size=(11, 3)),
    limits=JointLimits(joint_names=NAMES, max_velocity=[2.0] * 3, max_jerk=[50.0] * 3),
    channels={"kp": np.full((11, 3), 40.0), "mode": np.arange(11) % 3},
    header=Header(stamp=12.0, frame_id="base_link"),
)
POSE_TRAJ = PoseTrajectory.from_poses(
    [0.0, 0.5, 1.0],
    [POSE, POSE @ Pose.from_xyz_rpy(0.1, 0, 0, 0, 0, 0.2), POSE],
    child_frame_id="tool0",
    header=Header(frame_id="base_link"),
).with_derivatives()
HYPOTHESIS = ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("bottle", 0.91), pose=POSE)
BOX2D = BoundingBox2D.from_xyxy(10, 20, 110, 220)
BOX3D = BoundingBox3D(center=POSE, size=Vector3(0.07, 0.07, 0.25))
DETECTION2D = Detection2D(header=HEADER, results=(HYPOTHESIS,), bbox=BOX2D, id="7")
DETECTION3D = Detection3D(header=HEADER, results=(HYPOTHESIS,), bbox=BOX3D, id="7")
CAMERA = CameraInfo.from_intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480, header=HEADER)

EXAMPLES = [
    HEADER,
    Vector3(1.0, -2.0, 0.5),
    Point(0.1, 0.2, 0.3),
    Quaternion.from_euler(0.1, 0.2, 0.3),
    POSE,
    Pose2D(1.0, 2.0, 0.5),
    POSE.as_transform(),
    Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0.5)),
    Wrench(Vector3(0, 0, -9.81), Vector3(0.1, 0, 0)),
    Accel(Vector3(0, 0, 9.81), Vector3()),
    PoseStamped(HEADER, POSE),
    TransformStamped(HEADER, "tool0", POSE.as_transform()),
    JointState(name=NAMES, position=[0.1, 0.2, 0.3], velocity=[0.0, 0.1, 0.0], header=HEADER),
    Image.from_array(RNG.integers(0, 255, size=(48, 64, 3), dtype=np.uint8), header=HEADER),
    CAMERA,
    PointCloud(
        points=RNG.normal(size=(100, 3)).astype(np.float32),
        channels={"rgb": RNG.integers(0, 255, size=(100, 3), dtype=np.uint8)},
        header=HEADER,
    ),
    Imu(
        orientation=Quaternion(),
        angular_velocity=Vector3(0, 0, 0.1),
        linear_acceleration=Vector3(0, 0, 9.81),
        orientation_covariance=np.eye(3) * 1e-3,
        header=HEADER,
    ),
    BOX2D,
    BOX3D,
    ObjectHypothesis("bottle", 0.91),
    HYPOTHESIS,
    DETECTION2D,
    Detection2DArray(header=HEADER, detections=(DETECTION2D, DETECTION2D.replace(id="8"))),
    DETECTION3D,
    Detection3DArray(header=HEADER, detections=(DETECTION3D,)),
    Trajectory(t=[0.0, 0.1, 0.2], channels={"temperature": [20.0, 20.5, 21.0]}),
    JOINT_TRAJ[4],
    JOINT_TRAJ.limits,
    JOINT_TRAJ,
    POSE_TRAJ[1],
    POSE_TRAJ,
    Trajectory(t=[0.0], channels={"x": [1.0]})[0],
]

assert isinstance(EXAMPLES[-1], TrajectoryPoint)
assert isinstance(JOINT_TRAJ[4], JointTrajectoryPoint) and isinstance(
    POSE_TRAJ[1], PoseTrajectoryPoint
)
