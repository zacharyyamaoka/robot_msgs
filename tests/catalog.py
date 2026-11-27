"""One representative instance of every message type. The contract tests require full coverage,
so adding a type without adding it here fails the suite."""

import math

import numpy as np

from robot_msgs.behavior import BTState
from robot_msgs.compare import (
    Dataset,
    JointTolerance,
    PoseTolerance,
    Recording,
    Run,
    Tolerances,
    aggregate,
    check,
    compare,
    compare_dataset,
    error_stats,
    estimate_time_offset,
    goal,
)
from robot_msgs.control import (
    EffortStall,
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    ServoCommand,
    ServoState,
    Settled,
    Stall,
)
from robot_msgs.fiducial import (
    FiducialArray,
    FiducialBoard,
    FiducialDetection,
    FiducialMap,
    FiducialMount,
    FiducialPose,
    FiducialTrack,
    FiducialTrackArray,
)
from robot_msgs.geometry import (
    Accel,
    Inertia,
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
from robot_msgs.manipulation import (
    AllowedCollisionMatrix,
    CartesianLimits,
    CollisionElement,
    CollisionScene,
    GraspPose,
    NamedPose,
    Waypoint,
    WaypointAction,
)
from robot_msgs.robot import RobotConfig, RobotState
from robot_msgs.sensor import CameraInfo, CameraSettings, Image, Imu, JointState, PointCloud
from robot_msgs.shape import MeshResource, Plane, SolidPrimitive
from robot_msgs.std import ColorRGBA, DiagnosticRecord, Header, LogRecord, Metrics
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    JointLimits,
    JointTrajectory,
    JointTrajectoryPoint,
    PoseTrajectory,
    PoseTrajectoryPoint,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    Trajectory,
    TrajectoryPoint,
    WrenchTrajectory,
    joint_limit_violations,
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
# A physical stop, a software bound and an unstated one, and what JOINT_TRAJ's run does to them.
KIND_LIMITS = JointLimits(
    joint_names=NAMES,
    min_position=[-1.0, -2.0, -np.inf],
    max_position=[1.0, 2.0, np.inf],
    position_kind=("physical", "software", None),
)
LIMIT_VIOLATIONS = joint_limit_violations(JOINT_TRAJ, KIND_LIMITS, quantities=("position",))
POSE_TRAJ = PoseTrajectory.from_poses(
    [0.0, 0.5, 1.0],
    [POSE, POSE @ Pose.from_xyz_rpy(0.1, 0, 0, 0, 0, 0.2), POSE],
    child_frame_id="tool0",
    header=Header(frame_id="base_link"),
).with_derivatives()
TOLERANCES = Tolerances(
    path=(JointTolerance(name="j1", position=0.01), JointTolerance(name="j2", position=0.05)),
    goal=(JointTolerance(name="j1", position=0.01, velocity=0.05),),
    goal_time_tolerance=0.5,
)
OUTCOME = check(JOINT_TRAJ, JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions * 1.1), TOLERANCES)
HYPOTHESIS = ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("bottle", 0.91), pose=POSE)
BOX2D = BoundingBox2D.from_xyxy(10, 20, 110, 220)
BOX3D = BoundingBox3D(center=POSE, size=Vector3(0.07, 0.07, 0.25))
DETECTION2D = Detection2D(header=HEADER, results=(HYPOTHESIS,), bbox=BOX2D, id="7")
DETECTION3D = Detection3D(header=HEADER, results=(HYPOTHESIS,), bbox=BOX3D, id="7")
CAMERA = CameraInfo.from_intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480, header=HEADER)

# The recorded streams: what a driver, an IMU, an F/T sensor and a camera log on sim and on real.
SERVO_STATES = ServoStateTrajectory.from_servo_states(
    [
        ServoState(
            name=NAMES,
            position=[0.1 * k, 0.2, -0.1 * k],
            velocity=[0.1, 0.0, -0.1],
            effort=[0.5, -0.2, 0.1 * k],
            temperature=[31.0 + 0.1 * k, 30.5, 29.0],
            voltage=[24.1, 24.0, 24.1],
            fault=[0, 0, 0 if k < 4 else 2],
            seq=800 + k,
            header=HEADER.replace(stamp=HEADER.stamp + 0.01 * k),
        )
        for k in range(6)
    ]
)
SERVO_COMMANDS = ServoCommandTrajectory.from_servo_commands(
    [
        ServoCommand(
            name=NAMES,
            position=[0.1 * k + 0.01, 0.2, float("nan")],
            velocity=[0.1, 0.0, -0.1],
            stiffness=[30.0, 30.0, 5.0],
            state_stamp=None if k == 0 else HEADER.stamp + 0.01 * k - 0.002,
            header=HEADER.replace(stamp=HEADER.stamp + 0.01 * k),
        )
        for k in range(6)
    ]
)
IMU_TRAJ = ImuTrajectory(
    t=np.linspace(0.0, 0.5, 6),
    orientations=[Quaternion.from_euler(0.0, 0.0, 0.1 * k).as_array() for k in range(6)],
    angular_velocities=np.tile([0.0, 0.0, 0.2], (6, 1)),
    linear_accelerations=np.tile([0.0, 0.1, 9.81], (6, 1)),
    header=Header(stamp=12.0, frame_id="imu_link"),
)
WRENCH_TRAJ = WrenchTrajectory(
    t=np.linspace(0.0, 0.5, 6),
    forces=np.column_stack([np.zeros(6), np.zeros(6), np.linspace(-1.0, -9.0, 6)]),
    torques=np.tile([0.05, -0.02, 0.0], (6, 1)),
    channels={"saturated": np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)},
    header=Header(stamp=12.0, frame_id="ft_sensor"),
)
IMAGE_TRAJ = ImageTrajectory(
    t=[0.0, 1 / 30, 2 / 30],
    encoding="rgb8",
    frames=RNG.integers(0, 255, size=(3, 12, 16, 3), dtype=np.uint8),
    channels={"exposure_s": [0.008, 0.008, 0.01]},
    header=HEADER,
)
CLOUD = PointCloud(
    points=RNG.normal(size=(100, 3)).astype(np.float32),
    channels={"rgb": RNG.integers(0, 255, size=(100, 3), dtype=np.uint8)},
    header=HEADER,
)
# A run as sim-to-real records it: the target, then each source's streams under the same names.
RUN = Run(
    name="pick_0.5",
    recordings={
        "target": Recording(streams={"joints": JOINT_TRAJ}),
        "sim": Recording(
            streams={
                "joints": JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions + 0.01),
                "servo_state": SERVO_STATES,
                "imu": IMU_TRAJ,
                "camera": IMAGE_TRAJ,
            },
            header=Header(meta={"engine": "mujoco 3.3"}),
        ),
        "real": Recording(
            streams={
                "joints": JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions - 0.02),
                "servo_state": SERVO_STATES.replace(positions=SERVO_STATES.positions + 0.003),
                "imu": IMU_TRAJ.replace(angular_velocities=IMU_TRAJ.angular_velocities * 1.1),
                "scan": CLOUD,
            },
            header=Header(meta={"rig": "bam_fb_v2"}),
        ),
    },
    tags={"speed": 0.5, "payload_kg": 0.2, "gripper": "claw", "ok": True},
    time_offset_s={"real": 0.0},
    title="pick at half speed",
)
DATASET = Dataset(
    runs=(RUN, RUN.replace(name="pick_1.0", tags={**RUN.tags, "speed": 1.0})),
    title="speed sweep",
)
DATASET_COMPARISON = compare_dataset(DATASET)

L_PRISM = SolidPrimitive.prism(0.4, [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])

FIDUCIAL_DETECTION = FiducialDetection(
    id=7,
    dictionary="DICT_4X4_50",
    corners=[[600.5, 300.25], [680.0, 302.0], [678.5, 381.0], [598.0, 379.5]],
    marker_length=0.0275,
    decision_margin=41.5,
    hamming=0,
).with_corner_sigma(0.3)
FIDUCIAL_POSE = FiducialPose.from_candidates(
    7, "DICT_4X4_50", [POSE, Pose.from_xyz_rpy(0.4, -0.1, 0.25, 0.0, 0.0, 0.0)], [0.42, 0.61]
).replace(covariance=np.diag([1e-6, 1e-6, 4e-6, 1e-4, 1e-4, 1e-5]))
FIDUCIAL_MOUNT = FiducialMount(
    id=7,
    dictionary="DICT_4X4_50",
    marker_length=0.0275,
    pose=Pose.from_xyz_rpy(0.0, 0.03, 0.05, np.pi / 2, 0.0, 0.0),
    covariance=np.diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]),
    header=Header(frame_id="link_6", meta={"source": "cad"}),
)
FIDUCIAL_BOARD = FiducialBoard(
    layout="charuco",
    dictionary="DICT_4X4_50",
    cols=3,
    rows=3,
    marker_length=0.017,
    square_length=0.023,
    header=Header(frame_id="wrist_board"),
)

BT_STATE = BTState(
    tree_id="Bringup",
    node_status={"Bringup/Sequence": "RUNNING", "Bringup/Sequence/Home": "SUCCESS",
                 "Bringup/Sequence/WarmUp": "RUNNING", "Bringup/Sequence/Pick": "IDLE"},
    blackboard={"homed": True, "warmed_up": False, "last_active_s": 1795828140.4,
                "bins": ["paper", "plastic"], "limits": {"speed": 0.5, "tool": None}},
    tick=42,
    clock_s=12.5,
    header=HEADER,
)  # fmt: skip

EXAMPLES = [
    BT_STATE,
    RobotState(
        components={
            "bt": BT_STATE,
            "joints": JointState(name=("j1", "j2"), position=[0.1, -1.2], velocity=[0.0, 0.3]),
        },
        header=HEADER,
    ),
    RobotConfig(components={"camera": CameraSettings(exposure_time=0.01)}),
    CameraSettings(
        exposure_time=0.008,
        gain=6.0,
        white_balance=4500.0,
        gamma=1.0,
        auto_exposure=False,
        auto_white_balance=False,
        focus=120.0,
        frame_rate=30.0,
        header=Header(stamp=1_761_600_000.0, frame_id="camera_color_optical_frame"),
    ),
    FIDUCIAL_DETECTION,
    FIDUCIAL_POSE,
    FiducialArray(header=HEADER, detections=(FIDUCIAL_DETECTION,), poses=(FIDUCIAL_POSE,)),
    FIDUCIAL_MOUNT,
    FiducialMap(mounts=(FIDUCIAL_MOUNT,) + FIDUCIAL_BOARD.mounts().mounts),
    FIDUCIAL_BOARD,
    FiducialTrack(
        id=7,
        dictionary="DICT_4X4_50",
        pose=POSE,
        covariance=np.eye(6) * 1e-6,
        observed=True,
        last_seen=1_761_600_000.25,
        nis=3.2,
    ),
    FiducialTrackArray(
        header=Header(stamp=1_761_600_000.25, frame_id="base_link"),
        tracks=(
            FiducialTrack(
                id=7, dictionary="DICT_4X4_50", pose=POSE, observed=True, last_seen=1_761_600_000.25
            ),
            FiducialTrack(id=8, dictionary="DICT_4X4_50", pose=POSE, last_seen=1_761_599_999.0),
        ),
    ),
    HEADER,
    ColorRGBA(0.5, 0.25, 0.75, 1.0),
    LogRecord(
        header=Header(stamp=1_761_600_000.5),
        path="pick_cell.arm.jtc",
        tick=4812,
        level=30,
        message="input 'state' is 12.0 ms old (limit 5.0 ms)",
    ),
    DiagnosticRecord(
        header=Header(stamp=1_761_600_001.0),
        path="pick_cell.arm.jtc",
        level="warn",
        code="InputStale",
        message="input 'state' is 12.0 ms old (limit 5.0 ms)",
        mode="operational",
        expected_period_s=1.0,
        uptime_s=93.5,
        stale_inputs=("state",),
    ),
    Metrics(
        header=Header(stamp=1_761_600_001.0),
        path="pick_cell.arm.jtc",
        names=("steps", "step_p99_s", "overruns", "input_age_s.state"),
        values=[1000.0, 0.000183, 0.0, float("nan")],
    ),
    Vector3(1.0, -2.0, 0.5),
    Point(0.1, 0.2, 0.3),
    Quaternion.from_euler(0.1, 0.2, 0.3),
    POSE,
    Pose2D(1.0, 2.0, 0.5),
    POSE.as_transform(),
    Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0.5)),
    Wrench(Vector3(0, 0, -9.81), Vector3(0.1, 0, 0)),
    Accel(Vector3(0, 0, 9.81), Vector3()),
    Inertia(0.75, Vector3(0.0, 0.01, 0.04), 1.2e-3, 1.0e-6, -2.0e-6, 1.1e-3, 3.0e-6, 4.0e-4),
    PoseStamped(HEADER, POSE),
    TransformStamped(HEADER, "tool0", POSE.as_transform()),
    JointState(name=NAMES, position=[0.1, 0.2, 0.3], velocity=[0.0, 0.1, 0.0], header=HEADER),
    ServoState(
        name=NAMES,
        position=[0.1, 0.2, 0.3],
        velocity=[0.0, 0.1, 0.0],
        effort=[0.5, -0.2, 0.0],
        temperature=[31.0, 30.5, 29.0],
        voltage=[24.1, 24.0, 24.1],
        fault=[0, 0, 0],
        seq=812,
        header=HEADER,
    ),
    ServoCommand(
        name=NAMES,
        position=[0.11, 0.21, float("nan")],
        velocity=[0.1, 0.1, 0.2],
        feedforward_effort=[0.4, -0.1, 0.0],
        stiffness=[30.0, 30.0, 5.0],
        damping=[1.0, 1.0, 0.3],
        max_effort=[4.0, 4.0, 2.0],
        state_stamp=HEADER.stamp,
        state_seq=812,
        header=HEADER.replace(stamp=HEADER.stamp + 0.0021),
    ),
    JointCommand.to("j2", 1.9, speed=0.3, max_effort=4.0, feedforward_effort=0.5),
    Elapsed(duration=3.0),
    Settled(joints=("j2", "c1"), speed=0.01, dwell=0.5),
    Reached(joints=("j2",), tolerance=0.01),
    Stall(joint="c1", lead=0.08),
    EffortStall(joint="j1", effort=9.0),
    JointCommandStep(
        name="define",
        commands=(JointCommand.hold("j2", max_effort=10.0), JointCommand.limp("c1")),
        done_when=(Stall(joint="j2"), Elapsed(duration=5.0)),
        timeout=8.0,
        define_position={"j2": 1.9967},
    ),
    JointCommandSequence(
        name="home",
        joint_names=("j2", "c1"),
        max_speed=3.0,
        steps=(
            JointCommandStep(
                name="settle",
                commands=(JointCommand.limp("j2"), JointCommand.limp("c1")),
                done_when=(
                    Elapsed(duration=3.0),
                    Settled(joints=("j2", "c1"), speed=0.01, dwell=0.5),
                ),
                timeout=10.0,
            ),
            JointCommandStep(
                name="seek stop",
                commands=(
                    JointCommand.move_by("j2", 2.0, speed=0.15, max_effort=8.0),
                    JointCommand.hold("c1", max_effort=6.0),
                ),
                done_when=(Stall(joint="j2"),),
                timeout=30.0,
                define_position={"j2": 1.9967},
            ),
            JointCommandStep(
                name="home",
                commands=(
                    JointCommand.to("j2", 1.9, speed=0.3, max_effort=4.0, feedforward_effort=0.5),
                    JointCommand.hold("c1", max_effort=6.0),
                ),
                done_when=(Reached(joints=("j2",), tolerance=0.01),),
                timeout=20.0,
            ),
        ),
    ),
    GraspPose(
        pose=Pose.from_xyz_rpy(0.42, -0.05, 0.03, math.pi, 0.0, 0.3),
        width=0.012,
        pre_grasp_width=0.06,
        score=0.87,
        object_id="bottle",
        header=HEADER.replace(frame_id="world"),
    ),
    CartesianLimits(max_linear_velocity=0.5, max_angular_jerk=math.inf),
    AllowedCollisionMatrix.from_pairs(
        ("base_link", "shoulder_link", "upper_arm_link", "forearm_link", "conveyor"),
        [
            ("base_link", "shoulder_link", "Adjacent"),
            ("base_link", "conveyor", "Default"),
            ("shoulder_link", "forearm_link", "Never"),
            ("upper_arm_link", "conveyor"),
        ],
        why="robot_collision fast, 10000 samples, seed 0",
    ),
    CollisionElement(
        id="glove",
        source="object",
        pose=Pose.from_xyz_rpy(0.1, 0.0, 0.62, 0.0, 0.0, 0.4),
        primitives=(SolidPrimitive.box(0.20, 0.10, 0.03),),
        primitive_poses=(Pose.identity(),),
        motion="moving",
        compliance="deformable",
        margin=0.01,
        label="glove",
        why="detector box, 5 mm noise",
    ),
    CollisionScene(
        name="conveyor cell",
        elements=(
            CollisionElement(
                id="floor",
                source="scene",
                planes=(Plane.from_point_normal((0, 0, 0), (0, 0, 1)),),
                plane_poses=(Pose.identity(),),
                label="concrete floor",
            ),
            CollisionElement(
                id="conveyor",
                source="scene",
                pose=Pose.from_xyz_rpy(0.0, 0.0, 0.3, 0, 0, 0),
                primitives=(SolidPrimitive.box(0.55, 2.4, 0.6),),
                primitive_poses=(Pose.identity(),),
                margin=0.005,
            ),
            CollisionElement(
                id="bottle",
                source="object",
                pose=Pose.from_xyz_rpy(0.05, 0.3, 0.7, 0, 0, 0),
                primitives=(SolidPrimitive.cylinder(0.2, 0.04),),
                primitive_poses=(Pose.identity(),),
                motion="moving",
                compliance="deformable",
                margin=0.01,
                label="bottle",
            ),
            CollisionElement(
                id="forearm_link",
                source="self",
                link="forearm_link",
                pose=Pose.from_xyz_rpy(-0.3, 0.0, 1.0, 0, 1.2, 0),
                primitives=(SolidPrimitive.cylinder(0.39, 0.045),),
                primitive_poses=(Pose.identity(),),
                motion="moving",
            ),
        ),
        robot_state=JointState(name=("shoulder", "elbow"), position=[0.1, -1.2]),
        acm=AllowedCollisionMatrix.from_pairs(
            ("floor", "conveyor", "forearm_link"), [("floor", "conveyor", "Adjacent")]
        ),
        why="v1: static scene + detected objects + self at q",
    ),
    NamedPose.from_degrees(
        name="ready",
        group="arm",
        degrees={"joint_1": -12.7, "joint_2": -24.754, "joint_3": 90.0},
        tolerance_deg=3.0,
        why="where the arm waits between picks",
    ),
    NamedPose(
        name="over_bin",
        group="arm",
        joint_names=("joint_1", "joint_2", "joint_3"),
        positions=[0.0, -0.5, 1.2],
        tool_pose=PoseStamped.from_xyz_rpy(0.42, -0.05, 0.40, math.pi, 0.0, 0.0, frame_id="world"),
        tip_link="tcp",
        why="the claw 40 mm over the bin's drop, plumb",
    ),
    Waypoint(
        name="closed_grasp",
        label="close",
        pose=Pose.from_xyz_rpy(0.42, -0.05, 0.03, math.pi, 0.0, 0.3),
        gripper_width=0.012,
        gripper_max_effort=20.0,
        dwell=0.2,
        target_velocity=Vector3(0.1, 0.0, 0.0),
    ),
    WaypointAction(
        tool_frame="tcp_world",
        header=HEADER.replace(frame_id="world"),
        start_state=JointState(
            name=("joint_1", "joint_2", "gripper_joint"),
            position=[0.0, -0.5, 0.03],
            velocity=[0, 0, 0],
        ),
        arm_joints=("joint_1", "joint_2"),
        hand_joints=("gripper_joint",),
        cartesian_limits=CartesianLimits(),
        arm_limits=JointLimits(joint_names=("joint_1", "joint_2"), max_velocity=[2.0, 2.0]),
        sync_target="closed_grasp",
        waypoints=(
            Waypoint(
                name="start", label="start", pose=Pose.from_xyz_rpy(0.3, 0.0, 0.3, math.pi, 0, 0)
            ),
            Waypoint(
                name="pre_grasp",
                label="pre_grasp",
                pose=Pose.from_xyz_rpy(0.42, -0.05, 0.10, math.pi, 0.0, 0.3),
                gripper_width=0.06,
                blend_radius=0.02,
                blend_points=4,
                target_velocity=Vector3(0.1, 0.0, 0.0),
            ),
            Waypoint(
                name="closed_grasp",
                label="close",
                pose=Pose.from_xyz_rpy(0.42, -0.05, 0.03, math.pi, 0.0, 0.3),
                gripper_width=0.012,
                interpolation_points=3,
                duration=0.8,
                target_velocity=Vector3(0.1, 0.0, 0.0),
            ),
            Waypoint(
                name="lift",
                label="lift",
                pose=Pose.from_xyz_rpy(0.42, -0.05, 0.15, math.pi, 0.0, 0.3),
                target_velocity=Vector3(0.1, 0.0, 0.0),
                sync_at_target_time=True,
            ),
        ),
    ),
    Image.from_array(RNG.integers(0, 255, size=(48, 64, 3), dtype=np.uint8), header=HEADER),
    CAMERA,
    CLOUD,
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
    KIND_LIMITS,
    LIMIT_VIOLATIONS[0],
    JOINT_TRAJ,
    POSE_TRAJ[1],
    POSE_TRAJ,
    error_stats([0.002, -0.001, 0.003]),
    compare(JOINT_TRAJ, JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions * 1.1)),
    compare(POSE_TRAJ, POSE_TRAJ.shifted(dz=0.002), time_offset_s=0.0),
    SolidPrimitive.box(0.3, 0.2, 0.5),
    SolidPrimitive.sphere(0.25),
    SolidPrimitive.cylinder(0.6, 0.15),
    SolidPrimitive.cone(0.5, 0.2),
    L_PRISM,
    SolidPrimitive.cone(0.5, 0.2).to_mesh(12),
    Plane.from_point_normal(Point(0.0, 0.0, 0.75), Vector3(0.0, 0.0, 1.0)),
    MeshResource("package://bam_arm/meshes/link1.stl", scale=Vector3(0.001, 0.001, 0.001)),
    JointTolerance(name="j1", position=0.01, velocity=0.1),
    PoseTolerance(position=0.005, rotation=0.02),
    TOLERANCES,
    OUTCOME.violations[0],
    OUTCOME,
    aggregate(JOINT_TRAJ, [JOINT_TRAJ, JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions + 0.01)]),
    estimate_time_offset(JOINT_TRAJ, JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions * 1.1)),
    goal(JOINT_TRAJ, JOINT_TRAJ.replace(positions=JOINT_TRAJ.positions * 1.1), band=0.05)[
        "positions/j1"
    ],
    Trajectory(t=[0.0], channels={"x": [1.0]})[0],
    SERVO_STATES,
    SERVO_STATES[4],
    SERVO_COMMANDS,
    SERVO_COMMANDS[1],
    IMU_TRAJ,
    IMU_TRAJ[2],
    WRENCH_TRAJ,
    WRENCH_TRAJ[5],
    IMAGE_TRAJ,
    IMAGE_TRAJ[1],
    compare(IMAGE_TRAJ, IMAGE_TRAJ.replace(frames=IMAGE_TRAJ.frames // 2)),
    compare(CLOUD, CLOUD.transformed(POSE.as_transform().replace(rotation=Quaternion()))),
    RUN.recordings["sim"],
    RUN,
    DATASET,
    DATASET_COMPARISON.runs["pick_0.5"],
    DATASET_COMPARISON,
]

assert OUTCOME.violations, "the catalog's Outcome must carry a Violation to exemplify"
assert LIMIT_VIOLATIONS, "the catalog's run must leave KIND_LIMITS somewhere to exemplify"
assert any(type(example) is TrajectoryPoint for example in EXAMPLES)
assert isinstance(JOINT_TRAJ[4], JointTrajectoryPoint) and isinstance(
    POSE_TRAJ[1], PoseTrajectoryPoint
)
