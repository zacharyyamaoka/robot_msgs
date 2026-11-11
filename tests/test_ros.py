"""robot_msgs.ros: conversion to and from real ROS 2 (Jazzy) messages, and rosbag2 bags."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from catalog import EXAMPLES, HEADER, JOINT_TRAJ, POSE, POSE_TRAJ
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from robot_msgs import allclose
from robot_msgs.geometry import Inertia, Pose, PoseStamped, Twist, Vector3
from robot_msgs.ros import from_ros, read_bag, ros_type, to_ros, typestore, write_bag
from robot_msgs.sensor import CameraInfo, Image, Imu, JointState, PointCloud
from robot_msgs.shape import Mesh, Plane, SolidPrimitive
from robot_msgs.std import ColorRGBA, Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory, Trajectory
from robot_msgs.vision import Detection2DArray, ObjectHypothesisWithPose

UNMAPPED = {
    "Trajectory",
    "TrajectoryPoint",
    "JointTrajectoryPoint",
    "PoseTrajectoryPoint",
    "JointLimits",
    # robot_msgs.compare results: ROS has no message for a trajectory comparison
    "ErrorStats",
    "Comparison",
    # a file reference: shape_msgs has no file mesh (only visualization_msgs Marker MESH_RESOURCE,
    # which also needs a pose, colour and header of its own)
    "MeshResource",
    # control_msgs/JointTolerance exists, but the Jazzy typestore has no control_msgs
    "JointTolerance",
    "PoseTolerance",
    "Tolerances",
    "Violation",
    "Outcome",
    "Aggregate",
    "TimeOffsetEstimate",
    "GoalStats",
    # robot_msgs.control: ROS has no servo message (ros2_control keeps these inside the process)
    "ServoState",
    "ServoCommand",
}


def over_the_wire(ros_msg):
    """CDR-serialise and parse again, as a bag or a DDS network would."""
    store = typestore()
    raw = store.serialize_cdr(ros_msg, ros_msg.__msgtype__)
    return store.deserialize_cdr(bytes(raw), ros_msg.__msgtype__)


def round_trip(msg, **options):
    return from_ros(over_the_wire(to_ros(msg, **options)))


MAPPED = [m for m in EXAMPLES if type(m).__name__ not in UNMAPPED]


# ── every catalog type ───────────────────────────────────────────────────────


def test_catalog_is_split_into_mapped_and_unmapped():
    assert {type(m).__name__ for m in EXAMPLES if type(m).__name__ in UNMAPPED} == UNMAPPED
    assert len(MAPPED) + 19 == len(EXAMPLES)


@pytest.mark.parametrize("msg", MAPPED, ids=lambda m: type(m).__name__)
def test_round_trip_through_cdr(msg):
    if isinstance(msg, JointTrajectory):
        # ROS has nowhere for channels or limits
        assert allclose(round_trip(msg, drop_extras=True), msg.replace(channels={}, limits=None))
    elif isinstance(msg, PoseTrajectory):
        # a Path keeps neither velocities nor the child frame ...
        path = round_trip(msg, drop_extras=True)
        expected = msg.replace(linear_velocities=None, angular_velocities=None, child_frame_id="")
        assert allclose(path, expected)
        # ... MultiDOF keeps everything
        assert allclose(round_trip(msg, as_="multidof"), msg)
    else:
        assert allclose(round_trip(msg), msg)


@pytest.mark.parametrize("msg", MAPPED, ids=lambda m: type(m).__name__)
def test_ros_type_matches_the_converted_message(msg):
    name = ros_type(msg)
    assert ros_type(type(msg)) == name
    assert to_ros(msg, drop_extras=True).__msgtype__ == name


@pytest.mark.parametrize(
    "msg", [m for m in EXAMPLES if type(m).__name__ in UNMAPPED], ids=lambda m: type(m).__name__
)
def test_types_without_a_ros_message_raise_type_error(msg):
    with pytest.raises(TypeError, match=f"no standard ROS message for {type(msg).__name__}"):
        to_ros(msg)
    with pytest.raises(TypeError, match="no standard ROS message for"):
        ros_type(msg)


def test_from_ros_rejects_unknown_messages():
    store = typestore()
    with pytest.raises(TypeError, match="no robot_msgs type for ROS message std_msgs/msg/String"):
        from_ros(store.types["std_msgs/msg/String"](data="hi"))
    with pytest.raises(TypeError, match="not a rosbags message"):
        from_ros(object())


def test_ros_type_names():
    assert ros_type(JointState) == "sensor_msgs/msg/JointState"
    assert ros_type(Inertia) == "geometry_msgs/msg/Inertia"
    assert ros_type(PointCloud) == "sensor_msgs/msg/PointCloud2"
    assert ros_type(PoseTrajectory) == "nav_msgs/msg/Path"
    assert ros_type(Detection2DArray) == "vision_msgs/msg/Detection2DArray"
    assert ros_type(SolidPrimitive) == "shape_msgs/msg/SolidPrimitive"
    assert ros_type(Mesh) == "shape_msgs/msg/Mesh"
    assert ros_type(Plane) == "shape_msgs/msg/Plane"
    assert ros_type(ColorRGBA) == "std_msgs/msg/ColorRGBA"


# ── typestore ────────────────────────────────────────────────────────────────


def test_typestore_is_cached_jazzy_with_vision_msgs():
    store = typestore()
    assert store is typestore()
    jazzy = get_typestore(Stores.ROS2_JAZZY)
    assert set(jazzy.types) <= set(store.types)
    assert {n for n in store.types if n.startswith("vision_msgs/")} >= {
        "vision_msgs/msg/Point2D",
        "vision_msgs/msg/Pose2D",
        "vision_msgs/msg/BoundingBox2D",
        "vision_msgs/msg/BoundingBox3D",
        "vision_msgs/msg/ObjectHypothesis",
        "vision_msgs/msg/ObjectHypothesisWithPose",
        "vision_msgs/msg/Detection2D",
        "vision_msgs/msg/Detection2DArray",
        "vision_msgs/msg/Detection3D",
        "vision_msgs/msg/Detection3DArray",
    }


def test_point_field_codes_match_the_ros_constants():
    from robot_msgs.ros.convert import _POINT_FIELD_DTYPES

    field = typestore().types["sensor_msgs/msg/PointField"]
    for name in ("INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT32", "FLOAT64"):
        assert _POINT_FIELD_DTYPES[getattr(field, name)] == np.dtype(name.lower())


# ── layout spot checks against the ROS message ───────────────────────────────


def test_header_splits_stamp_into_sec_and_nanosec():
    ros = to_ros(Header(stamp=1_761_600_000.25, frame_id="map"))
    assert (ros.stamp.sec, ros.stamp.nanosec, ros.frame_id) == (1_761_600_000, 250_000_000, "map")
    assert from_ros(ros).stamp == 1_761_600_000.25


def test_nanosecond_stamps_survive_the_wire():
    # float64 steps 256 ns at this epoch, so the wire's nanosec is the nearest float, not 789
    header = Header.from_ns(1_761_600_000_123_456_789, "map")
    ros = over_the_wire(to_ros(header))
    assert ros.stamp.sec == 1_761_600_000 and abs(ros.stamp.nanosec - 123_456_789) <= 128
    assert round_trip(header) == header


def test_joint_state_uses_empty_arrays_for_absent_fields():
    state = JointState(name=("a", "b"), position=[1.0, 2.0], header=HEADER)
    ros = to_ros(state)
    assert list(ros.name) == ["a", "b"] and ros.position.dtype == np.float64
    assert len(ros.velocity) == 0 and len(ros.effort) == 0
    back = round_trip(state)
    assert back.velocity is None and back.effort is None


def test_image_fields():
    ros = to_ros(catalog_example(Image))
    assert (ros.height, ros.width, ros.encoding, ros.is_bigendian) == (48, 64, "rgb8", 0)
    assert ros.step == 64 * 3 and len(ros.data) == 48 * 64 * 3 and ros.data.dtype == np.uint8
    assert bytes(ros.data[:3]) == catalog_example(Image).data[0, 0].tobytes()


@pytest.mark.parametrize(
    ("array", "encoding"),
    [
        (np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000, "16UC1"),
        (np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4), "32FC1"),
        (np.arange(12, dtype=np.uint8).reshape(3, 4), "mono8"),
        (np.arange(24, dtype=np.uint8).reshape(3, 2, 4), "bgra8"),
    ],
)
def test_image_encodings_round_trip(array, encoding):
    image = Image.from_array(array, encoding=encoding, header=HEADER)
    ros = to_ros(image)
    assert ros.step == image.step
    assert allclose(round_trip(image), image)


def test_image_with_row_padding_and_big_endian_from_ros():
    store = typestore()
    pixels = np.array([[1, 2, 3], [4, 5, 6]], dtype=">u2")
    padded = np.zeros((2, 10), dtype=np.uint8)  # step 10 bytes, only 6 are pixels
    padded[:, :6] = pixels.view(np.uint8).reshape(2, 6)
    ros = store.types["sensor_msgs/msg/Image"](
        header=to_ros(Header()), height=2, width=3, encoding="16UC1",
        is_bigendian=1, step=10, data=padded.reshape(-1),
    )  # fmt: skip
    image = from_ros(ros)
    assert image.data.dtype == np.uint16 and image.data.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_camera_info_flattens_row_major():
    info = catalog_example(CameraInfo)
    ros = to_ros(info)
    assert list(ros.k) == [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
    assert list(ros.r) == list(np.eye(3).ravel())
    assert len(ros.p) == 12 and list(ros.p[:4]) == [615.0, 0.0, 320.0, 0.0]
    assert (ros.width, ros.height, ros.binning_x, ros.binning_y) == (640, 480, 0, 0)
    assert (ros.roi.x_offset, ros.roi.width, ros.roi.do_rectify) == (0, 0, False)
    assert ros.distortion_model == "plumb_bob" and len(ros.d) == 0


def test_camera_info_asymmetric_matrices_keep_their_orientation():
    info = CameraInfo(
        width=4, height=3, k=np.arange(9.0).reshape(3, 3), r=np.arange(9.0).reshape(3, 3) + 10,
        p=np.arange(12.0).reshape(3, 4) + 100, d=[0.1, 0.2, 0.3, 0.4, 0.5],
    )  # fmt: skip
    ros = to_ros(info)
    assert list(ros.k) == list(range(9)) and list(ros.p) == list(range(100, 112))
    assert allclose(round_trip(info), info)


def test_point_cloud2_layout_and_packed_rgb():
    cloud = catalog_example(PointCloud)
    ros = to_ros(cloud)
    assert (ros.height, ros.width, ros.is_bigendian) == (1, 100, False)
    assert [(f.name, f.offset, f.datatype, f.count) for f in ros.fields] == [
        ("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("rgb", 12, 7, 1),
    ]  # fmt: skip
    assert ros.point_step == 16 and ros.row_step == 1600 and len(ros.data) == 1600
    assert ros.is_dense is True
    # point 0 as bytes: xyz float32 then the PCL-packed 0x00RRGGBB
    red, green, blue = (int(c) for c in cloud.channels["rgb"][0])
    record = bytes(ros.data[:16])
    assert record[:12] == cloud.points[0].astype("<f4").tobytes()
    assert int.from_bytes(record[12:], "little") == (red << 16) | (green << 8) | blue
    # a pure-red point is the float whose bits are 0x00FF0000
    pure = PointCloud(points=[[0, 0, 0]], channels={"rgb": np.array([[255, 0, 0]], np.uint8)})
    packed = bytes(to_ros(pure).data[12:16])
    assert packed == (0x00FF0000).to_bytes(4, "little")
    assert np.frombuffer(packed, "<f4")[0] == np.float32(
        np.frombuffer(packed, "<u4")[0].view("<f4")
    )


def test_point_cloud_numeric_channels_get_their_own_fields():
    cloud = PointCloud(
        points=np.arange(12, dtype=np.float32).reshape(4, 3),
        channels={
            "intensity": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
            "range": np.array([1.0, 2.0, 3.0, 4.0]),
            "ring": np.array([0, 1, 2, 3], dtype=np.uint8),
            "label": np.array([-1, 5, 7, 9], dtype=np.int32),
            "rgb": np.arange(12, dtype=np.uint8).reshape(4, 3),
        },
        header=HEADER,
    )
    ros = to_ros(cloud)
    layout = [(f.name, f.offset, f.datatype) for f in ros.fields]
    assert layout == [
        ("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("rgb", 12, 7),
        ("intensity", 16, 7), ("range", 20, 8), ("ring", 28, 2), ("label", 29, 5),
    ]  # fmt: skip
    assert ros.point_step == 33 and ros.row_step == 33 * 4
    back = round_trip(cloud)
    assert allclose(back, cloud)
    assert {name: values.dtype for name, values in back.channels.items()} == {
        name: values.dtype for name, values in cloud.channels.items()
    }


def test_point_cloud_with_nan_is_not_dense():
    cloud = PointCloud(points=[[0, 0, 0], [np.nan, 0, 1]])
    assert to_ros(cloud).is_dense is False


def test_empty_point_cloud_round_trips():
    cloud = PointCloud(points=np.zeros((0, 3)), channels={"rgb": np.zeros((0, 3), np.uint8)})
    assert allclose(round_trip(cloud), cloud)


def test_organized_point_cloud_with_row_padding_from_ros():
    store = typestore()
    field = store.types["sensor_msgs/msg/PointField"]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    table = np.zeros((2, 3), dtype=dtype)  # height 2, width 3
    table["x"], table["y"], table["z"] = 1.0, 2.0, 3.0
    table["intensity"] = np.arange(6, dtype=np.float32).reshape(2, 3)
    names = dtype.names
    row_step = 3 * 16 + 8  # 8 bytes of padding per row
    data = np.zeros((2, row_step), dtype=np.uint8)
    data[:, : 3 * 16] = table.view(np.uint8).reshape(2, 48)
    ros = store.types["sensor_msgs/msg/PointCloud2"](
        header=to_ros(Header()), height=2, width=3,
        fields=[field(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(names)],
        is_bigendian=False, point_step=16, row_step=row_step, data=data.reshape(-1), is_dense=True,
    )  # fmt: skip
    cloud = from_ros(ros)
    assert cloud.points.shape == (6, 3) and cloud.points.dtype == np.float32
    assert cloud.channels["intensity"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_point_cloud2_without_xyz_is_rejected():
    store = typestore()
    field = store.types["sensor_msgs/msg/PointField"]
    ros = store.types["sensor_msgs/msg/PointCloud2"](
        header=to_ros(Header()), height=1, width=0,
        fields=[field(name="x", offset=0, datatype=7, count=1)],
        is_bigendian=False, point_step=4, row_step=0, data=np.zeros(0, np.uint8), is_dense=True,
    )  # fmt: skip
    with pytest.raises(ValueError, match="no 'y' field"):
        from_ros(ros)


def test_imu_unknown_covariance_is_all_zeros():
    imu = catalog_example(Imu)
    ros = to_ros(imu)
    assert list(ros.orientation_covariance) == list((np.eye(3) * 1e-3).ravel())
    assert list(ros.angular_velocity_covariance) == [0.0] * 9
    assert list(ros.linear_acceleration_covariance) == [0.0] * 9
    back = round_trip(imu)
    assert back.angular_velocity_covariance is None and back.orientation_covariance.shape == (3, 3)


def test_joint_trajectory_points_carry_duration_and_per_point_arrays():
    traj = JointTrajectory(
        t=[0.0, 0.25, 1.500000001],
        joint_names=("a", "b"),
        positions=[[0, 1], [2, 3], [4, 5]],
        velocities=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        header=Header(stamp=3.5, frame_id="base"),
    )
    ros = to_ros(traj)
    assert list(ros.joint_names) == ["a", "b"]
    assert (ros.header.stamp.sec, ros.header.stamp.nanosec) == (3, 500_000_000)
    durations = [(p.time_from_start.sec, p.time_from_start.nanosec) for p in ros.points]
    assert durations == [(0, 0), (0, 250_000_000), (1, 500_000_001)]
    assert list(ros.points[1].positions) == [2.0, 3.0]
    assert list(ros.points[2].velocities) == [0.5, 0.6]
    assert len(ros.points[0].accelerations) == 0 and len(ros.points[0].effort) == 0
    assert ros.__msgtype__ == "trajectory_msgs/msg/JointTrajectory"


def test_joint_trajectory_all_four_per_point_arrays_round_trip():
    t = np.linspace(0, 1, 5)
    traj = JointTrajectory(
        t=t, joint_names=("a", "b"), positions=np.ones((5, 2)), velocities=np.full((5, 2), 2.0),
        accelerations=np.full((5, 2), 3.0), effort=np.full((5, 2), 4.0),
    )  # fmt: skip
    assert allclose(round_trip(traj), traj)


def test_empty_trajectories_round_trip():
    joints = JointTrajectory(t=[], joint_names=("a",), positions=np.zeros((0, 1)))
    assert allclose(round_trip(joints), joints)
    poses = PoseTrajectory(t=[], positions=np.zeros((0, 3)), orientations=np.zeros((0, 4)))
    assert allclose(round_trip(poses), poses)
    assert allclose(round_trip(poses, as_="multidof"), poses.replace(child_frame_id=""))


def test_joint_trajectory_with_ragged_points_from_ros_is_rejected():
    ros = to_ros(JOINT_TRAJ, drop_extras=True)
    ros.points[3].velocities = np.zeros(0)
    with pytest.raises(ValueError, match="velocities of sizes \\[0, 3\\]"):
        from_ros(ros)


def test_path_stamps_and_frames():
    traj = POSE_TRAJ.replace(header=Header(stamp=100.0, frame_id="base_link"))
    ros = to_ros(traj, drop_extras=True)
    assert ros.__msgtype__ == "nav_msgs/msg/Path"
    assert ros.header.frame_id == "base_link" and ros.header.stamp.sec == 100
    assert [(p.header.stamp.sec, p.header.stamp.nanosec) for p in ros.poses] == [
        (100, 0), (100, 500_000_000), (101, 0),
    ]  # fmt: skip
    assert {p.header.frame_id for p in ros.poses} == {"base_link"}
    assert ros.poses[1].pose.position.x == pytest.approx(traj.positions[1, 0])
    back = from_ros(over_the_wire(ros))
    assert back.header == Header(stamp=100.0, frame_id="base_link") and back.child_frame_id == ""
    assert back.t.tolist() == [0.0, 0.5, 1.0]


def test_path_makes_t_relative_to_its_first_stamp():
    traj = POSE_TRAJ.replace(t=POSE_TRAJ.t + 2.0, header=Header(stamp=10.0, frame_id="odom"))
    back = round_trip(traj, drop_extras=True)
    assert back.t.tolist() == [0.0, 0.5, 1.0]
    assert back.header == Header(stamp=12.0, frame_id="odom")


def test_multidof_layout():
    ros = to_ros(POSE_TRAJ, as_="multidof")
    assert ros.__msgtype__ == "trajectory_msgs/msg/MultiDOFJointTrajectory"
    assert list(ros.joint_names) == ["tool0"] and len(ros.points) == 3
    point = ros.points[1]
    assert len(point.transforms) == 1 and len(point.velocities) == 1
    assert (point.time_from_start.sec, point.time_from_start.nanosec) == (0, 500_000_000)
    assert point.transforms[0].translation.x == pytest.approx(POSE_TRAJ.positions[1, 0])
    assert point.transforms[0].rotation.w == pytest.approx(POSE_TRAJ.orientations[1, 3])
    assert point.velocities[0].linear.x == pytest.approx(POSE_TRAJ.linear_velocities[1, 0])
    assert point.velocities[0].angular.z == pytest.approx(POSE_TRAJ.angular_velocities[1, 2])
    without_velocity = POSE_TRAJ.replace(linear_velocities=None, angular_velocities=None)
    assert all(len(p.velocities) == 0 for p in to_ros(without_velocity, as_="multidof").points)


# ── error paths ──────────────────────────────────────────────────────────────


def test_joint_trajectory_channels_without_drop_extras():
    with pytest.raises(ValueError, match=r"channels \['kp', 'mode'\] and limits"):
        to_ros(JOINT_TRAJ)
    with pytest.raises(ValueError, match="channels"):
        to_ros(JOINT_TRAJ.replace(limits=None))
    with pytest.raises(ValueError, match="limits"):
        to_ros(JOINT_TRAJ.replace(channels={}))
    plain = JOINT_TRAJ.replace(channels={}, limits=None)
    assert to_ros(plain).__msgtype__ == "trajectory_msgs/msg/JointTrajectory"
    assert to_ros(JOINT_TRAJ, drop_extras=True).__msgtype__ == "trajectory_msgs/msg/JointTrajectory"


def test_pose_trajectory_loss_is_reported_not_silent():
    with pytest.raises(ValueError, match="velocities"):
        to_ros(POSE_TRAJ)
    with pytest.raises(ValueError, match="channels"):
        to_ros(POSE_TRAJ.with_channel("temperature", [1.0, 2.0, 3.0]), as_="multidof")
    bare = POSE_TRAJ.replace(linear_velocities=None, angular_velocities=None)
    assert to_ros(bare).__msgtype__ == "nav_msgs/msg/Path"


def test_as_applies_only_to_pose_trajectories_and_knows_two_values():
    with pytest.raises(ValueError, match="only applies to PoseTrajectory"):
        to_ros(POSE, as_="multidof")
    with pytest.raises(ValueError, match="'path' or 'multidof'"):
        to_ros(POSE_TRAJ, as_="tf", drop_extras=True)
    assert to_ros(POSE_TRAJ, as_="path", drop_extras=True).__msgtype__ == "nav_msgs/msg/Path"


def test_bare_trajectory_has_no_ros_message():
    with pytest.raises(TypeError, match="no standard ROS message for Trajectory"):
        to_ros(Trajectory(t=[0.0, 1.0]))
    with pytest.raises(TypeError, match="no standard ROS message for JointLimits"):
        to_ros(JointLimits(joint_names=("a",), max_velocity=[1.0]))


def test_point_cloud_channels_without_a_field_are_reported():
    cloud = PointCloud(
        points=np.zeros((2, 3)),
        channels={
            "normals": np.zeros((2, 3)),
            "id": np.array([1, 2], dtype=np.int64),
            "ring": np.array([1, 2], np.uint8),
        },
    )
    with pytest.raises(ValueError, match=r"'normals' \(float64 \(2, 3\)\), 'id' \(int64 \(2,\)\)"):
        to_ros(cloud)
    dropped = round_trip(cloud, drop_extras=True)
    assert set(dropped.channels) == {"ring"}
    assert allclose(dropped, cloud.replace(channels={"ring": cloud.channels["ring"]}))


def test_point_cloud_rgb_must_be_n_by_3_uint8():
    cloud = PointCloud(points=np.zeros((2, 3)), channels={"rgb": np.zeros(2, np.float32)})
    with pytest.raises(ValueError, match="'rgb'"):
        to_ros(cloud)
    assert set(round_trip(cloud, drop_extras=True).channels) == set()


def test_point_cloud_channel_named_like_a_coordinate_is_reported():
    cloud = PointCloud(points=np.zeros((2, 3)), channels={"x": np.zeros(2)})
    with pytest.raises(ValueError, match="clashes with the coordinate"):
        to_ros(cloud)


# ── vision_msgs ──────────────────────────────────────────────────────────────


def test_vision_msgs_layout():
    detection = catalog_example(Detection2DArray).detections[0]
    ros = to_ros(detection)
    assert ros.__msgtype__ == "vision_msgs/msg/Detection2D"
    assert ros.bbox.center.position.x == 60.0 and ros.bbox.center.position.y == 120.0
    assert ros.bbox.center.theta == 0.0 and (ros.bbox.size_x, ros.bbox.size_y) == (100.0, 200.0)
    result = ros.results[0]
    assert (result.hypothesis.class_id, result.hypothesis.score) == ("bottle", 0.91)
    assert result.pose.pose.position.x == pytest.approx(0.4)
    assert len(result.pose.covariance) == 36 and not result.pose.covariance.any()
    assert ros.id == "7" and ros.header.frame_id == HEADER.frame_id


def test_hypothesis_covariance_is_flattened_row_major_and_zeros_mean_none():
    covariance = np.arange(36.0).reshape(6, 6)
    hypothesis = ObjectHypothesisWithPose(
        hypothesis=catalog_example(ObjectHypothesisWithPose).hypothesis,
        pose=POSE,
        covariance=covariance,
    )
    ros = to_ros(hypothesis)
    assert list(ros.pose.covariance) == list(range(36))
    assert allclose(round_trip(hypothesis), hypothesis)
    assert round_trip(hypothesis.replace(covariance=None)).covariance is None


def test_detection_array_keeps_results_and_order():
    array = catalog_example(Detection2DArray)
    back = round_trip(array)
    assert [d.id for d in back.detections] == ["7", "8"]
    assert allclose(back, array)
    assert allclose(round_trip(Detection2DArray(header=HEADER)), Detection2DArray(header=HEADER))


# ── bags ─────────────────────────────────────────────────────────────────────


def build_stream():
    states = [
        (
            "/joint_states",
            JointState(
                name=("j1", "j2"), position=[0.1 * i, -0.1 * i], velocity=[1.0, 2.0],
                header=Header(stamp=100.0 + 0.1 * i, frame_id="base"),
            ),
        )
        for i in range(5)
    ]  # fmt: skip
    trajectory = JOINT_TRAJ.replace(
        channels={}, limits=None, header=Header(stamp=100.25, frame_id="base")
    )
    image = catalog_example(Image).replace(header=Header(stamp=100.3, frame_id="cam"))
    detections = catalog_example(Detection2DArray).replace(
        header=Header(stamp=100.35, frame_id="cam")
    )
    messages = [
        *states,
        ("/plan", trajectory),
        ("/camera/image", image),
        ("/detections", detections),
    ]
    return sorted(messages, key=lambda item: item[1].header.stamp)


@pytest.mark.parametrize("storage", ["mcap", "sqlite3"])
def test_bag_round_trip(tmp_path, storage):
    stream = build_stream()
    path = tmp_path / "bag"
    write_bag(path, stream, storage=storage)

    with Reader(path) as reader:
        types = {c.topic: c.msgtype for c in reader.connections}
        counts = {c.topic: c.msgcount for c in reader.connections}
        times = [timestamp for _, timestamp, _ in reader.messages()]
    assert types == {
        "/joint_states": "sensor_msgs/msg/JointState",
        "/plan": "trajectory_msgs/msg/JointTrajectory",
        "/camera/image": "sensor_msgs/msg/Image",
        "/detections": "vision_msgs/msg/Detection2DArray",
    }
    assert counts == {"/joint_states": 5, "/plan": 1, "/camera/image": 1, "/detections": 1}
    assert times == sorted(times) and times[0] == 100_000_000_000  # log time = header stamp

    back = list(read_bag(path))
    assert [topic for topic, _ in back] == [topic for topic, _ in stream]
    for (_, original), (_, restored) in zip(stream, back, strict=True):
        assert allclose(restored, original)


def test_mcap_bag_is_standard_ros2(tmp_path):
    path = tmp_path / "bag"
    write_bag(path, build_stream())
    files = sorted(p.name for p in path.iterdir())
    assert "metadata.yaml" in files and any(name.endswith(".mcap") for name in files)
    mcap = next(p for p in path.iterdir() if p.suffix == ".mcap").read_bytes()
    assert mcap.startswith(b"\x89MCAP") and b"ros2msg" in mcap and b"cdr" in mcap
    assert b"vision_msgs/msg/Detection2DArray" in mcap
    assert "storage_identifier: mcap" in (path / "metadata.yaml").read_text()


def test_bag_log_time_is_a_running_counter_without_a_header(tmp_path):
    path = tmp_path / "bag"
    write_bag(
        path,
        [("/pose", POSE), ("/pose", Pose.identity()), ("/twist", Twist(Vector3(1, 0, 0))),
         ("/stamped", PoseStamped(Header(stamp=7.0), POSE))],
    )  # fmt: skip
    with Reader(path) as reader:
        logged = {(c.topic, t) for c, t, _ in reader.messages()}
    assert logged == {("/pose", 0), ("/pose", 1), ("/twist", 2), ("/stamped", 7_000_000_000)}


def test_bag_topic_keeps_one_type(tmp_path):
    with pytest.raises(ValueError, match="topic '/x' carries geometry_msgs/msg/Pose"):
        write_bag(tmp_path / "bag", [("/x", POSE), ("/x", Twist())])
    with pytest.raises(ValueError, match="storage"):
        write_bag(tmp_path / "bag2", [], storage="rosbag1")


def test_bag_accepts_ready_made_ros_messages(tmp_path):
    path = tmp_path / "bag"
    multidof = to_ros(POSE_TRAJ, as_="multidof")
    write_bag(path, [("/tool_path", multidof)])
    ((topic, back),) = read_bag(path)
    assert topic == "/tool_path" and allclose(back, POSE_TRAJ)


def test_read_bag_names_the_topic_it_cannot_convert(tmp_path):
    path = tmp_path / "bag"
    write_bag(path, [("/chatter", typestore().types["std_msgs/msg/String"](data="hi"))])
    with pytest.raises(
        TypeError, match="topic '/chatter': no robot_msgs type for std_msgs/msg/String"
    ):
        list(read_bag(path))


# ── import contracts ─────────────────────────────────────────────────────────


def run_python(code):
    source = str(
        Path(__file__).resolve().parents[1] / "src"
    )  # a fresh interpreter, not pytest's path
    env = {**os.environ, "PYTHONPATH": source}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_importing_robot_msgs_does_not_import_rosbags():
    result = run_python(
        "import sys, robot_msgs\n"
        "import robot_msgs.geometry, robot_msgs.sensor, robot_msgs.trajectory, robot_msgs.vision\n"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'rosbags'], 'rosbags imported'\n"
    )
    assert result.returncode == 0, result.stderr


def test_ros_subpackage_explains_a_missing_rosbags():
    result = run_python(
        "import sys\n"
        "sys.modules['rosbags'] = None  # makes 'import rosbags' raise ImportError\n"
        "try:\n"
        "    import robot_msgs.ros\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    assert result.stdout.strip() == "robot_msgs.ros needs rosbags: pip install 'robot-msgs[ros]'"


def catalog_example(cls):
    for message in EXAMPLES:
        if type(message) is cls:
            return message
    raise LookupError(cls)


# ── shape_msgs and ColorRGBA ─────────────────────────────────────────────────


def test_solid_primitive_constants_and_indices_are_the_ros_ones():
    ros = typestore().types["shape_msgs/msg/SolidPrimitive"]
    for name in (
        "BOX", "SPHERE", "CYLINDER", "CONE", "PRISM",
        "BOX_X", "BOX_Y", "BOX_Z", "SPHERE_RADIUS", "CYLINDER_HEIGHT", "CYLINDER_RADIUS",
        "CONE_HEIGHT", "CONE_RADIUS", "PRISM_HEIGHT",
    ):  # fmt: skip
        assert getattr(SolidPrimitive, name) == getattr(ros, name), name


def test_solid_primitive_fields():
    cylinder = to_ros(SolidPrimitive.cylinder(height=0.6, radius=0.15))
    assert cylinder.type == 3 and list(cylinder.dimensions) == [0.6, 0.15]
    assert list(cylinder.polygon.points) == []
    prism = to_ros(SolidPrimitive.prism(0.4, [(0, 0), (2, 0), (1, 1)]))
    assert prism.type == 5 and list(prism.dimensions) == [0.4]
    assert [(p.x, p.y, p.z) for p in prism.polygon.points] == [(0, 0, 0), (2, 0, 0), (1, 1, 0)]
    back = from_ros(over_the_wire(prism))
    assert back.polygon.tolist() == [[0, 0], [2, 0], [1, 1]] and back.height == 0.4


def test_mesh_fields_are_triangles_of_vertex_indices_and_points():
    mesh = SolidPrimitive.box(1.0, 2.0, 3.0).to_mesh()
    ros = to_ros(mesh)
    assert len(ros.vertices) == 8 and len(ros.triangles) == 12
    assert [list(t.vertex_indices) for t in ros.triangles] == mesh.triangles.tolist()
    assert [(p.x, p.y, p.z) for p in ros.vertices] == [tuple(v) for v in mesh.vertices.tolist()]
    assert round_trip(mesh) == mesh


def test_plane_coef_and_colour_channels():
    assert list(to_ros(Plane([0.0, 0.0, 1.0, -0.5])).coef) == [0.0, 0.0, 1.0, -0.5]
    ros = to_ros(ColorRGBA(0.5, 0.25, 0.75, 0.125))
    assert (ros.r, ros.g, ros.b, ros.a) == (0.5, 0.25, 0.75, 0.125)
    # std_msgs/ColorRGBA is float32: 1/255 steps survive to the nearest float32
    color = ColorRGBA.from_hex("#336699")
    assert round_trip(color).to_hex() == "#336699"


def test_an_empty_ros_mesh_and_a_prism_without_polygon_points():
    store = typestore()
    empty = store.types["shape_msgs/msg/Mesh"](triangles=[], vertices=[])
    assert len(from_ros(empty)) == 0
    bare = store.types["shape_msgs/msg/SolidPrimitive"](
        type=5,
        dimensions=np.array([1.0]),
        polygon=store.types["geometry_msgs/msg/Polygon"](points=[]),
    )
    with pytest.raises(ValueError, match="PRISM needs a polygon"):
        from_ros(bare)
