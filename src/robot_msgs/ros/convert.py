"""Conversion between robot_msgs values and rosbags' ROS 2 (Jazzy) message dataclasses.

The mapping is field for field wherever ROS has the field (names are verbatim). Where ROS has
no place for something robot_msgs holds (`channels`, `limits`, velocities on a `Path`), the
conversion raises instead of dropping it silently, unless the caller says `drop_extras=True`.
`None` in robot_msgs is an empty array in ROS; fixed-size covariances use all zeros, the
convention `sensor_msgs/Imu` documents for "unknown".
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from robot_msgs._core import Msg
from robot_msgs.geometry import (
    Accel,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
    Vector3,
    Wrench,
)
from robot_msgs.ros.vision_msgs import definitions as vision_msgs_definitions
from robot_msgs.sensor import CameraInfo, Image, Imu, JointState, PointCloud
from robot_msgs.sensor.image import IMAGE_ENCODINGS
from robot_msgs.std import Header
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory
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

_NS = 1_000_000_000


@functools.cache
def typestore() -> Any:
    """The ROS 2 Jazzy typestore with vision_msgs registered (built once, then shared)."""
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    store.register(vision_msgs_definitions())
    return store


def _ros(name: str) -> Any:
    return typestore().types[name]


# ── registry ─────────────────────────────────────────────────────────────────

_TO_ROS: dict[type, tuple[str, Callable[..., Any], bool]] = {}
_FROM_ROS: dict[str, Callable[[Any], Msg]] = {}


def _to(cls: type, ros_name: str, *, options: bool = False) -> Callable:
    """Register a robot_msgs -> ROS converter. `options` converters also take the
    `drop_extras` and `as_` keywords of `to_ros`."""

    def register(function: Callable) -> Callable:
        _TO_ROS[cls] = (ros_name, function, options)
        return function

    return register


def _from(*ros_names: str) -> Callable:
    def register(function: Callable) -> Callable:
        for ros_name in ros_names:
            _FROM_ROS[ros_name] = function
        return function

    return register


def _entry(msg_or_cls: Any) -> tuple[str, Callable[..., Any], bool]:
    cls = msg_or_cls if isinstance(msg_or_cls, type) else type(msg_or_cls)
    for base in cls.__mro__:  # WHY the MRO: a subclass of a mapped type converts as its parent
        if base in _TO_ROS:
            return _TO_ROS[base]
    raise TypeError(f"no standard ROS message for {cls.__name__}")


def ros_type(msg_or_cls: Any) -> str:
    """The ROS type name of a robot_msgs value or class, e.g. ``"sensor_msgs/msg/JointState"``.

    A `PoseTrajectory` is a ``nav_msgs/msg/Path`` (the default of `to_ros`).
    """
    return _entry(msg_or_cls)[0]


def to_ros(msg: Msg, *, drop_extras: bool = False, as_: str | None = None) -> Any:
    """A robot_msgs value as the rosbags dataclass of its ROS 2 message.

    `drop_extras` lets a conversion discard what ROS cannot hold (see each type) instead of
    raising `ValueError`. `as_="multidof"` converts a `PoseTrajectory` to
    ``trajectory_msgs/msg/MultiDOFJointTrajectory`` instead of ``nav_msgs/msg/Path``.
    """
    _, convert, options = _entry(msg)
    if as_ is not None and not isinstance(msg, PoseTrajectory):
        raise ValueError(
            f"to_ros(as_={as_!r}) only applies to PoseTrajectory, got {type(msg).__name__}"
        )
    if options:
        return convert(msg, drop_extras=drop_extras, as_=as_)
    return convert(msg)


def from_ros(ros_msg: Any) -> Msg:
    """A rosbags message dataclass as a robot_msgs value, dispatched on its ``__msgtype__``."""
    msgtype = getattr(ros_msg, "__msgtype__", None)
    if msgtype is None:
        raise TypeError(f"{type(ros_msg).__name__} is not a rosbags message")
    try:
        convert = _FROM_ROS[msgtype]
    except KeyError:
        raise TypeError(f"no robot_msgs type for ROS message {msgtype}") from None
    return convert(ros_msg)


# ── std ──────────────────────────────────────────────────────────────────────


def _time(stamp_ns: int, name: str = "builtin_interfaces/msg/Time") -> Any:
    sec, nanosec = divmod(stamp_ns, _NS)
    return _ros(name)(sec=sec, nanosec=nanosec)


def _stamp_ns(time: Any) -> int:
    return time.sec * _NS + time.nanosec


def _header_ns(stamp_ns: int, frame_id: str) -> Any:
    return _ros("std_msgs/msg/Header")(stamp=_time(stamp_ns), frame_id=frame_id)


@_to(Header, "std_msgs/msg/Header")
def _header_to_ros(header: Header) -> Any:
    # WHY `meta` is dropped here, always and without drop_extras: std_msgs/Header has only stamp
    # and frame_id, and every stamped message exports through this one function. ROS consumers
    # never had the context meta carries; robot_msgs files (to_dict, npz) keep it.
    sec, nanosec = header.to_sec_nanosec()
    return _ros("std_msgs/msg/Header")(
        stamp=_ros("builtin_interfaces/msg/Time")(sec=sec, nanosec=nanosec),
        frame_id=header.frame_id,
    )


@_from("std_msgs/msg/Header")
def _header_from_ros(ros: Any) -> Header:
    return Header.from_sec_nanosec(ros.stamp.sec, ros.stamp.nanosec, ros.frame_id)


# ── geometry ─────────────────────────────────────────────────────────────────


@_to(Vector3, "geometry_msgs/msg/Vector3")
def _vector3_to_ros(v: Vector3) -> Any:
    return _ros("geometry_msgs/msg/Vector3")(x=v.x, y=v.y, z=v.z)


@_from("geometry_msgs/msg/Vector3")
def _vector3_from_ros(ros: Any) -> Vector3:
    return Vector3(float(ros.x), float(ros.y), float(ros.z))


@_to(Point, "geometry_msgs/msg/Point")
def _point_to_ros(p: Point) -> Any:
    return _ros("geometry_msgs/msg/Point")(x=p.x, y=p.y, z=p.z)


@_from("geometry_msgs/msg/Point")
def _point_from_ros(ros: Any) -> Point:
    return Point(float(ros.x), float(ros.y), float(ros.z))


@_to(Quaternion, "geometry_msgs/msg/Quaternion")
def _quaternion_to_ros(q: Quaternion) -> Any:
    return _ros("geometry_msgs/msg/Quaternion")(x=q.x, y=q.y, z=q.z, w=q.w)


@_from("geometry_msgs/msg/Quaternion")
def _quaternion_from_ros(ros: Any) -> Quaternion:
    return Quaternion(float(ros.x), float(ros.y), float(ros.z), float(ros.w))


@_to(Pose, "geometry_msgs/msg/Pose")
def _pose_to_ros(pose: Pose) -> Any:
    return _ros("geometry_msgs/msg/Pose")(
        position=_point_to_ros(pose.position), orientation=_quaternion_to_ros(pose.orientation)
    )


@_from("geometry_msgs/msg/Pose")
def _pose_from_ros(ros: Any) -> Pose:
    return Pose(_point_from_ros(ros.position), _quaternion_from_ros(ros.orientation))


@_to(Pose2D, "geometry_msgs/msg/Pose2D")
def _pose2d_to_ros(p: Pose2D) -> Any:
    return _ros("geometry_msgs/msg/Pose2D")(x=p.x, y=p.y, theta=p.theta)


@_from("geometry_msgs/msg/Pose2D")
def _pose2d_from_ros(ros: Any) -> Pose2D:
    return Pose2D(float(ros.x), float(ros.y), float(ros.theta))


@_to(Transform, "geometry_msgs/msg/Transform")
def _transform_to_ros(t: Transform) -> Any:
    return _ros("geometry_msgs/msg/Transform")(
        translation=_vector3_to_ros(t.translation), rotation=_quaternion_to_ros(t.rotation)
    )


@_from("geometry_msgs/msg/Transform")
def _transform_from_ros(ros: Any) -> Transform:
    return Transform(_vector3_from_ros(ros.translation), _quaternion_from_ros(ros.rotation))


@_to(Twist, "geometry_msgs/msg/Twist")
def _twist_to_ros(t: Twist) -> Any:
    return _ros("geometry_msgs/msg/Twist")(
        linear=_vector3_to_ros(t.linear), angular=_vector3_to_ros(t.angular)
    )


@_from("geometry_msgs/msg/Twist")
def _twist_from_ros(ros: Any) -> Twist:
    return Twist(_vector3_from_ros(ros.linear), _vector3_from_ros(ros.angular))


@_to(Wrench, "geometry_msgs/msg/Wrench")
def _wrench_to_ros(w: Wrench) -> Any:
    return _ros("geometry_msgs/msg/Wrench")(
        force=_vector3_to_ros(w.force), torque=_vector3_to_ros(w.torque)
    )


@_from("geometry_msgs/msg/Wrench")
def _wrench_from_ros(ros: Any) -> Wrench:
    return Wrench(_vector3_from_ros(ros.force), _vector3_from_ros(ros.torque))


@_to(Accel, "geometry_msgs/msg/Accel")
def _accel_to_ros(a: Accel) -> Any:
    return _ros("geometry_msgs/msg/Accel")(
        linear=_vector3_to_ros(a.linear), angular=_vector3_to_ros(a.angular)
    )


@_from("geometry_msgs/msg/Accel")
def _accel_from_ros(ros: Any) -> Accel:
    return Accel(_vector3_from_ros(ros.linear), _vector3_from_ros(ros.angular))


@_to(PoseStamped, "geometry_msgs/msg/PoseStamped")
def _pose_stamped_to_ros(p: PoseStamped) -> Any:
    return _ros("geometry_msgs/msg/PoseStamped")(
        header=_header_to_ros(p.header), pose=_pose_to_ros(p.pose)
    )


@_from("geometry_msgs/msg/PoseStamped")
def _pose_stamped_from_ros(ros: Any) -> PoseStamped:
    return PoseStamped(_header_from_ros(ros.header), _pose_from_ros(ros.pose))


@_to(TransformStamped, "geometry_msgs/msg/TransformStamped")
def _transform_stamped_to_ros(t: TransformStamped) -> Any:
    return _ros("geometry_msgs/msg/TransformStamped")(
        header=_header_to_ros(t.header),
        child_frame_id=t.child_frame_id,
        transform=_transform_to_ros(t.transform),
    )


@_from("geometry_msgs/msg/TransformStamped")
def _transform_stamped_from_ros(ros: Any) -> TransformStamped:
    return TransformStamped(
        _header_from_ros(ros.header), str(ros.child_frame_id), _transform_from_ros(ros.transform)
    )


# ── sensor ───────────────────────────────────────────────────────────────────


def _floats(values: Any) -> np.ndarray:
    return np.array(values, dtype=np.float64).reshape(-1)


def _optional(values: Any) -> np.ndarray | None:
    """An empty ROS array is an absent field."""
    array = _floats(values)
    return array if array.size else None


@_to(JointState, "sensor_msgs/msg/JointState")
def _joint_state_to_ros(state: JointState) -> Any:
    return _ros("sensor_msgs/msg/JointState")(
        header=_header_to_ros(state.header),
        name=list(state.name),
        position=_floats(state.position),
        velocity=_floats([] if state.velocity is None else state.velocity),
        effort=_floats([] if state.effort is None else state.effort),
    )


@_from("sensor_msgs/msg/JointState")
def _joint_state_from_ros(ros: Any) -> JointState:
    return JointState(
        name=tuple(ros.name),
        position=_floats(ros.position),
        velocity=_optional(ros.velocity),
        effort=_optional(ros.effort),
        header=_header_from_ros(ros.header),
    )


@_to(Image, "sensor_msgs/msg/Image")
def _image_to_ros(image: Image) -> Any:
    # WHY '<': ROS says is_bigendian=0 for little-endian pixels whatever the host is.
    data = np.ascontiguousarray(image.data, dtype=image.data.dtype.newbyteorder("<"))
    return _ros("sensor_msgs/msg/Image")(
        header=_header_to_ros(image.header),
        height=image.height,
        width=image.width,
        encoding=image.encoding,
        is_bigendian=0,
        step=image.step,
        data=np.frombuffer(data.tobytes(), dtype=np.uint8),
    )


@_from("sensor_msgs/msg/Image")
def _image_from_ros(ros: Any) -> Image:
    if ros.encoding not in IMAGE_ENCODINGS:
        raise ValueError(
            f"Image.encoding {ros.encoding!r} is not supported; "
            f"use one of {sorted(IMAGE_ENCODINGS)}"
        )
    dtype, channels = IMAGE_ENCODINGS[ros.encoding]
    height, width, step = int(ros.height), int(ros.width), int(ros.step)
    row_bytes = width * channels * dtype.itemsize
    if step < row_bytes or len(ros.data) < height * step:
        raise ValueError(
            f"Image {width}x{height} {ros.encoding} needs step >= {row_bytes} and "
            f"{height * step} data bytes, got step {step} and {len(ros.data)} bytes"
        )
    rows = np.frombuffer(ros.data, dtype=np.uint8)[: height * step].reshape(height, step)
    packed = np.ascontiguousarray(rows[:, :row_bytes])  # drops any row padding
    stored = dtype.newbyteorder(">" if ros.is_bigendian else "<")
    array = packed.view(stored).astype(dtype)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return Image(
        data=array.reshape(shape), encoding=ros.encoding, header=_header_from_ros(ros.header)
    )


@_to(CameraInfo, "sensor_msgs/msg/CameraInfo")
def _camera_info_to_ros(info: CameraInfo) -> Any:
    roi = _ros("sensor_msgs/msg/RegionOfInterest")(
        x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
    )
    return _ros("sensor_msgs/msg/CameraInfo")(
        header=_header_to_ros(info.header),
        height=info.height,
        width=info.width,
        distortion_model=info.distortion_model,
        d=_floats(info.d),
        k=_floats(info.k),
        r=_floats(info.r),
        p=_floats(info.p),
        binning_x=0,
        binning_y=0,
        roi=roi,
    )


@_from("sensor_msgs/msg/CameraInfo")
def _camera_info_from_ros(ros: Any) -> CameraInfo:
    return CameraInfo(
        width=ros.width,
        height=ros.height,
        k=_floats(ros.k).reshape(3, 3),
        d=_floats(ros.d),
        distortion_model=ros.distortion_model,
        r=_floats(ros.r).reshape(3, 3),
        p=_floats(ros.p).reshape(3, 4),
        header=_header_from_ros(ros.header),
    )


# PointField.datatype codes from sensor_msgs/PointField.msg (cross-checked in the tests).
_POINT_FIELD_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}
_POINT_FIELD_CODES = {dtype: code for code, dtype in _POINT_FIELD_DTYPES.items()}


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 -> (N,) float32 whose bits are 0x00RRGGBB, the PCL convention."""
    wide = rgb.astype(np.uint32)
    return ((wide[:, 0] << 16) | (wide[:, 1] << 8) | wide[:, 2]).view(np.float32)


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(packed).view(np.uint32)
    return np.stack([(bits >> 16) & 255, (bits >> 8) & 255, bits & 255], axis=1).astype(np.uint8)


@_to(PointCloud, "sensor_msgs/msg/PointCloud2", options=True)
def _point_cloud_to_ros(
    cloud: PointCloud, *, drop_extras: bool = False, as_: str | None = None
) -> Any:
    count = len(cloud)
    columns: list[tuple[str, np.ndarray]] = [
        ("x", cloud.points[:, 0]),
        ("y", cloud.points[:, 1]),
        ("z", cloud.points[:, 2]),
    ]
    unsupported: list[str] = []
    for name, values in cloud.channels.items():  # rgb is placed right after xyz (PCL's layout)
        if name in ("x", "y", "z"):
            unsupported.append(f"{name!r} (clashes with the coordinate field)")
        elif name == "rgb":
            if values.dtype == np.uint8 and values.shape == (count, 3):
                columns.insert(3, ("rgb", _pack_rgb(values)))
            else:
                unsupported.append(
                    f"'rgb' (needs shape ({count}, 3) uint8, got {values.dtype} {values.shape})"
                )
        elif values.ndim == 1 and values.dtype in _POINT_FIELD_CODES:
            columns.append((name, values))
        else:
            unsupported.append(f"{name!r} ({values.dtype} {values.shape})")
    if unsupported and not drop_extras:
        raise ValueError(
            f"PointCloud.channels {', '.join(unsupported)} have no PointCloud2 field "
            "(rgb as (N, 3) uint8 and numeric (N,) channels do); pass drop_extras=True to drop them"
        )
    offsets, offset = [], 0
    for _, values in columns:
        offsets.append(offset)
        offset += values.dtype.itemsize
    point_step = offset
    layout = np.dtype(
        {
            "names": [name for name, _ in columns],
            "formats": [values.dtype.newbyteorder("<") for _, values in columns],
            "offsets": offsets,
            "itemsize": point_step,
        }
    )
    table = np.zeros(count, dtype=layout)
    for name, values in columns:
        table[name] = values
    field_type = _ros("sensor_msgs/msg/PointField")
    fields = [
        field_type(name=name, offset=at, datatype=_POINT_FIELD_CODES[values.dtype], count=1)
        for (name, values), at in zip(columns, offsets, strict=True)
    ]
    return _ros("sensor_msgs/msg/PointCloud2")(
        header=_header_to_ros(cloud.header),
        height=1,
        width=count,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * count,
        data=np.frombuffer(table.tobytes(), dtype=np.uint8),
        is_dense=bool(np.all(np.isfinite(cloud.points))),
    )


@_from("sensor_msgs/msg/PointCloud2")
def _point_cloud_from_ros(ros: Any) -> PointCloud:
    order = ">" if ros.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for point_field in ros.fields:
        if point_field.datatype not in _POINT_FIELD_DTYPES:
            raise ValueError(
                f"PointCloud2 field {point_field.name!r} has unknown datatype "
                f"{point_field.datatype}"
            )
        if point_field.count == 0:
            continue
        base = _POINT_FIELD_DTYPES[point_field.datatype].newbyteorder(order)
        names.append(point_field.name)
        formats.append(base if point_field.count == 1 else (base, (point_field.count,)))
        offsets.append(point_field.offset)
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PointCloud2 has no {axis!r} field, so it has no points")
    width, height = int(ros.width), int(ros.height)
    point_step, row_step = int(ros.point_step), int(ros.row_step)
    count = width * height
    layout = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    if count:
        if row_step < width * point_step or len(ros.data) < height * row_step:
            raise ValueError(
                f"PointCloud2 {width}x{height} with point_step {point_step} needs row_step >= "
                f"{width * point_step} and {height * row_step} data bytes, "
                f"got row_step {row_step} and {len(ros.data)} bytes"
            )
        rows = np.frombuffer(ros.data, dtype=np.uint8)[: height * row_step]
        rows = rows.reshape(height, row_step)[:, : width * point_step]
        table = np.ascontiguousarray(rows).reshape(-1).view(layout)
    else:
        table = np.zeros(0, dtype=layout)

    def column(name: str) -> np.ndarray:
        values = table[name]
        return values.astype(values.dtype.newbyteorder("="))

    points = np.stack([column(axis) for axis in ("x", "y", "z")], axis=1).astype(np.float32)
    channels: dict[str, np.ndarray] = {}
    for name in names:
        if name in ("x", "y", "z"):
            continue
        values = column(name)
        packed = name == "rgb" and values.ndim == 1 and values.dtype in (np.float32, np.uint32)
        channels[name] = _unpack_rgb(values) if packed else values
    return PointCloud(points=points, channels=channels, header=_header_from_ros(ros.header))


def _covariance_to_ros(covariance: np.ndarray | None) -> np.ndarray:
    return np.zeros(9) if covariance is None else _floats(covariance)


def _covariance_from_ros(values: Any) -> np.ndarray | None:
    array = _floats(values).reshape(3, 3)
    return array if np.any(array) else None  # all zeros is ROS's "unknown"


@_to(Imu, "sensor_msgs/msg/Imu")
def _imu_to_ros(imu: Imu) -> Any:
    return _ros("sensor_msgs/msg/Imu")(
        header=_header_to_ros(imu.header),
        orientation=_quaternion_to_ros(imu.orientation),
        orientation_covariance=_covariance_to_ros(imu.orientation_covariance),
        angular_velocity=_vector3_to_ros(imu.angular_velocity),
        angular_velocity_covariance=_covariance_to_ros(imu.angular_velocity_covariance),
        linear_acceleration=_vector3_to_ros(imu.linear_acceleration),
        linear_acceleration_covariance=_covariance_to_ros(imu.linear_acceleration_covariance),
    )


@_from("sensor_msgs/msg/Imu")
def _imu_from_ros(ros: Any) -> Imu:
    return Imu(
        orientation=_quaternion_from_ros(ros.orientation),
        angular_velocity=_vector3_from_ros(ros.angular_velocity),
        linear_acceleration=_vector3_from_ros(ros.linear_acceleration),
        orientation_covariance=_covariance_from_ros(ros.orientation_covariance),
        angular_velocity_covariance=_covariance_from_ros(ros.angular_velocity_covariance),
        linear_acceleration_covariance=_covariance_from_ros(ros.linear_acceleration_covariance),
        header=_header_from_ros(ros.header),
    )


# ── trajectory ───────────────────────────────────────────────────────────────


def _duration(seconds: float) -> Any:
    return _time(round(float(seconds) * _NS), "builtin_interfaces/msg/Duration")


def _duration_seconds(duration: Any) -> float:
    return _stamp_ns(duration) / _NS


@_to(JointTrajectory, "trajectory_msgs/msg/JointTrajectory", options=True)
def _joint_trajectory_to_ros(
    traj: JointTrajectory, *, drop_extras: bool = False, as_: str | None = None
) -> Any:
    lost = []
    if traj.channels:
        lost.append(f"channels {list(traj.channels)}")
    if traj.limits is not None:
        lost.append("limits")
    if lost and not drop_extras:
        raise ValueError(
            f"JointTrajectory {' and '.join(lost)} have no field in "
            "trajectory_msgs/JointTrajectory; pass drop_extras=True to drop them"
        )

    def row(values: np.ndarray | None, index: int) -> np.ndarray:
        return _floats([] if values is None else values[index])

    point_type = _ros("trajectory_msgs/msg/JointTrajectoryPoint")
    points = [
        point_type(
            positions=row(traj.positions, i),
            velocities=row(traj.velocities, i),
            accelerations=row(traj.accelerations, i),
            effort=row(traj.effort, i),
            time_from_start=_duration(traj.t[i]),
        )
        for i in range(len(traj))
    ]
    return _ros("trajectory_msgs/msg/JointTrajectory")(
        header=_header_to_ros(traj.header), joint_names=list(traj.joint_names), points=points
    )


@_from("trajectory_msgs/msg/JointTrajectory")
def _joint_trajectory_from_ros(ros: Any) -> JointTrajectory:
    names = tuple(ros.joint_names)
    joints = len(names)

    def stacked(attribute: str, *, required: bool = False) -> np.ndarray | None:
        rows = [_floats(getattr(point, attribute)) for point in ros.points]
        if not rows:
            return np.zeros((0, joints)) if required else None
        sizes = {len(row) for row in rows}
        if sizes == {0} and not required:
            return None
        if sizes != {joints}:
            raise ValueError(
                f"JointTrajectory points carry {attribute} of sizes {sorted(sizes)}; "
                f"every point needs {joints} (one per joint) or none"
            )
        return np.stack(rows)

    positions = stacked("positions", required=True)
    return JointTrajectory(
        t=[_duration_seconds(point.time_from_start) for point in ros.points],
        joint_names=names,
        positions=positions,
        velocities=stacked("velocities"),
        accelerations=stacked("accelerations"),
        effort=stacked("effort"),
        header=_header_from_ros(ros.header),
    )


@_to(PoseTrajectory, "nav_msgs/msg/Path", options=True)
def _pose_trajectory_to_ros(
    traj: PoseTrajectory, *, drop_extras: bool = False, as_: str | None = None
) -> Any:
    if as_ not in (None, "path", "multidof"):
        raise ValueError(f"to_ros(as_={as_!r}): PoseTrajectory converts as 'path' or 'multidof'")
    multidof = as_ == "multidof"
    lost = []
    if traj.channels:
        lost.append(f"channels {list(traj.channels)}")
    if not multidof and (traj.linear_velocities is not None or traj.angular_velocities is not None):
        lost.append("velocities (use as_='multidof' to keep them)")
    if lost and not drop_extras:
        target = "MultiDOFJointTrajectory" if multidof else "nav_msgs/Path"
        raise ValueError(
            f"PoseTrajectory {' and '.join(lost)} have no field in {target}; "
            "pass drop_extras=True to drop them"
        )
    if multidof:
        return _pose_trajectory_to_multidof(traj)
    base_sec, base_nanosec = traj.header.to_sec_nanosec()
    base_ns = base_sec * _NS + base_nanosec
    poses = [
        _ros("geometry_msgs/msg/PoseStamped")(
            header=_header_ns(base_ns + round(float(traj.t[i]) * _NS), traj.header.frame_id),
            pose=_pose_to_ros(_sample_pose(traj, i)),
        )
        for i in range(len(traj))
    ]
    return _ros("nav_msgs/msg/Path")(header=_header_to_ros(traj.header), poses=poses)


def _sample_pose(traj: PoseTrajectory, index: int) -> Pose:
    return Pose(
        Point.from_array(traj.positions[index]), Quaternion.from_array(traj.orientations[index])
    )


def _pose_trajectory_to_multidof(traj: PoseTrajectory) -> Any:
    def vector(values: np.ndarray | None, index: int) -> Vector3:
        return Vector3() if values is None else Vector3.from_array(values[index])

    has_twist = traj.linear_velocities is not None or traj.angular_velocities is not None
    point_type = _ros("trajectory_msgs/msg/MultiDOFJointTrajectoryPoint")
    points = []
    for i in range(len(traj)):
        pose = _sample_pose(traj, i)
        twist = Twist(vector(traj.linear_velocities, i), vector(traj.angular_velocities, i))
        points.append(
            point_type(
                transforms=[_transform_to_ros(pose.as_transform())],
                velocities=[_twist_to_ros(twist)] if has_twist else [],
                accelerations=[],
                time_from_start=_duration(traj.t[i]),
            )
        )
    return _ros("trajectory_msgs/msg/MultiDOFJointTrajectory")(
        header=_header_to_ros(traj.header), joint_names=[traj.child_frame_id], points=points
    )


@_from("nav_msgs/msg/Path")
def _path_from_ros(ros: Any) -> PoseTrajectory:
    """`t` counts from the first pose's stamp, which becomes `header.stamp`; the child frame is
    unknown (a Path only names the reference frame)."""
    header = _header_from_ros(ros.header)
    if not ros.poses:
        return PoseTrajectory(
            t=[], positions=np.zeros((0, 3)), orientations=np.zeros((0, 4)), header=header
        )
    stamps = [_stamp_ns(posed.header.stamp) for posed in ros.poses]
    poses = [_pose_from_ros(posed.pose) for posed in ros.poses]
    return PoseTrajectory.from_poses(
        [(ns - stamps[0]) / _NS for ns in stamps],
        poses,
        header=Header(stamp=stamps[0] / _NS, frame_id=header.frame_id),
    )


@_from("trajectory_msgs/msg/MultiDOFJointTrajectory")
def _multidof_from_ros(ros: Any) -> PoseTrajectory:
    """One joint only: its name becomes `child_frame_id`. Accelerations have no place in a
    PoseTrajectory and are ignored."""
    if len(ros.joint_names) != 1:
        raise ValueError(
            "a PoseTrajectory is one frame; MultiDOFJointTrajectory has joints "
            f"{list(ros.joint_names)}"
        )
    header = _header_from_ros(ros.header)
    child = str(ros.joint_names[0])
    if not ros.points:
        return PoseTrajectory(
            t=[],
            positions=np.zeros((0, 3)),
            orientations=np.zeros((0, 4)),
            child_frame_id=child,
            header=header,
        )
    poses = []
    for index, point in enumerate(ros.points):
        if len(point.transforms) != 1:
            raise ValueError(
                f"MultiDOFJointTrajectory point {index} has {len(point.transforms)} transforms, "
                "expected 1 (one joint)"
            )
        poses.append(_transform_from_ros(point.transforms[0]).as_pose())
    t = [_duration_seconds(point.time_from_start) for point in ros.points]
    twists = [len(point.velocities) for point in ros.points]
    changes: dict[str, Any] = {}
    if set(twists) == {1}:
        velocities = [_twist_from_ros(point.velocities[0]) for point in ros.points]
        changes["linear_velocities"] = [v.linear.as_array() for v in velocities]
        changes["angular_velocities"] = [v.angular.as_array() for v in velocities]
    elif set(twists) != {0}:
        raise ValueError(
            f"MultiDOFJointTrajectory points carry {sorted(set(twists))} velocities; "
            "every point needs exactly 1 or none"
        )
    return PoseTrajectory.from_poses(t, poses, child_frame_id=child, header=header, **changes)


# ── vision ───────────────────────────────────────────────────────────────────


@_to(ObjectHypothesis, "vision_msgs/msg/ObjectHypothesis")
def _hypothesis_to_ros(h: ObjectHypothesis) -> Any:
    return _ros("vision_msgs/msg/ObjectHypothesis")(class_id=h.class_id, score=h.score)


@_from("vision_msgs/msg/ObjectHypothesis")
def _hypothesis_from_ros(ros: Any) -> ObjectHypothesis:
    return ObjectHypothesis(str(ros.class_id), float(ros.score))


@_to(ObjectHypothesisWithPose, "vision_msgs/msg/ObjectHypothesisWithPose")
def _hypothesis_with_pose_to_ros(h: ObjectHypothesisWithPose) -> Any:
    covariance = np.zeros(36) if h.covariance is None else _floats(h.covariance)
    return _ros("vision_msgs/msg/ObjectHypothesisWithPose")(
        hypothesis=_hypothesis_to_ros(h.hypothesis),
        pose=_ros("geometry_msgs/msg/PoseWithCovariance")(
            pose=_pose_to_ros(h.pose), covariance=covariance
        ),
    )


@_from("vision_msgs/msg/ObjectHypothesisWithPose")
def _hypothesis_with_pose_from_ros(ros: Any) -> ObjectHypothesisWithPose:
    covariance = _floats(ros.pose.covariance).reshape(6, 6)
    return ObjectHypothesisWithPose(
        hypothesis=_hypothesis_from_ros(ros.hypothesis),
        pose=_pose_from_ros(ros.pose.pose),
        covariance=covariance if np.any(covariance) else None,  # all zeros: unknown
    )


@_to(BoundingBox2D, "vision_msgs/msg/BoundingBox2D")
def _box2d_to_ros(box: BoundingBox2D) -> Any:
    center = _ros("vision_msgs/msg/Pose2D")(
        position=_ros("vision_msgs/msg/Point2D")(x=box.center.x, y=box.center.y),
        theta=box.center.theta,
    )
    return _ros("vision_msgs/msg/BoundingBox2D")(
        center=center, size_x=box.size_x, size_y=box.size_y
    )


@_from("vision_msgs/msg/BoundingBox2D")
def _box2d_from_ros(ros: Any) -> BoundingBox2D:
    center = Pose2D(
        float(ros.center.position.x), float(ros.center.position.y), float(ros.center.theta)
    )
    return BoundingBox2D(center, float(ros.size_x), float(ros.size_y))


@_to(BoundingBox3D, "vision_msgs/msg/BoundingBox3D")
def _box3d_to_ros(box: BoundingBox3D) -> Any:
    return _ros("vision_msgs/msg/BoundingBox3D")(
        center=_pose_to_ros(box.center), size=_vector3_to_ros(box.size)
    )


@_from("vision_msgs/msg/BoundingBox3D")
def _box3d_from_ros(ros: Any) -> BoundingBox3D:
    return BoundingBox3D(_pose_from_ros(ros.center), _vector3_from_ros(ros.size))


@_to(Detection2D, "vision_msgs/msg/Detection2D")
def _detection2d_to_ros(d: Detection2D) -> Any:
    return _ros("vision_msgs/msg/Detection2D")(
        header=_header_to_ros(d.header),
        results=[_hypothesis_with_pose_to_ros(r) for r in d.results],
        bbox=_box2d_to_ros(d.bbox),
        id=d.id,
    )


@_from("vision_msgs/msg/Detection2D")
def _detection2d_from_ros(ros: Any) -> Detection2D:
    return Detection2D(
        header=_header_from_ros(ros.header),
        results=tuple(_hypothesis_with_pose_from_ros(r) for r in ros.results),
        bbox=_box2d_from_ros(ros.bbox),
        id=str(ros.id),
    )


@_to(Detection2DArray, "vision_msgs/msg/Detection2DArray")
def _detection2d_array_to_ros(array: Detection2DArray) -> Any:
    return _ros("vision_msgs/msg/Detection2DArray")(
        header=_header_to_ros(array.header),
        detections=[_detection2d_to_ros(d) for d in array.detections],
    )


@_from("vision_msgs/msg/Detection2DArray")
def _detection2d_array_from_ros(ros: Any) -> Detection2DArray:
    return Detection2DArray(
        header=_header_from_ros(ros.header),
        detections=tuple(_detection2d_from_ros(d) for d in ros.detections),
    )


@_to(Detection3D, "vision_msgs/msg/Detection3D")
def _detection3d_to_ros(d: Detection3D) -> Any:
    return _ros("vision_msgs/msg/Detection3D")(
        header=_header_to_ros(d.header),
        results=[_hypothesis_with_pose_to_ros(r) for r in d.results],
        bbox=_box3d_to_ros(d.bbox),
        id=d.id,
    )


@_from("vision_msgs/msg/Detection3D")
def _detection3d_from_ros(ros: Any) -> Detection3D:
    return Detection3D(
        header=_header_from_ros(ros.header),
        results=tuple(_hypothesis_with_pose_from_ros(r) for r in ros.results),
        bbox=_box3d_from_ros(ros.bbox),
        id=str(ros.id),
    )


@_to(Detection3DArray, "vision_msgs/msg/Detection3DArray")
def _detection3d_array_to_ros(array: Detection3DArray) -> Any:
    return _ros("vision_msgs/msg/Detection3DArray")(
        header=_header_to_ros(array.header),
        detections=[_detection3d_to_ros(d) for d in array.detections],
    )


@_from("vision_msgs/msg/Detection3DArray")
def _detection3d_array_from_ros(ros: Any) -> Detection3DArray:
    return Detection3DArray(
        header=_header_from_ros(ros.header),
        detections=tuple(_detection3d_from_ros(d) for d in ros.detections),
    )
