"""sensor messages: JointState, Image, CameraInfo, PointCloud, Imu."""

import dataclasses
import json

import numpy as np
import pytest

from robot_msgs._core import load_npz, save_npz
from robot_msgs.geometry import Quaternion, Transform, Vector3
from robot_msgs.sensor import CameraInfo, Image, Imu, JointState, PointCloud
from robot_msgs.sensor.image import IMAGE_ENCODINGS
from robot_msgs.std import Header


def round_trip(msg, tmp_path):
    """to_dict → JSON → from_dict, and save_npz → load_npz: both must give back an equal value."""
    wire = json.loads(json.dumps(msg.to_dict()))
    assert wire["_type"] == msg.type_name
    assert type(msg).from_dict(wire) == msg
    save_npz(msg, tmp_path / "msg.npz")
    assert load_npz(tmp_path / "msg.npz") == msg


# ── JointState ───────────────────────────────────────────────────────────────


def make_joint_state():
    return JointState(
        name=("shoulder", "elbow", "wrist"),
        position=[0.1, 0.2, 0.3],
        velocity=[1.0, 2.0, 3.0],
        effort=[10.0, 20.0, 30.0],
        header=Header(2.5, "base"),
    )


def test_joint_state_helpers():
    state = make_joint_state()
    assert state.position_of("elbow") == 0.2
    assert state.as_dict() == {"shoulder": 0.1, "elbow": 0.2, "wrist": 0.3}
    assert len(state) == 3


def test_joint_state_select_reorders_every_array_and_keeps_the_header():
    picked = make_joint_state().select(["wrist", "shoulder"])
    assert picked.name == ("wrist", "shoulder")
    assert picked.position.tolist() == [0.3, 0.1]
    assert picked.velocity.tolist() == [3.0, 1.0]
    assert picked.effort.tolist() == [30.0, 10.0]
    assert picked.header == Header(2.5, "base")


def test_joint_state_select_keeps_absent_arrays_absent():
    picked = JointState(name=("a", "b"), position=[1.0, 2.0]).select(["b"])
    assert picked.velocity is None and picked.effort is None
    assert picked.position.tolist() == [2.0]


def test_joint_state_unknown_joint_raises_key_error_naming_it():
    with pytest.raises(KeyError, match="gripper"):
        make_joint_state().position_of("gripper")
    with pytest.raises(KeyError, match="gripper"):
        make_joint_state().select(["elbow", "gripper"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(name=("a", "a"), position=[0.0, 1.0]), "JointState.name has duplicate names"),
        (dict(name=("a", "b"), position=[0.0]), "JointState.position has 1 values, name has 2"),
        (
            dict(name=("a", "b"), position=[0.0, 1.0], velocity=[0.0]),
            "JointState.velocity has 1 values, name has 2",
        ),
        (
            dict(name=("a",), position=[0.0], effort=[1.0, 2.0]),
            "JointState.effort has 2 values, name has 1",
        ),
        (dict(name=("a",), position=[[0.0]]), "JointState.position must be 1-D"),
    ],
)
def test_joint_state_validation_names_the_field(kwargs, message):
    with pytest.raises(ValueError, match=message):
        JointState(**kwargs)


def test_joint_state_is_an_immutable_value():
    state = make_joint_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.position = np.zeros(3)
    with pytest.raises(ValueError):
        state.position[0] = 9.0
    assert state == make_joint_state() and hash(state) == hash(make_joint_state())
    assert state != state.replace(position=[0.1, 0.2, 0.31])


def test_joint_state_copies_the_callers_array():
    mine = np.array([1.0, 2.0])
    state = JointState(name=("a", "b"), position=mine)
    mine[0] = 99.0
    assert state.position[0] == 1.0


def test_joint_state_round_trip(tmp_path):
    round_trip(make_joint_state(), tmp_path)
    round_trip(JointState(name=("a",), position=[1.0]), tmp_path)
    round_trip(JointState(name=(), position=[]), tmp_path)


# ── Image ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("encoding", sorted(IMAGE_ENCODINGS))
def test_image_accepts_every_supported_encoding_and_keeps_the_dtype(encoding):
    dtype, channels = IMAGE_ENCODINGS[encoding]
    shape = (4, 5) if channels == 1 else (4, 5, channels)
    image = Image(data=np.zeros(shape, dtype=dtype), encoding=encoding)
    assert image.data.dtype == dtype
    assert (image.height, image.width, image.channels) == (4, 5, channels)
    assert image.step == 5 * channels * dtype.itemsize


def test_image_step_is_bytes_per_row():
    assert Image(data=np.zeros((2, 7, 3), np.uint8), encoding="rgb8").step == 21
    assert Image(data=np.zeros((2, 7), np.uint16), encoding="16UC1").step == 14
    assert Image(data=np.zeros((2, 7), np.float32), encoding="32FC1").step == 28


@pytest.mark.parametrize(
    "data, encoding, message",
    [
        (np.zeros((2, 2, 3), np.uint8), "rgba8", "needs 4"),
        (np.zeros((2, 2), np.uint8), "rgb8", "1 channel.*needs 3"),
        (np.zeros((2, 2, 3), np.float64), "rgb8", "dtype float64.*needs uint8"),
        (np.zeros((2, 2), np.uint8), "32FC1", "needs float32"),
        (np.zeros((2, 2), np.uint16), "mono8", "needs uint8"),
        (np.zeros(4, np.uint8), "mono8", "must be \\(H, W\\)"),
        (np.zeros((2, 2, 1, 1), np.uint8), "mono8", "must be \\(H, W\\)"),
        (np.zeros((2, 2), np.uint8), "yuv422", "not supported"),
    ],
)
def test_image_validates_data_against_its_encoding(data, encoding, message):
    with pytest.raises(ValueError, match=f"Image.*{message}"):
        Image(data=data, encoding=encoding)


def test_image_accepts_single_channel_with_trailing_axis():
    image = Image(data=np.zeros((3, 4, 1), np.uint8), encoding="mono8")
    assert image.channels == 1 and image.step == 4


def test_image_does_not_cast_to_float64():
    image = Image(data=np.arange(6, dtype=np.uint16).reshape(2, 3), encoding="16UC1")
    assert image.data.dtype == np.uint16


def test_image_from_array_infers_the_unambiguous_cases():
    cases = [
        (np.zeros((2, 3, 3), np.uint8), "rgb8"),
        (np.zeros((2, 3), np.uint8), "mono8"),
        (np.zeros((2, 3, 1), np.uint8), "mono8"),
        (np.zeros((2, 3), np.uint16), "16UC1"),
        (np.zeros((2, 3), np.float32), "32FC1"),
    ]
    for array, expected in cases:
        assert Image.from_array(array).encoding == expected


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 3, 4), np.uint8),  # rgba8 or bgra8?
        np.zeros((2, 3), np.float64),
        np.zeros((2, 3, 3), np.float32),
        np.zeros(5, np.uint8),
    ],
)
def test_image_from_array_raises_when_ambiguous(array):
    with pytest.raises(ValueError, match="cannot infer an encoding"):
        Image.from_array(array)


def test_image_from_array_with_explicit_encoding_and_header():
    array = np.zeros((2, 3, 4), np.uint8)
    image = Image.from_array(array, "bgra8", header=Header(1.0, "cam"))
    assert image.encoding == "bgra8" and image.header.frame_id == "cam"
    with pytest.raises(ValueError):
        Image.from_array(array, "rgb8")


def test_image_to_rgb_reorders_and_drops_alpha():
    pixel = np.array([[[10, 20, 30]]], np.uint8)
    assert Image(data=pixel, encoding="rgb8").to_rgb().tolist() == [[[10, 20, 30]]]
    assert Image(data=pixel, encoding="bgr8").to_rgb().tolist() == [[[30, 20, 10]]]
    alpha = np.array([[[10, 20, 30, 255]]], np.uint8)
    assert Image(data=alpha, encoding="rgba8").to_rgb().tolist() == [[[10, 20, 30]]]
    assert Image(data=alpha, encoding="bgra8").to_rgb().tolist() == [[[30, 20, 10]]]


def test_image_to_rgb_stacks_grey_and_normalises_depth():
    grey = Image(data=np.array([[0, 128, 255]], np.uint8), encoding="mono8").to_rgb()
    assert grey.shape == (1, 3, 3) and grey.dtype == np.uint8
    assert grey[0, 1].tolist() == [128, 128, 128]
    depth = Image(data=np.array([[1000, 2000, 3000]], np.uint16), encoding="16UC1").to_rgb()
    assert depth[0, :, 0].tolist() == [0, 128, 255]
    metres = np.array([[np.nan, 1.0, 3.0]], np.float32)
    assert Image(data=metres, encoding="32FC1").to_rgb()[0, :, 0].tolist() == [0, 0, 255]
    flat = Image(data=np.full((2, 2), 7, np.uint16), encoding="16UC1").to_rgb()
    assert not flat.any()


def test_image_to_rgb_returns_a_writable_copy_that_leaves_the_image_alone():
    image = Image(data=np.zeros((1, 1, 3), np.uint8), encoding="rgb8")
    shown = image.to_rgb()
    shown[0, 0, 0] = 255
    assert image.data[0, 0, 0] == 0


def test_image_crop_returns_the_window():
    data = np.arange(20, dtype=np.uint8).reshape(4, 5)
    image = Image(data=data, encoding="mono8", header=Header(1.0, "cam"))
    window = image.crop(x_offset=1, y_offset=2, width=3, height=2)
    assert window.data.tolist() == [[11, 12, 13], [16, 17, 18]]
    assert (window.width, window.height) == (3, 2)
    assert window.header == image.header and window.encoding == "mono8"
    assert np.shares_memory(window.data, image.data)  # a view, not a copy


def test_image_crop_rejects_windows_outside_the_image():
    image = Image(data=np.zeros((4, 5), np.uint8), encoding="mono8")
    for window in [(3, 0, 3, 1), (0, 3, 1, 2), (-1, 0, 1, 1), (0, 0, 0, 1), (0, 0, 1, 0)]:
        with pytest.raises(ValueError, match="Image.crop"):
            image.crop(*window)
    assert image.crop(0, 0, 5, 4) == image


def test_image_is_an_immutable_value():
    image = Image(data=np.zeros((2, 2), np.uint8), encoding="mono8")
    with pytest.raises(ValueError):
        image.data[0, 0] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        image.encoding = "8UC1"
    assert image == Image(data=np.zeros((2, 2), np.uint8), encoding="mono8")
    assert image != Image(data=np.zeros((2, 2), np.uint8), encoding="8UC1")
    assert image != Image(data=np.ones((2, 2), np.uint8), encoding="mono8")


def test_image_round_trip_keeps_dtype(tmp_path):
    rng = np.random.default_rng(0)
    for array, encoding in [
        (rng.integers(0, 255, (3, 4, 3), dtype=np.uint8), "bgr8"),
        (rng.integers(0, 60000, (3, 4), dtype=np.uint16), "16UC1"),
        (rng.random((3, 4), dtype=np.float32), "32FC1"),
    ]:
        image = Image(data=array, encoding=encoding, header=Header(3.0, "cam"))
        round_trip(image, tmp_path)
        assert Image.from_dict(image.to_dict()).data.dtype == array.dtype


# ── CameraInfo ───────────────────────────────────────────────────────────────


def make_camera(**kwargs):
    return CameraInfo.from_intrinsics(500.0, 400.0, 320.0, 240.0, 640, 480, **kwargs)


def test_camera_info_from_intrinsics_builds_k_and_defaults():
    info = make_camera()
    assert info.k.tolist() == [[500.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
    assert (info.fx, info.fy, info.cx, info.cy) == (500.0, 400.0, 320.0, 240.0)
    assert info.d.shape == (0,)
    assert info.distortion_model == "plumb_bob"
    assert np.array_equal(info.r, np.eye(3))
    assert np.array_equal(info.p, np.hstack([info.k, np.zeros((3, 1))]))
    assert (info.width, info.height) == (640, 480)


def test_camera_info_keeps_distortion_and_explicit_p():
    info = make_camera(d=[0.1, -0.2, 0.0, 0.0, 0.01])
    assert info.d.tolist() == [0.1, -0.2, 0.0, 0.0, 0.01]
    p = np.hstack([info.k, [[1.0], [2.0], [3.0]]])
    assert CameraInfo(width=2, height=2, k=info.k, p=p).p[:, 3].tolist() == [1.0, 2.0, 3.0]


def test_camera_info_project_matches_hand_computed_pixels():
    info = make_camera()
    pixels = info.project([[0.0, 0.0, 2.0], [0.2, -0.1, 1.0], [-1.0, 1.0, 4.0]])
    expected = [[320.0, 240.0], [420.0, 200.0], [195.0, 340.0]]
    assert np.allclose(pixels, expected)


def test_camera_info_project_of_a_point_at_z_zero_is_not_finite():
    pixels = make_camera().project([[1.0, 1.0, 0.0]])
    assert not np.isfinite(pixels).any()


def test_camera_info_unproject_matches_hand_computed_points():
    info = make_camera()
    points = info.unproject([[320.0, 240.0], [420.0, 200.0]], [2.0, 1.0])
    assert np.allclose(points, [[0.0, 0.0, 2.0], [0.2, -0.1, 1.0]])


def test_camera_info_project_unproject_round_trip():
    info = make_camera()
    rng = np.random.default_rng(1)
    points = rng.uniform([-1, -1, 0.3], [1, 1, 5], size=(50, 3))
    pixels = info.project(points)
    assert np.allclose(info.unproject(pixels, points[:, 2]), points)
    # and the other way: pixels → points → pixels
    uv = rng.uniform([0, 0], [640, 480], size=(50, 2))
    assert np.allclose(info.project(info.unproject(uv, np.full(50, 1.7))), uv)


def test_camera_info_project_and_unproject_validate_shapes():
    info = make_camera()
    with pytest.raises(ValueError, match="points"):
        info.project(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="pixels"):
        info.unproject(np.zeros((4, 3)), np.ones(4))
    with pytest.raises(ValueError, match="depth"):
        info.unproject(np.zeros((4, 2)), np.ones(3))


def test_camera_info_scaled_for_a_resized_image():
    info = make_camera(d=[0.1, 0.0, 0.0, 0.0, 0.0])
    half = info.scaled(0.5)
    assert (half.width, half.height) == (320, 240)
    assert (half.fx, half.fy, half.cx, half.cy) == (250.0, 200.0, 160.0, 120.0)
    assert half.k[2].tolist() == [0.0, 0.0, 1.0]
    assert half.p[0].tolist() == [250.0, 0.0, 160.0, 0.0]
    assert half.d.tolist() == info.d.tolist()
    # the same 3D point lands on the proportional pixel
    point = [[0.3, -0.2, 2.0]]
    assert np.allclose(half.project(point), info.project(point) * 0.5)
    assert info.width == 640  # the original is untouched


def test_camera_info_scaled_rounds_the_size_and_scales_by_the_real_ratio():
    info = CameraInfo.from_intrinsics(100.0, 100.0, 50.0, 50.0, 101, 101).scaled(0.5)
    assert (info.width, info.height) == (50, 50)  # round(50.5) banker's rounding
    assert info.fx == pytest.approx(100.0 * 50 / 101)
    with pytest.raises(ValueError, match="factor"):
        make_camera().scaled(0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(width=0, height=4, k=np.eye(3)), "CameraInfo.width/height"),
        (dict(width=4, height=-1, k=np.eye(3)), "CameraInfo.width/height"),
        (dict(width=4, height=4, k=np.eye(4)), "CameraInfo.k must have shape \\(3, 3\\)"),
        (dict(width=4, height=4, k=np.eye(3), r=np.eye(2)), "CameraInfo.r must have shape"),
        (dict(width=4, height=4, k=np.eye(3), p=np.eye(3)), "CameraInfo.p must have shape"),
        (dict(width=4, height=4, k=np.eye(3), d=np.zeros((2, 2))), "CameraInfo.d must be 1-D"),
    ],
)
def test_camera_info_validation_names_the_field(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CameraInfo(**kwargs)


def test_camera_info_is_an_immutable_value():
    info = make_camera()
    with pytest.raises(ValueError):
        info.k[0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.width = 1
    assert info == make_camera() and hash(info) == hash(make_camera())
    assert info != CameraInfo.from_intrinsics(501.0, 400.0, 320.0, 240.0, 640, 480)


def test_camera_info_round_trip(tmp_path):
    round_trip(make_camera(), tmp_path)
    round_trip(
        make_camera(d=[0.1, 0.2, 0.3, 0.4, 0.5]).replace(header=Header(4.0, "optical")), tmp_path
    )


# ── PointCloud ───────────────────────────────────────────────────────────────


def test_point_cloud_stores_float32_points_and_keeps_channel_dtypes():
    cloud = PointCloud(
        points=np.zeros((3, 3), np.float64),
        channels={"rgb": np.zeros((3, 3), np.uint8), "intensity": np.arange(3.0)},
    )
    assert cloud.points.dtype == np.float32
    assert cloud.channels["rgb"].dtype == np.uint8
    assert len(cloud) == 3
    assert len(PointCloud(points=np.zeros((0, 3)))) == 0


def test_point_cloud_validation_names_the_field():
    with pytest.raises(ValueError, match="PointCloud.points must have shape \\(N, 3\\)"):
        PointCloud(points=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="PointCloud.points"):
        PointCloud(points=np.zeros(3))
    with pytest.raises(ValueError, match="PointCloud.channels\\['rgb'\\] has leading dimension 2"):
        PointCloud(points=np.zeros((3, 3)), channels={"rgb": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="PointCloud.channels\\['i'\\]"):
        PointCloud(points=np.zeros((3, 3)), channels={"i": np.float64(1.0)})


def test_point_cloud_channels_are_copied_and_read_only():
    mine = np.arange(3.0)
    cloud = PointCloud(points=np.zeros((3, 3)), channels={"intensity": mine})
    mine[0] = 9.0
    assert cloud.channels["intensity"][0] == 0.0
    with pytest.raises(ValueError):
        cloud.channels["intensity"][0] = 1.0
    with pytest.raises(ValueError):
        cloud.points[0, 0] = 1.0


def test_point_cloud_from_depth_matches_a_hand_computed_cloud():
    # fx = fy = 2, cx = cy = 1: x = (u - 1) / 2 * z, y = (v - 1) / 2 * z
    info = CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)
    metres = np.array([[1.0, 0.0, 2.0], [np.nan, 1.0, 1.0], [4.0, -1.0, 0.0]], np.float32)
    header = Header(5.0, "optical")
    cloud = PointCloud.from_depth(Image(data=metres, encoding="32FC1", header=header), info)
    expected = [  # row-major: (u, v) = (0,0) (2,0) (1,1) (2,1) (0,2)
        [-0.5, -0.5, 1.0],
        [1.0, -1.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [-2.0, 2.0, 4.0],
    ]
    assert np.allclose(cloud.points, expected)
    assert cloud.points.dtype == np.float32
    assert cloud.header == header
    assert cloud.channels == {}


def test_point_cloud_from_depth_reads_16uc1_as_millimetres():
    info = CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)
    millimetres = np.zeros((3, 3), np.uint16)
    millimetres[0, 0], millimetres[1, 1], millimetres[2, 0] = 1000, 1500, 4000
    cloud = PointCloud.from_depth(Image(data=millimetres, encoding="16UC1"), info)
    assert np.allclose(cloud.points, [[-0.5, -0.5, 1.0], [0.0, 0.0, 1.5], [-2.0, 2.0, 4.0]])


def test_point_cloud_from_depth_attaches_aligned_rgb():
    info = CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)
    metres = np.zeros((3, 3), np.float32)
    metres[0, 1], metres[2, 2] = 1.0, 2.0
    colours = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    cloud = PointCloud.from_depth(
        Image(data=metres, encoding="32FC1"), info, rgb=Image(data=colours, encoding="bgr8")
    )
    assert cloud.channels["rgb"].dtype == np.uint8
    # bgr → rgb reverses the last axis; kept pixels are (row 0, col 1) and (row 2, col 2)
    assert cloud.channels["rgb"].tolist() == [
        colours[0, 1, ::-1].tolist(),
        colours[2, 2, ::-1].tolist(),
    ]


def test_point_cloud_from_depth_validates_inputs():
    info = CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)
    with pytest.raises(ValueError, match="16UC1 or 32FC1"):
        PointCloud.from_depth(Image(data=np.zeros((3, 3), np.uint8), encoding="mono8"), info)
    depth = Image(data=np.ones((3, 3), np.float32), encoding="32FC1")
    with pytest.raises(ValueError, match="depth image is 3x3, camera_info is 4x4"):
        PointCloud.from_depth(depth, CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 4, 4))
    small = Image(data=np.zeros((2, 2, 3), np.uint8), encoding="rgb8")
    with pytest.raises(ValueError, match="rgb image is 2x2"):
        PointCloud.from_depth(depth, info, rgb=small)


def test_point_cloud_from_depth_of_an_empty_image_is_an_empty_cloud():
    info = CameraInfo.from_intrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)
    cloud = PointCloud.from_depth(Image(data=np.zeros((3, 3), np.float32), encoding="32FC1"), info)
    assert len(cloud) == 0 and cloud.points.shape == (0, 3)


def test_point_cloud_from_depth_then_project_recovers_the_pixels():
    info = CameraInfo.from_intrinsics(300.0, 310.0, 31.5, 23.5, 64, 48)
    rng = np.random.default_rng(2)
    metres = rng.uniform(0.5, 3.0, (48, 64)).astype(np.float32)
    cloud = PointCloud.from_depth(Image(data=metres, encoding="32FC1"), info)
    columns, rows = np.meshgrid(np.arange(64), np.arange(48))
    pixels = np.stack([columns.ravel(), rows.ravel()], axis=1)
    assert np.allclose(info.project(cloud.points), pixels, atol=1e-2)


def test_point_cloud_transformed_rotates_then_translates_and_keeps_channels():
    cloud = PointCloud(
        points=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        channels={"intensity": [0.5, 0.7]},
        header=Header(1.0, "camera"),
    )
    moved = cloud.transformed(Transform.from_xyz_rpy(10.0, 0.0, 1.0, 0.0, 0.0, np.pi / 2))
    assert np.allclose(moved.points, [[10.0, 1.0, 1.0], [8.0, 0.0, 1.0]], atol=1e-6)
    assert moved.channels["intensity"].tolist() == [0.5, 0.7]
    assert moved.points.dtype == np.float32
    assert cloud.points[0].tolist() == [1.0, 0.0, 0.0]  # original untouched


def test_point_cloud_transformed_by_inverse_round_trips():
    rng = np.random.default_rng(3)
    cloud = PointCloud(points=rng.uniform(-2, 2, (20, 3)))
    transform = Transform.from_xyz_rpy(0.3, -0.2, 1.1, 0.4, -0.5, 1.2)
    back = cloud.transformed(transform).transformed(transform.inverse())
    assert np.allclose(back.points, cloud.points, atol=1e-5)
    assert cloud.transformed(Transform.identity()) == cloud


def test_point_cloud_cropped_filters_points_and_channels_together():
    cloud = PointCloud(
        points=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 5, 1]],
        channels={
            "intensity": [0.0, 1.0, 2.0, 3.0],
            "rgb": np.arange(12, dtype=np.uint8).reshape(4, 3),
        },
    )
    inside = cloud.cropped([0.5, 0.5, 0.5], [2.0, 2.0, 2.0])  # edges are inclusive
    assert inside.points.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert inside.channels["intensity"].tolist() == [1.0, 2.0]
    assert inside.channels["rgb"].tolist() == [[3, 4, 5], [6, 7, 8]]
    assert len(cloud.cropped([9, 9, 9], [10, 10, 10])) == 0
    with pytest.raises(ValueError, match="min_xyz"):
        cloud.cropped([0, 0], [1, 1, 1])


def test_point_cloud_downsampled_keeps_the_first_point_per_voxel_in_order():
    cloud = PointCloud(
        points=[[0.9, 0.1, 0.1], [0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [1.9, 0.9, 0.9], [-0.1, 0, 0]],
        channels={"intensity": [0.0, 1.0, 2.0, 3.0, 4.0]},
    )
    thinned = cloud.downsampled(1.0)
    # voxels: (0,0,0): points 0, 1 → keep 0; (1,0,0): points 2, 3 → keep 2; (-1,0,0): point 4
    assert thinned.channels["intensity"].tolist() == [0.0, 2.0, 4.0]
    assert np.allclose(thinned.points, cloud.points[[0, 2, 4]])
    assert len(cloud.downsampled(0.01)) == 5
    assert len(cloud.downsampled(100.0)) == 2  # the origin voxel and the voxel below it
    assert len(PointCloud(points=np.zeros((0, 3))).downsampled(1.0)) == 0
    with pytest.raises(ValueError, match="voxel_size"):
        cloud.downsampled(0.0)


def test_point_cloud_is_an_immutable_value():
    cloud = PointCloud(points=[[1.0, 2.0, 3.0]], channels={"i": [1.0]})
    assert cloud == PointCloud(points=[[1, 2, 3]], channels={"i": [1.0]})
    assert hash(cloud) == hash(PointCloud(points=[[1, 2, 3]], channels={"i": [1.0]}))
    assert cloud != PointCloud(points=[[1, 2, 3]], channels={"j": [1.0]})
    assert cloud != PointCloud(points=[[1, 2, 3]])
    with pytest.raises(dataclasses.FrozenInstanceError):
        cloud.points = np.zeros((1, 3))


def test_point_cloud_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    cloud = PointCloud(
        points=rng.random((6, 3)),
        channels={"rgb": rng.integers(0, 255, (6, 3), dtype=np.uint8), "intensity": rng.random(6)},
        header=Header(1.0, "lidar"),
    )
    round_trip(cloud, tmp_path)
    assert PointCloud.from_dict(cloud.to_dict()).channels["rgb"].dtype == np.uint8
    round_trip(PointCloud(points=np.zeros((0, 3))), tmp_path)


# ── Imu ──────────────────────────────────────────────────────────────────────


def make_imu(**kwargs):
    return Imu(
        orientation=Quaternion.from_euler(0.1, 0.2, 0.3),
        angular_velocity=Vector3(0.0, 0.1, 0.2),
        linear_acceleration=Vector3(0.0, 0.0, 9.81),
        header=Header(1.0, "imu"),
        **kwargs,
    )


def test_imu_covariances_default_to_none_and_are_stored_as_matrices():
    assert make_imu().orientation_covariance is None
    imu = make_imu(
        orientation_covariance=np.eye(3) * 0.01,
        angular_velocity_covariance=np.eye(3) * 0.02,
        linear_acceleration_covariance=np.eye(3) * 0.03,
    )
    assert imu.angular_velocity_covariance[1, 1] == 0.02
    with pytest.raises(ValueError):
        imu.orientation_covariance[0, 0] = 1.0


@pytest.mark.parametrize(
    "field_name",
    ["orientation_covariance", "angular_velocity_covariance", "linear_acceleration_covariance"],
)
def test_imu_covariance_shape_is_validated_per_field(field_name):
    with pytest.raises(ValueError, match=f"Imu.{field_name} must have shape \\(3, 3\\)"):
        make_imu(**{field_name: np.zeros(9)})


def test_imu_is_an_immutable_value():
    imu = make_imu(orientation_covariance=np.eye(3))
    assert imu == make_imu(orientation_covariance=np.eye(3)) and hash(imu) == hash(
        make_imu(orientation_covariance=np.eye(3))
    )
    assert imu != make_imu()
    with pytest.raises(dataclasses.FrozenInstanceError):
        imu.orientation = Quaternion()


def test_imu_round_trip(tmp_path):
    round_trip(make_imu(), tmp_path)
    round_trip(make_imu(orientation_covariance=np.arange(9.0).reshape(3, 3)), tmp_path)
