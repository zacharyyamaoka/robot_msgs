"""robot_msgs.compare for every recorded stream, single messages and point clouds.

Known answers built by hand from each fixture (a constant offset → that rms; an image 5 levels
darker → mae 5; a cloud shifted 1 cm → 1 cm), with scipy as the outside oracle for rotations and
nearest neighbours. Sign convention as everywhere: error = reference - measured.
"""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from robot_msgs.compare import (
    Comparison,
    aggregate,
    check,
    compare,
    frechet,
    goal,
)
from robot_msgs.control import ServoCommand, ServoState
from robot_msgs.geometry import Quaternion, Transform, Vector3, Wrench
from robot_msgs.sensor import Image, Imu, JointState, PointCloud
from robot_msgs.std import Header
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    JointTrajectory,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    WrenchTrajectory,
)

NAMES = ("j1", "j2")
EPOCH = 1_761_600_000.25  # a real wall-clock stamp: errors must survive its float spacing
T = np.linspace(0.0, 1.0, 101)


def swing(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(t), 0.5 * np.cos(2.0 * t)])


def servo_states(positions: np.ndarray, **fields) -> ServoStateTrajectory:
    return ServoStateTrajectory(t=T, joint_names=NAMES, positions=positions, **fields)


# ── ServoStateTrajectory ─────────────────────────────────────────────────────


def test_a_constant_offset_on_every_servo_is_its_rms_and_the_error_keeps_the_type():
    reference = servo_states(
        swing(T), velocities=np.zeros((101, 2)), temperature=np.full((101, 2), 40.0),
        header=Header(stamp=EPOCH),
    )  # fmt: skip
    measured = reference.replace(
        positions=swing(T) + [0.1, -0.1], temperature=np.full((101, 2), 38.5)
    )
    result = compare(reference, measured)
    assert isinstance(result.error, ServoStateTrajectory)
    assert result.metrics["positions/j1"].rms == pytest.approx(0.1)
    assert result.metrics["positions/j1"].mean == pytest.approx(-0.1)  # reference - measured
    assert result.metrics["positions/j2"].mean == pytest.approx(0.1)
    assert result.metrics["temperature/j2"].mean == pytest.approx(1.5)
    assert result.metrics["velocities/j1"].rms == 0.0
    # only fields both sides carry are compared
    assert not any(name.startswith(("effort", "voltage", "fault")) for name in result.metrics)


def test_servo_states_are_matched_by_name_not_column():
    reference = servo_states(swing(T))
    measured = servo_states(swing(T) + [0.0, 0.2]).select(["j2", "j1"])
    result = compare(reference, measured)
    assert result.error.joint_names == ("j2", "j1")
    assert result.metrics["positions/j2"].mean == pytest.approx(-0.2)
    assert result.metrics["positions/j1"].rms == pytest.approx(0.0)
    with pytest.raises(ValueError, match="joints differ"):
        compare(reference, reference.select(["j1"]))


def test_fault_codes_are_copied_and_their_disagreement_is_a_fraction():
    codes = np.zeros((101, 2), dtype=np.int64)
    measured_codes = codes.copy()
    measured_codes[80:, 1] = 7  # j2 faults for the last 21 of 101 samples
    reference = servo_states(swing(T), fault=codes)
    result = compare(reference, servo_states(swing(T), fault=measured_codes))
    np.testing.assert_array_equal(result.error.fault, measured_codes)  # copied, never differenced
    assert result.error.fault.dtype == np.int64
    assert result.metrics["fault_mismatch"].mean == pytest.approx(21 / 101)
    assert result.metrics["fault_mismatch"].max == 1.0
    assert result.metrics["fault_mismatch/j1"].mean == 0.0
    assert result.metrics["fault_mismatch/j2"].n == 101
    assert "channels/fault_mismatch" not in result.metrics


def test_from_servo_states_reorders_by_name_keeps_seq_and_state_at_reads_back():
    states = [
        ServoState(
            name=NAMES if k % 2 == 0 else NAMES[::-1],
            position=[k, 10.0 * k] if k % 2 == 0 else [10.0 * k, k],
            fault=[0, k],
            seq=40 + k,
            header=Header(stamp=EPOCH + 0.01 * k, frame_id="rig"),
        )
        for k in range(4)
    ]
    traj = ServoStateTrajectory.from_servo_states(states)
    np.testing.assert_array_equal(traj.positions[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(traj.positions[:, 1], [0, 10, 20, 30])
    # fault [0, k] is in each message's own order: j2's code at even k, j1's at odd k
    np.testing.assert_array_equal(traj.fault, [[0, 0], [1, 0], [0, 2], [3, 0]])
    np.testing.assert_array_equal(traj.channels["seq"], [40, 41, 42, 43])
    assert traj.header == Header(stamp=EPOCH, frame_id="rig")
    state = traj.state_at(0.015)
    assert state.position.tolist() == pytest.approx([1.5, 15.0])
    assert state.fault.tolist() == [1, 0]  # held, never 0.5
    assert state.seq == 41
    assert state.header.stamp == pytest.approx(EPOCH + 0.015)
    assert traj.joint_trajectory().positions is traj.positions
    with pytest.raises(ValueError, match="differ from the first"):
        ServoStateTrajectory.from_servo_states(
            [states[0], states[1].replace(name=("j1", "j9"))]
        )


# ── ServoCommandTrajectory ───────────────────────────────────────────────────


def test_every_command_field_is_compared_per_servo_and_no_target_is_skipped():
    positions = swing(T)
    positions[:, 1] = np.nan  # j2 runs velocity-only: no position target
    reference = ServoCommandTrajectory(
        t=T, joint_names=NAMES, positions=positions, velocities=np.ones((101, 2)),
        stiffness=np.full((101, 2), 30.0), channels={"latency": np.full(101, 0.002)},
    )  # fmt: skip
    measured = reference.replace(
        positions=positions - 0.05, stiffness=np.full((101, 2), 25.0),
        channels={"latency": np.full(101, 0.005)},
    )  # fmt: skip
    result = compare(reference, measured)
    assert isinstance(result.error, ServoCommandTrajectory)
    assert result.metrics["positions/j1"].mean == pytest.approx(0.05)
    assert result.metrics["positions/j2"].n == 0  # NaN target: missing data, never 0
    assert result.metrics["stiffness/j2"].mean == pytest.approx(5.0)
    assert result.metrics["velocities/j1"].rms == 0.0
    assert result.metrics["channels/latency"].mean == pytest.approx(-0.003)


def test_from_servo_commands_records_latency_and_command_at_restores_it():
    commands = [
        ServoCommand(
            name=NAMES,
            position=[0.1 * k, 0.0],
            state_stamp=None if k == 0 else EPOCH + 0.01 * k - 0.002,
            header=Header(stamp=EPOCH + 0.01 * k),
        )
        for k in range(3)
    ]
    traj = ServoCommandTrajectory.from_servo_commands(commands)
    assert np.isnan(traj.channels["latency"][0])
    assert traj.channels["latency"][1:] == pytest.approx([0.002, 0.002], abs=1e-6)
    command = traj.command_at(0.015)
    assert command.latency == pytest.approx(0.002, abs=1e-6)
    assert ServoCommandTrajectory.from_servo_commands(commands[:1]).channels == {}


def test_servo_command_modes_round_trip_through_a_recorded_stream():
    """WHY: before the mode column a recorded stop or brake came back as a position command."""
    modes = [None, ("stop", "position"), ("position", "brake"), None]
    commands = [
        ServoCommand(
            name=NAMES, position=[0.1 * k, 0.0], mode=mode, header=Header(stamp=EPOCH + 0.01 * k)
        )
        for k, mode in enumerate(modes)
    ]
    traj = ServoCommandTrajectory.from_servo_commands(commands)
    assert traj.mode.tolist() == [[0, 0], [1, 0], [0, 2], [0, 0]]
    assert traj.command_at(0.01).mode == ("stop", "position")
    assert traj.command_at(0.025).mode == ("position", "brake")  # held, never interpolated
    assert traj.select(["j2"]).command_at(0.02).mode == ("brake",)
    reordered = commands[1].replace(name=NAMES[::-1], position=[0.0, 0.1], mode=("position", "stop"))
    assert ServoCommandTrajectory.from_servo_commands([commands[0], reordered]).mode[1].tolist() == [1, 0]
    assert ServoCommandTrajectory.from_servo_commands([commands[0]]).mode is None  # absent = position
    assert ServoCommandTrajectory.from_servo_commands([commands[0]]).command_at(0.0).mode is None
    assert compare(traj, traj).error.mode.tolist() == traj.mode.tolist()
    with pytest.raises(ValueError, match="SERVO_MODES"):
        traj.replace(mode=np.full((4, 2), 3))


# ── ImuTrajectory ────────────────────────────────────────────────────────────


def test_imu_orientation_error_is_a_rotation_vector_and_rates_are_per_axis():
    yaw = 0.3 * T
    reference = ImuTrajectory(
        t=T,
        orientations=Rotation.from_euler("z", yaw[:, None]).as_quat(),
        angular_velocities=np.tile([0.0, 0.0, 0.3], (101, 1)),
        linear_accelerations=np.tile([0.0, 0.0, 9.81], (101, 1)),
    )
    measured = reference.replace(
        orientations=Rotation.from_euler("z", (yaw + 0.1)[:, None]).as_quat(),  # 0.1 rad ahead in yaw
        angular_velocities=np.tile([0.02, 0.0, 0.3], (101, 1)),
        linear_accelerations=np.tile([0.0, 0.0, 9.71], (101, 1)),
    )
    result = compare(reference, measured)
    assert isinstance(result.error, ImuTrajectory)
    oracle = (
        Rotation.from_quat(reference.orientations) * Rotation.from_quat(measured.orientations).inv()
    ).as_rotvec()
    np.testing.assert_allclose(result.error.channels["rotation_vector"], oracle, atol=1e-12)
    assert result.metrics["rotation"].rms == pytest.approx(0.1)
    assert result.metrics["rz"].mean == pytest.approx(-0.1)
    assert result.metrics["rx"].max_abs == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["angular_velocities/x"].mean == pytest.approx(-0.02)
    assert result.metrics["angular_velocity"].rms == pytest.approx(0.02)
    assert result.metrics["linear_accelerations/z"].mean == pytest.approx(0.1)
    assert result.metrics["linear_acceleration"].rms == pytest.approx(0.1)


def test_imu_stream_from_messages_and_back():
    imus = [
        Imu(
            orientation=Quaternion.from_euler(0.0, 0.0, 0.1 * k),
            angular_velocity=Vector3(0, 0, 0.1),
            linear_acceleration=Vector3(0, 0, 9.81),
            orientation_covariance=np.eye(3),
            header=Header(stamp=EPOCH + 0.1 * k, frame_id="imu"),
        )
        for k in range(3)
    ]
    traj = ImuTrajectory.from_imus(imus)
    middle = traj.imu_at(0.15)
    assert middle.orientation.to_euler()[2] == pytest.approx(0.15)  # slerped
    assert middle.header.frame_id == "imu"
    assert middle.orientation_covariance is None  # covariances describe the sensor: dropped


# ── WrenchTrajectory ─────────────────────────────────────────────────────────


def test_wrench_error_is_per_axis_and_magnitude():
    reference = WrenchTrajectory.from_wrenches(
        T, [Wrench(Vector3(0, 0, -10.0), Vector3(0.1, 0, 0))] * 101
    )
    measured = WrenchTrajectory.from_wrenches(
        T, [Wrench(Vector3(0, 0, -12.0), Vector3(0.1, 0.03, 0))] * 101
    )
    result = compare(reference, measured)
    assert isinstance(result.error, WrenchTrajectory)
    assert result.metrics["forces/z"].mean == pytest.approx(2.0)
    assert result.metrics["force"].rms == pytest.approx(2.0)
    assert result.metrics["torques/y"].mean == pytest.approx(-0.03)
    assert result.metrics["torque"].rms == pytest.approx(0.03)
    assert reference.wrench_at(0.5) == Wrench(Vector3(0, 0, -10.0), Vector3(0.1, 0, 0))


# ── ImageTrajectory ──────────────────────────────────────────────────────────


def frames(value: int, count: int = 3) -> np.ndarray:
    return np.full((count, 6, 8, 3), value, dtype=np.uint8)


def test_an_image_stream_five_levels_darker_has_mae_and_rmse_five():
    reference = ImageTrajectory(t=[0.0, 0.1, 0.2], encoding="rgb8", frames=frames(100))
    result = compare(reference, reference.replace(frames=frames(95)))
    assert isinstance(result.error, ImageTrajectory)
    assert result.error.encoding == "32FC3"
    assert result.error.frames.dtype == np.float32
    assert result.metrics["pixels"].mean == pytest.approx(5.0)
    assert result.metrics["pixels"].n == 3 * 6 * 8 * 3
    assert math.isnan(result.metrics["pixels"].iae)  # a pixel is not a sample in time
    assert result.metrics["mae"].mean == pytest.approx(5.0)
    assert result.metrics["rmse"].max == pytest.approx(5.0)
    assert result.metrics["psnr"].mean == pytest.approx(10 * np.log10(255**2 / 25))


def test_the_error_is_signed_and_never_wraps_around_uint8():
    reference = ImageTrajectory(t=[0.0], encoding="mono8", frames=np.full((1, 2, 2), 3, np.uint8))
    measured = reference.replace(frames=np.full((1, 2, 2), 250, np.uint8))
    result = compare(reference, measured)
    assert result.error.encoding == "32FC1"
    assert result.error.frames.min() == -247.0


def test_the_reference_frame_is_held_and_an_epoch_stamp_does_not_slip_a_frame():
    # 10 Hz reference, 30 Hz measured, both on a wall clock: each measured frame is compared with
    # the latest reference frame at or before it, and one stamped exactly on a reference frame
    # uses that frame (not the previous one, a few float ulps earlier).
    reference = ImageTrajectory(
        t=[0.0, 0.1, 0.2], encoding="mono8",
        frames=np.stack([np.full((2, 2), v, np.uint8) for v in (10, 20, 30)]),
        header=Header(stamp=EPOCH),
    )  # fmt: skip
    measured_t = np.array([0.0, 1, 2, 3, 4, 5, 6]) / 30.0
    measured = ImageTrajectory(
        t=measured_t + EPOCH - (EPOCH - 5.0),  # its own stamp, same instants
        encoding="mono8",
        frames=np.zeros((7, 2, 2), np.uint8),
        header=Header(stamp=EPOCH - 5.0),
    )
    result = compare(reference, measured)
    held = result.error.frames[:, 0, 0]
    assert held.tolist() == [10, 10, 10, 20, 20, 20, 30]


def test_frames_of_another_shape_or_encoding_raise():
    reference = ImageTrajectory(t=[0.0], encoding="rgb8", frames=frames(1, 1))
    with pytest.raises(ValueError, match="shape"):
        compare(reference, reference.replace(frames=np.zeros((1, 6, 9, 3), np.uint8)))
    with pytest.raises(ValueError, match="encoding"):
        compare(reference, reference.replace(encoding="bgr8"))


def test_depth_in_millimetres_is_compared_in_metres():
    reference = ImageTrajectory(t=[0.0], encoding="16UC1", frames=np.full((1, 2, 2), 1000, np.uint16))
    result = compare(reference, reference.replace(frames=np.full((1, 2, 2), 990, np.uint16)))
    assert result.error.encoding == "32FC1"
    assert result.metrics["pixels"].mean == pytest.approx(0.01)
    assert "psnr" not in result.metrics


def test_image_at_returns_the_held_frame_with_its_own_stamp():
    images = [
        Image(data=np.full((2, 3), k, np.uint8), encoding="mono8", header=Header(stamp=EPOCH + k))
        for k in range(3)
    ]
    traj = ImageTrajectory.from_images(images)
    shown = traj.image_at(1.6)
    assert shown.data[0, 0] == 1
    assert shown.header.stamp == EPOCH + 1.0  # taken at 1.0, not restamped to 1.6
    assert traj[2].image == images[2]
    assert ImageTrajectory.from_points(list(traj)) == traj


# ── single messages: values only ─────────────────────────────────────────────


def test_two_joint_states_compare_by_value_whatever_their_stamps():
    reference = JointState(name=NAMES, position=[0.1, 0.2], header=Header(stamp=EPOCH))
    measured = JointState(name=NAMES[::-1], position=[0.5, 0.0], header=Header(stamp=3.0))
    result = reference.compare(measured)
    assert isinstance(result.error, JointTrajectory) and len(result.error) == 1
    assert result.metrics["positions/j1"].final == pytest.approx(0.1)
    assert result.metrics["positions/j2"].final == pytest.approx(-0.3)


@pytest.mark.parametrize(
    ("reference", "measured", "metric", "value"),
    [
        (
            ServoState(name=NAMES, position=[1.0, 2.0], header=Header(stamp=1.0)),
            ServoState(name=NAMES, position=[0.9, 2.0], header=Header(stamp=99.0)),
            "positions/j1",
            0.1,
        ),
        (
            ServoCommand(name=NAMES, position=[1.0, 2.0], state_stamp=0.998, header=Header(1.0)),
            ServoCommand(name=NAMES, position=[1.0, 2.0], state_stamp=4.995, header=Header(5.0)),
            "channels/latency",
            -0.003,
        ),
        (
            Imu(orientation=Quaternion(), angular_velocity=Vector3(0, 0, 1.0),
                linear_acceleration=Vector3(0, 0, 9.8)),
            Imu(orientation=Quaternion.from_euler(0, 0, 0.2), angular_velocity=Vector3(0, 0, 1.0),
                linear_acceleration=Vector3(0, 0, 9.8), header=Header(stamp=7.0)),
            "rz",
            -0.2,
        ),
        (Wrench(Vector3(0, 0, 5.0)), Wrench(Vector3(0, 0, 4.0)), "forces/z", 1.0),
        (
            Image(data=np.full((2, 2), 9, np.uint8), encoding="mono8"),
            Image(data=np.full((2, 2), 4, np.uint8), encoding="mono8", header=Header(stamp=2.0)),
            "pixels",
            5.0,
        ),
    ],
    ids=["ServoState", "ServoCommand", "Imu", "Wrench", "Image"],
)
def test_single_messages_compare_as_one_sample_streams(reference, measured, metric, value):
    result = reference.compare(measured)
    assert isinstance(result, Comparison) and len(result.error) == 1
    assert result.metrics[metric].final == pytest.approx(value, abs=1e-9)


def test_two_different_message_types_refuse_to_compare():
    with pytest.raises(TypeError, match="both must be"):
        compare(JointState(name=NAMES, position=[0, 0]), ServoState(name=NAMES, position=[0, 0]))
    with pytest.raises(TypeError, match="both must be"):
        compare(ImuTrajectory(t=[0.0], orientations=[[0, 0, 0, 1.0]],
                              angular_velocities=[[0, 0, 0.0]], linear_accelerations=[[0, 0, 0.0]]),
                WrenchTrajectory(t=[0.0], forces=[[0, 0, 0.0]], torques=[[0, 0, 0.0]]))  # fmt: skip


# ── point clouds: nearest neighbours ─────────────────────────────────────────


def grid_cloud(**kwargs) -> PointCloud:
    axis = np.arange(0.0, 1.0, 0.1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return PointCloud(points=points, **kwargs)


def test_a_cloud_shifted_one_centimetre_is_one_centimetre_away_both_ways():
    reference = grid_cloud()
    measured = reference.transformed(Transform(Vector3(0.01, 0.0, 0.0)))
    result = reference.compare(measured)
    assert isinstance(result.error, PointCloud)
    assert result.error.points is measured.points or np.array_equal(
        result.error.points, measured.points
    )
    for name in ("distance", "distance_reverse", "chamfer"):
        assert result.metrics[name].mean == pytest.approx(0.01, abs=1e-6)
        assert result.metrics[name].max == pytest.approx(0.01, abs=1e-6)
    assert result.metrics["chamfer"].n == 2 * len(reference)
    np.testing.assert_allclose(result.error.channels["distance"], 0.01, atol=1e-6)


def test_cloud_distances_match_a_kd_tree():
    rng = np.random.default_rng(7)
    reference = PointCloud(points=rng.normal(size=(1500, 3)))
    measured = PointCloud(points=rng.normal(size=(900, 3)) * 1.1)
    result = compare(reference, measured)
    forward, _ = cKDTree(reference.points.astype(np.float64)).query(measured.points)
    reverse, _ = cKDTree(measured.points.astype(np.float64)).query(reference.points)
    np.testing.assert_allclose(result.error.channels["distance"], forward, atol=1e-9)
    assert result.metrics["distance_reverse"].rms == pytest.approx(np.sqrt(np.mean(reverse**2)))
    assert result.metrics["chamfer"].max == pytest.approx(max(forward.max(), reverse.max()))


def test_a_partial_scan_is_close_one_way_and_far_the_other():
    reference = grid_cloud()
    measured = PointCloud(points=reference.points[reference.points[:, 0] < 0.45])
    result = compare(reference, measured)
    assert result.metrics["distance"].max == 0.0  # every measured point lies on the reference
    assert result.metrics["distance_reverse"].max == pytest.approx(0.5)  # x = 0.9 → x = 0.4


def test_clouds_in_different_frames_or_empty_refuse():
    with pytest.raises(ValueError, match="different frames"):
        compare(grid_cloud(header=Header(frame_id="a")), grid_cloud(header=Header(frame_id="b")))
    with pytest.raises(ValueError, match="empty"):
        compare(grid_cloud(), PointCloud(points=np.zeros((0, 3))))
    with pytest.raises(ValueError, match="shadow"):
        compare(grid_cloud(), grid_cloud(channels={"distance": np.zeros(1000)}))


# ── Comparison holds a cloud error; the trajectory-only analyses refuse the new types ─


def test_comparison_takes_a_trajectory_or_a_cloud_and_nothing_else():
    with pytest.raises(TypeError, match="Trajectory"):
        Comparison(error=JointState(name=NAMES, position=[0, 0]))
    assert isinstance(Comparison(error=grid_cloud()).error, PointCloud)


def test_the_joint_and_pose_analyses_name_what_they_take():
    states = servo_states(swing(T))
    with pytest.raises(TypeError):
        aggregate(states, [states, states])
    with pytest.raises(TypeError):
        check(states, states, None)
    with pytest.raises(TypeError):
        goal(states, states, band=0.1)
    with pytest.raises(TypeError):
        frechet(states, states)
