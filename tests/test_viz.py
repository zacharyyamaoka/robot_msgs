"""robot_msgs.viz: every plot is checked against the data it was given, not just for existing."""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot: the suite never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from catalog import (  # noqa: E402
    BOX3D,
    CAMERA,
    DETECTION2D,
    DETECTION3D,
    EXAMPLES,
    HEADER,
    JOINT_TRAJ,
    NAMES,
    POSE,
    POSE_TRAJ,
)
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image as PilImage  # noqa: E402

import robot_msgs  # noqa: E402
from robot_msgs.geometry import PoseStamped, TransformStamped  # noqa: E402
from robot_msgs.geometry.rotations import quat_to_euler  # noqa: E402
from robot_msgs.sensor import Image, JointState, PointCloud  # noqa: E402
from robot_msgs.shape import MeshResource, Plane, SolidPrimitive  # noqa: E402
from robot_msgs.std import ColorRGBA, Header  # noqa: E402
from robot_msgs.trajectory import (  # noqa: E402
    JointLimits,
    JointTrajectory,
    PoseTrajectory,
    Trajectory,
    sample,
)
from robot_msgs.vision import Detection2DArray, Detection3DArray  # noqa: E402
from robot_msgs.viz import animate, plot  # noqa: E402
from robot_msgs.viz.movie import _build_scene  # noqa: E402

REPO = Path(robot_msgs.__file__).parents[1]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def has_plot(message) -> bool:
    return plot.dispatch(type(message)) is not plot.registry[object]


def lines(fig: Figure, prefix: str) -> list:
    """Every line whose gid starts with `prefix`, in axes then drawing order."""
    return [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if (line.get_gid() or "").startswith(prefix)
    ]


def png_size(fig: Figure, tmp_path: Path) -> int:
    target = tmp_path / "figure.png"
    fig.savefig(target, dpi=50)
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    return target.stat().st_size


# ── every catalog example ────────────────────────────────────────────────────

DRAWABLE = [example for example in EXAMPLES if has_plot(example)]


@pytest.mark.parametrize("message", DRAWABLE, ids=[type(m).__name__ for m in DRAWABLE])
def test_every_registered_plot_returns_a_figure_that_saves_to_png(message, tmp_path):
    fig = plot(message)
    assert isinstance(fig, Figure)
    assert png_size(fig, tmp_path) > 1000


def test_the_standard_types_all_have_a_plot():
    wanted = {
        "Trajectory",
        "JointTrajectory",
        "PoseTrajectory",
        "JointState",
        "Pose",
        "PoseStamped",
        "Transform",
        "TransformStamped",
        "Image",
        "Detection2DArray",
        "Detection3DArray",
        "BoundingBox3D",
        "PointCloud",
        "CameraInfo",
        "SolidPrimitive",
        "Mesh",
        "Plane",
        "ColorRGBA",
    }
    assert wanted <= {type(m).__name__ for m in DRAWABLE}


def test_a_type_without_a_plot_says_so():
    with pytest.raises(TypeError, match="no plot for Header"):
        plot(HEADER)


def test_unknown_options_are_rejected():
    with pytest.raises(TypeError):
        plot(POSE, colour="red")


# ── JointTrajectory ──────────────────────────────────────────────────────────


def test_joint_trajectory_lines_are_the_data_one_colour_per_joint():
    fig = plot(JOINT_TRAJ)
    for row, array in (("positions", JOINT_TRAJ.positions), ("velocities", JOINT_TRAJ.velocities)):
        drawn = lines(fig, row + ":")
        assert [line.get_gid() for line in drawn] == [f"{row}:{n}" for n in NAMES]
        for column, line in enumerate(drawn):
            np.testing.assert_array_equal(line.get_xdata(), JOINT_TRAJ.t)
            np.testing.assert_array_equal(line.get_ydata(), array[:, column])
    positions, velocities = lines(fig, "positions:"), lines(fig, "velocities:")
    assert [a.get_color() for a in positions] == [b.get_color() for b in velocities]
    assert len({line.get_color() for line in positions}) == len(NAMES)
    assert lines(fig, "accelerations:") == [] and lines(fig, "effort:") == []


def test_joint_names_appear_once_in_a_figure_legend_and_the_title_names_the_frame():
    fig = plot(JOINT_TRAJ)
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == list(NAMES)
    assert "base_link" in fig.legends[0].get_title().get_text()
    assert all(ax.get_legend() is None for ax in fig.axes if ax.get_ylabel().startswith("pos"))


def test_limits_are_dashed_lines_at_plus_minus_the_bound_in_the_joint_colour():
    fig = plot(JOINT_TRAJ)
    colors = {line.get_gid().split(":")[1]: line.get_color() for line in lines(fig, "positions:")}
    for name in NAMES:
        drawn = lines(fig, f"limit:max_velocity:{name}")
        assert sorted(float(line.get_ydata()[0]) for line in drawn) == [-2.0, 2.0]
        assert all(
            line.get_linestyle() != "-" and line.get_color() == colors[name] for line in drawn
        )
    assert lines(fig, "limit:max_position") == [] and lines(fig, "limit:max_acceleration") == []


def test_position_limits_are_drawn_at_their_values():
    limits = JointLimits(joint_names=NAMES, min_position=[-1.0, -2.0, -3.0], max_position=[1, 2, 3])
    fig = plot(JOINT_TRAJ.replace(limits=limits))
    for column, name in enumerate(NAMES):
        low = lines(fig, f"limit:min_position:{name}")[0].get_ydata()[0]
        high = lines(fig, f"limit:max_position:{name}")[0].get_ydata()[0]
        assert (low, high) == (limits.min_position[column], limits.max_position[column])


def test_a_limit_far_outside_the_data_stays_drawn_but_off_scale_and_named():
    far = JointLimits(joint_names=NAMES, max_position=[100.0] * 3)
    fig = plot(JOINT_TRAJ.replace(limits=far))
    assert len(lines(fig, "limit:max_position")) == 3
    ax = next(a for a in fig.axes if a.get_ylabel().startswith("position"))
    assert ax.get_ylim()[1] < 100.0
    assert any("max_position 100 rad off scale" in text.get_text() for text in ax.texts)


def test_channels_become_rows_floats_as_lines_ints_as_steps():
    fig = plot(JOINT_TRAJ)
    kp = lines(fig, "kp:")
    assert [line.get_color() for line in kp] == [
        line.get_color() for line in lines(fig, "positions:")
    ]
    np.testing.assert_array_equal(kp[1].get_ydata(), JOINT_TRAJ.channels["kp"][:, 1])
    (mode,) = lines(fig, "mode:")
    assert mode.get_drawstyle() == "steps-post"
    np.testing.assert_array_equal(mode.get_ydata(), JOINT_TRAJ.channels["mode"])


def test_joints_option_keeps_each_joints_colour():
    full = {line.get_gid(): line.get_color() for line in lines(plot(JOINT_TRAJ), "positions:")}
    subset = lines(plot(JOINT_TRAJ, joints=["j3", "j1"]), "positions:")
    assert [line.get_gid() for line in subset] == ["positions:j3", "positions:j1"]
    assert [line.get_color() for line in subset] == [full["positions:j3"], full["positions:j1"]]
    np.testing.assert_array_equal(subset[0].get_ydata(), JOINT_TRAJ.positions[:, 2])
    with pytest.raises(KeyError):
        plot(JOINT_TRAJ, joints=["nope"])


def test_waypoints_are_markers_on_the_position_row_labelled_by_name():
    waypoints = JOINT_TRAJ[0:11:5].with_channel("name", np.array(["start", "middle", "end"]))
    fig = plot(JOINT_TRAJ, waypoints=waypoints)
    for column, name in enumerate(NAMES):
        (marker,) = lines(fig, f"waypoints:{name}")
        assert marker.get_marker() == "o" and marker.get_linestyle() == "None"
        np.testing.assert_array_equal(marker.get_xdata(), waypoints.t)
        np.testing.assert_array_equal(marker.get_ydata(), waypoints.positions[:, column])
    fig.canvas.draw()
    top = next(a for a in fig.axes if a.get_ylabel().startswith("position")).child_axes[0]
    assert [label.get_text() for label in top.get_xticklabels()] == ["start", "middle", "end"]


def test_waypoints_without_names_are_markers_only():
    fig = plot(JOINT_TRAJ, waypoints=JOINT_TRAJ[::5])
    assert len(lines(fig, "waypoints:")) == 3
    assert all(not ax.child_axes for ax in fig.axes)


def test_joint_trajectory_subclass_uses_the_joint_plot():
    class Arm(JointTrajectory):
        pass

    assert plot.dispatch(Arm) is plot.dispatch(JointTrajectory)


# ── generic Trajectory and fallback dispatch ─────────────────────────────────


def test_generic_trajectory_has_one_row_per_numeric_array_columns_as_lines_ints_as_steps():
    trajectory = Trajectory(
        t=[0.0, 0.5, 1.0],
        channels={
            "temperature": [20.0, 20.5, 21.0],
            "pair": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
            "mode": [0, 1, 1],
            "label": np.array(["a", "b", "c"]),  # strings cannot be drawn against time
        },
    )
    fig = plot(trajectory)
    assert [ax.get_ylabel() for ax in fig.axes] == ["temperature", "pair", "mode"]
    (temperature,) = lines(fig, "temperature:")
    np.testing.assert_array_equal(temperature.get_ydata(), [20.0, 20.5, 21.0])
    pair = lines(fig, "pair:")
    assert len(pair) == 2
    np.testing.assert_array_equal(pair[1].get_ydata(), [2.0, 2.5, 3.0])
    assert lines(fig, "mode:")[0].get_drawstyle() == "steps-post"


def test_a_user_subclass_falls_back_to_the_generic_trajectory_plot():
    from dataclasses import dataclass

    from robot_msgs._core import Array

    @dataclass(frozen=True, eq=False, repr=False, kw_only=True)
    class GripperTrajectory(Trajectory):
        width: Array = sample()
        grasping: Array | None = sample(default=None, interp="hold", dtype=bool)

    trajectory = GripperTrajectory(
        t=[0.0, 1.0, 2.0], width=[0.08, 0.04, 0.0], grasping=[False, False, True]
    )
    assert plot.dispatch(GripperTrajectory) is plot.dispatch(Trajectory)
    fig = plot(trajectory)
    assert [ax.get_ylabel() for ax in fig.axes] == ["width", "grasping"]
    np.testing.assert_array_equal(lines(fig, "width:")[0].get_ydata(), [0.08, 0.04, 0.0])
    grasping = lines(fig, "grasping:")[0]
    assert grasping.get_drawstyle() == "steps-post"
    np.testing.assert_array_equal(grasping.get_ydata(), [0.0, 0.0, 1.0])
    assert "GripperTrajectory" in fig._suptitle.get_text()


def test_multi_panel_types_ignore_ax():
    _, ax = plt.subplots()
    fig = plot(JOINT_TRAJ, ax=ax)
    assert fig is not ax.figure


# ── PoseTrajectory ───────────────────────────────────────────────────────────


def helix(count: int = 41) -> PoseTrajectory:
    t = np.linspace(0.0, 4.0, count)
    yaw = np.linspace(0.0, 5.0, count)  # sweeps past +pi: the plotted yaw must not jump
    orientations = np.stack(
        [np.zeros(count), np.zeros(count), np.sin(yaw / 2), np.cos(yaw / 2)], axis=1
    )
    positions = np.stack([0.3 * np.cos(t), 0.3 * np.sin(t), 0.1 * t], axis=1)
    return PoseTrajectory(
        t=t,
        positions=positions,
        orientations=orientations,
        child_frame_id="tool0",
        header=Header(frame_id="base_link"),
    )


def test_pose_trajectory_path_triads_and_series_match_the_data():
    trajectory = helix()
    fig = plot(trajectory)
    ax3d = next(ax for ax in fig.axes if ax.name == "3d")
    (path,) = [c for c in ax3d.collections if c.get_gid() == "path"]
    segments = np.asarray(path._segments3d)  # get_segments() is the 2-D projection
    np.testing.assert_allclose(segments[:, 0], trajectory.positions[:-1])
    np.testing.assert_allclose(segments[:, 1], trajectory.positions[1:])
    np.testing.assert_allclose(path.get_array(), 0.5 * (trajectory.t[:-1] + trajectory.t[1:]))
    x_axes = lines(fig, "triad:x")
    assert len(x_axes) == 10
    rotations = trajectory.matrices()[:, :3, :3]
    indices = np.unique(np.linspace(0, len(trajectory) - 1, 10).round().astype(int))
    for line, index in zip(x_axes, indices, strict=True):
        start, end = (np.array(a) for a in zip(*[c for c in line.get_data_3d()], strict=True))
        np.testing.assert_allclose(start, trajectory.positions[index])
        direction = (end - start) / np.linalg.norm(end - start)
        np.testing.assert_allclose(direction, rotations[index][:, 0], atol=1e-12)
    for axis, name in enumerate("xyz"):
        np.testing.assert_array_equal(
            lines(fig, f"position:{name}")[0].get_ydata(), trajectory.positions[:, axis]
        )
    colors = [lines(fig, f"triad:{name}")[0].get_color() for name in "xyz"]
    assert colors[0] != colors[1] != colors[2]


def test_roll_pitch_yaw_are_unwrapped():
    trajectory = helix()
    fig = plot(trajectory)
    yaw = lines(fig, "rpy:yaw")[0].get_ydata()
    raw = quat_to_euler(trajectory.orientations)[:, 2]
    assert np.abs(np.diff(raw)).max() > np.pi  # the input really does wrap
    np.testing.assert_allclose(yaw, np.unwrap(raw))
    assert np.abs(np.diff(yaw)).max() < 0.5 and yaw[-1] == pytest.approx(5.0)


def test_pose_trajectory_axes_are_equal_scale_and_the_title_names_both_frames():
    fig = plot(helix())
    ax3d = next(ax for ax in fig.axes if ax.name == "3d")
    spans = np.array([np.ptp(ax3d.get_xlim()), np.ptp(ax3d.get_ylim()), np.ptp(ax3d.get_zlim())])
    np.testing.assert_allclose(
        np.array(ax3d.get_box_aspect()[:3]) / spans,
        np.full(3, (ax3d.get_box_aspect()[:3] / spans)[0]),
        rtol=1e-9,
    )
    assert "tool0 in base_link" in fig._suptitle.get_text()


def test_single_sample_pose_trajectory_still_plots():
    assert isinstance(plot(helix()[:1]), Figure)


# ── JointState ───────────────────────────────────────────────────────────────


def test_joint_state_bars_are_the_positions_labelled_by_joint():
    state = JointState(name=("a", "b", "c"), position=[0.5, -1.0, 2.0], header=HEADER)
    fig = plot(state)
    ax = fig.axes[0]
    bars = [p for p in ax.patches if p.get_gid() == "position"]
    np.testing.assert_allclose([bar.get_width() for bar in bars], state.position)
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b", "c"]
    assert ax.get_xlabel() == "position (rad)" and "camera_color_optical_frame" in ax.get_title(
        "left"
    )


def test_joint_state_limits_shade_each_joints_range():
    state = JointState(name=("a", "b"), position=[0.5, -1.0])
    limits = JointLimits(joint_names=("b", "a"), min_position=[-2.0, -3.0], max_position=[2.0, 4.0])
    ax = plot(state, limits=limits).axes[0]
    ranges = [
        (p.get_x(), p.get_x() + p.get_width()) for p in ax.patches if p.get_gid() == "limit_range"
    ]
    assert ranges == [(-3.0, 4.0), (-2.0, 2.0)]  # a then b, matched by name


def test_joint_state_draws_into_a_given_axes():
    _, ax = plt.subplots()
    assert plot(JointState(name=("a",), position=[1.0]), ax=ax) is ax.figure


# ── Pose, PoseStamped, Transform, TransformStamped ───────────────────────────


@pytest.mark.parametrize(
    "message, position, rotation, text",
    [
        (POSE, POSE.position.as_array(), POSE.orientation.to_matrix(), "Pose"),
        (
            PoseStamped(HEADER, POSE),
            POSE.position.as_array(),
            POSE.orientation.to_matrix(),
            HEADER.frame_id,
        ),
        (POSE.as_transform(), POSE.position.as_array(), POSE.orientation.to_matrix(), "Transform"),
        (
            TransformStamped(HEADER, "tool0", POSE.as_transform()),
            POSE.position.as_array(),
            POSE.orientation.to_matrix(),
            "tool0 in " + HEADER.frame_id,
        ),
    ],
    ids=["Pose", "PoseStamped", "Transform", "TransformStamped"],
)
def test_frames_are_triads_at_the_pose_with_the_right_axes(message, position, rotation, text):
    fig = plot(message)
    ax = fig.axes[0]
    assert text in ax.get_title("left")
    length = max(0.05, 0.3 * np.linalg.norm(position))
    for axis, name in enumerate("xyz"):
        (line,) = lines(fig, f"triad:{name}")
        start, end = np.array(line.get_data_3d()).T
        np.testing.assert_allclose(start, position)
        np.testing.assert_allclose(end - start, rotation[:, axis] * length, atol=1e-12)
    (origin_x,) = lines(fig, "parent:x")
    np.testing.assert_allclose(np.array(origin_x.get_data_3d()).T[0], np.zeros(3))
    if "tool0" in text:
        assert any("tool0" in t.get_text() for t in ax.texts)


def test_a_3d_plot_needs_a_3d_axes():
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="3-D axes"):
        plot(POSE, ax=ax)
    figure = plt.figure()
    assert plot(POSE, ax=figure.add_subplot(projection="3d")) is figure


# ── Image, Detection2DArray, CameraInfo ──────────────────────────────────────


def test_image_axes_array_equals_to_rgb_and_the_title_has_encoding_and_size():
    bgr = Image.from_array(np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3), "bgr8")
    ax = plot(bgr).axes[0]
    np.testing.assert_array_equal(ax.images[0].get_array(), bgr.to_rgb())
    assert "bgr8 3×2" in ax.get_title("left")
    depth = Image.from_array(np.arange(12, dtype=np.uint16).reshape(3, 4))
    np.testing.assert_array_equal(plot(depth).axes[0].images[0].get_array(), depth.to_rgb())


def test_image_with_detections_draws_boxes_at_to_xyxy_with_class_score_labels():
    image = next(m for m in EXAMPLES if isinstance(m, Image))
    detections = next(m for m in EXAMPLES if isinstance(m, Detection2DArray))
    ax = plot(image, detections=detections).axes[0]
    boxes = [p for p in ax.patches if p.get_gid() == "box"]
    assert len(boxes) == len(detections)
    for patch, detection in zip(boxes, detections, strict=True):
        x0, y0, x1, y1 = detection.bbox.to_xyxy()
        assert (
            patch.get_x(),
            patch.get_y(),
            patch.get_x() + patch.get_width(),
            patch.get_y() + patch.get_height(),
        ) == pytest.approx((x0, y0, x1, y1))
    assert {t.get_text() for t in ax.texts} == {"bottle 0.91"}
    assert len(ax.images) == 1


def test_detection_array_draws_boxes_with_or_without_a_background_image():
    detections = Detection2DArray(header=HEADER, detections=(DETECTION2D,))
    ax = plot(detections).axes[0]
    (patch,) = [p for p in ax.patches if p.get_gid() == "box"]
    assert (patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()) == (
        10,
        20,
        100,
        200,
    )
    assert len(ax.images) == 0 and ax.yaxis_inverted()
    image = Image.from_array(np.zeros((240, 320, 3), dtype=np.uint8))
    assert len(plot(detections, image=image).axes[0].images) == 1
    assert plot(DETECTION2D).axes[0].patches[0].get_gid() == "box"


def test_rotated_boxes_keep_their_angle():
    from robot_msgs.geometry import Pose2D
    from robot_msgs.vision import BoundingBox2D, Detection2D

    tilted = Detection2D(bbox=BoundingBox2D(Pose2D(50.0, 40.0, np.pi / 6), 40.0, 20.0))
    (patch,) = [p for p in plot(tilted).axes[0].patches if p.get_gid() == "box"]
    assert patch.angle == pytest.approx(30.0)
    np.testing.assert_allclose(patch.get_corners().mean(axis=0), [50.0, 40.0])


def test_camera_info_marks_the_rectangle_and_principal_point():
    fig = plot(CAMERA)
    ax = fig.axes[0]
    (rectangle,) = [p for p in ax.patches if p.get_gid() == "image_rect"]
    assert (rectangle.get_width(), rectangle.get_height()) == (CAMERA.width, CAMERA.height)
    (point,) = lines(fig, "principal_point")
    assert (point.get_xdata()[0], point.get_ydata()[0]) == (CAMERA.cx, CAMERA.cy)
    assert "fx=615" in ax.get_title("left") and "fy=615" in ax.get_title("left")


# ── 3-D boxes ────────────────────────────────────────────────────────────────


def wireframe_vertices(collection) -> np.ndarray:
    segments = np.asarray(collection._segments3d)  # get_segments() is the 2-D projection
    assert segments.shape == (12, 2, 3)
    return np.unique(segments.reshape(-1, 3).round(9), axis=0)


def test_bounding_box_3d_is_a_wireframe_of_its_corners():
    ax = plot(BOX3D).axes[0]
    (wireframe,) = [c for c in ax.collections if c.get_gid() == "box3d"]
    np.testing.assert_allclose(
        wireframe_vertices(wireframe), np.unique(BOX3D.corners().round(9), axis=0)
    )


def test_detection_3d_array_has_one_labelled_wireframe_per_detection():
    other = DETECTION3D.replace(
        id="8", bbox=BOX3D.__class__(center=POSE.shifted(0.3, 0.0, 0.0), size=BOX3D.size)
    )
    detections = Detection3DArray(header=HEADER, detections=(DETECTION3D, other))
    ax = plot(detections).axes[0]
    wireframes = [c for c in ax.collections if c.get_gid() == "box3d"]
    assert len(wireframes) == 2
    for wireframe, detection in zip(wireframes, detections, strict=True):
        np.testing.assert_allclose(
            wireframe_vertices(wireframe), np.unique(detection.bbox.corners().round(9), axis=0)
        )
    assert [t.get_text() for t in ax.texts] == ["bottle 0.91", "bottle 0.91"]
    assert "2 detections" in ax.get_title("left") and HEADER.frame_id in ax.get_title("left")
    assert (
        wireframes[0].get_edgecolor().tolist() == wireframes[1].get_edgecolor().tolist()
    )  # same class


# ── PointCloud ───────────────────────────────────────────────────────────────


def scatter_of(fig: Figure):
    (scatter,) = [c for ax in fig.axes for c in ax.collections if c.get_gid() == "points"]
    return scatter, np.array(scatter._offsets3d).T  # 3-D offsets have no public getter


def test_point_cloud_colours_by_its_rgb_channel():
    cloud = next(m for m in EXAMPLES if isinstance(m, PointCloud))
    scatter, points = scatter_of(plot(cloud))
    np.testing.assert_allclose(points, cloud.points.astype(float))
    colors = np.asarray(scatter._facecolors)  # get_facecolor() on a 3-D scatter is depth-processed
    np.testing.assert_allclose(colors[:, :3], cloud.channels["rgb"] / 255.0)


def test_point_cloud_without_rgb_is_coloured_by_z():
    cloud = PointCloud(points=np.random.default_rng(1).normal(size=(50, 3)))
    fig = plot(cloud)
    scatter, points = scatter_of(fig)
    np.testing.assert_allclose(scatter.get_array(), cloud.points[:, 2])
    assert any(ax.get_ylabel() == "z (m)" for ax in fig.axes)  # the colour bar


def test_big_clouds_are_subsampled_to_at_most_20k_points_repeatably():
    points = np.random.default_rng(2).normal(size=(60_000, 3)).astype(np.float32)
    cloud = PointCloud(points=points, header=Header(frame_id="base_link"))
    fig = plot(cloud)
    _, first = scatter_of(fig)
    assert len(first) <= 20_000
    _, second = scatter_of(plot(cloud))
    np.testing.assert_array_equal(first, second)
    assert set(map(tuple, first.astype(np.float32).tolist())) <= set(map(tuple, points.tolist()))
    assert "of 60,000 points" in fig.axes[0].get_title("left")
    _, fewer = scatter_of(plot(cloud, max_points=100))
    assert len(fewer) == 100


def test_empty_point_cloud_plots():
    assert isinstance(plot(PointCloud(points=np.zeros((0, 3)))), Figure)


# ── animate ──────────────────────────────────────────────────────────────────


def frame_count(path: Path) -> int:
    with PilImage.open(path) as gif:
        return gif.n_frames


@pytest.mark.parametrize(
    "trajectories",
    [(JOINT_TRAJ,), (POSE_TRAJ,), (JOINT_TRAJ, POSE_TRAJ), (POSE_TRAJ, JOINT_TRAJ)],
    ids=["joints", "pose", "joints+pose", "pose+joints"],
)
def test_animate_writes_a_gif_with_round_duration_times_fps_frames(trajectories, tmp_path):
    target = tmp_path / "out" / "motion.gif"
    assert animate(*trajectories, path=target, fps=4, duration=1.5, dpi=30) == target
    assert frame_count(target) == 6


def test_animate_defaults_to_real_time():
    scene = _build_scene(JOINT_TRAJ, None, dpi=30)
    assert (scene.start, scene.end) == (0.0, 1.0)


def test_animate_writes_an_mp4(tmp_path):
    target = tmp_path / "motion.mp4"
    animate(JOINT_TRAJ, POSE_TRAJ, path=target, fps=4, duration=1.0, dpi=31)  # odd size: padded
    data = target.read_bytes()
    assert len(data) > 1000 and data[4:8] == b"ftyp"


def test_the_cursor_markers_and_triad_follow_at_t():
    scene = _build_scene(JOINT_TRAJ, POSE_TRAJ, dpi=30)
    scene.update(0.37)
    point = JOINT_TRAJ.at(0.37)
    all_lines = [line for ax in scene.figure.axes for line in ax.get_lines()]

    def named(prefix):
        return [line for line in all_lines if (line.get_gid() or "").startswith(prefix)]

    cursors = named("cursor")
    assert len(cursors) == 2 and all(list(c.get_xdata()) == [0.37, 0.37] for c in cursors)
    for column, name in enumerate(NAMES):
        (marker,) = named(f"marker:positions:{name}")
        assert (marker.get_xdata()[0], marker.get_ydata()[0]) == (0.37, point.positions[column])
        (marker,) = named(f"marker:velocities:{name}")
        assert marker.get_ydata()[0] == point.velocities[column]
    pose = POSE_TRAJ.at(0.37)
    rotation = robot_msgs.geometry.Quaternion.from_array(pose.orientation).to_matrix()
    for axis, name in enumerate("xyz"):
        (triad,) = named(f"moving:{name}")
        start, end = np.array(triad.get_data_3d()).T
        np.testing.assert_allclose(start, pose.position)
        np.testing.assert_allclose(
            (end - start) / np.linalg.norm(end - start), rotation[:, axis], atol=1e-12
        )
    ax3d = next(ax for ax in scene.figure.axes if ax.name == "3d")
    assert "0.37" in next(t for t in ax3d.texts if t.get_gid() == "clock").get_text()
    scene.update(0.9)  # moving on again moves everything on again
    assert all(list(c.get_xdata()) == [0.9, 0.9] for c in cursors)


def test_animate_rejects_what_it_cannot_draw(tmp_path):
    with pytest.raises(TypeError, match="JointTrajectory, a PoseTrajectory, or one of each"):
        animate(Trajectory(t=[0.0, 1.0]), path=tmp_path / "x.gif")
    with pytest.raises(TypeError):
        animate(JOINT_TRAJ, JOINT_TRAJ, path=tmp_path / "x.gif")
    with pytest.raises(TypeError):
        animate(path=tmp_path / "x.gif")
    with pytest.raises(ValueError, match="same span"):
        animate(JOINT_TRAJ, POSE_TRAJ.time_shifted(5.0), path=tmp_path / "x.gif")
    with pytest.raises(ValueError, match=r"\.mp4 or \.gif"):
        animate(JOINT_TRAJ, path=tmp_path / "x.avi")
    with pytest.raises(ValueError, match="two samples"):
        animate(JOINT_TRAJ[:1], path=tmp_path / "x.gif")


# ── the optional dependency ──────────────────────────────────────────────────


def test_importing_viz_without_matplotlib_names_the_extra(monkeypatch):
    for name in [
        m for m in sys.modules if m == "robot_msgs.viz" or m.startswith("robot_msgs.viz.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # None makes `import matplotlib` fail
    with pytest.raises(
        ImportError,
        match=re.escape("robot_msgs.viz needs matplotlib: pip install 'robot-msgs[viz]'"),
    ):
        importlib.import_module("robot_msgs.viz")


def test_importing_robot_msgs_never_imports_viz_or_matplotlib():
    probe = (
        "import sys, robot_msgs; "
        "print(sorted(m for m in sys.modules if m.split('.')[:2] == ['robot_msgs', 'viz'] "
        "or m.split('.')[0] == 'matplotlib'))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ── shapes ───────────────────────────────────────────────────────────────────


def faces_of(collection) -> np.ndarray:
    return np.asarray(collection._faces)  # (M, 3, 3): the triangles in 3-D, not their projection


def test_mesh_is_one_polygon_per_triangle_at_the_vertices():
    mesh = SolidPrimitive.cone(0.5, 0.2).to_mesh(12)
    ax = plot(mesh).axes[0]
    (collection,) = [c for c in ax.collections if c.get_gid() == "mesh"]
    np.testing.assert_allclose(faces_of(collection), mesh.vertices[mesh.triangles])
    assert f"{len(mesh)} triangles" in ax.get_title("left")


def test_solid_primitive_draws_its_mesh_and_names_its_type():
    solid = SolidPrimitive.box(0.3, 0.2, 0.5)
    ax = plot(solid).axes[0]
    (collection,) = [c for c in ax.collections if c.get_gid() == "mesh"]
    np.testing.assert_allclose(
        faces_of(collection), solid.to_mesh().vertices[solid.to_mesh().triangles]
    )
    assert "BOX" in ax.get_title("left")
    # equal axes: the longest dimension spans the longest axis, the same metre on every axis
    assert max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_zlim()[1] - ax.get_zlim()[0]) > 0.5


def test_plane_is_a_patch_in_the_plane_with_its_normal_arrow():
    plane = Plane.from_point_normal([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
    ax = plot(plane).axes[0]
    (patch,) = [c for c in ax.collections if c.get_gid() == "plane"]
    corners = faces_of(patch).reshape(-1, 3)
    assert plane.distance(corners) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(corners[:, 0]) > 0 and np.ptp(corners[:, 1]) > 0
    (arrow,) = lines(plot(plane), "normal")
    start, end = np.asarray(arrow._verts3d).T
    assert (end - start) / np.linalg.norm(end - start) == pytest.approx([0, 0, 1])
    assert np.linalg.norm(end - start) > 0


def test_color_is_a_swatch_of_its_own_colour_with_its_hex():
    color = ColorRGBA(0.2, 0.4, 0.6, 0.5)
    ax = plot(color).axes[0]
    (swatch,) = [p for p in ax.patches if p.get_gid() == "swatch"]
    assert tuple(swatch.get_facecolor()) == pytest.approx((0.2, 0.4, 0.6, 0.5))
    assert color.to_hex() in [t.get_text() for t in ax.texts]


def test_mesh_resource_has_no_plot_because_a_file_is_not_read_here():
    assert not has_plot(MeshResource("a.stl"))
