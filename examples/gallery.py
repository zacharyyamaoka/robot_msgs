"""A picture of every kind of message robot_msgs.viz draws, plus the hero video.

    uv run --directory /home/bam/robot_msgs --extra viz python examples/gallery.py OUTDIR

Writes PNGs for the UR5e pick (joint and tool trajectories, a JointState), a synthetic image with
detections, a point cloud built from a synthetic depth image, a CameraInfo and 3-D detections,
and `hero.mp4` / `hero.gif`: the pick played back with a moving cursor and triad.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # a gallery never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ur5e_pick import UR5E_LIMITS, minimum_jerk, tool_trajectory, waypoint_trajectory  # noqa: E402

from robot_msgs.geometry import (  # noqa: E402
    Pose,
    PoseStamped,
    Transform,
    TransformStamped,
    Vector3,  # noqa: E402
)
from robot_msgs.sensor import CameraInfo, Image, PointCloud  # noqa: E402
from robot_msgs.std import Header  # noqa: E402
from robot_msgs.vision import (  # noqa: E402
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from robot_msgs.viz import animate, plot  # noqa: E402

CAMERA = CameraInfo.from_intrinsics(
    525.0, 525.0, 319.5, 239.5, 640, 480, header=Header(frame_id="camera_color_optical_frame")
)


def detection(
    class_id: str, score: float, box: BoundingBox2D, header: Header, id: str
) -> Detection2D:
    return Detection2D(
        header=header,
        results=(ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id, score)),),
        bbox=box,
        id=id,
    )


def synthetic_image() -> tuple[Image, Detection2DArray]:
    """A 640×480 scene of coloured shapes on a gradient, and boxes around them."""
    height, width = 480, 640
    rows, columns = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 215 - 40 * rows // height
    image[..., 1] = 218 - 30 * rows // height
    image[..., 2] = 222 - 20 * rows // height
    image[(rows > 330)] = (190, 176, 150)  # table
    image[(rows > 70) & (rows < 332) & (columns > 80) & (columns < 150)] = (75, 85, 110)  # bottle
    image[(rows > 70) & (rows < 110) & (columns > 100) & (columns < 130)] = (45, 55, 80)  # neck
    image[((rows - 300) ** 2 + (columns - 360) ** 2) < 70**2] = (205, 70, 60)  # ball
    image[(rows > 250) & (rows < 380) & (columns > 450) & (columns < 590)] = (230, 170, 40)  # box
    header = Header(stamp=12.5, frame_id="camera_color_optical_frame")
    detections = (
        detection("bottle", 0.94, BoundingBox2D.from_xyxy(78, 68, 152, 332), header, "1"),
        detection("ball", 0.88, BoundingBox2D.from_xyxy(288, 228, 432, 372), header, "2"),
        detection("box", 0.71, BoundingBox2D.from_xyxy(448, 248, 592, 382), header, "3"),
    )
    return Image.from_array(image, header=header), Detection2DArray(
        header=header, detections=detections
    )


def synthetic_cloud() -> tuple[PointCloud, PointCloud]:
    """A table and a box seen by a camera 0.7 m up, pitched 40° down: via a 16UC1 depth image.

    Returns the cloud with an rgb channel and the same cloud without one, in `base_link`.
    """
    pitch = np.radians(40.0)
    optical_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    tilt = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rotation = tilt @ optical_to_body
    eye = np.array([0.0, 0.0, 0.7])
    rows, columns = np.mgrid[0 : CAMERA.height, 0 : CAMERA.width]
    rays = (
        np.stack(
            [
                (columns - CAMERA.cx) / CAMERA.fx,
                (rows - CAMERA.cy) / CAMERA.fy,
                np.ones(columns.shape),
            ],
            axis=-1,
        )
        @ rotation.T
    )  # rays in the base frame
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.where(rays[..., 2] < 0, -eye[2] / rays[..., 2], np.inf)
        low, high = np.array([0.45, -0.1, 0.0]), np.array([0.65, 0.1, 0.12])  # the box
        near = (low - eye) / rays
        far = (high - eye) / rays
        enter = np.minimum(near, far).max(axis=-1)
        leave = np.maximum(near, far).min(axis=-1)
        box = np.where((enter <= leave) & (leave > 0), np.maximum(enter, 0), np.inf)
    depth = np.minimum(table, box)
    depth[~np.isfinite(depth)] = 0.0
    hit = eye + rays * depth[..., None]
    checker = ((np.floor(hit[..., 0] * 10) + np.floor(hit[..., 1] * 10)) % 2).astype(bool)
    colour = np.where(checker[..., None], (205, 200, 190), (160, 155, 148)).astype(np.uint8)
    colour[box < table] = (225, 140, 40)
    colour[depth == 0.0] = 0
    header = Header(frame_id=CAMERA.header.frame_id)
    depth_image = Image.from_array(np.round(depth * 1000).astype(np.uint16), header=header)
    rgb_image = Image.from_array(colour, header=header)
    to_base = Transform.from_matrix(np.block([[rotation, eye[:, None]], [np.zeros((1, 3)), 1.0]]))
    world = Header(frame_id="base_link")
    corner_low, corner_high = np.array([-0.2, -0.7, -0.01]), np.array([1.1, 0.7, 0.5])
    clouds = []
    for rgb in (rgb_image, None):
        cloud = PointCloud.from_depth(depth_image, CAMERA, rgb=rgb).transformed(to_base)
        clouds.append(cloud.replace(header=world).cropped(corner_low, corner_high))
    return clouds[0], clouds[1]


def detections_3d() -> Detection3DArray:
    header = Header(frame_id="base_link")
    objects = (
        (
            "bottle",
            0.93,
            Pose.from_xyz_rpy(0.50, -0.20, 0.125, 0.0, 0.0, 0.3),
            Vector3(0.07, 0.07, 0.25),
        ),
        (
            "box",
            0.81,
            Pose.from_xyz_rpy(0.60, 0.10, 0.06, 0.0, 0.0, -0.5),
            Vector3(0.16, 0.12, 0.12),
        ),
        (
            "can",
            0.67,
            Pose.from_xyz_rpy(0.35, 0.15, 0.06, 0.0, 0.0, 0.0),
            Vector3(0.066, 0.066, 0.12),
        ),
    )
    return Detection3DArray(
        header=header,
        detections=tuple(
            Detection3D(
                header=header,
                id=str(index),
                results=(
                    ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(name, score), pose=pose),
                ),
                bbox=BoundingBox3D(center=pose, size=size),
            )
            for index, (name, score, pose, size) in enumerate(objects)
        ),
    )


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    waypoints = waypoint_trajectory()
    joints = minimum_jerk(waypoints)
    tool = tool_trajectory(joints)
    image, detections = synthetic_image()
    cloud_rgb, cloud_z = synthetic_cloud()
    grasp = tool.at(2.4)
    figures = {
        "ur5e_joints": plot(joints, waypoints=waypoints),
        "ur5e_tool": plot(tool),
        "joint_state": plot(joints.joint_state_at(2.4), limits=UR5E_LIMITS),
        "pose_stamped": plot(PoseStamped(Header(frame_id="base_link"), grasp.pose)),
        "transform_stamped": plot(
            TransformStamped(Header(frame_id="base_link"), "tool0", grasp.pose.as_transform())
        ),
        "image_detections": plot(image, detections=detections),
        "camera_info": plot(CAMERA),
        "point_cloud": plot(cloud_z),
        "point_cloud_rgb": plot(cloud_rgb),
        "detections_3d": plot(detections_3d()),
    }
    for name, figure in figures.items():
        figure.savefig(out / f"{name}.png", dpi=130)
        plt.close(figure)
        print(out / f"{name}.png")
    print(animate(joints, tool, path=out / "hero.mp4", duration=8.0))
    print(animate(joints, tool, path=out / "hero.gif", fps=15, duration=8.0, dpi=60))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/gallery.py OUTDIR")
    main(Path(sys.argv[1]))
