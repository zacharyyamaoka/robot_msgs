"""PoseTrajectory — one frame's pose over time (a Cartesian / end-effector trajectory).

`header.frame_id` is the reference frame, `child_frame_id` the frame that moves. Maps to
nav_msgs/Path (a list of PoseStamped) and to a one-joint trajectory_msgs/MultiDOFJointTrajectory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robot_msgs._core import Array, adopt, set_field
from robot_msgs.geometry import Point, Pose, Quaternion, Transform
from robot_msgs.geometry.rotations import (
    matrix_to_pose,
    pose_to_matrix,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_rotvec,
)
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class PoseTrajectoryPoint(TrajectoryPoint):
    position: Array | None = None
    orientation: Array | None = None
    linear_velocity: Array | None = None
    angular_velocity: Array | None = None

    @property
    def pose(self) -> Pose:
        return Pose(Point.from_array(self.position), Quaternion.from_array(self.orientation))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class PoseTrajectory(Trajectory):
    """`positions (N, 3)` in metres, `orientations (N, 4)` xyzw, optional twist per sample.

    Orientations are slerped between samples and normalised at construction. Velocities are
    expressed in the reference frame (`header.frame_id`).
    """

    child_frame_id: str = ""
    positions: Array = sample(point_name="position")
    orientations: Array = sample(interp="slerp", point_name="orientation")
    linear_velocities: Array | None = sample(default=None, point_name="linear_velocity")
    angular_velocities: Array | None = sample(default=None, point_name="angular_velocity")

    point_type: ClassVar[type[TrajectoryPoint]] = PoseTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        set_field(self, "child_frame_id", str(self.child_frame_id))
        expected = {
            "positions": 3,
            "orientations": 4,
            "linear_velocities": 3,
            "angular_velocities": 3,
        }
        for name, width in expected.items():
            values = getattr(self, name)
            if values is not None and (values.ndim != 2 or values.shape[1] != width):
                raise ValueError(
                    f"PoseTrajectory.{name} has shape {values.shape}, expected ({len(self)}, {width})"
                )
        if len(self):
            norms = np.linalg.norm(self.orientations, axis=1)
            if np.any(norms < 1e-9):
                raise ValueError("PoseTrajectory.orientations contains a zero quaternion")
            if not np.allclose(norms, 1.0, atol=1e-12):
                set_field(self, "orientations", adopt(quat_normalize(self.orientations)))

    @classmethod
    def from_poses(cls, t: Sequence[float], poses: Sequence[Pose], **static: object) -> PoseTrajectory:
        arrays = np.array([pose.as_array() for pose in poses]).reshape(-1, 7)
        return cls(t=t, positions=arrays[:, :3], orientations=arrays[:, 3:], **static)  # type: ignore[arg-type]

    @classmethod
    def from_matrices(cls, t: Sequence[float], matrices: np.ndarray, **static: object) -> PoseTrajectory:
        """From ``(N, 4, 4)`` homogeneous transforms (what FK usually returns)."""
        positions, orientations = matrix_to_pose(matrices)
        return cls(t=t, positions=positions, orientations=orientations, **static)  # type: ignore[arg-type]

    def poses(self) -> tuple[Pose, ...]:
        return tuple(point.pose for point in self)

    def pose_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> Pose:
        return self.at(time, extrapolate=extrapolate).pose

    def matrices(self) -> np.ndarray:
        """``(N, 4, 4)`` homogeneous transforms, reference ← child."""
        return pose_to_matrix(self.positions, self.orientations)

    def transformed(self, transform: Transform | Pose, *, frame_id: str | None = None) -> PoseTrajectory:
        """Re-express in another frame: ``T_new_child = transform @ T_old_child``."""
        offset = transform.as_array()
        translation, rotation = offset[:3], offset[3:]
        header = self.header if frame_id is None else self.header.replace(frame_id=frame_id)
        rotated = {
            name: None if getattr(self, name) is None else quat_rotate(rotation, getattr(self, name))
            for name in ("linear_velocities", "angular_velocities")
        }
        return self.replace(
            positions=quat_rotate(rotation, self.positions) + translation,
            orientations=quat_multiply(rotation, self.orientations),
            header=header,
            **rotated,
        )

    def path_length(self) -> float:
        """Distance travelled by the origin of the child frame, in metres."""
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    def with_derivatives(self, *, overwrite: bool = False) -> PoseTrajectory:
        """Fill missing linear/angular velocity by finite differences on the real time grid.

        Angular velocity comes from the rotation between neighbouring samples
        (``q[i+1] · q[i]⁻¹`` as a rotation vector over the step), averaged onto the samples.
        """
        if len(self) < 2:
            raise ValueError("derivatives need at least two samples")
        linear = self.linear_velocities
        if linear is None or overwrite:
            edge_order = 2 if len(self) >= 3 else 1
            linear = np.gradient(self.positions, self.t, axis=0, edge_order=edge_order)
        angular = self.angular_velocities
        if angular is None or overwrite:
            step = quat_multiply(self.orientations[1:], quat_conjugate(self.orientations[:-1]))
            rates = quat_to_rotvec(step) / np.diff(self.t)[:, None]
            angular = np.empty_like(self.positions)
            angular[0], angular[-1] = rates[0], rates[-1]
            angular[1:-1] = 0.5 * (rates[:-1] + rates[1:])
        return self.replace(linear_velocities=linear, angular_velocities=angular)
