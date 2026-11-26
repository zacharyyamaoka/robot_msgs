"""FiducialPose — the pose of one detected marker in the camera, with both PnP solutions.

A single square marker has two poses that reproject almost equally well (the planar ambiguity,
Collins & Bartoli 2014, "Infinitesimal Plane-Based Pose Estimation"); OpenCV's
``solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)`` returns both. Keeping both lets a consumer with a
prior (robot FK, a filter) pick the right one instead of inheriting a flip. Mirrors
aruco_msgs/Marker's pose and fiducial_msgs/FiducialTransform's errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial.covariance import checked_covariance
from robot_msgs.fiducial.fiducial_detection import checked_key
from robot_msgs.fiducial.fiducial_mount import marker_object_points
from robot_msgs.geometry import Pose
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.sensor import CameraInfo


def _error(name: str, value: object) -> float | None:
    if value is None:
        return None
    error = float(value)  # type: ignore[arg-type]
    if not np.isfinite(error) or error < 0.0:
        raise ValueError(f"FiducialPose.{name} must be finite and >= 0 px, got {error}")
    return error


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialPose(Msg):
    """`pose` is ``T_camera_marker``: the marker frame (`marker_object_points`) in the enclosing
    `FiducialArray`'s camera frame (``header.frame_id``, the optical frame: x right, y down,
    z forward).

    - `reprojection_error`: RMS corner reprojection error of `pose`, px.
    - `alt_pose` / `alt_reprojection_error`: the other planar solution, or both `None` when the
      solver gave one. `pose` is the chosen solution: the lower error from a solver, or whichever a
      consumer picked (`flipped()` swaps them).
    - `covariance`: optional (6, 6) of `pose`, local-tangent convention
      (`robot_msgs.fiducial.covariance`).
    """

    id: int
    dictionary: str
    pose: Pose
    reprojection_error: float | None = None
    alt_pose: Pose | None = None
    alt_reprojection_error: float | None = None
    covariance: Array | None = None

    def __post_init__(self) -> None:
        marker_id, dictionary = checked_key("FiducialPose", self.id, self.dictionary)
        set_field(self, "id", marker_id)
        set_field(self, "dictionary", dictionary)
        expect_type("FiducialPose.pose", self.pose, Pose)
        if self.alt_pose is not None:
            expect_type("FiducialPose.alt_pose", self.alt_pose, Pose)
        set_field(self, "reprojection_error", _error("reprojection_error", self.reprojection_error))
        alt_error = _error("alt_reprojection_error", self.alt_reprojection_error)
        if alt_error is not None and self.alt_pose is None:
            raise ValueError("FiducialPose.alt_reprojection_error needs an alt_pose")
        set_field(self, "alt_reprojection_error", alt_error)
        set_field(self, "covariance", checked_covariance("FiducialPose", self.covariance))

    @classmethod
    def from_candidates(
        cls,
        id: int,
        dictionary: str,
        poses: Sequence[Pose] | Array,
        reprojection_errors: Sequence[float] | Array,
    ) -> FiducialPose:
        """From a solver's 1 or 2 candidates (`Pose`s or (k, 4, 4) matrices) and their errors;
        the lower error becomes `pose` (a stable sort: a tie keeps the solver's order)."""
        candidates = [p if isinstance(p, Pose) else Pose.from_matrix(p) for p in poses]
        errors = [float(e) for e in reprojection_errors]
        if len(candidates) not in (1, 2) or len(errors) != len(candidates):
            raise ValueError(
                f"FiducialPose.from_candidates takes 1 or 2 poses with one error each, "
                f"got {len(candidates)} poses and {len(errors)} errors"
            )
        order = sorted(range(len(candidates)), key=lambda i: errors[i])
        best = order[0]
        alt = order[1] if len(order) == 2 else None
        return cls(
            id=id,
            dictionary=dictionary,
            pose=candidates[best],
            reprojection_error=errors[best],
            alt_pose=None if alt is None else candidates[alt],
            alt_reprojection_error=None if alt is None else errors[alt],
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.dictionary, self.id)

    @property
    def ambiguity(self) -> float | None:
        """``reprojection_error / alt_reprojection_error``: near 0 is unambiguous, near 1 means
        both solutions fit the corners equally well (trust neither alone). `None` without both."""
        if self.reprojection_error is None or self.alt_reprojection_error is None:
            return None
        if self.alt_reprojection_error == 0.0:
            return 1.0 if self.reprojection_error == 0.0 else float("inf")
        return self.reprojection_error / self.alt_reprojection_error

    def flipped(self) -> FiducialPose:
        """The same detection with the two solutions swapped (the covariance belongs to the old
        choice, so it is dropped)."""
        if self.alt_pose is None:
            raise ValueError(f"FiducialPose {self.key} has no alt_pose to flip to")
        return self.replace(
            pose=self.alt_pose,
            alt_pose=self.pose,
            reprojection_error=self.alt_reprojection_error,
            alt_reprojection_error=self.reprojection_error,
            covariance=None,
        )

    def candidate_matrices(self) -> Array:
        """(2, 4, 4) ``T_camera_marker`` for `pose` then `alt_pose` (NaN when absent)."""
        alt = np.full((4, 4), np.nan) if self.alt_pose is None else self.alt_pose.to_matrix()
        return freeze_new(np.stack([self.pose.to_matrix(), alt]))

    def corners_in_camera(self, marker_length: float) -> Array:
        """(4, 3) corners of the marker in the camera frame, m, top-left first."""
        matrix = self.pose.to_matrix()
        return freeze_new(marker_object_points(marker_length) @ matrix[:3, :3].T + matrix[:3, 3])

    def project(self, camera_info: CameraInfo, marker_length: float) -> Array:
        """(4, 2) pixels where `pose` puts the corners (pinhole, `CameraInfo.project`)."""
        return camera_info.project(self.corners_in_camera(marker_length))
