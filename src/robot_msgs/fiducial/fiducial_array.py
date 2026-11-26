"""FiducialArray — every marker found in one image, and the poses solved for them.

Mirrors aruco_msgs/MarkerArray, fiducial_msgs/FiducialArray + FiducialTransformArray and
apriltag_msgs/AprilTagDetectionArray in one message: a pipeline that only detects fills
`detections`; one that also solves PnP adds `poses`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial._lookup import find, keys_of
from robot_msgs.fiducial.fiducial_detection import FiducialDetection
from robot_msgs.fiducial.fiducial_pose import FiducialPose
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialArray(Msg):
    """`header` is the image's: its `stamp` is the capture time (mid-exposure when known) and its
    `frame_id` the camera's optical frame, so a solver can join the markers to the robot's
    `JointState` at the same stamp and to the camera's `CameraInfo` by frame. Capture settings
    (exposure, gain, …) ride in `header.meta`.

    - `detections`: every marker decoded, in the detector's order.
    - `poses`: a `FiducialPose` for each detection that was solved; each must name a detected
      marker. Join them by `key` (`pose_of`, `detection_of`).
    """

    header: Header = field(default_factory=Header)
    detections: tuple[FiducialDetection, ...] = ()
    poses: tuple[FiducialPose, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "detections", tuple(self.detections))
        set_field(self, "poses", tuple(self.poses))
        for detection in self.detections:
            expect_type("FiducialArray.detections[]", detection, FiducialDetection)
        detected = set(keys_of(self.detections))
        for pose in self.poses:
            expect_type("FiducialArray.poses[]", pose, FiducialPose)
            if pose.key not in detected:
                raise ValueError(f"FiducialArray.poses has {pose.key}, which was not detected")

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[FiducialDetection]:
        return iter(self.detections)

    @property
    def keys(self) -> tuple[tuple[str, int], ...]:
        """``(dictionary, id)`` of each detection, in order."""
        return keys_of(self.detections)

    @property
    def ids(self) -> Array:
        """(k,) int64 ids of the detections, in order."""
        return freeze_new(np.array([d.id for d in self.detections], dtype=np.int64))

    def detection_of(self, id: int, dictionary: str | None = None) -> FiducialDetection:
        return find(self.detections, int(id), dictionary, "FiducialArray.detections")

    def pose_of(self, id: int, dictionary: str | None = None) -> FiducialPose:
        return find(self.poses, int(id), dictionary, "FiducialArray.poses")

    def filtered(
        self, ids: Iterable[int] | None = None, dictionary: str | None = None
    ) -> FiducialArray:
        """Only the markers with an id in `ids` and/or from `dictionary` (poses follow)."""
        wanted = None if ids is None else {int(i) for i in ids}

        def keep(item: FiducialDetection | FiducialPose) -> bool:
            return (wanted is None or item.id in wanted) and (
                dictionary is None or item.dictionary == dictionary
            )

        return self.replace(
            detections=tuple(d for d in self.detections if keep(d)),
            poses=tuple(p for p in self.poses if keep(p)),
        )

    def as_arrays(self) -> dict[str, Array]:
        """The batch view a vectorised filter or solver takes, one row per detection:

        - ``ids`` (k,) int64, ``corners`` (k, 4, 2) px;
        - ``solved`` (k,) bool: whether that detection has a pose;
        - ``T_camera_marker`` (k, 2, 4, 4): `pose` then `alt_pose` (NaN when absent or unsolved);
        - ``reprojection_error`` (k, 2) px (NaN when absent).
        """
        count = len(self.detections)
        matrices = np.full((count, 2, 4, 4), np.nan)
        errors = np.full((count, 2), np.nan)
        solved = np.zeros(count, dtype=bool)
        by_key = {pose.key: pose for pose in self.poses}
        for row, detection in enumerate(self.detections):
            pose = by_key.get(detection.key)
            if pose is None:
                continue
            solved[row] = True
            matrices[row] = pose.candidate_matrices()
            errors[row] = [
                np.nan if e is None else e
                for e in (pose.reprojection_error, pose.alt_reprojection_error)
            ]
        corners = np.array([d.corners for d in self.detections], dtype=float).reshape(count, 4, 2)
        return {
            "ids": self.ids,
            "corners": freeze_new(corners),
            "solved": freeze_new(solved),
            "T_camera_marker": freeze_new(matrices),
            "reprojection_error": freeze_new(errors),
        }
