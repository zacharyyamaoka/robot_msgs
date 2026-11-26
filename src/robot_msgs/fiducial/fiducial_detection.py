"""FiducialDetection — one fiducial marker found in one image: its id and four corner pixels.

The measurement, and nothing the detector inferred from it (a pose is a `FiducialPose`). Mirrors
the per-marker part of ROS aruco_msgs/Marker, fiducial_msgs/Fiducial and
apriltag_msgs/AprilTagDetection, in OpenCV's corner order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, frozen_array, set_field
from robot_msgs.vision import BoundingBox2D

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def checked_key(owner: str, marker_id: object, dictionary: object) -> tuple[int, str]:
    if isinstance(marker_id, bool) or int(marker_id) != marker_id or int(marker_id) < 0:  # type: ignore[call-overload]
        raise ValueError(f"{owner}.id must be a non-negative integer, got {marker_id!r}")
    name = str(dictionary)
    if not name:
        raise ValueError(f"{owner}.dictionary must name the marker family, e.g. 'DICT_4X4_50'")
    return int(marker_id), name  # type: ignore[call-overload]


def checked_length(owner: str, name: str, value: object, *, optional: bool) -> float | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{owner}.{name} is required")
    length = float(value)  # type: ignore[arg-type]
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"{owner}.{name} must be a positive length in metres, got {length}")
    return length


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialDetection(Msg):
    """One marker in one image.

    - `id`, `dictionary`: the decoded marker. `dictionary` is OpenCV's name for the family
      (``"DICT_4X4_50"``, ``"DICT_APRILTAG_36h11"``); ``(dictionary, id)`` is the marker's key.
    - `corners`: (4, 2) pixels (u = column, v = row) of the marker's black square in OpenCV's
      order, top-left, top-right, bottom-right, bottom-left *as printed* (so the order survives
      the marker being upside down in the image).
    - `marker_length`: side of the black square (border included), m, when the detector knew it —
      what pose estimation scales by; `None` when it did not.
    - `corner_covariance`: optional (4, 2, 2) pixel² covariance per corner;
      `with_corner_sigma(σ)` builds the isotropic one.
    - `decision_margin` (AprilTag) / `hamming` (bits corrected) are the decoder's confidence,
      `None` when the detector does not report them.

    The image it came from is the enclosing `FiducialArray`'s header.
    """

    id: int
    dictionary: str
    corners: Array
    marker_length: float | None = None
    corner_covariance: Array | None = None
    decision_margin: float | None = None
    hamming: int | None = None

    def __post_init__(self) -> None:
        marker_id, dictionary = checked_key("FiducialDetection", self.id, self.dictionary)
        set_field(self, "id", marker_id)
        set_field(self, "dictionary", dictionary)
        corners = frozen_array(self.corners)
        if corners.shape != (4, 2):
            raise ValueError(
                f"FiducialDetection.corners must be (4, 2) pixels, got {corners.shape}"
            )
        if not np.all(np.isfinite(corners)):
            raise ValueError("FiducialDetection.corners must be finite")
        set_field(self, "corners", corners)
        set_field(
            self,
            "marker_length",
            checked_length("FiducialDetection", "marker_length", self.marker_length, optional=True),
        )
        if self.corner_covariance is not None:
            covariance = frozen_array(self.corner_covariance)
            if covariance.shape != (4, 2, 2):
                raise ValueError(
                    "FiducialDetection.corner_covariance must be (4, 2, 2) pixel², "
                    f"got {covariance.shape}"
                )
            set_field(self, "corner_covariance", covariance)
        if self.decision_margin is not None:
            set_field(self, "decision_margin", float(self.decision_margin))
        if self.hamming is not None:
            hamming = int(self.hamming)
            if hamming < 0:
                raise ValueError(f"FiducialDetection.hamming must be >= 0, got {hamming}")
            set_field(self, "hamming", hamming)

    @property
    def key(self) -> tuple[str, int]:
        """``(dictionary, id)``: what identifies a marker across images and maps."""
        return (self.dictionary, self.id)

    @property
    def center(self) -> Array:
        """(2,) pixel where the diagonals cross (the projected marker centre, exact under any
        homography, unlike the corners' mean)."""
        p0, p1, p2, p3 = (np.asarray(c, dtype=float) for c in self.corners)
        d1, d2 = p2 - p0, p3 - p1
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denominator) < 1e-12:
            return freeze_new(self.corners.mean(axis=0))
        s = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denominator
        return freeze_new(p0 + s * d1)

    @property
    def side_lengths_px(self) -> Array:
        """(4,) pixel lengths of the sides top, right, bottom, left."""
        return freeze_new(np.linalg.norm(np.roll(self.corners, -1, axis=0) - self.corners, axis=1))

    @property
    def area_px(self) -> float:
        """Pixel area of the quadrilateral (shoelace), positive when the corners run clockwise
        in the image, as a marker facing the camera does."""
        x, y = self.corners[:, 0], self.corners[:, 1]
        return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def bbox(self) -> BoundingBox2D:
        """The axis-aligned box around the four corners."""
        (x0, y0), (x1, y1) = self.corners.min(axis=0), self.corners.max(axis=0)
        return BoundingBox2D.from_xyxy(float(x0), float(y0), float(x1), float(y1))

    def with_corner_sigma(self, sigma_px: float) -> FiducialDetection:
        """This detection with an isotropic per-corner pixel noise σ (px) on both axes."""
        sigma = float(sigma_px)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma_px must be finite and >= 0, got {sigma}")
        return self.replace(corner_covariance=np.tile(np.eye(2) * sigma**2, (4, 1, 1)))
