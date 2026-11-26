"""FiducialMount — where one physical marker is: its key, its size and its pose on a parent frame.

The description-side record (a sticker on a robot link, a marker on a board, a tag on a wall):
what a detector, a tracker or a bundle adjustment needs to turn corner pixels into poses. Mirrors
fiducial_msgs/FiducialMapEntry, with the parent frame, the size and the dictionary made explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial.covariance import checked_covariance
from robot_msgs.fiducial.fiducial_detection import checked_key, checked_length
from robot_msgs.geometry import Pose
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std import Header


def marker_object_points(marker_length: float) -> Array:
    """(4, 3) corners of a marker in its own frame, m: OpenCV's ``SOLVEPNP_IPPE_SQUARE`` frame.

    Origin at the centre of the black square, x right and y up as printed, z out of the face (toward
    a camera that sees it). Corners top-left, top-right, bottom-right, bottom-left, the same order
    as `FiducialDetection.corners`: ``(-h, h, 0), (h, h, 0), (h, -h, 0), (-h, -h, 0)``, h = L / 2.
    """
    half = checked_length("marker_object_points", "marker_length", marker_length, optional=False)
    half = half / 2.0  # type: ignore[operator]
    return freeze_new(
        np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
    )


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialMount(Msg):
    """One marker, placed: `pose` is ``T_parent_marker``, the marker frame
    (`marker_object_points`) in `header.frame_id` — a robot link, a board, the world.

    - `marker_length`: side of the black square, border included, m.
    - `covariance`: optional (6, 6) uncertainty of `pose`, in the package's local-tangent
      convention (`robot_msgs.fiducial.covariance`): a CAD mount's prior, or what a bundle
      adjustment estimated.
    - `header.meta` carries provenance (``source``: ``cad`` / ``estimated``, residuals, n_obs).
    """

    id: int
    dictionary: str
    marker_length: float
    pose: Pose = field(default_factory=Pose)
    covariance: Array | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        marker_id, dictionary = checked_key("FiducialMount", self.id, self.dictionary)
        set_field(self, "id", marker_id)
        set_field(self, "dictionary", dictionary)
        set_field(
            self,
            "marker_length",
            checked_length("FiducialMount", "marker_length", self.marker_length, optional=False),
        )
        expect_type("FiducialMount.pose", self.pose, Pose)
        set_field(self, "covariance", checked_covariance("FiducialMount", self.covariance))

    @property
    def key(self) -> tuple[str, int]:
        return (self.dictionary, self.id)

    @property
    def frame_id(self) -> str:
        """The parent frame the pose is in (`header.frame_id`)."""
        return self.header.frame_id

    def object_points(self) -> Array:
        """(4, 3) corners in the marker's own frame (`marker_object_points`)."""
        return marker_object_points(self.marker_length)

    def corners_in_parent(self) -> Array:
        """(4, 3) corners in the parent frame, m, top-left first."""
        matrix = self.pose.to_matrix()
        return freeze_new(self.object_points() @ matrix[:3, :3].T + matrix[:3, 3])

    def transformed(self, parent_pose: Pose, frame_id: str) -> FiducialMount:
        """This mount re-expressed in `frame_id`, given ``parent_pose = T_frame_parent``.

        The local-tangent covariance is unchanged by a rigid change of parent (that is its point).
        """
        return self.replace(
            pose=parent_pose @ self.pose, header=self.header.replace(frame_id=str(frame_id))
        )
