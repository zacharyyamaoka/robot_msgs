"""FiducialTrack — a tracker's estimate of one marker: filtered pose, uncertainty, and whether it
was seen this frame or only predicted.

What a filter fusing camera detections with robot kinematics reports for every marker it knows,
detected or not (occluded frames are where fusion earns its keep, so the flag is explicit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot_msgs._core import Array, Msg, set_field
from robot_msgs.fiducial.covariance import checked_covariance
from robot_msgs.fiducial.fiducial_detection import checked_key
from robot_msgs.geometry import Pose
from robot_msgs.geometry.vector3 import expect_type


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialTrack(Msg):
    """`pose` is the marker frame in the enclosing `FiducialTrackArray`'s frame.

    - `covariance`: (6, 6) of `pose`, local-tangent convention (`robot_msgs.fiducial.covariance`).
    - `observed`: whether a detection of this marker updated the estimate this frame (`False`:
      predicted only).
    - `last_seen`: absolute stamp (s) of the last detection that updated it, `None` if never.
    - `nis`: normalised innovation squared of this frame's update (`None` when not observed) — the
      truth-free consistency check (χ² with the measurement's dimension).
    """

    id: int
    dictionary: str
    pose: Pose = field(default_factory=Pose)
    covariance: Array | None = None
    observed: bool = False
    last_seen: float | None = None
    nis: float | None = None

    def __post_init__(self) -> None:
        marker_id, dictionary = checked_key("FiducialTrack", self.id, self.dictionary)
        set_field(self, "id", marker_id)
        set_field(self, "dictionary", dictionary)
        expect_type("FiducialTrack.pose", self.pose, Pose)
        set_field(self, "covariance", checked_covariance("FiducialTrack", self.covariance))
        set_field(self, "observed", bool(self.observed))
        if self.last_seen is not None:
            set_field(self, "last_seen", float(self.last_seen))
        elif self.observed:
            raise ValueError("FiducialTrack.observed needs last_seen (the stamp it was seen at)")
        if self.nis is not None:
            nis = float(self.nis)
            if not math.isfinite(nis) or nis < 0.0:
                raise ValueError(f"FiducialTrack.nis must be finite and >= 0, got {nis}")
            set_field(self, "nis", nis)

    @property
    def key(self) -> tuple[str, int]:
        return (self.dictionary, self.id)

    def age(self, stamp: float) -> float:
        """Seconds since `last_seen` at `stamp` (inf if never seen)."""
        return math.inf if self.last_seen is None else float(stamp) - self.last_seen
