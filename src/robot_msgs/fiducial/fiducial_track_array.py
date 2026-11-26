"""FiducialTrackArray — a tracker's output for one frame: every known marker's `FiducialTrack`."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial._lookup import find, keys_of
from robot_msgs.fiducial.fiducial_track import FiducialTrack
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialTrackArray(Msg):
    """`header.stamp` is the estimate's time, `header.frame_id` the frame every pose is in (the
    robot base, the world). One track per marker key."""

    header: Header = field(default_factory=Header)
    tracks: tuple[FiducialTrack, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "tracks", tuple(self.tracks))
        seen: set[tuple[str, int]] = set()
        for track in self.tracks:
            expect_type("FiducialTrackArray.tracks[]", track, FiducialTrack)
            if track.key in seen:
                raise ValueError(f"FiducialTrackArray has marker {track.key} twice")
            seen.add(track.key)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[FiducialTrack]:
        return iter(self.tracks)

    @property
    def keys(self) -> tuple[tuple[str, int], ...]:
        return keys_of(self.tracks)

    def get(self, id: int, dictionary: str | None = None) -> FiducialTrack:
        return find(self.tracks, int(id), dictionary, "FiducialTrackArray")

    def observed(self) -> FiducialTrackArray:
        """Only the tracks a detection updated this frame."""
        return self.replace(tracks=tuple(t for t in self.tracks if t.observed))

    def predicted(self) -> FiducialTrackArray:
        """Only the tracks that were predicted, not seen, this frame."""
        return self.replace(tracks=tuple(t for t in self.tracks if not t.observed))

    def as_arrays(self) -> dict[str, Array]:
        """``ids`` (N,) int64, ``T_frame_marker`` (N, 4, 4), ``covariance`` (N, 6, 6) (NaN when
        absent), ``observed`` (N,) bool, ``last_seen`` (N,) s (NaN never), ``nis`` (N,) (NaN)."""
        count = len(self.tracks)
        covariance = np.full((count, 6, 6), np.nan)
        for row, track in enumerate(self.tracks):
            if track.covariance is not None:
                covariance[row] = track.covariance

        def optional(values: list[float | None]) -> np.ndarray:
            return np.array([np.nan if v is None else v for v in values], dtype=float)

        return {
            "ids": freeze_new(np.array([t.id for t in self.tracks], dtype=np.int64)),
            "T_frame_marker": freeze_new(
                np.array([t.pose.to_matrix() for t in self.tracks]).reshape(count, 4, 4)
            ),
            "covariance": freeze_new(covariance),
            "observed": freeze_new(np.array([t.observed for t in self.tracks], dtype=bool)),
            "last_seen": freeze_new(optional([t.last_seen for t in self.tracks])),
            "nis": freeze_new(optional([t.nis for t in self.tracks])),
        }
