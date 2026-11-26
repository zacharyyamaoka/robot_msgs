"""FiducialMap — every known marker and where it is mounted: a robot's stickers, a board, a room.

Mirrors fiducial_msgs/FiducialMapEntryArray, with mounts on any frame (a robot link, a board,
the world), so one map can hold the markers of a whole robot description.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial._lookup import find, keys_of
from robot_msgs.fiducial.fiducial_mount import FiducialMount
from robot_msgs.geometry.vector3 import expect_type


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialMap(Msg):
    """`mounts`, one per marker; a marker key ``(dictionary, id)`` appears once (two physical
    copies of one marker cannot be told apart by a detector, so a map refuses them)."""

    mounts: tuple[FiducialMount, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "mounts", tuple(self.mounts))
        seen: set[tuple[str, int]] = set()
        for mount in self.mounts:
            expect_type("FiducialMap.mounts[]", mount, FiducialMount)
            if mount.key in seen:
                raise ValueError(f"FiducialMap has marker {mount.key} twice")
            seen.add(mount.key)

    def __len__(self) -> int:
        return len(self.mounts)

    def __iter__(self) -> Iterator[FiducialMount]:
        return iter(self.mounts)

    def __contains__(self, key: object) -> bool:
        return key in set(self.keys)

    @property
    def keys(self) -> tuple[tuple[str, int], ...]:
        return keys_of(self.mounts)

    @property
    def frames(self) -> tuple[str, ...]:
        """The parent frames, in first-appearance order."""
        return tuple(dict.fromkeys(m.frame_id for m in self.mounts))

    def get(self, id: int, dictionary: str | None = None) -> FiducialMount:
        return find(self.mounts, int(id), dictionary, "FiducialMap")

    def in_frame(self, frame_id: str) -> FiducialMap:
        """The mounts whose parent is `frame_id`."""
        return self.replace(mounts=tuple(m for m in self.mounts if m.frame_id == frame_id))

    def merged(self, other: FiducialMap) -> FiducialMap:
        """Both maps' mounts (a marker in both is refused, like any duplicate)."""
        return self.replace(mounts=self.mounts + other.mounts)

    def as_arrays(self) -> dict[str, Array]:
        """``ids`` (N,) int64, ``T_parent_marker`` (N, 4, 4), ``marker_length`` (N,) m,
        ``frame_ids`` (N,) str — the batch view a vectorised prior takes."""
        count = len(self.mounts)
        matrices = np.array([m.pose.to_matrix() for m in self.mounts]).reshape(count, 4, 4)
        return {
            "ids": freeze_new(np.array([m.id for m in self.mounts], dtype=np.int64)),
            "T_parent_marker": freeze_new(matrices),
            "marker_length": freeze_new(np.array([m.marker_length for m in self.mounts])),
            "frame_ids": freeze_new(np.array([m.frame_id for m in self.mounts], dtype=np.str_)),
        }
