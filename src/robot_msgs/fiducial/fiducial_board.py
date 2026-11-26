"""FiducialBoard — a printed grid or ChArUco board, the way OpenCV defines it.

The board is a *specification*; `mounts()` lowers it to the `FiducialMap` of its markers in the
board frame, so a board is solved like any other set of mounted markers. Geometry matches
OpenCV 4.6+'s ``cv2.aruco.GridBoard`` / ``CharucoBoard`` (tests check it against OpenCV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from robot_msgs._core import Array, Msg, freeze_new, set_field
from robot_msgs.fiducial.fiducial_detection import checked_key, checked_length
from robot_msgs.fiducial.fiducial_map import FiducialMap
from robot_msgs.fiducial.fiducial_mount import FiducialMount
from robot_msgs.geometry import Point, Pose, Quaternion
from robot_msgs.std import Header

BoardLayout = Literal["grid", "charuco"]
# WHY Rx(pi): the board frame is OpenCV's (origin at the top-left outer corner as printed, x right,
# y down, z into the board); a marker's own frame has y up and z out of the face.
_MARKER_IN_BOARD = Quaternion(1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class FiducialBoard(Msg):
    """A board of `cols` × `rows`, `header.frame_id` its frame.

    - ``layout="grid"``: `cols` × `rows` markers, `separation` m between neighbours.
    - ``layout="charuco"``: `cols` × `rows` chessboard squares of `square_length` m, a marker on
      every white square; `legacy` is OpenCV's ``setLegacyPattern`` (pre-4.6 boards, which differ
      only when `rows` is even).
    - `marker_length`: side of each marker's black square, m.
    - `ids`: the markers' ids in OpenCV's row-major order; defaults to ``first_id, first_id+1, …``.
    """

    layout: BoardLayout
    dictionary: str
    cols: int
    rows: int
    marker_length: float
    separation: float | None = None
    square_length: float | None = None
    legacy: bool = False
    first_id: int = 0
    ids: tuple[int, ...] | None = (
        None  # WHY None: the default depends on the size; a tuple after init
    )
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        if self.layout not in ("grid", "charuco"):
            raise ValueError(
                f"FiducialBoard.layout must be 'grid' or 'charuco', got {self.layout!r}"
            )
        _, dictionary = checked_key("FiducialBoard", self.first_id, self.dictionary)
        set_field(self, "dictionary", dictionary)
        for name in ("cols", "rows"):
            value = int(getattr(self, name))
            if value < 1 or (self.layout == "charuco" and value < 2):
                raise ValueError(f"FiducialBoard.{name} is too small for a {self.layout}: {value}")
            set_field(self, name, value)
        length = checked_length(
            "FiducialBoard", "marker_length", self.marker_length, optional=False
        )
        set_field(self, "marker_length", length)
        set_field(self, "legacy", bool(self.legacy))
        set_field(self, "first_id", int(self.first_id))
        if self.layout == "grid":
            separation = float(self.separation if self.separation is not None else math.nan)
            if not math.isfinite(separation) or separation < 0.0:
                raise ValueError("FiducialBoard.separation (m, >= 0) is required for a grid")
            if self.square_length is not None:
                raise ValueError("FiducialBoard.square_length is for a charuco board, not a grid")
            set_field(self, "separation", separation)
        else:
            square = checked_length(
                "FiducialBoard", "square_length", self.square_length, optional=False
            )
            if square <= length:  # type: ignore[operator]
                raise ValueError(
                    f"FiducialBoard.marker_length {length} must be under square_length {square}"
                )
            if self.separation is not None:
                raise ValueError("FiducialBoard.separation is for a grid, not a charuco board")
            set_field(self, "square_length", square)
        count = len(self._cells())
        ids = tuple(range(self.first_id, self.first_id + count)) if self.ids is None else self.ids
        ids = tuple(checked_key("FiducialBoard", i, dictionary)[0] for i in ids)
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"FiducialBoard.ids needs {count} distinct ids, got {ids}")
        set_field(self, "ids", ids)

    def _cells(self) -> list[tuple[int, int]]:
        """(col, row) of each marker, row-major."""
        if self.layout == "grid":
            return [(c, r) for r in range(self.rows) for c in range(self.cols)]
        # WHY: OpenCV puts the first marker on the second square of the top row; the legacy
        # (pre-4.6) pattern starts on the first square when the board has an even number of rows.
        parity = 0 if (self.legacy and self.rows % 2 == 0) else 1
        return [(c, r) for r in range(self.rows) for c in range(self.cols) if (c + r) % 2 == parity]

    @property
    def size(self) -> tuple[float, float]:
        """Board (width, height), m: the outer edge of the markers or squares."""
        if self.layout == "grid":
            pitch = self.marker_length + self.separation  # type: ignore[operator]
            return (self.cols * pitch - self.separation, self.rows * pitch - self.separation)  # type: ignore[operator]
        return (self.cols * self.square_length, self.rows * self.square_length)  # type: ignore[operator]

    def mounts(self) -> FiducialMap:
        """Each marker as a `FiducialMount` in the board frame (`header.frame_id`)."""
        half = self.marker_length / 2.0
        if self.layout == "grid":
            pitch = self.marker_length + self.separation  # type: ignore[operator]
            offset = 0.0
        else:
            pitch = self.square_length  # type: ignore[assignment]
            offset = (self.square_length - self.marker_length) / 2.0  # type: ignore[operator]
        mounts = tuple(
            FiducialMount(
                id=marker_id,
                dictionary=self.dictionary,
                marker_length=self.marker_length,
                pose=Pose(
                    Point(col * pitch + offset + half, row * pitch + offset + half, 0.0),
                    _MARKER_IN_BOARD,
                ),
                header=Header(frame_id=self.header.frame_id),
            )
            for marker_id, (col, row) in zip(self.ids, self._cells(), strict=True)  # type: ignore[arg-type]
        )
        return FiducialMap(mounts=mounts)

    def chessboard_corners(self) -> Array:
        """(M, 3) inner chessboard corners of a charuco board in the board frame, m, row-major —
        the points ChArUco calibration uses, index = ChArUco corner id."""
        if self.layout != "charuco":
            raise ValueError("FiducialBoard.chessboard_corners is for a charuco board")
        square = self.square_length
        points = [
            ((c + 1) * square, (r + 1) * square, 0.0)  # type: ignore[operator]
            for r in range(self.rows - 1)
            for c in range(self.cols - 1)
        ]
        return freeze_new(np.array(points, dtype=float).reshape(-1, 3))
