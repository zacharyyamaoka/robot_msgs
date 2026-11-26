"""ImageTrajectory — a camera stream: frames at strictly increasing times, one encoding.

`frames` is one ``(N, H, W)`` or ``(N, H, W, C)`` array in the dtype `encoding` names (kept as
is, like `Image.data`), held between samples: a frame is never blended with the next.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from robot_msgs._core import Array, set_field
from robot_msgs.sensor import Image
from robot_msgs.sensor.image import IMAGE_ENCODINGS
from robot_msgs.std import Header
from robot_msgs.trajectory.trajectory import (
    Extrapolation,
    Trajectory,
    TrajectoryPoint,
    sample,
)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ImageTrajectoryPoint(TrajectoryPoint):
    """One frame of an `ImageTrajectory`; `image` is it as a sensor `Image`.

    `encoding` is the trajectory's, carried so the frame can be read on its own.
    """

    frame: Array | None = field(default=None, metadata={"dtype": None})
    encoding: str = ""

    @property
    def image(self) -> Image:
        """This frame as an `Image`, stamped ``header.stamp + t``."""
        return Image(
            data=self.frame,
            encoding=self.encoding,
            header=Header(stamp=self.header.stamp + self.t, frame_id=self.header.frame_id),
        )


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ImageTrajectory(Trajectory):
    """Frames `(N, H, W[, C])` of one `encoding` (a sensor_msgs/image_encodings name, as `Image`).

    ``traj[i]`` is a point like every trajectory's; ``traj[i].image`` and ``traj.image_at(t)`` are
    the frame as an `Image`. `at` / `resample` hold the previous frame (``"hold"``); float
    channels (exposure, a per-frame score) interpolate as usual.
    """

    encoding: str
    frames: Array = sample(interp="hold", point_name="frame", dtype=None)

    point_type: ClassVar[type[TrajectoryPoint]] = ImageTrajectoryPoint

    def __post_init__(self) -> None:
        super().__post_init__()
        encoding = str(self.encoding)
        if encoding not in IMAGE_ENCODINGS:
            raise ValueError(
                f"ImageTrajectory.encoding {encoding!r} is not supported; "
                f"use one of {sorted(IMAGE_ENCODINGS)}"
            )
        set_field(self, "encoding", encoding)
        dtype, channels = IMAGE_ENCODINGS[encoding]
        frames = self.frames
        if frames.dtype != dtype:
            raise ValueError(
                f"ImageTrajectory.frames has dtype {frames.dtype}, "
                f"encoding {encoding} needs {dtype}"
            )
        if frames.ndim == 3:
            actual = 1
        elif frames.ndim == 4:
            actual = frames.shape[3]
        else:
            raise ValueError(
                "ImageTrajectory.frames must be (N, H, W) or (N, H, W, C), "
                f"got shape {frames.shape}"
            )
        if actual != channels:
            raise ValueError(
                f"ImageTrajectory.frames has {actual} channel(s), "
                f"encoding {encoding} needs {channels}"
            )

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """One frame's shape, ``(H, W)`` or ``(H, W, C)``."""
        return tuple(int(n) for n in self.frames.shape[1:])

    def _point(self, index: int) -> TrajectoryPoint:
        point = super()._point(index)
        set_field(point, "encoding", self.encoding)
        return point

    def images(self) -> tuple[Image, ...]:
        return tuple(point.image for point in self)

    def image_at(self, time: float, *, extrapolate: Extrapolation = "raise") -> Image:
        """The frame shown at `time` (the latest one at or before it) as an `Image`.

        Stamped with that frame's own time, not `time`. WHY: a held frame was taken when it was
        taken; restamping it would claim a picture nobody took.
        """
        lower, upper, alpha = self._bracket(np.array([float(time)]), extrapolate)
        return self[int(upper[0] if alpha[0] >= 1.0 else lower[0])].image

    @classmethod
    def from_points(  # type: ignore[override]
        cls, points: Sequence[TrajectoryPoint], **static: Any
    ) -> ImageTrajectory:
        """Stack frames back into a stream; `encoding` defaults to the first point's."""
        if points and "encoding" not in static:
            static["encoding"] = getattr(points[0], "encoding", "")
        return super().from_points(points, **static)

    @classmethod
    def from_images(cls, images: Sequence[Image], **static: object) -> ImageTrajectory:
        """A recorded stream from a log of Images of one encoding and size: stamps become `t`,
        the first one the header's stamp."""
        if not images:
            raise ValueError("from_images needs at least one Image")
        first = images[0]
        for image in images[1:]:
            if image.encoding != first.encoding or image.data.shape != first.data.shape:
                raise ValueError(
                    f"every Image must be {first.encoding} {first.data.shape}, got "
                    f"{image.encoding} {image.data.shape}"
                )
        stamps = np.array([image.header.stamp for image in images], dtype=np.float64)
        header = Header(stamp=float(stamps[0]), frame_id=first.header.frame_id)
        return cls(
            t=stamps - stamps[0],
            encoding=first.encoding,
            frames=np.stack([image.data for image in images]),
            header=static.pop("header", header),  # type: ignore[arg-type]
            **static,  # type: ignore[arg-type]
        )
