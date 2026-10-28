"""Image — a pixel grid with its encoding. Mirrors sensor_msgs/Image."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field
from robot_msgs.std import Header

IMAGE_ENCODINGS: dict[str, tuple[np.dtype, int]] = {
    "rgb8": (np.dtype(np.uint8), 3),
    "bgr8": (np.dtype(np.uint8), 3),
    "rgba8": (np.dtype(np.uint8), 4),
    "bgra8": (np.dtype(np.uint8), 4),
    "mono8": (np.dtype(np.uint8), 1),
    "mono16": (np.dtype(np.uint16), 1),
    "8UC1": (np.dtype(np.uint8), 1),
    "8UC3": (np.dtype(np.uint8), 3),
    "16UC1": (np.dtype(np.uint16), 1),
    "32FC1": (np.dtype(np.float32), 1),
}
"""sensor_msgs/image_encodings that we support: name → (numpy dtype, channels)."""


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Image(Msg):
    """`data` is (H, W) or (H, W, C) in the dtype `encoding` names; the dtype is kept as is.

    Depth conventions follow ROS: 16UC1 is millimetres, 32FC1 is metres.
    """

    data: Array
    encoding: str
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        encoding = str(self.encoding)
        if encoding not in IMAGE_ENCODINGS:
            raise ValueError(
                f"Image.encoding {encoding!r} is not supported; "
                f"use one of {sorted(IMAGE_ENCODINGS)}"
            )
        dtype, channels = IMAGE_ENCODINGS[encoding]
        data = frozen_array(self.data, dtype=None)
        if data.dtype != dtype:
            raise ValueError(
                f"Image.data has dtype {data.dtype}, encoding {encoding} needs {dtype}"
            )
        if data.ndim == 2:
            actual = 1
        elif data.ndim == 3:
            actual = data.shape[2]
        else:
            raise ValueError(f"Image.data must be (H, W) or (H, W, C), got shape {data.shape}")
        if actual != channels:
            raise ValueError(
                f"Image.data has {actual} channel(s), encoding {encoding} needs {channels}"
            )
        set_field(self, "encoding", encoding)
        set_field(self, "data", data)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return IMAGE_ENCODINGS[self.encoding][1]

    @property
    def step(self) -> int:
        """Bytes per row, as in ROS."""
        return self.width * self.channels * self.data.dtype.itemsize

    @classmethod
    def from_array(
        cls, array: Array, encoding: str | None = None, header: Header | None = None
    ) -> Image:
        """Wrap an array. Without `encoding`, infer it where only one reading is sensible.

        uint8 HxWx3 → rgb8, uint8 HxW → mono8, uint16 HxW → 16UC1, float32 HxW → 32FC1.
        Anything else (uint8 HxWx4 could be rgba8 or bgra8) raises: name the encoding.
        """
        array = np.asarray(array)
        if encoding is None:
            channels = 1 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else None
            inferred = {
                (np.dtype(np.uint8), 3): "rgb8",
                (np.dtype(np.uint8), 1): "mono8",
                (np.dtype(np.uint16), 1): "16UC1",
                (np.dtype(np.float32), 1): "32FC1",
            }.get((array.dtype, channels))
            if inferred is None:
                raise ValueError(
                    f"cannot infer an encoding for dtype {array.dtype} shape {array.shape}; "
                    "pass encoding="
                )
            encoding = inferred
        return cls(data=array, encoding=encoding, header=header if header is not None else Header())

    def to_rgb(self) -> Array:
        """(H, W, 3) uint8 for display. Depth and 16-bit images are min-max normalised to grey."""
        data = self.data
        if self.encoding in ("rgb8", "8UC3"):
            return np.array(data, dtype=np.uint8)
        if self.encoding == "bgr8":
            return np.ascontiguousarray(data[..., ::-1])
        if self.encoding == "rgba8":
            return np.ascontiguousarray(data[..., :3])
        if self.encoding == "bgra8":
            return np.ascontiguousarray(data[..., 2::-1])
        grey = data.reshape(self.height, self.width)
        if grey.dtype != np.uint8:
            grey = _normalised_uint8(grey)
        return np.repeat(grey[..., None], 3, axis=2)

    def crop(self, x_offset: int, y_offset: int, width: int, height: int) -> Image:
        """The `width`×`height` window whose top-left pixel is (x_offset, y_offset)."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Image.crop size must be positive, got {width}x{height}")
        if (
            x_offset < 0
            or y_offset < 0
            or x_offset + width > self.width
            or y_offset + height > self.height
        ):
            raise ValueError(
                f"Image.crop window {width}x{height} at ({x_offset}, {y_offset}) "
                f"falls outside the {self.width}x{self.height} image"
            )
        window = self.data[y_offset : y_offset + height, x_offset : x_offset + width]
        return self.replace(data=window)


def _normalised_uint8(grey: Array) -> Array:
    """Scale the finite values to 0..255; non-finite pixels and constant images become 0."""
    values = grey.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(grey.shape, dtype=np.uint8)
    low, high = values[finite].min(), values[finite].max()
    if high == low:
        return np.zeros(grey.shape, dtype=np.uint8)
    scaled = np.where(finite, (values - low) / (high - low), 0.0)
    return np.round(scaled * 255.0).astype(np.uint8)
