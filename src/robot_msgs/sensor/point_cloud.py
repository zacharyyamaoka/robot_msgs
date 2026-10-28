"""PointCloud — N points with per-point channels. Mirrors sensor_msgs/PointCloud2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, FrozenDict, Msg, frozen_array, set_field
from robot_msgs.geometry import Transform
from robot_msgs.sensor.camera_info import CameraInfo
from robot_msgs.sensor.image import Image
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class PointCloud(Msg):
    """`points` is (N, 3) float32 in `header.frame_id`; each channel has leading dimension N.

    Channels keep their own dtype: `rgb` (N, 3) uint8, `intensity` (N,), `normals` (N, 3) ...

    WHY float32 points: half the memory of float64, and it is what PointCloud2 carries.
    """

    points: Array
    channels: Mapping[str, Array] = field(default_factory=dict)
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        points = frozen_array(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"PointCloud.points must have shape (N, 3), got {points.shape}")
        channels: dict[str, Array] = {}
        for name, values in self.channels.items():
            values = frozen_array(values, dtype=None)
            if values.ndim < 1 or values.shape[0] != len(points):
                raise ValueError(
                    f"PointCloud.channels[{name!r}] has leading dimension "
                    f"{values.shape[0] if values.ndim else 'none'}, points has {len(points)}"
                )
            channels[str(name)] = values
        set_field(self, "points", points)
        set_field(self, "channels", FrozenDict(channels))

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def from_depth(
        cls, depth: Image, camera_info: CameraInfo, rgb: Image | None = None
    ) -> PointCloud:
        """Back-project a depth image through the pinhole model, row-major pixel order.

        16UC1 depth is millimetres and 32FC1 is metres; pixels with zero, negative or NaN depth
        are dropped. An aligned `rgb` image becomes the `rgb` channel (N, 3) uint8.
        """
        if depth.encoding not in ("16UC1", "32FC1"):
            raise ValueError(f"from_depth needs a 16UC1 or 32FC1 depth image, got {depth.encoding}")
        for label, image in (("depth", depth), ("rgb", rgb)):
            if image is not None and (image.width, image.height) != (
                camera_info.width,
                camera_info.height,
            ):
                raise ValueError(
                    f"{label} image is {image.width}x{image.height}, "
                    f"camera_info is {camera_info.width}x{camera_info.height}"
                )
        metres = depth.data.reshape(depth.height, depth.width).astype(np.float64)
        if depth.encoding == "16UC1":
            metres = metres / 1000.0
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(metres) & (metres > 0.0)
        rows, columns = np.nonzero(valid)
        pixels = np.stack([columns, rows], axis=1).astype(np.float64)
        points = camera_info.unproject(pixels, metres[rows, columns])
        channels = {} if rgb is None else {"rgb": rgb.to_rgb()[rows, columns]}
        return cls(points=points, channels=channels, header=depth.header)

    def transformed(self, transform: Transform) -> PointCloud:
        """Rotate then translate the points. Channels are unchanged; set the new `header`."""
        matrix = transform.to_matrix()
        moved = self.points.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        return self.replace(points=moved)

    def cropped(self, min_xyz: Array, max_xyz: Array) -> PointCloud:
        """Points inside the axis-aligned box (edges included), channels filtered alike."""
        low = _as_corner(min_xyz, "min_xyz")
        high = _as_corner(max_xyz, "max_xyz")
        keep = np.all((self.points >= low) & (self.points <= high), axis=1)
        return self._select(keep)

    def downsampled(self, voxel_size: float) -> PointCloud:
        """One point per `voxel_size` cube: the first point (in cloud order) that falls in it."""
        if not voxel_size > 0:
            raise ValueError(
                f"PointCloud.downsampled voxel_size must be positive, got {voxel_size}"
            )
        if len(self) == 0:
            return self
        cells = np.floor(self.points.astype(np.float64) / voxel_size).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        return self._select(np.sort(first))

    def _select(self, index: Array) -> PointCloud:
        return self.replace(
            points=self.points[index],
            channels={name: values[index] for name, values in self.channels.items()},
        )


def _as_corner(value: Array, name: str) -> Array:
    corner = np.asarray(value, dtype=np.float64)
    if corner.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {corner.shape}")
    return corner
