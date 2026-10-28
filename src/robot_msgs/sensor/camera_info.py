"""CameraInfo — intrinsics of a pinhole camera. Mirrors sensor_msgs/CameraInfo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CameraInfo(Msg):
    """Image size, camera matrix `k`, distortion `d`, rectification `r`, projection `p`.

    `k`, `r` and `p` are stored as matrices (ROS flattens them to 9, 9 and 12 numbers;
    converters flatten). Pixel coordinates are (u = column, v = row), x right, y down, z forward.
    `p` defaults to `[k | 0]` when left out.
    """

    width: int
    height: int
    k: Array
    d: Array = field(default_factory=lambda: np.zeros(0))
    distortion_model: str = "plumb_bob"
    r: Array = field(default_factory=lambda: np.eye(3))
    p: Array | None = None  # WHY None: the default depends on k; always an array after init
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        width, height = int(self.width), int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"CameraInfo.width/height must be positive, got {width}x{height}")
        set_field(self, "width", width)
        set_field(self, "height", height)
        k = frozen_array(self.k)
        if k.shape != (3, 3):
            raise ValueError(f"CameraInfo.k must have shape (3, 3), got {k.shape}")
        set_field(self, "k", k)
        d = frozen_array(self.d)
        if d.ndim != 1:
            raise ValueError(f"CameraInfo.d must be 1-D, got shape {d.shape}")
        set_field(self, "d", d)
        r = frozen_array(self.r)
        if r.shape != (3, 3):
            raise ValueError(f"CameraInfo.r must have shape (3, 3), got {r.shape}")
        set_field(self, "r", r)
        p = np.hstack([k, np.zeros((3, 1))]) if self.p is None else frozen_array(self.p)
        if p.shape != (3, 4):
            raise ValueError(f"CameraInfo.p must have shape (3, 4), got {p.shape}")
        set_field(self, "p", frozen_array(p))
        set_field(self, "distortion_model", str(self.distortion_model))

    @classmethod
    def from_intrinsics(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        d: Array | None = None,
        header: Header | None = None,
    ) -> CameraInfo:
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return cls(
            width=width,
            height=height,
            k=k,
            d=np.zeros(0) if d is None else d,
            header=header if header is not None else Header(),
        )

    @property
    def fx(self) -> float:
        return float(self.k[0, 0])

    @property
    def fy(self) -> float:
        return float(self.k[1, 1])

    @property
    def cx(self) -> float:
        return float(self.k[0, 2])

    @property
    def cy(self) -> float:
        return float(self.k[1, 2])

    def project(self, points: Array) -> Array:
        """(N, 3) points in the camera optical frame → (N, 2) pixels (u, v).

        WHY pinhole only: v1 ignores `d` (distortion), so this is exact only for rectified
        images. Points with z == 0 come back non-finite.
        """
        points = _as_points(points, "points")
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * points[:, 0] / points[:, 2] + self.cx
            v = self.fy * points[:, 1] / points[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def unproject(self, pixels: Array, depth: Array) -> Array:
        """(N, 2) pixels (u, v) with (N,) depth z along the optical axis → (N, 3) points."""
        pixels = np.asarray(pixels, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}")
        if depth.shape != (len(pixels),):
            raise ValueError(f"depth must have shape ({len(pixels)},), got {depth.shape}")
        x = (pixels[:, 0] - self.cx) / self.fx * depth
        y = (pixels[:, 1] - self.cy) / self.fy * depth
        return np.stack([x, y, depth], axis=1)

    def scaled(self, factor: float) -> CameraInfo:
        """Intrinsics for the same image resized by `factor` (sizes round to whole pixels).

        WHY plain proportional scaling of k and p (as ROS binning does): the exact map for a
        pixel-centre convention is (c + 0.5) * s - 0.5, which differs by under a pixel.
        """
        if not factor > 0:
            raise ValueError(f"CameraInfo.scaled factor must be positive, got {factor}")
        width = max(1, round(self.width * factor))
        height = max(1, round(self.height * factor))
        scale_x, scale_y = width / self.width, height / self.height
        row_scale = np.array([[scale_x], [scale_y], [1.0]])
        return self.replace(width=width, height=height, k=self.k * row_scale, p=self.p * row_scale)


def _as_points(points: Array, name: str) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    return points
