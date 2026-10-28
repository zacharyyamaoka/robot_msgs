"""BoundingBox2D — a rectangle in an image. Mirrors vision_msgs/BoundingBox2D."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry import Pose2D


@dataclass(frozen=True, eq=False, repr=False)
class BoundingBox2D(Msg):
    """`center` is the box centre in pixels with rotation `center.theta`; sizes are in pixels."""

    center: Pose2D
    size_x: float
    size_y: float

    def __post_init__(self) -> None:
        for name in ("size_x", "size_y"):
            size = float(getattr(self, name))
            if not math.isfinite(size) or size < 0.0:
                raise ValueError(f"BoundingBox2D.{name} must be finite and >= 0, got {size}")
            set_field(self, name, size)

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> BoundingBox2D:
        """Axis-aligned box from its top-left (x0, y0) and bottom-right (x1, y1) corners."""
        return cls(Pose2D((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0), x1 - x0, y1 - y0)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        self._require_axis_aligned("to_xyxy")
        half_x, half_y = self.size_x / 2.0, self.size_y / 2.0
        return (
            self.center.x - half_x,
            self.center.y - half_y,
            self.center.x + half_x,
            self.center.y + half_y,
        )

    @property
    def area(self) -> float:
        return self.size_x * self.size_y

    def iou(self, other: BoundingBox2D) -> float:
        """Intersection over union of two axis-aligned boxes (0.0 when both are empty)."""
        ax0, ay0, ax1, ay1 = self.to_xyxy()
        bx0, by0, bx1, by1 = other.to_xyxy()
        overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))
        intersection = overlap_x * overlap_y
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def contains(self, x: float, y: float) -> bool:
        """Whether pixel (x, y) is inside the box (edges included), honouring `center.theta`."""
        dx, dy = x - self.center.x, y - self.center.y
        cos_theta, sin_theta = math.cos(self.center.theta), math.sin(self.center.theta)
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy
        return abs(local_x) <= self.size_x / 2.0 and abs(local_y) <= self.size_y / 2.0

    def _require_axis_aligned(self, operation: str) -> None:
        if self.center.theta != 0.0:
            raise ValueError(
                f"BoundingBox2D.{operation} needs an axis-aligned box, theta is {self.center.theta}"
            )
