"""Detection3DArray — all detections in one frame. Mirrors vision_msgs/Detection3DArray."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import overload

import numpy as np

from robot_msgs._core import Array, Msg, adopt, set_field
from robot_msgs.std import Header
from robot_msgs.vision.detection3d import Detection3D


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Detection3DArray(Msg):
    """The detections of one frame; `header` is the sensor frame's header."""

    header: Header = field(default_factory=Header)
    detections: tuple[Detection3D, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "detections", tuple(self.detections))

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection3D]:
        return iter(self.detections)

    @overload
    def __getitem__(self, index: int) -> Detection3D: ...
    @overload
    def __getitem__(self, index: slice) -> Detection3DArray: ...
    def __getitem__(self, index: int | slice) -> Detection3D | Detection3DArray:
        if isinstance(index, slice):
            return self.replace(detections=self.detections[index])
        return self.detections[index]

    def as_arrays(self) -> dict[str, Array]:
        """The batch view: `centers` (N, 3), `sizes` (N, 3), `scores` (N,), `class_ids` (N,).

        Each detection contributes its best hypothesis (`""` and 0.0 when it has none).
        Box orientations are not in this view; read them from `detections`.
        """
        centers = np.array([d.bbox.center.position.as_array() for d in self.detections])
        sizes = np.array([d.bbox.size.as_array() for d in self.detections])
        scores = np.array([d.score for d in self.detections], dtype=np.float64)
        class_ids = np.array([d.class_id for d in self.detections], dtype=np.str_)
        return {
            "centers": adopt(centers.reshape(len(self), 3).astype(np.float64)),
            "sizes": adopt(sizes.reshape(len(self), 3).astype(np.float64)),
            "scores": adopt(scores),
            "class_ids": adopt(class_ids),
        }

    def filtered(
        self, min_score: float | None = None, class_ids: Iterable[str] | None = None
    ) -> Detection3DArray:
        """Detections whose best score is at least `min_score` and whose class is in `class_ids`."""
        wanted = None if class_ids is None else frozenset(class_ids)
        kept = tuple(
            d
            for d in self.detections
            if (min_score is None or d.score >= min_score)
            and (wanted is None or d.class_id in wanted)
        )
        return self.replace(detections=kept)

    def nms(self, iou_threshold: float) -> Detection3DArray:
        """Non-maximum suppression, class-agnostic: highest scores win, overlaps above
        `iou_threshold` are dropped. The survivors keep their original order.

        Overlap is the volume IoU of axis-aligned boxes, so every box must have an identity
        orientation (up to sign).
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        for detection in self.detections:
            q = detection.bbox.center.orientation
            if q.x != 0.0 or q.y != 0.0 or q.z != 0.0:
                raise ValueError(
                    "Detection3DArray.nms needs axis-aligned boxes (identity rotation)"
                )
        arrays = self.as_arrays()
        low = arrays["centers"] - arrays["sizes"] / 2.0
        high = arrays["centers"] + arrays["sizes"] / 2.0
        volumes = np.prod(arrays["sizes"], axis=1)
        by_score = sorted(range(len(self)), key=lambda i: -arrays["scores"][i])
        kept: list[int] = []
        for candidate in by_score:
            overlap = np.clip(
                np.minimum(high[candidate], high[kept]) - np.maximum(low[candidate], low[kept]),
                0.0,
                None,
            )
            intersection = np.prod(overlap, axis=1)
            union = volumes[candidate] + volumes[kept] - intersection
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(union > 0.0, intersection / union, 0.0)
            if np.all(iou <= iou_threshold):
                kept.append(candidate)
        return self.replace(detections=tuple(self.detections[i] for i in sorted(kept)))
