"""Detection2DArray — all detections in one image. Mirrors vision_msgs/Detection2DArray."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import overload

import numpy as np

from robot_msgs._core import Array, Msg, adopt, set_field
from robot_msgs.std import Header
from robot_msgs.vision.detection2d import Detection2D


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Detection2DArray(Msg):
    """The detections of one frame; `header` is the image's header."""

    header: Header = field(default_factory=Header)
    detections: tuple[Detection2D, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "detections", tuple(self.detections))

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection2D]:
        return iter(self.detections)

    @overload
    def __getitem__(self, index: int) -> Detection2D: ...
    @overload
    def __getitem__(self, index: slice) -> Detection2DArray: ...
    def __getitem__(self, index: int | slice) -> Detection2D | Detection2DArray:
        if isinstance(index, slice):
            return self.replace(detections=self.detections[index])
        return self.detections[index]

    def as_arrays(self) -> dict[str, Array]:
        """The batch view: `boxes_xyxy` (N, 4), `scores` (N,), `class_ids` (N,) strings.

        Each detection contributes its best hypothesis (`""` and 0.0 when it has none).
        Boxes must be axis-aligned.
        """
        boxes = np.array([d.bbox.to_xyxy() for d in self.detections], dtype=np.float64)
        scores = np.array([d.score for d in self.detections], dtype=np.float64)
        class_ids = np.array([d.class_id for d in self.detections], dtype=np.str_)
        return {
            "boxes_xyxy": adopt(boxes.reshape(len(self), 4)),
            "scores": adopt(scores),
            "class_ids": adopt(class_ids),
        }

    def filtered(
        self, min_score: float | None = None, class_ids: Iterable[str] | None = None
    ) -> Detection2DArray:
        """Detections whose best score is at least `min_score` and whose class is in `class_ids`."""
        wanted = None if class_ids is None else frozenset(class_ids)
        kept = tuple(
            d
            for d in self.detections
            if (min_score is None or d.score >= min_score)
            and (wanted is None or d.class_id in wanted)
        )
        return self.replace(detections=kept)

    def nms(self, iou_threshold: float) -> Detection2DArray:
        """Non-maximum suppression, class-agnostic: highest scores win, overlaps above
        `iou_threshold` are dropped. The survivors keep their original order."""
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        by_score = sorted(range(len(self)), key=lambda i: -self.detections[i].score)
        kept: list[int] = []
        for candidate in by_score:
            box = self.detections[candidate].bbox
            if all(box.iou(self.detections[k].bbox) <= iou_threshold for k in kept):
                kept.append(candidate)
        return self.replace(detections=tuple(self.detections[i] for i in sorted(kept)))
