"""Detection2D — one detected object in an image. Mirrors vision_msgs/Detection2D."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_msgs._core import Msg, set_field
from robot_msgs.std import Header
from robot_msgs.vision.bounding_box2d import BoundingBox2D
from robot_msgs.vision.object_hypothesis_with_pose import ObjectHypothesisWithPose


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Detection2D(Msg):
    """`results` are the class hypotheses for the box; `id` tracks the object across frames."""

    header: Header = field(default_factory=Header)
    results: tuple[ObjectHypothesisWithPose, ...] = ()
    bbox: BoundingBox2D
    id: str = ""

    def __post_init__(self) -> None:
        set_field(self, "results", tuple(self.results))
        set_field(self, "id", str(self.id))

    @property
    def best(self) -> ObjectHypothesisWithPose | None:
        """The highest-scoring hypothesis (first one wins a tie), or `None` without results."""
        if not self.results:
            return None
        return max(self.results, key=lambda result: result.hypothesis.score)

    @property
    def class_id(self) -> str:
        """Class of the best hypothesis, `""` without results."""
        best = self.best
        return "" if best is None else best.hypothesis.class_id

    @property
    def score(self) -> float:
        """Score of the best hypothesis, `0.0` without results."""
        best = self.best
        return 0.0 if best is None else best.hypothesis.score
