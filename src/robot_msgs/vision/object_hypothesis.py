"""ObjectHypothesis — a class guess with a confidence. Mirrors vision_msgs/ObjectHypothesis."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_msgs._core import Msg, set_field


@dataclass(frozen=True, eq=False, repr=False)
class ObjectHypothesis(Msg):
    """`class_id` names the class (a label or an id string); `score` is the confidence."""

    class_id: str
    score: float

    def __post_init__(self) -> None:
        score = float(self.score)
        if math.isnan(score):
            raise ValueError("ObjectHypothesis.score must not be NaN")
        set_field(self, "class_id", str(self.class_id))
        set_field(self, "score", score)
