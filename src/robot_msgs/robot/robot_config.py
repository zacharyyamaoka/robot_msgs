"""RobotConfig — how the robot is set up, as each component's own configuration message."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg
from robot_msgs.robot._components import Components
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class RobotConfig(Components, Msg):
    """Component name → that component's configuration message, e.g.
    ``RobotConfig(components={"limits": JointLimits(...), "camera": CameraSettings(...)})``.

    Same shape and methods as `RobotState`: ``config["camera"]``, ``with_components``, ``without``.

    WHY separate from `RobotState`, saved together only when you bundle them: keras keeps config
    (``get_config`` / ``from_config``, ``config.json``) apart from state (``model.weights.h5``)
    and ``model.save`` zips both into one ``.keras`` file; torch keeps ``state_dict()`` apart from
    the code and arguments that build the model, and Hugging Face's ``save_pretrained`` writes
    ``config.json`` beside the weights. Both: two things, one artifact on demand. Here:
    ``robot_msgs.save({"config": config, "state": state}, "arm.npz")`` (Zach asked, 2026-11-26).
    """

    components: Mapping[str, Msg] = field(default_factory=FrozenDict)
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        self._freeze_components()
