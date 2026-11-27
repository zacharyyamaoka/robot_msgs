"""robot — the whole robot as named components: `RobotState` (what it is now) and `RobotConfig`
(how it is set up). Each component keeps its own message (`JointState`, `BTState`, …)."""

from robot_msgs.robot.robot_config import RobotConfig
from robot_msgs.robot.robot_state import RobotState

__all__ = ["RobotConfig", "RobotState"]
