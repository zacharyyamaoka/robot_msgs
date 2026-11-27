"""behavior — what a behaviour tree exposes about itself.

`BTState` is the tree's state at one instant (node statuses, blackboard, tick, clock), so a tree
sits in a `RobotState` beside the joints and the camera like any other component.
"""

from robot_msgs.behavior.bt_state import FAILURE, IDLE, RUNNING, STATUSES, SUCCESS, BTState

__all__ = ["BTState", "FAILURE", "IDLE", "RUNNING", "STATUSES", "SUCCESS"]
