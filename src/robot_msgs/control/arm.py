"""Deprecated names for `robot_msgs.control.hardware_interface` (renamed 2026-11-28: Arm -> HardwareInterface).

Kept so code written against c72d664 still imports; new code uses HardwareInterface / HardwareBackend /
HardwareDeployment.
"""

from robot_msgs.control.hardware_interface import (
    BackendUnavailable,
    HardwareBackend,
    HardwareDeployment,
    HardwareInterface,
)

Arm = HardwareInterface
ArmBackend = HardwareBackend
ArmDeployment = HardwareDeployment

__all__ = ["Arm", "ArmBackend", "ArmDeployment", "BackendUnavailable"]
