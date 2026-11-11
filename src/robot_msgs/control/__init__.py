"""control — what crosses the seam between a controller and the servos it drives.

A controller loops: read a `ServoState`, emit a `ServoCommand`. The driver behind the seam (a
moteus board, a MuJoCo twin, a Damiao or RobStride motor) is the same interface on sim and real.
"""

from robot_msgs.control.servo_state import ServoState

# isort: split
# WHY after ServoState: servo_command imports ServoState for `from_state`.
from robot_msgs.control.servo_command import ServoCommand

__all__ = ["ServoCommand", "ServoState"]
