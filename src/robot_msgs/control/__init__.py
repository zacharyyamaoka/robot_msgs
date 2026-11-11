"""control — what crosses the seam between a controller and the servos it drives.

A controller loops: read a `ServoState`, emit a `ServoCommand`. The driver behind the seam (a
moteus board, a MuJoCo twin, a Damiao or RobStride motor) is the same interface on sim and real.

A routine that is a list of steps (homing, recovery) is a `JointCommandSequence`, run by a
`JointCommandSequenceController`.
"""

from robot_msgs.control.servo_state import ServoState

# isort: split
# WHY after ServoState: servo_command imports ServoState for `from_state`.
from robot_msgs.control.servo_command import ServoCommand

# isort: split
from robot_msgs.control.joint_command import (
    MODES,
    DoneWhen,
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    Settled,
    Stall,
)
from robot_msgs.control.sequence_controller import (
    JointCommandSequenceController,
    JointController,
    StepEvent,
)

__all__ = [
    "MODES",
    "DoneWhen",
    "Elapsed",
    "JointCommand",
    "JointCommandSequence",
    "JointCommandSequenceController",
    "JointCommandStep",
    "JointController",
    "Reached",
    "ServoCommand",
    "ServoState",
    "Settled",
    "Stall",
    "StepEvent",
]
