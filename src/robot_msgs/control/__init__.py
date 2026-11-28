"""control — what crosses the seam between a controller and the servos it drives.

A controller loops: read a `ServoState`, emit a `ServoCommand`. The driver behind the seam (a
moteus board, a MuJoCo twin, a Damiao or RobStride motor) is the same interface on sim and real.

A routine that is a list of steps (homing, recovery) is a `JointCommandSequence`. Since 0.8 it is written
as three primitive step kinds (`Command`, `Delay`, `Condition`), built with the helper FUNCTIONS in
`robot_msgs.control.helpers` (`find_hardstop`, `move_to`, `stop_all`, ...), and run by the
`JointCommandController`: the one writer to the servos (stop, brake, hold, follow; one writer per joint; hold,
don't drop). The 0.3 `JointCommandSequenceController` still runs step sequences as before.

`robot_msgs.control.recipe` is the short way to write those helper calls: one vocabulary of keywords that say
what each number is (`pos`, `by`, `vel`, `max_tau`, `ff`, ...), a `Joint` that carries its defaults
(`J2.move_by(deg(20))`), a `Group` addressed all / a set / one (`Arm.stop()`, `Arm[ALL]`, `Arm["elbow"]`) and
tview-style text lines (`script("elbow> move_by 20deg")`). Same primitives; only `call` reads as written.

A `JointTable` is the thin Arm between them and the bus: joint name -> `Servo` (bus, CAN id,
sign, offset), so controllers address joints by name and only the table knows a CAN id. An `Arm` is that table in
front of a backend (`ArmDeployment`: ``Arm(UR5E, backend="sim")`` / ``"real"``), and a simulated `ServoState`
carries the simulator's `GroundTruth` in `truth` (None on real).

Joint names are cell-unique slash paths (`robot_msgs.control.joint_path`): an arm's table names `shoulder`,
`JointTable.mounted("right_arm")` makes it `right_arm/shoulder`, `JointTable.combine` joins arms into one cell
table, and `select(command, "right_arm")` is the same joint-named message viewed at one level (all / a set / one).
"""

from robot_msgs.control.ground_truth import GroundTruth
from robot_msgs.control.servo_state import ServoState

# isort: split
# WHY after ServoState: servo_command imports ServoState for `from_state`.
from robot_msgs.control.servo_command import SERVO_MODES, ServoCommand

# isort: split
# WHY after ServoCommand: joint_path and joint_table filter / map ServoCommand / ServoState.
from robot_msgs.control import joint_path
from robot_msgs.control.joint_path import add_prefix, is_under, join_path, select, split_path, strip_prefix
from robot_msgs.control.joint_table import JointTable, Servo, parse_servo_key, servo_key

# isort: split
from robot_msgs.control.arm import Arm, ArmBackend, ArmDeployment, BackendUnavailable

# isort: split
from robot_msgs.control import helpers
from robot_msgs.control.controller_manager import ControllerManager, ManagerTick
from robot_msgs.control.joint_command import (
    ALL_JOINTS,
    MODES,
    DoneWhen,
    EffortStall,
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    Settled,
    Stall,
)
from robot_msgs.control.joint_command_controller import (
    Event,
    FollowRequest,
    HardstopReport,
    JointCommandController,
    JointCommandResult,
    JointCommandTick,
    SampledFollower,
)
from robot_msgs.control.primitives import (
    PRIMITIVES,
    Command,
    Condition,
    Delay,
    FailWhen,
    FollowEnded,
    InCollision,
    Record,
    Reduce,
    TravelEnd,
)
from robot_msgs.control.sequence_controller import (
    JointCommandSequenceController,
    JointController,
    StepEvent,
)

# isort: split
# WHY last: recipe is a way to WRITE helpers' primitives (short keywords, Joint, Group, text lines).
from robot_msgs.control import recipe

__all__ = [
    "ALL_JOINTS",
    "Arm",
    "ArmBackend",
    "ArmDeployment",
    "BackendUnavailable",
    "GroundTruth",
    "Command",
    "Condition",
    "ControllerManager",
    "Delay",
    "Event",
    "FailWhen",
    "FollowEnded",
    "FollowRequest",
    "HardstopReport",
    "InCollision",
    "JointCommandController",
    "JointCommandResult",
    "JointCommandTick",
    "JointTable",
    "ManagerTick",
    "PRIMITIVES",
    "Record",
    "Reduce",
    "SampledFollower",
    "TravelEnd",
    "helpers",
    "MODES",
    "DoneWhen",
    "Elapsed",
    "JointCommand",
    "JointCommandSequence",
    "JointCommandSequenceController",
    "JointCommandStep",
    "JointController",
    "Reached",
    "SERVO_MODES",
    "Servo",
    "ServoCommand",
    "ServoState",
    "add_prefix",
    "is_under",
    "join_path",
    "joint_path",
    "select",
    "split_path",
    "strip_prefix",
    "Settled",
    "EffortStall",
    "Stall",
    "StepEvent",
    "parse_servo_key",
    "recipe",
    "servo_key",
]
