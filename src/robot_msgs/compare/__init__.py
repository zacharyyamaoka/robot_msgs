"""compare — one trajectory measured against another: the error signal and its numbers.

    from robot_msgs.compare import compare
    result = compare(target, real)            # or target.compare(real)
    result.error                              # same type as the inputs: reference - measured
    result.metrics["positions/j2"].rms        # ErrorStats per joint / axis / channel

Beside it, each a function over the same trajectories: `check` / `ToleranceMonitor` (a verdict in
ROS 2 joint_trajectory_controller terms), `aggregate` (ISO 9283 accuracy vs repeatability over N
runs), `estimate_time_offset` (latency), `goal` (settling, overshoot), `relative_error` (drift over
windows, evo RPE) and `frechet` (path shape without time).

Pure numpy, like the rest of robot_msgs. Viewers (HTML report, Rerun) live outside and read
`Comparison` values; see docs/DESIGN.md, "Comparing trajectories".
"""

from robot_msgs.compare._dispatch import compare
from robot_msgs.compare.aggregate import Aggregate, aggregate
from robot_msgs.compare.align import align
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.difference import (
    TIME_BASE_CHANNELS,
    contour_errors,
    track_errors,
)
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.compare.frechet import frechet
from robot_msgs.compare.goal import GoalStats, goal
from robot_msgs.compare.relative import relative_error
from robot_msgs.compare.timing import TimeOffsetEstimate, estimate_time_offset
from robot_msgs.compare.tolerances import (
    GOAL_TOLERANCE_VIOLATED,
    PATH_TOLERANCE_VIOLATED,
    SUCCESSFUL,
    JointTolerance,
    Outcome,
    PoseTolerance,
    ToleranceMonitor,
    Tolerances,
    Violation,
    check,
)

__all__ = [
    "GOAL_TOLERANCE_VIOLATED",
    "PATH_TOLERANCE_VIOLATED",
    "SUCCESSFUL",
    "TIME_BASE_CHANNELS",
    "Aggregate",
    "Comparison",
    "ErrorStats",
    "GoalStats",
    "JointTolerance",
    "Outcome",
    "PoseTolerance",
    "TimeOffsetEstimate",
    "ToleranceMonitor",
    "Tolerances",
    "Violation",
    "aggregate",
    "align",
    "check",
    "compare",
    "contour_errors",
    "error_stats",
    "estimate_time_offset",
    "frechet",
    "goal",
    "relative_error",
    "track_errors",
]
