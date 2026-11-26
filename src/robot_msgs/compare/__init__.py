"""compare — one message measured against another: the error signal and its numbers.

    from robot_msgs.compare import compare
    result = compare(target, real)            # or target.compare(real)
    result.error                              # same type as the inputs: reference - measured
    result.metrics["positions/j2"].rms        # ErrorStats per joint / axis / channel

Every recorded stream compares the same way (`ServoStateTrajectory`, `ServoCommandTrajectory`,
`ImuTrajectory`, `WrenchTrajectory`, `ImageTrajectory`), single messages as one-sample streams,
point clouds by nearest neighbour. Many runs are a `Dataset` of `Run`s, each one `Recording` per
source: `compare_dataset(dataset)` → `DatasetComparison` (rows across runs, `worst` to drill in).

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
from robot_msgs.compare.dataset import Dataset
from robot_msgs.compare.dataset_comparison import DatasetComparison, compare_dataset
from robot_msgs.compare.difference import (
    TIME_BASE_CHANNELS,
    contour_errors,
    track_errors,
)
from robot_msgs.compare.error_stats import ErrorStats, error_stats
from robot_msgs.compare.frechet import frechet
from robot_msgs.compare.goal import GoalStats, goal
from robot_msgs.compare.recording import Recording
from robot_msgs.compare.relative import relative_error
from robot_msgs.compare.run import Run, default_pairs
from robot_msgs.compare.run_comparison import (
    RunComparison,
    compare_run,
    comparison_key,
    split_key,
)
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
    "Dataset",
    "DatasetComparison",
    "ErrorStats",
    "GoalStats",
    "JointTolerance",
    "Outcome",
    "PoseTolerance",
    "Recording",
    "Run",
    "RunComparison",
    "TimeOffsetEstimate",
    "ToleranceMonitor",
    "Tolerances",
    "Violation",
    "aggregate",
    "align",
    "check",
    "compare",
    "compare_dataset",
    "compare_run",
    "comparison_key",
    "contour_errors",
    "default_pairs",
    "error_stats",
    "estimate_time_offset",
    "frechet",
    "goal",
    "relative_error",
    "split_key",
    "track_errors",
]
