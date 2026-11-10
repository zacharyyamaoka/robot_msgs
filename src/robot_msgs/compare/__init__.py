"""compare — one trajectory measured against another: the error signal and its numbers.

    from robot_msgs.compare import compare
    result = compare(target, real)            # or target.compare(real)
    result.error                              # same type as the inputs: reference - measured
    result.metrics["positions/j2"].rms        # ErrorStats per joint / axis / channel

Pure numpy, like the rest of robot_msgs. Viewers (HTML report, Rerun) live outside and read
`Comparison` values; see docs/DESIGN.md, "Comparing trajectories".
"""

from robot_msgs.compare._dispatch import compare
from robot_msgs.compare.align import align
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.difference import track_errors
from robot_msgs.compare.error_stats import ErrorStats, error_stats

__all__ = ["Comparison", "ErrorStats", "align", "compare", "error_stats", "track_errors"]
