"""Comparison: one trajectory measured against a reference — the error signal and its numbers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.compare.error_stats import ErrorStats
from robot_msgs.trajectory import Trajectory


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Comparison(Msg):
    """What `robot_msgs.compare.compare(reference, measured)` returns.

    - `error` is a trajectory of the *same type* as the inputs, on the measured run's own
      samples, holding ``reference - measured`` (the ROS 2 controller-state convention): joint
      error is a `JointTrajectory`, pose error a `PoseTrajectory`. Derived error forms (rotation
      vector, lag, cross-track, …) ride along as `channels`. WHY one type, not a parallel
      "error" schema: plots, resampling, slicing and serialisation already work on it.
    - `metrics` maps a stable name (``"positions/j2"``, ``"position"``, ``"lag"``,
      ``"channels/closure_gap"``) to its `ErrorStats`.
    - `time_offset_s`, `dropped` and `coverage` record how the two were lined up, so a number is
      never separated from the decision that produced it: the offset applied to the measured
      clock, how many measured samples fell outside the reference (never extrapolated), and the
      fraction of the reference's duration the measured run covers.
    """

    error: Trajectory
    metrics: Mapping[str, ErrorStats] = field(default_factory=dict)
    time_offset_s: float = 0.0
    dropped: int = 0
    coverage: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.error, Trajectory):
            raise TypeError(
                f"Comparison.error must be a Trajectory, got {type(self.error).__name__}"
            )
        for name, stats in self.metrics.items():
            if not isinstance(stats, ErrorStats):
                raise TypeError(f"Comparison.metrics[{name!r}] must be ErrorStats")
        set_field(self, "metrics", FrozenDict({str(k): v for k, v in self.metrics.items()}))
        set_field(self, "time_offset_s", float(self.time_offset_s))
        set_field(self, "dropped", int(self.dropped))
        set_field(self, "coverage", float(self.coverage))

    def rows(self) -> list[dict[str, Any]]:
        """The metrics as tidy rows — one dict per metric — for tables and data-ui."""
        return [{"metric": name, **stats.as_dict()} for name, stats in self.metrics.items()]
