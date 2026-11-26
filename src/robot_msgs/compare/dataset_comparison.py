"""DatasetComparison — every run of a `Dataset` compared: the table across runs, its worst row."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.compare.dataset import Dataset
from robot_msgs.compare.error_stats import ErrorStats
from robot_msgs.compare.run_comparison import RunComparison, compare_run, split_key

_STATS = tuple(ErrorStats.__dataclass_fields__)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class DatasetComparison(Msg):
    """What `compare_dataset(dataset)` returns: one `RunComparison` per run, by run name.

    ``rows()`` is the whole table (one row per run, pair, stream and metric) for a data-ui table
    or a groupby on tags; ``worst(metric)`` names the run, pair and stream to drill into, and
    ``runs[run].get(a, b, stream)`` is that `Comparison`.
    """

    runs: Mapping[str, RunComparison] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        for name, run in self.runs.items():
            if not isinstance(run, RunComparison):
                raise TypeError(f"DatasetComparison.runs[{name!r}] must be a RunComparison")
            if run.run != name:
                raise ValueError(f"DatasetComparison.runs[{name!r}] is run {run.run!r}")
        set_field(self, "runs", FrozenDict({str(k): v for k, v in self.runs.items()}))
        set_field(self, "title", str(self.title))

    def __repr__(self) -> str:
        compared = sum(len(run.comparisons) for run in self.runs.values())
        skipped = sum(len(run.skipped) for run in self.runs.values())
        title = f"title={self.title!r}, " if self.title else ""
        return (
            f"DatasetComparison({title}runs={list(self.runs)}, comparisons={compared}, "
            f"skipped={skipped})"
        )

    def rows(self) -> list[dict[str, Any]]:
        """Every run's `RunComparison.rows()`, in run order."""
        return [row for run in self.runs.values() for row in run.rows()]

    def skipped(self) -> list[dict[str, str]]:
        """Every stream not compared, as rows ``run a b stream reason``."""
        out = []
        for run in self.runs.values():
            for key, reason in run.skipped.items():
                a, b, stream = split_key(key)
                out.append({"run": run.run, "a": a, "b": b, "stream": stream, "reason": reason})
        return out

    def worst(self, metric: str, stat: str = "rms") -> tuple[str, str, str, str, float]:
        """``(run, a, b, stream, value)`` of the comparison whose `metric` has the largest
        ``|stat|``, over every run, pair and stream; NaN (no data) never wins.

        Raises `KeyError` when no comparison has `metric`, `ValueError` when none has a finite
        value of it.
        """
        if stat not in _STATS:
            raise ValueError(f"stat must be an ErrorStats field {_STATS}, got {stat!r}")
        best: tuple[str, str, str, str, float] | None = None
        found = False
        for run in self.runs.values():
            for key, comparison in run.comparisons.items():
                stats = comparison.metrics.get(metric)
                if stats is None:
                    continue
                found = True
                value = float(getattr(stats, stat))
                if math.isfinite(value) and (best is None or abs(value) > abs(best[4])):
                    best = (run.run, *split_key(key), value)
        if not found:
            raise KeyError(f"no comparison has metric {metric!r}")
        if best is None:
            raise ValueError(f"metric {metric!r} has no finite {stat} in any comparison")
        return best


def compare_dataset(
    dataset: Dataset,
    *,
    contour_window_s: float = 0.25,
    errors: Literal["skip", "raise"] = "skip",
) -> DatasetComparison:
    """`compare_run` on every run of `dataset`."""
    runs = {
        run.name: compare_run(run, contour_window_s=contour_window_s, errors=errors)
        for run in dataset.runs
    }
    return DatasetComparison(runs=runs, title=dataset.title)
