"""ErrorStats: the summary numbers of one error signal, and the function that computes them."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ErrorStats(Msg):
    """Summary of one error signal over its finite samples, in the signal's own SI unit.

    `mean`, `median`, `min`, `max` and `final` keep the error's sign; `rms`, `max_abs` and
    `p95_abs` are of its magnitude. WHY this set: evo's rmse / mean / median / std / min / max
    (the SLAM-evaluation standard), plus what tracking needs on top: the worst magnitude, a p95
    that ignores a single glitch sample, and `final`, the error left when the motion ends.
    Missing data is NaN with ``n == 0``, never 0.
    """

    n: int
    rms: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    max_abs: float
    p95_abs: float
    final: float

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def error_stats(values: Any) -> ErrorStats:
    """`ErrorStats` of a 1-D error signal; NaN samples (no reference there) are skipped."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = series[np.isfinite(series)]
    if not finite.size:
        nan = float("nan")
        return ErrorStats(
            n=0, rms=nan, mean=nan, median=nan, std=nan, min=nan, max=nan,
            max_abs=nan, p95_abs=nan, final=nan,
        )  # fmt: skip
    magnitude = np.abs(finite)
    return ErrorStats(
        n=int(finite.size),
        rms=float(np.sqrt(np.mean(finite**2))),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        max_abs=float(np.max(magnitude)),
        p95_abs=float(np.percentile(magnitude, 95.0)),
        final=float(finite[-1]),
    )
