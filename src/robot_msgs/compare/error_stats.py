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

    `iae`, `ise` and `itae` are the control-loop integral criteria (∫|e|, ∫e², ∫(t - t₀)|e|, in
    the signal's unit × s, unit² × s and unit × s²): trapezoids over the intervals whose both ends
    were observed. WHY NaN by default: a series given without its time axis has no integral, and
    data serialised before these fields existed still loads.
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
    iae: float = float("nan")
    ise: float = float("nan")
    itae: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def _integral_criteria(series: np.ndarray, t: Any) -> tuple[float, float, float]:
    """IAE, ISE, ITAE by the trapezoid rule over intervals with both endpoints finite."""
    nan = float("nan")
    if t is None:
        return nan, nan, nan
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    if times.shape != series.shape:
        raise ValueError(f"t has {times.size} samples, the error series {series.size}")
    observed = np.isfinite(series[:-1]) & np.isfinite(series[1:])
    if not observed.any():
        return nan, nan, nan
    width = np.diff(times)[observed]
    first, second = series[:-1][observed], series[1:][observed]
    elapsed = times - times[0]
    start, end = elapsed[:-1][observed], elapsed[1:][observed]
    iae = np.sum(0.5 * width * (np.abs(first) + np.abs(second)))
    ise = np.sum(0.5 * width * (first**2 + second**2))
    itae = np.sum(0.5 * width * (start * np.abs(first) + end * np.abs(second)))
    return float(iae), float(ise), float(itae)


def error_stats(values: Any, t: Any = None) -> ErrorStats:
    """`ErrorStats` of a 1-D error signal; NaN samples (no reference there) are skipped.

    `t` is the signal's time axis; with it the integral criteria are filled, without it they
    are NaN.
    """
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    iae, ise, itae = _integral_criteria(series, t)
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
        iae=iae,
        ise=ise,
        itae=itae,
    )
