"""coverage — how much of a binned input space a set of points visits.

    from robot_msgs.coverage import binned, marginal
    axes = [q_edges, qd_edges, ("levels", "PET", "HDPE")]  # float edges, or categorical levels
    result = binned(points, axes, reachable=lambda centre: ...)
    result.pct, result.visited, result.reachable   # % of reachable cells with a point

One rule for every device: an axis is a list of float edges (continuous) or a tuple
``("levels", a, b, ...)`` (categorical); a point is one value per axis; coverage is the share of
the reachable cells that at least one point visits. Bins follow numpy.histogramdd: [lo, hi) per
bin, the last bin [lo, hi], points outside every edge dropped and counted in ``outside``.

WHY only the textbook part lives here (Zach, 2026-11-10, "Sim2Real as a Pure Library on
robot_msgs", proposal 2): robot_msgs is the slowest-changing package, so it takes the nouns and
the math with one definition. Which axes a device type has, its edges, which cells it can reach,
and any target or gate are policy that churns weekly; they stay in sim2real (its cards), which
calls these functions. Moved from sim2real/type_cards/coverage.py unchanged in behaviour.

Plain Python on purpose: the points are often a few hundred plan rows mixed with categorical
levels, and the results must stay bit-identical to the sim2real numbers already published.
"""

from __future__ import annotations

import bisect
import itertools
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CoverageResult",
    "bin_index",
    "binned",
    "centres",
    "marginal",
    "n_cells",
    "occupancy",
    "random_plan",
    "signed",
]


@dataclass(frozen=True)
class CoverageResult:
    """The coverage of one point set over one grid.

    WHY a plain frozen dataclass and not a `Msg`: it is a computed result read by value
    (``==``, ``asdict``, ``__dict__``) by sim2real's site and reports, which were built on
    exactly this shape; a `Msg` would make it compare by identity and its dict fields read-only
    mappings, changing what every caller sees.
    """

    pct: float  # visited reachable cells / reachable cells, %, 2 decimals
    visited: int  # reachable cells with at least one point
    reachable: int  # the denominator
    samples: int  # points offered
    outside: int = 0  # points outside every edge (dropped, as histogramdd does)
    axes: dict = field(default_factory=dict)  # extra axis -> its marginal %, None if not varied
    unplanned: tuple = ()  # axes the caller declares the plan cannot cover (said, never hidden)
    cells: dict = field(default_factory=dict)  # cell index tuple -> count (for plots)
    note: str = ""


def bin_index(value: float, edges: Sequence[float]) -> int | None:
    """numpy.histogramdd's convention: [lo, hi) per bin, the last bin [lo, hi]; outside -> None."""
    if value == edges[-1]:
        return len(edges) - 2
    i = bisect.bisect_right(edges, value) - 1
    return i if 0 <= i < len(edges) - 1 else None


def occupancy(points: Sequence[Sequence[Any]], axes: Sequence[Sequence[Any]]) -> tuple[dict, int]:
    """Points per cell, and how many fell outside. An axis is a list of float edges
    (continuous) or ``("levels", a, b, ...)`` (categorical)."""
    counts: dict[tuple, int] = {}
    outside = 0
    for point in points:
        cell = []
        for value, axis in zip(point, axes, strict=False):
            if axis and axis[0] == "levels":
                i = axis.index(value) - 1 if value in axis[1:] else None
            else:
                i = bin_index(float(value), axis)
            if i is None:
                break
            cell.append(i)
        else:
            counts[tuple(cell)] = counts.get(tuple(cell), 0) + 1
            continue
        outside += 1
    return counts, outside


def n_cells(axis: Sequence[Any]) -> int:
    """Bins on one axis (edges - 1, or the number of levels)."""
    return len(axis) - 1


def centres(axis: Sequence[Any]) -> list:
    """Each bin's centre, or each level's name."""
    if axis and axis[0] == "levels":
        return list(axis[1:])
    return [0.5 * (a + b) for a, b in zip(axis[:-1], axis[1:], strict=False)]


def binned(
    points: Sequence[Sequence[Any]],
    grid_axes: Sequence[Sequence[Any]],
    reachable: Callable[[tuple], bool] | None = None,
    reach: frozenset | set | None = None,
    **extra: Any,
) -> CoverageResult:
    """Coverage over the product of `grid_axes`. `reachable(centre tuple)` drops cells the device
    cannot hold (None: every cell counts), or `reach` names the reachable cells already (a set the
    caller computed once for the same axes). `extra` fills CoverageResult's other fields."""
    counts, outside = occupancy(points, grid_axes)
    if reach is None:
        grid = [centres(a) for a in grid_axes]
        reach = {
            cell
            for cell in itertools.product(*(range(n_cells(a)) for a in grid_axes))
            if reachable is None or reachable(tuple(grid[k][i] for k, i in enumerate(cell)))
        }
    visited = sum(1 for cell in counts if cell in reach)
    total = max(1, len(reach))
    return CoverageResult(
        pct=round(100.0 * visited / total, 2),
        visited=visited,
        reachable=len(reach),
        samples=len(points),
        outside=outside,
        cells=counts,
        **extra,
    )


def marginal(
    values: Sequence[float], edges: Sequence[float], keep: Callable[[float], bool] | None = None
) -> float:
    """One axis alone: % of its (kept) bins any value visits."""
    counts, _ = occupancy([(v,) for v in values], [edges])
    bins = [i for i, c in enumerate(centres(edges)) if keep is None or keep(c)]
    return round(100.0 * sum(1 for i in bins if (i,) in counts) / max(1, len(bins)), 2)


def signed(fractions: Sequence[float], scale: float) -> list[float]:
    """Edges mirrored about zero: ``fractions`` of ``scale`` on each side (a dead cell around 0
    when the first fraction is above zero)."""
    positive = [scale * f for f in fractions]
    return [-v for v in reversed(positive)] + positive


def random_plan(axes: Sequence[Sequence[Any]], n: int, seed: int = 0) -> list[tuple]:
    """n points drawn uniformly over the FULL space the axes span (a continuous axis from its
    first edge to its last, a categorical one over its levels), reproducible from `seed`.
    The cells are only the ruler; a plan need not sit on them."""
    rng = random.Random(seed)
    return [
        tuple(
            rng.choice(axis[1:])
            if axis and axis[0] == "levels"
            else rng.uniform(float(axis[0]), float(axis[-1]))
            for axis in axes
        )
        for _ in range(n)
    ]
