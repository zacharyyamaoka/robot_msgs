"""robot_msgs.coverage: how much of a binned input space a set of points visits.

Written before the code. The bin convention is numpy.histogramdd's ([lo, hi) per bin, the last
bin [lo, hi], points outside every edge dropped), so histogramdd is the independent oracle for
every count here; the percentages are worked by hand.

What is NOT here, on purpose: which axes a device type has, its edges, which cells it can reach,
and any target or gate. Those are policy and live in sim2real (its cards); this module only
answers "which cells did these points visit, out of the cells you call reachable".
"""

import dataclasses
import itertools

import numpy as np
import pytest

from robot_msgs.coverage import (
    CoverageResult,
    bin_index,
    binned,
    centres,
    marginal,
    n_cells,
    occupancy,
    random_plan,
    signed,
)

EDGES = [0.0, 1.0, 2.0, 4.0]  # three bins, uneven on purpose


# ── one value into one bin ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (0.999, 0), (1.0, 1), (2.0, 2), (3.9, 2), (4.0, 2), (-0.001, None), (4.001, None)],
)
def test_bin_index_follows_histogramdd(value, expected):
    assert bin_index(value, EDGES) == expected
    counts, _ = np.histogramdd([[value]], bins=[EDGES])
    hit = np.flatnonzero(counts)
    assert (int(hit[0]) if hit.size else None) == expected


def test_occupancy_matches_histogramdd_on_random_points():
    rng = np.random.default_rng(7)
    axes = [[-1.0, -0.2, 0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0]]
    points = rng.uniform(-1.2, 1.2, size=(400, 2))
    points[:5] = [[1.0, 1.0], [-1.0, 0.0], [0.0, 0.5], [1.0, 0.0], [-1.0, 1.0]]  # every edge case
    counts, outside = occupancy(points, axes)
    oracle, _ = np.histogramdd(points, bins=axes)
    assert {cell: n for cell, n in counts.items()} == {
        tuple(int(i) for i in cell): int(oracle[cell]) for cell in zip(*np.nonzero(oracle))
    }
    assert outside == len(points) - int(oracle.sum())


def test_categorical_axis_is_a_levels_tuple():
    axes = [("levels", "metal", "PET", "film"), [0.0, 1.0, 2.0]]
    counts, outside = occupancy([("PET", 0.5), ("PET", 1.5), ("film", 0.1), ("glass", 0.1)], axes)
    assert counts == {(1, 0): 1, (1, 1): 1, (2, 0): 1}
    assert outside == 1  # an unknown level is outside, like a value past the edges
    assert n_cells(axes[0]) == 3
    assert centres(axes[0]) == ["metal", "PET", "film"]
    assert centres(axes[1]) == [0.5, 1.5]


# ── coverage over a grid ─────────────────────────────────────────────────────


def test_binned_counts_reachable_cells_only():
    axes = [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]  # 2 x 2 cells, centres 0.5 / 1.5

    def reachable(centre):
        return centre != (1.5, 1.5)  # a device that cannot hold the far corner

    points = [(0.2, 0.2), (0.3, 0.1), (1.5, 1.5), (5.0, 0.0)]
    result = binned(points, axes, reachable)
    assert isinstance(result, CoverageResult)
    assert (result.visited, result.reachable, result.samples, result.outside) == (1, 3, 4, 1)
    assert result.pct == 33.33  # 1 of 3 reachable cells, 2 decimals
    assert result.cells == {(0, 0): 2, (1, 1): 1}  # the unreachable visit is counted, not credited
    # the same set given up front gives the same answer (callers cache it)
    assert binned(points, axes, reach={(0, 0), (0, 1), (1, 0)}) == result
    # no rule: every cell counts
    assert binned(points, axes).pct == 50.0


def test_binned_passes_extra_fields_through():
    result = binned([(0.5,)], [[0.0, 1.0]], axes={"temperature": None}, unplanned=("temperature",), note="n")
    assert (result.pct, result.axes, result.unplanned, result.note) == (100.0, {"temperature": None}, ("temperature",), "n")


def test_empty_plan_is_zero_not_an_error():
    result = binned([], [[0.0, 1.0, 2.0]])
    assert (result.pct, result.visited, result.reachable, result.samples) == (0.0, 0, 2, 0)


def test_marginal_is_one_axis_alone():
    assert marginal([0.5, 0.6, 3.0], EDGES) == 66.67  # bins 0 and 2 of 3
    assert marginal([0.5, 3.0], EDGES, keep=lambda centre: centre < 2.0) == 50.0  # kept: bins 0, 1


def test_coverage_result_is_a_frozen_value():
    result = binned([(0.5,)], [[0.0, 1.0]])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.pct = 0.0  # type: ignore[misc]
    assert [f.name for f in dataclasses.fields(CoverageResult)] == [
        "pct", "visited", "reachable", "samples", "outside", "axes", "unplanned", "cells", "note",
    ]  # fmt: skip


# ── building edges and plans ─────────────────────────────────────────────────


def test_signed_mirrors_fractions_about_zero():
    assert signed((0.1, 0.5, 1.0), 2.0) == [-2.0, -1.0, -0.2, 0.2, 1.0, 2.0]


def test_random_plan_spans_the_full_space_reproducibly():
    axes = [[-1.0, 0.0, 3.0], ("levels", "a", "b")]
    plan = random_plan(axes, 500, seed=3)
    assert plan == random_plan(axes, 500, seed=3)
    assert plan != random_plan(axes, 500, seed=4)
    assert all(-1.0 <= x <= 3.0 and level in ("a", "b") for x, level in plan)
    assert binned(plan, axes).pct == 100.0  # 500 uniform draws visit all 4 cells


def test_policy_stays_out():
    import robot_msgs.coverage as coverage

    device_words = ("servo", "actuator", "driver", "arm", "camera", "conveyor", "joint", "gate", "target")
    public = [name for name in dir(coverage) if not name.startswith("_")]
    assert not [n for n in public if any(w in n.lower() for w in device_words)], public
    assert set(itertools.chain(coverage.__all__)) == {
        "CoverageResult", "bin_index", "binned", "centres", "marginal", "n_cells", "occupancy",
        "random_plan", "signed",
    }  # fmt: skip
