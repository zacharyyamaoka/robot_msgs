"""TimeRepair — what `Trajectory.from_samples` changed to make recorded stamps strictly increasing.

A trajectory's `t` must strictly increase (interpolation needs it), and the constructor refuses
anything else. Recorded data breaks that: a loop that logs the same stamp twice, a clock that
steps back. `from_samples(..., duplicate_times=...)` is the one explicit door for such data, and
this report is its receipt: every dropped or merged sample, by its index in the samples as given.

WHY a receipt and not a quiet fix: data is rendered truthfully. A cleaned trajectory that does
not say it was cleaned is indistinguishable from a clean recording, so the report is returned
*and* written into the cleaned trajectory's `header.meta` (`as_meta()`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

DuplicateTimes = Literal["error", "drop_later", "average"]
POLICIES: tuple[str, ...] = ("error", "drop_later", "average")
META_PREFIX = "time_repair."


@dataclass(frozen=True)
class TimeRepair:
    """`dropped`: indices removed. `averaged`: groups of indices merged into one sample.

    Under ``"drop_later"`` a sample whose stamp is not after every earlier kept stamp is dropped,
    so the first of a repeat survives and a backwards stamp goes. Under ``"average"`` samples at
    exactly the same stamp are merged into one (floats averaged, quaternions averaged on the short
    arc, held values must agree); a backwards stamp has no partner and is dropped.
    """

    policy: str
    samples_in: int
    samples_out: int
    dropped: tuple[int, ...] = ()
    dropped_t: tuple[float, ...] = ()
    averaged: tuple[tuple[int, ...], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.dropped or self.averaged)

    def as_meta(self) -> dict[str, str]:
        """The receipt as `Header.meta` entries (``time_repair.*``); empty when nothing changed."""
        if not self.changed:
            return {}
        meta = {
            f"{META_PREFIX}policy": self.policy,
            f"{META_PREFIX}samples_in": str(self.samples_in),
            f"{META_PREFIX}samples_out": str(self.samples_out),
        }
        if self.dropped:
            meta[f"{META_PREFIX}dropped"] = json.dumps(list(self.dropped))
            meta[f"{META_PREFIX}dropped_t"] = json.dumps(list(self.dropped_t))
        if self.averaged:
            meta[f"{META_PREFIX}averaged"] = json.dumps([list(group) for group in self.averaged])
        return meta

    def __str__(self) -> str:
        if not self.changed:
            return f"{self.policy}: all {self.samples_in} stamps strictly increase, nothing changed"
        parts = []
        if self.dropped:
            parts.append(
                f"dropped {len(self.dropped)} of {self.samples_in} samples "
                f"(repeated or backwards stamps) at indices {list(self.dropped)}"
            )
        if self.averaged:
            merged = sum(len(group) for group in self.averaged)
            parts.append(f"averaged {merged} samples into {len(self.averaged)} at repeated stamps")
        return f"{self.policy}: " + "; ".join(parts)


def plan(t: np.ndarray, policy: str) -> tuple[np.ndarray, TimeRepair]:
    """Group label per input sample (-1 = dropped; equal labels merge) and the receipt.

    Labels count up from 0 in output order. Under ``"error"`` nothing is planned: the caller
    constructs directly and the constructor's own message names the first bad pair.
    """
    if policy not in POLICIES:
        raise ValueError(f"duplicate_times must be one of {POLICIES}, got {policy!r}")
    count = t.shape[0]
    labels = np.arange(count)
    if policy == "error" or count < 2:
        return labels, TimeRepair(policy, count, count)
    # WHY a running maximum: the last kept stamp is always the maximum so far (a dropped stamp is
    # never above a kept one), so "after every earlier kept stamp" is t[i] > max(t[:i]).
    previous_max = np.maximum.accumulate(t)[:-1]
    later = t[1:]
    keep = np.concatenate([[True], later > previous_max])
    joins = np.concatenate([[False], later == previous_max]) if policy == "average" else None
    labels = np.cumsum(keep) - 1
    if joins is not None:
        labels = np.where(keep | joins, labels, -1)
    else:
        labels = np.where(keep, labels, -1)
    dropped = tuple(int(i) for i in np.flatnonzero(labels < 0))
    averaged: tuple[tuple[int, ...], ...] = ()
    if joins is not None and joins.any():
        groups = {int(label) for label in labels[joins]}
        averaged = tuple(tuple(int(i) for i in np.flatnonzero(labels == g)) for g in sorted(groups))
    repair = TimeRepair(
        policy=policy,
        samples_in=count,
        samples_out=int(keep.sum()),
        dropped=dropped,
        dropped_t=tuple(float(t[i]) for i in dropped),
        averaged=averaged,
    )
    return labels, repair


def merge(array: np.ndarray, labels: np.ndarray, groups: int, kind: str, name: str) -> np.ndarray:
    """One per-sample array reduced to one row per group (`labels` from `plan`)."""
    kept = labels >= 0
    rows, members = array[kept], labels[kept]
    # labels never decrease (a join only ever equals the latest kept stamp), so each group's first
    # member is where its label first appears
    representative = rows[np.searchsorted(members, np.arange(groups), side="left")]
    if len(members) == groups:
        return representative  # nothing merged: a pure drop
    if kind == "hold" or array.dtype.kind not in "fc":
        disagree = np.any((rows != representative[members]).reshape(len(members), -1), axis=1)
        if disagree.any():
            raise ValueError(
                f"cannot average {name!r}: it holds discrete values (interp 'hold') and samples "
                "at the same stamp disagree; use duplicate_times='drop_later'"
            )
        return representative
    values = rows.astype(np.result_type(rows.dtype, np.float64))
    if kind == "slerp":
        # q and -q are one rotation: flip each member onto its group's first before summing
        signs = np.sign(np.sum(values * representative[members], axis=-1, keepdims=True))
        values = values * np.where(signs == 0, 1.0, signs)
    sums = np.zeros((groups, *values.shape[1:]), dtype=values.dtype)
    np.add.at(sums, members, values)
    counts = np.bincount(members, minlength=groups).reshape(-1, *([1] * (values.ndim - 1)))
    means = sums / counts
    if kind == "slerp":
        means = means / np.linalg.norm(means, axis=-1, keepdims=True)
    return means
