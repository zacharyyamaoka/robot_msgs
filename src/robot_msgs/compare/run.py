"""Run — one experiment: the recordings of every source (target, sim, real) and how to pair them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.compare.error_stats import ErrorStats
from robot_msgs.compare.recording import Recording

Tag = str | int | float | bool
"""A tag value: a speed, a payload, a firmware name, a pass/fail flag."""

KEY_SEPARATOR = "→"
"""Between the two recordings of a pair in a comparison key: ``"target→real/servo_state"``."""

RESERVED_COLUMNS = frozenset(
    {"run", "a", "b", "stream", "metric", *ErrorStats.__dataclass_fields__}
)
"""Columns `RunComparison.rows()` writes itself, so no tag may take their names."""


def default_pairs(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """The first recording (the reference) against every other, then every other against each
    later one: ``(target, sim), (target, real), (sim, real)``. traj_compare's `default_pairs`."""
    reference, *others = names
    pairs = [(reference, name) for name in others]
    pairs += [(others[i], others[j]) for i in range(len(others)) for j in range(i + 1, len(others))]
    return tuple(pairs)


def _tag(run: str, key: str, value: object) -> Tag:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(
            f"Run {run!r} tag {key!r} must be a str, int, float or bool, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Run(Msg):
    """One run of one experiment: every source's `Recording`, and which pairs to compare.

    - `recordings` by source name, in order; the first is the reference (``"target"``), as in
      ``{"target": plan, "sim": sim, "real": real}``. Names may not contain ``/`` or ``→`` (they
      key the comparisons).
    - `pairs` is which (a, b) to compare, `a` the reference of each; `None` = `default_pairs`
      (the first against every other, then every other pair), the rule traj_compare's dataset
      uses.
    - `time_offset_s` is how late a recording's clock runs (`compare`'s ``time_offset_s``); a
      pair (a, b) is compared with ``offset[b] - offset[a]``. A missing source is 0.
    - `tags` are the run's knobs and conditions (``speed=0.5``, ``payload_kg=1.2``), copied onto
      every row of its comparison so a table can be grouped by them.
    """

    name: str
    recordings: Mapping[str, Recording]
    tags: Mapping[str, Tag] = field(default_factory=dict)
    pairs: tuple[tuple[str, str], ...] | None = None
    time_offset_s: Mapping[str, float] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        name = str(self.name)
        set_field(self, "name", name)
        set_field(self, "title", str(self.title))
        if not self.recordings:
            raise ValueError(f"Run {name!r} needs at least one recording")
        recordings = {}
        for source, recording in self.recordings.items():
            source = str(source)
            if "/" in source or KEY_SEPARATOR in source:
                raise ValueError(f"Run {name!r} recording name {source!r} contains '/' or '→'")
            if not isinstance(recording, Recording):
                raise TypeError(
                    f"Run {name!r} recordings[{source!r}] must be a Recording, "
                    f"got {type(recording).__name__}"
                )
            recordings[source] = recording
        set_field(self, "recordings", FrozenDict(recordings))
        clash = sorted(set(self.tags) & RESERVED_COLUMNS)
        if clash:
            raise ValueError(f"Run {name!r} tags {clash} would shadow comparison row columns")
        tags = {str(key): _tag(name, str(key), value) for key, value in self.tags.items()}
        set_field(self, "tags", FrozenDict(tags))
        if self.pairs is not None:
            pairs = tuple((str(a), str(b)) for a, b in self.pairs)
            for a, b in pairs:
                unknown = sorted({a, b} - set(recordings))
                if unknown:
                    raise ValueError(f"Run {name!r} pair ({a}, {b}) names no recording {unknown}")
                if a == b:
                    raise ValueError(f"Run {name!r} pairs {a!r} with itself")
            set_field(self, "pairs", pairs)
        unknown = sorted(set(self.time_offset_s) - set(recordings))
        if unknown:
            raise ValueError(f"Run {name!r} time_offset_s names no recording {unknown}")
        offsets = {str(k): float(v) for k, v in self.time_offset_s.items()}
        set_field(self, "time_offset_s", FrozenDict(offsets))

    @property
    def reference(self) -> str:
        """The first recording's name: the reference of the default pairs."""
        return next(iter(self.recordings))

    def resolved_pairs(self) -> tuple[tuple[str, str], ...]:
        """`pairs`, or `default_pairs` of the recordings when it is `None`."""
        return self.pairs if self.pairs is not None else default_pairs(tuple(self.recordings))

    def offset(self, a: str, b: str) -> float:
        """The ``time_offset_s`` pair (a, b) is compared with: ``offset[b] - offset[a]``."""
        return self.time_offset_s.get(b, 0.0) - self.time_offset_s.get(a, 0.0)
