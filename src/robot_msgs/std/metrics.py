"""Metrics — named numbers one component reports at one instant (step counts, latencies, drops)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std.header import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Metrics(Msg):
    """`values[i]` (float64) is the metric `names[i]`, as `path` measured it at `header.stamp`.

    - `path`: the component, as a dotted path (``"pick_cell.arm.jtc"``).
    - `names`: free strings, unique; SI units, with the unit as a suffix when it is not a count
      (``"steps"``, ``"step_p99_s"``, ``"input_age_s.state"``).
    - `values`: one per name. NaN means "not measured this period" (a percentile of no steps).

    WHY two parallel fields, not a dict: the numbers sit in one contiguous float64 array, so the
    record crosses a process as a list of strings and one buffer, with no object per entry.
    `metrics["steps"]` and `as_dict()` give the by-name view.
    The field order is the cross-process layout (like_keras's codec reads it), so it never changes.
    """

    header: Header
    path: str
    names: tuple[str, ...]
    values: Array

    def __post_init__(self) -> None:
        expect_type("Metrics.header", self.header, Header)
        set_field(self, "path", str(self.path))
        if isinstance(self.names, str):
            raise TypeError(
                f"Metrics.names is a tuple of names; got the string {self.names!r} "
                f"(write ({self.names!r},))"
            )
        names = tuple(str(name) for name in self.names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Metrics.names has duplicate names: {duplicates}")
        set_field(self, "names", names)
        values = frozen_array(self.values)
        if values.ndim != 1 or len(values) != len(names):
            raise ValueError(
                f"Metrics.values must hold one number per name: {len(names)} names, "
                f"values of shape {values.shape}"
            )
        set_field(self, "values", values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float], *, header: Header, path: str) -> Metrics:
        """Build from ``{name: value}``, keeping the mapping's order."""
        return cls(
            header=header,
            path=path,
            names=tuple(values),
            values=np.fromiter(values.values(), dtype=np.float64, count=len(values)),
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> float:
        """The value called `name`, as a Python float."""
        try:
            return float(self.values[self.names.index(name)])
        except ValueError:
            raise KeyError(f"no metric {name!r}; names are {self.names}") from None

    def as_dict(self) -> dict[str, float]:
        """``{name: value}`` in the record's order."""
        return dict(zip(self.names, self.values.tolist(), strict=True))
