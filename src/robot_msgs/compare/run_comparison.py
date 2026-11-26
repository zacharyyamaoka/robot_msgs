"""RunComparison — every stream of every pair of one `Run`, compared; and `compare_run`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.compare._dispatch import compare
from robot_msgs.compare.comparison import Comparison
from robot_msgs.compare.run import KEY_SEPARATOR, Run, Tag


def comparison_key(a: str, b: str, stream: str) -> str:
    """``"<a>→<b>/<stream>"``: the key of stream `stream` of `b` measured against `a`.

    WHY one string, not a tuple: it is a dict key in JSON (`to_dict`) and a column in a table,
    and it reads as the direction of the comparison. Recording names hold neither ``/`` nor
    ``→`` (`Run` refuses them), so the key splits back unambiguously; a stream name may hold ``/``.
    """
    return f"{a}{KEY_SEPARATOR}{b}/{stream}"


def split_key(key: str) -> tuple[str, str, str]:
    """``(a, b, stream)`` of a `comparison_key`."""
    a, rest = key.split(KEY_SEPARATOR, 1)
    b, stream = rest.split("/", 1)
    return a, b, stream


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class RunComparison(Msg):
    """What `compare_run(run)` returns: one `Comparison` per pair and shared stream.

    - `comparisons` keyed ``"<a>→<b>/<stream>"`` (`comparison_key`), pairs in the run's order,
      streams in `a`'s order; ``get(a, b, stream)`` reads one.
    - `skipped` keyed the same way: every stream of a pair that was *not* compared, with why
      (only one side recorded it, the two are different types, or `compare` refused them: no
      overlap, different joints, different frame sizes). Never silent.
    - `tags` are the run's, copied onto every row.
    """

    run: str
    comparisons: Mapping[str, Comparison] = field(default_factory=dict)
    skipped: Mapping[str, str] = field(default_factory=dict)
    tags: Mapping[str, Tag] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_field(self, "run", str(self.run))
        for key, comparison in self.comparisons.items():
            if not isinstance(comparison, Comparison):
                raise TypeError(f"RunComparison.comparisons[{key!r}] must be a Comparison")
            split_key(key)
        set_field(self, "comparisons", FrozenDict({str(k): v for k, v in self.comparisons.items()}))
        set_field(self, "skipped", FrozenDict({str(k): str(v) for k, v in self.skipped.items()}))
        set_field(self, "tags", FrozenDict(dict(self.tags)))

    def __repr__(self) -> str:
        # WHY short: every Comparison inside prints its metrics; the summary is what a REPL needs
        return (
            f"RunComparison(run={self.run!r}, comparisons={list(self.comparisons)}, "
            f"skipped={len(self.skipped)})"
        )

    def get(self, a: str, b: str, stream: str) -> Comparison:
        key = comparison_key(a, b, stream)
        if key not in self.comparisons:
            why = self.skipped.get(key)
            raise KeyError(f"{key!r} was skipped: {why}" if why else f"no comparison {key!r}")
        return self.comparisons[key]

    def rows(self) -> list[dict[str, Any]]:
        """One tidy row per (pair, stream, metric): ``run a b stream metric`` + every `ErrorStats`
        field + the run's tags. Skipped streams have no rows (they are in `skipped`)."""
        out = []
        for key, comparison in self.comparisons.items():
            a, b, stream = split_key(key)
            for row in comparison.rows():
                out.append({"run": self.run, "a": a, "b": b, "stream": stream, **row, **self.tags})
        return out


def compare_run(
    run: Run,
    *,
    contour_window_s: float = 0.25,
    errors: Literal["skip", "raise"] = "skip",
) -> RunComparison:
    """Every stream both recordings of each pair carry, compared (`compare`), on one clock.

    Pairs come from `run.resolved_pairs()`; pair (a, b) is ``compare(a's stream, b's stream,
    time_offset_s=run.offset(a, b))``. A stream only one side has, two streams of different
    types, and (with ``errors="skip"``, the default) a pair `compare` refuses with a `ValueError`
    or `TypeError` go into `skipped` with the reason; ``errors="raise"`` lets that error through.
    WHY skip by default: one dead stream in one run must not hide a hundred good comparisons, and
    the reason travels with the result.
    """
    if errors not in ("skip", "raise"):
        raise ValueError(f"errors must be 'skip' or 'raise', got {errors!r}")
    comparisons: dict[str, Comparison] = {}
    skipped: dict[str, str] = {}
    for a, b in run.resolved_pairs():
        first, second = run.recordings[a].streams, run.recordings[b].streams
        names = list(first) + [name for name in second if name not in first]
        for stream in names:
            key = comparison_key(a, b, stream)
            if stream not in second:
                skipped[key] = f"only {a!r} recorded {stream!r}"
                continue
            if stream not in first:
                skipped[key] = f"only {b!r} recorded {stream!r}"
                continue
            reference, measured = first[stream], second[stream]
            if type(reference) is not type(measured):
                skipped[key] = (
                    f"different types: {type(reference).__name__} vs {type(measured).__name__}"
                )
                continue
            try:
                comparisons[key] = compare(
                    reference,
                    measured,
                    time_offset_s=run.offset(a, b),
                    contour_window_s=contour_window_s,
                )
            except (ValueError, TypeError) as error:
                if errors == "raise":
                    raise
                skipped[key] = f"{type(error).__name__}: {error}"
    return RunComparison(run=run.name, comparisons=comparisons, skipped=skipped, tags=run.tags)
