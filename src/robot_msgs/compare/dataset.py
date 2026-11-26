"""Dataset — many runs of an experiment: what a sim-to-real analysis rolls up and drills into."""

from __future__ import annotations

from dataclasses import dataclass

from robot_msgs._core import Msg, set_field
from robot_msgs.compare.run import Run


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Dataset(Msg):
    """Runs with unique names, and a `title`.

    `compare_dataset(dataset)` compares every run (`DatasetComparison`): the table across runs,
    and from its `worst` row the run, pair and stream to drill into.

    WHY here, generalising traj_compare's ``Run`` list (Zach, daily note 2026-11-25, "Sim to real
    agent": run sim and the real robot, record every sensor and actuator stream, align them,
    compare each one, then "a bunch of runs, analysis across them, drill down into one"): the
    dataset was a joint-trajectory-only type in traj_compare and bam_ws; robot_msgs takes the
    nouns and the math, so every tool reads one shape for any stream.
    """

    runs: tuple[Run, ...] = ()
    title: str = ""

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        for run in runs:
            if not isinstance(run, Run):
                raise TypeError(f"Dataset.runs holds Runs, got {type(run).__name__}")
        names = [run.name for run in runs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Dataset.runs has duplicate names: {duplicates}")
        set_field(self, "runs", runs)
        set_field(self, "title", str(self.title))

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.runs)

    def run(self, name: str) -> Run:
        for run in self.runs:
            if run.name == name:
                return run
        raise KeyError(f"no run {name!r}; runs are {[run.name for run in self.runs]}")
