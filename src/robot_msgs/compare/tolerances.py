"""Tolerances and a verdict, in the ROS 2 joint_trajectory_controller vocabulary.

Prior art: ROS 2 ``joint_trajectory_controller`` (path / goal tolerances, ``goal_time_tolerance``,
``stopped_velocity_tolerance``; control.ros.org) and ``control_msgs/FollowJointTrajectory``
(result codes, ``JointTolerance``). `check` is the offline verdict on two recorded runs;
`ToleranceMonitor` is what a controller runs each tick. They share one per-sample rule
(`_Judge.step`), so an online run and the same run checked offline can never disagree.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from robot_msgs._core import FrozenDict, Msg, from_dict, set_field
from robot_msgs.compare._phases import in_goal_phase, split_phases
from robot_msgs.compare.align import match_joints
from robot_msgs.geometry.rotations import quat_conjugate, quat_multiply, quat_to_rotvec
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

# control_msgs/action/FollowJointTrajectory Result
SUCCESSFUL = 0
PATH_TOLERANCE_VIOLATED = -4
GOAL_TOLERANCE_VIOLATED = -5

_CODE_NAMES = {
    SUCCESSFUL: "SUCCESSFUL",
    PATH_TOLERANCE_VIOLATED: "PATH_TOLERANCE_VIOLATED",
    GOAL_TOLERANCE_VIOLATED: "GOAL_TOLERANCE_VIOLATED",
}


# ── messages ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class JointTolerance(Msg):
    """control_msgs/JointTolerance: limits on one joint's ``|reference - measured|``.

    A value above 0 is a limit; 0 or -1 means not checked. WHY: ROS reads 0 as "use the
    controller default" and -1 as "no limit"; here a `Tolerances` *is* that default set, and
    merging a per-goal tolerance onto it is the caller's job, so both mean "this is not checked".
    """

    name: str
    position: float = 0.0  # rad or m
    velocity: float = 0.0  # rad/s or m/s
    acceleration: float = 0.0  # rad/s² or m/s²

    def __post_init__(self) -> None:
        set_field(self, "name", str(self.name))
        for name in ("position", "velocity", "acceleration"):
            set_field(self, name, float(getattr(self, name)))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class PoseTolerance(Msg):
    """Limits on a pose error's magnitudes: ``position`` (m) and ``rotation`` angle (rad).

    Same rule as `JointTolerance`: above 0 is a limit, 0 or -1 is not checked.
    """

    position: float = 0.0
    rotation: float = 0.0

    def __post_init__(self) -> None:
        set_field(self, "position", float(self.position))
        set_field(self, "rotation", float(self.rotation))


def _group(value: Any) -> tuple[JointTolerance, ...] | PoseTolerance:
    """A tolerance group as it should be stored; also what `from_dict` hands back for a union."""
    if isinstance(value, PoseTolerance):
        return value
    if isinstance(value, Mapping):
        return from_dict(value, PoseTolerance)
    entries = tuple(
        from_dict(entry, JointTolerance) if isinstance(entry, Mapping) else entry for entry in value
    )
    for entry in entries:
        if not isinstance(entry, JointTolerance):
            raise TypeError(f"a tolerance group holds JointTolerances, got {type(entry).__name__}")
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"a tolerance group names each joint once, got {names}")
    return entries


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Tolerances(Msg):
    """What a run must stay inside, as ``joint_trajectory_controller`` states it.

    - `path`: limits that must hold at every sample while the reference is running; a breach ends
      the check (JTC aborts the goal). `goal`: limits that must hold once the reference has ended.
      Each is a tuple of `JointTolerance`, or one `PoseTolerance` for a `PoseTrajectory`.
    - `goal_time_tolerance` (s after the reference ends): the goal must be met within it; 0 means
      wait as long as the data lasts (JTC: "potentially infinite").
    - `stopped_velocity_tolerance`: JTC's default goal velocity limit, applied to EVERY joint's
      measured ``|velocity|`` (a pose: ``|linear velocity|``) unless that joint's goal
      `JointTolerance.velocity` is above 0. It only applies when `goal` has a limit of its own,
      and a measured run with no velocities skips it and says so in `Outcome.unchecked`.
    """

    path: tuple[JointTolerance, ...] | PoseTolerance = ()
    goal: tuple[JointTolerance, ...] | PoseTolerance = ()
    goal_time_tolerance: float = 0.0
    stopped_velocity_tolerance: float = 0.01

    def __post_init__(self) -> None:
        set_field(self, "path", _group(self.path))
        set_field(self, "goal", _group(self.goal))
        set_field(self, "goal_time_tolerance", float(self.goal_time_tolerance))
        set_field(self, "stopped_velocity_tolerance", float(self.stopped_velocity_tolerance))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Violation(Msg):
    """One limit broken, at the first sample that broke it.

    `metric` uses compare's names (``positions/j1``, ``velocities/j2``, ``accelerations/j1``,
    ``position``, ``rotation``, ``linear_velocity``). `t` is the measured sample's time (the error
    trajectory's `t`). `error` is the signed ``reference - measured`` for a joint and the
    magnitude for a pose's position / rotation.
    """

    phase: str  # "path" | "goal"
    metric: str
    t: float
    error: float
    limit: float

    def __post_init__(self) -> None:
        set_field(self, "phase", str(self.phase))
        set_field(self, "metric", str(self.metric))
        for name in ("t", "error", "limit"):
            set_field(self, name, float(getattr(self, name)))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Outcome(Msg):
    """The verdict: a ``FollowJointTrajectory`` result code, the reason, and the room left.

    - `error_code` is `SUCCESSFUL`, `PATH_TOLERANCE_VIOLATED` or `GOAL_TOLERANCE_VIOLATED`,
      taken from the earliest violation; `error_string` names the joint / metric and the numbers.
    - `violations`: the first violation per (phase, metric), by time. Empty on success, even when
      the run was still settling early in the goal phase: the goal was met.
    - `margins`: ``path/<metric>`` = limit - worst ``|error|`` over the path phase;
      ``goal/<metric>`` = limit - ``|error|`` at the sample where the goal was decided. Negative
      = violated by that much.
    - `goal_time`: seconds after the reference ends at which the goal was met; NaN if not met.
    - `unchecked`: what a limit asked for but the data could not give (a measured run without
      velocities cannot be held to `stopped_velocity_tolerance`); never silently a pass.
    """

    error_code: int
    error_string: str = ""
    violations: tuple[Violation, ...] = ()
    margins: Mapping[str, float] = field(default_factory=dict)
    goal_time: float = math.nan
    unchecked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        set_field(self, "error_code", int(self.error_code))
        set_field(self, "error_string", str(self.error_string))
        set_field(self, "violations", tuple(self.violations))
        for violation in self.violations:
            if not isinstance(violation, Violation):
                raise TypeError(f"Outcome.violations holds Violations, got {violation!r}")
        set_field(self, "margins", FrozenDict({str(k): float(v) for k, v in self.margins.items()}))
        set_field(self, "goal_time", float(self.goal_time))
        set_field(self, "unchecked", tuple(str(name) for name in self.unchecked))

    @property
    def ok(self) -> bool:
        return self.error_code == SUCCESSFUL

    @property
    def code_name(self) -> str:
        return _CODE_NAMES.get(self.error_code, str(self.error_code))


# ── the per-sample rule, shared by check() and ToleranceMonitor ─────────────


class _Row(NamedTuple):
    """One side of one sample. Joints: `position` is the joint positions. Poses: the position."""

    position: Any = None
    orientation: Any = None
    velocity: Any = None  # joint velocities, or a pose's linear velocity
    acceleration: Any = None


@dataclass(frozen=True)
class _Limit:
    phase: str
    metric: str
    quantity: str  # positions | velocities | accelerations | position | rotation | linear_velocity
    column: int
    limit: float
    explicit: bool  # False: a default (stopped velocity) that a run without velocities may skip


def _kind_of(group: tuple[JointTolerance, ...] | PoseTolerance) -> str | None:
    if isinstance(group, PoseTolerance):
        return "pose"
    return "joint" if group else None


def _compile(
    tolerances: Tolerances, kind: str | None, names: Sequence[str]
) -> dict[str, list[_Limit]]:
    """Every limit that is actually set (above 0), by phase."""
    kinds = {k for k in (_kind_of(tolerances.path), _kind_of(tolerances.goal)) if k}
    if len(kinds) > 1:
        raise ValueError("path and goal tolerances mix JointTolerance and PoseTolerance")
    if kinds and kind is not None and kind not in kinds:
        raise ValueError(
            f"{'JointTolerance' if 'joint' in kinds else 'PoseTolerance'} tolerances cannot "
            f"judge a {kind} trajectory"
        )
    limits: dict[str, list[_Limit]] = {"path": [], "goal": []}
    for phase, group in (("path", tolerances.path), ("goal", tolerances.goal)):
        if isinstance(group, PoseTolerance):
            for quantity in ("position", "rotation"):
                value = getattr(group, quantity)
                if value > 0.0:
                    limits[phase].append(_Limit(phase, quantity, quantity, 0, value, True))
            continue
        for tolerance in group:
            if tolerance.name not in names:
                raise ValueError(
                    f"tolerance for joint {tolerance.name!r}; joints are {tuple(names)}"
                )
            column = list(names).index(tolerance.name)
            for quantity, value in (
                ("positions", tolerance.position),
                ("velocities", tolerance.velocity),
                ("accelerations", tolerance.acceleration),
            ):
                if value > 0.0:
                    metric = f"{quantity}/{tolerance.name}"
                    limits[phase].append(_Limit(phase, metric, quantity, column, value, True))
    stopped = tolerances.stopped_velocity_tolerance
    if limits["goal"] and stopped > 0.0:
        if isinstance(tolerances.goal, PoseTolerance):
            limits["goal"].append(
                _Limit("goal", "linear_velocity", "linear_velocity", 0, stopped, False)
            )
        else:
            overridden = {lim.column for lim in limits["goal"] if lim.quantity == "velocities"}
            for column, name in enumerate(names):
                if column not in overridden:
                    limits["goal"].append(
                        _Limit("goal", f"velocities/{name}", "velocities", column, stopped, False)
                    )
    return limits


def _missing(limit: _Limit, ref: _Row, meas: _Row) -> str | None:
    """The data this limit needs and the rows lack, or None when it can be evaluated."""
    quantity = limit.quantity
    if quantity in ("positions", "position"):
        return None if ref.position is not None and meas.position is not None else "positions"
    if quantity == "rotation":
        both = ref.orientation is not None and meas.orientation is not None
        return None if both else "orientations"
    if quantity == "linear_velocity":
        return None if meas.velocity is not None else "linear velocities"
    attribute = "velocity" if quantity == "velocities" else "acceleration"
    needed = (meas,) if limit.phase == "goal" else (ref, meas)
    return None if all(getattr(row, attribute) is not None for row in needed) else quantity


def _error(limit: _Limit, ref: _Row, meas: _Row) -> float:
    """``reference - measured`` for a joint, the magnitude for a pose.

    Goal phase: the reference's velocity and acceleration are 0 (held final point), so only the
    measured run is needed for them.
    """
    column = limit.column
    quantity = limit.quantity
    if quantity == "positions":
        return float(ref.position[column] - meas.position[column])
    if quantity in ("velocities", "accelerations"):
        field_name = "velocity" if quantity == "velocities" else "acceleration"
        measured = float(getattr(meas, field_name)[column])
        reference = 0.0 if limit.phase == "goal" else float(getattr(ref, field_name)[column])
        return reference - measured
    if quantity == "position":
        return float(np.linalg.norm(np.asarray(ref.position) - np.asarray(meas.position)))
    if quantity == "rotation":
        turn = quat_multiply(
            np.asarray(ref.orientation), quat_conjugate(np.asarray(meas.orientation))
        )
        return float(np.linalg.norm(quat_to_rotvec(turn)))
    return float(np.linalg.norm(np.asarray(meas.velocity)))  # linear_velocity


class _Judge:
    """State machine over a run's samples: the one rule `check` and `ToleranceMonitor` share."""

    def __init__(self, tolerances: Tolerances, limits: dict[str, list[_Limit]], end: float) -> None:
        self.tolerances = tolerances
        self.limits = limits
        self.end = end  # the reference's end on the clock `step` is given
        self.goal_active = any(limit.explicit for limit in limits["goal"])
        self.outcome: Outcome | None = None
        self.worst: dict[str, float] = {}  # path metric -> worst |error|
        self.latest: dict[str, float] = {}  # goal metric -> |error| at the latest goal sample
        self.first: dict[tuple[str, str], Violation] = {}
        self.last_breaches: list[Violation] = []
        self.unchecked: dict[str, None] = {}
        self.goal_samples = 0
        self.last_tau = math.nan
        self.last_age = math.nan

    def step(self, phase: str, t: float, tau: float, ref: _Row, meas: _Row) -> Outcome | None:
        """Judge one measured sample: `t` is the time to report, `tau` its time on the reference
        clock. Returns the Outcome once decided (and the same one on every later call)."""
        if self.outcome is not None:
            return self.outcome
        self.last_tau = tau
        breaches = self._evaluate(phase, t, ref, meas)
        if phase == "path":
            if breaches:
                return self._decide(PATH_TOLERANCE_VIOLATED, self._describe("path", breaches[0]))
            return None
        self.goal_samples += 1
        self.last_age = max(0.0, tau - self.end)
        self.last_breaches = breaches
        if breaches:
            return None
        return self._decide(SUCCESSFUL, "", goal_time=self.last_age)

    def finish(self, why: str) -> Outcome:
        """The verdict when no more samples will decide it: the data `ended` or the goal time
        `expired`."""
        if self.outcome is not None:
            return self.outcome
        if not self.goal_active:
            return self._decide(SUCCESSFUL, "")
        if self.goal_samples == 0:
            if why == "expired":
                text = (
                    f"no measured sample within goal_time_tolerance "
                    f"({self.tolerances.goal_time_tolerance:g} s) after the reference end: "
                    "the goal was never observed"
                )
            elif math.isnan(self.last_tau):
                text = (
                    "measured run has no sample inside the reference: the goal was never observed"
                )
            else:
                text = (
                    f"measured run ends {self.end - self.last_tau:.4g} s before the reference "
                    "end: the goal was never observed"
                )
            return self._decide(GOAL_TOLERANCE_VIOLATED, f"goal tolerance violated: {text}")
        text = self._describe("goal", self.last_breaches[0]) if self.last_breaches else ""
        reason = (
            f" within goal_time_tolerance ({self.tolerances.goal_time_tolerance:g} s)"
            if why == "expired"
            else f" by the end of the data ({self.last_age:.4g} s after the reference end)"
        )
        return self._decide(GOAL_TOLERANCE_VIOLATED, f"{text}; goal not met{reason}")

    def _evaluate(self, phase: str, t: float, ref: _Row, meas: _Row) -> list[Violation]:
        breaches = []
        for limit in self.limits[phase]:
            lacking = _missing(limit, ref, meas)
            if lacking is not None:
                if limit.explicit:
                    who = "both runs" if phase == "path" else "the measured run"
                    raise ValueError(
                        f"a {phase} tolerance on {limit.metric} needs {who} to carry {lacking}"
                    )
                self.unchecked[f"{phase}/{limit.quantity}"] = None
                continue
            error = _error(limit, ref, meas)
            magnitude = abs(error)
            if phase == "path":
                self.worst[limit.metric] = max(self.worst.get(limit.metric, 0.0), magnitude)
            else:
                self.latest[limit.metric] = magnitude
            if magnitude > limit.limit:
                violation = Violation(
                    phase=phase, metric=limit.metric, t=t, error=error, limit=limit.limit
                )
                self.first.setdefault((phase, limit.metric), violation)
                breaches.append(violation)
        return breaches

    @staticmethod
    def _describe(phase: str, violation: Violation) -> str:
        return (
            f"{phase} tolerance violated: {violation.metric} error {violation.error:+.4g} "
            f"exceeds limit {violation.limit:.4g} at t={violation.t:.4g} s"
        )

    def _decide(self, code: int, text: str, *, goal_time: float = math.nan) -> Outcome:
        margins = {}
        for limit in self.limits["path"]:
            if limit.metric in self.worst:
                margins[f"path/{limit.metric}"] = limit.limit - self.worst[limit.metric]
        for limit in self.limits["goal"]:
            if limit.metric in self.latest:
                margins[f"goal/{limit.metric}"] = limit.limit - self.latest[limit.metric]
        if code == SUCCESSFUL:
            violations: tuple[Violation, ...] = ()
        else:
            phase = "path" if code == PATH_TOLERANCE_VIOLATED else "goal"
            violations = tuple(
                sorted((v for (p, _), v in self.first.items() if p == phase), key=lambda v: v.t)
            )
        self.outcome = Outcome(
            error_code=code,
            error_string=text,
            violations=violations,
            margins=margins,
            goal_time=goal_time,
            unchecked=tuple(self.unchecked),
        )
        return self.outcome


# ── offline: check ───────────────────────────────────────────────────────────


def _require_fields(limits: dict[str, list[_Limit]], reference: Any, measured: Any) -> None:
    """Fail before the first sample when a set velocity / acceleration limit has no data."""
    for phase, group in limits.items():
        for limit in group:
            if not limit.explicit or limit.quantity not in ("velocities", "accelerations"):
                continue
            if getattr(measured, limit.quantity) is None or (
                phase == "path" and getattr(reference, limit.quantity) is None
            ):
                who = "both runs" if phase == "path" else "the measured run"
                raise ValueError(
                    f"a {phase} tolerance on {limit.metric} needs {who} to carry {limit.quantity}"
                )


def _rows(
    reference: JointTrajectory | PoseTrajectory,
    measured: JointTrajectory | PoseTrajectory,
    phases: Any,
    time_offset_s: float,
) -> Iterable[tuple[str, float, float, _Row, _Row]]:
    """Every judged measured sample in time order, with the reference it is judged against:
    the reference at that instant on the path, its FINAL point held in the goal phase."""
    joint = isinstance(reference, JointTrajectory)

    def measured_row(index: int) -> _Row:
        if joint:
            return _Row(
                measured.positions[index],
                None,
                None if measured.velocities is None else measured.velocities[index],
                None if measured.accelerations is None else measured.accelerations[index],
            )
        return _Row(
            measured.positions[index],
            measured.orientations[index],
            None if measured.linear_velocities is None else measured.linear_velocities[index],
        )

    path_index = np.flatnonzero(phases.path)
    if path_index.size:
        at = reference.resample(
            phases.tau[path_index] - reference.header.stamp, extrapolate="clamp"
        )
        for row, index in enumerate(path_index):
            if joint:
                ref = _Row(
                    at.positions[row],
                    None,
                    None if at.velocities is None else at.velocities[row],
                    None if at.accelerations is None else at.accelerations[row],
                )
            else:
                ref = _Row(
                    at.positions[row],
                    at.orientations[row],
                    None if at.linear_velocities is None else at.linear_velocities[row],
                )
            yield (
                "path",
                float(measured.t[index]),
                float(phases.tau[index]),
                ref,
                measured_row(index),
            )
    final = _Row(reference.positions[-1], None if joint else reference.orientations[-1])
    for index in np.flatnonzero(phases.goal):
        yield "goal", float(measured.t[index]), float(phases.tau[index]), final, measured_row(index)


def check(
    reference: JointTrajectory | PoseTrajectory,
    measured: JointTrajectory | PoseTrajectory,
    tolerances: Tolerances,
    *,
    time_offset_s: float = 0.0,
) -> Outcome:
    """Judge a recorded `measured` run against `reference` and `tolerances`.

    - The clock rules are `align`'s (`time_offset_s` = how late the measured clock runs); the
      phases are `split_phases`: path while the reference runs, goal from its end on, against its
      final point held.
    - The first path breach ends the check; the goal is met at the first goal-phase sample where
      every goal limit holds. A run that ends before observing the goal is
      `GOAL_TOLERANCE_VIOLATED`, never a pass.
    - A velocity / acceleration tolerance that is set while a side lacks that field raises
      `ValueError`.
    """
    if isinstance(reference, JointTrajectory) and isinstance(measured, JointTrajectory):
        reference = match_joints(reference, measured)
        names: Sequence[str] = measured.joint_names
        kind = "joint"
    elif isinstance(reference, PoseTrajectory) and isinstance(measured, PoseTrajectory):
        names, kind = (), "pose"
    else:
        raise TypeError(
            f"check judges a JointTrajectory or PoseTrajectory pair, got "
            f"{type(reference).__name__} and {type(measured).__name__}"
        )
    limits = _compile(tolerances, kind, names)
    _require_fields(limits, reference, measured)
    phases = split_phases(
        reference,
        measured,
        time_offset_s=time_offset_s,
        goal_time_tolerance=tolerances.goal_time_tolerance,
    )
    judge = _Judge(tolerances, limits, phases.end)
    for phase, t, tau, ref, meas in _rows(reference, measured, phases, time_offset_s):
        outcome = judge.step(phase, t, tau, ref, meas)
        if outcome is not None:
            return outcome
    horizon = tolerances.goal_time_tolerance
    expired = horizon > 0.0 and bool(np.any(phases.tau > phases.end + horizon))
    return judge.finish("expired" if expired else "ended")


# ── online: ToleranceMonitor ─────────────────────────────────────────────────


def _row_of(point: Any, kind: str) -> _Row:
    if kind == "joint":
        return _Row(
            getattr(point, "positions", None),
            None,
            getattr(point, "velocities", None),
            getattr(point, "accelerations", None),
        )
    return _Row(
        getattr(point, "position", None),
        getattr(point, "orientation", None),
        getattr(point, "linear_velocity", None),
    )


class ToleranceMonitor:
    """What a trajectory controller runs each tick: `step` returns None while undecided, then
    the `Outcome` (path violated, goal met, or `goal_time_tolerance` ran out) on that tick and
    on every later one.

    `end_time` is the reference's end on the clock of the `t` given to `step`. `joint_names`
    names the columns of the joint points (needed when the tolerances name joints). The caller
    passes the reference point it is tracking; once ``t >= end_time`` that is the held final
    point (positions are taken from it; velocity and acceleration references are 0), which is
    the controller's own goal-checking convention (see `split_phases`). With no goal limits set
    the run succeeds at the first tick from the reference's end on.
    """

    def __init__(
        self,
        tolerances: Tolerances,
        *,
        joint_names: Sequence[str] | None = None,
        end_time: float,
    ) -> None:
        kind = _kind_of(tolerances.path) or _kind_of(tolerances.goal)
        if kind == "joint" and joint_names is None:
            raise ValueError("ToleranceMonitor needs joint_names when the tolerances name joints")
        self._kind = kind or "joint"
        self._tolerances = tolerances
        self._end = float(end_time)
        self._tolerance = 1e-9 * max(1.0, abs(self._end))
        limits = _compile(tolerances, kind, tuple(joint_names or ()))
        self._judge = _Judge(tolerances, limits, self._end)

    def step(self, t: float, reference_point: Any, measured_point: Any) -> Outcome | None:
        t = float(t)
        if self._judge.outcome is not None:
            return self._judge.outcome
        horizon = self._tolerances.goal_time_tolerance
        if horizon > 0.0 and t > self._end + horizon + self._tolerance:
            return self._judge.finish("expired")
        phase = "goal" if in_goal_phase(t, self._end, self._tolerance) else "path"
        return self._judge.step(
            phase, t, t, _row_of(reference_point, self._kind), _row_of(measured_point, self._kind)
        )
