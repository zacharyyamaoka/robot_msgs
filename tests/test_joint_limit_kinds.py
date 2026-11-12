"""Physical vs software position bounds, and the after-the-fact check that reads them (2026-11-11).

Zach: "Physical limits actually prevent further rotation. software limits don't." and "the way to
enforce that check is just afterwards to check the position read out and make sure it stayed within
the limits the entire time."
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from dataclass_values import Basis, Value

from robot_msgs.trajectory import (
    PHYSICAL,
    SOFTWARE,
    JointLimits,
    JointLimitViolation,
    JointTrajectory,
    joint_limit_violations,
)

NAMES = ("shoulder", "elbow", "wrist")


def kind_limits() -> JointLimits:
    return JointLimits(
        joint_names=NAMES,
        min_position=[-1.0, -2.0, -3.0],
        max_position=[1.0, 2.0, 3.0],
        position_kind=(PHYSICAL, SOFTWARE, None),
    )


def tape(positions, t=None, **fields) -> JointTrajectory:
    positions = np.asarray(positions, dtype=float)
    t = np.arange(len(positions)) * 0.1 if t is None else t
    return JointTrajectory(t=t, joint_names=NAMES, positions=positions, **fields)


# ── the kind on JointLimits ──────────────────────────────────────────────────


def test_a_kind_per_joint_is_validated_and_kept_as_the_plain_word():
    cited = Value(SOFTWARE, basis=Basis.DATASHEET, source="UR5e technical specification")
    limits = JointLimits(joint_names=("a", "b"), position_kind=(PHYSICAL, cited))
    assert limits.position_kind == ("physical", "software")
    assert type(limits.position_kind[1]) is str  # the evidence stays in the description
    with pytest.raises(ValueError, match="2 entries, joint_names has 3"):
        JointLimits(joint_names=NAMES, position_kind=(PHYSICAL, SOFTWARE))
    with pytest.raises(ValueError, match="'hard'"):
        JointLimits(joint_names=("a",), position_kind=("hard",))


def test_no_stated_kind_has_one_spelling():
    assert JointLimits(joint_names=("a", "b"), position_kind=(None, None)).position_kind is None
    assert JointLimits(joint_names=("a", "b")) == JointLimits(
        joint_names=("a", "b"), position_kind=(None, None)
    )
    assert "position_kind" not in repr(JointLimits(joint_names=("a",), max_velocity=[1.0]))


def test_select_and_a_trajectory_carry_the_kinds_in_joint_order():
    limits = kind_limits()
    assert limits.select(("wrist", "shoulder")).position_kind == (None, PHYSICAL)
    assert limits.select(("wrist",)).position_kind is None
    reordered = JointLimits(
        joint_names=("wrist", "elbow", "shoulder"),
        max_position=[3.0, 2.0, 1.0],
        position_kind=(None, SOFTWARE, PHYSICAL),
    )
    attached = tape(np.zeros((3, 3)), limits=reordered).limits
    assert attached.joint_names == NAMES
    assert attached.position_kind == (PHYSICAL, SOFTWARE, None)


def test_kinds_round_trip_through_json():
    limits = kind_limits()
    assert JointLimits.from_dict(json.loads(json.dumps(limits.to_dict()))) == limits


# ── the check ────────────────────────────────────────────────────────────────


def test_a_tape_past_a_software_limit_gives_one_violation():
    positions = np.zeros((10, 3))
    positions[4:7, 1] = [2.1, 2.4, 2.2]  # the elbow leaves its software max of 2.0 once
    (violation,) = joint_limit_violations(tape(positions), kind_limits())
    assert violation.joint == "elbow" and violation.quantity == "position"
    assert violation.side == "max" and violation.kind == SOFTWARE
    assert violation.limit == 2.0 and violation.value == 2.4
    assert violation.excess == pytest.approx(0.4)
    assert (violation.t, violation.t_first, violation.t_last) == pytest.approx((0.5, 0.4, 0.6))
    assert violation.samples == 3
    assert str(violation) == "elbow: position reaches 2.4 > 2 at t=0.5 s"


def test_every_crossing_of_one_bound_is_one_record():
    positions = np.zeros((10, 3))
    positions[1, 1] = 2.05
    positions[8, 1] = 2.5  # left twice: one record, worst at t = 0.8, spanning both
    (violation,) = joint_limit_violations(tape(positions), kind_limits())
    assert violation.samples == 2 and violation.value == 2.5
    assert (violation.t_first, violation.t_last) == pytest.approx((0.1, 0.8))


def test_a_physical_bound_says_so_and_both_sides_are_reported():
    positions = np.zeros((4, 3))
    positions[0, 0] = -1.5  # past the physical min
    positions[3, 0] = 1.25  # past the physical max
    found = joint_limit_violations(tape(positions), kind_limits())
    assert [(v.joint, v.side, v.kind) for v in found] == [
        ("shoulder", "min", PHYSICAL),
        ("shoulder", "max", PHYSICAL),
    ]
    assert str(found[0]) == "shoulder: position reaches -1.5 < -1 at t=0 s"
    assert found[0].excess == pytest.approx(0.5)


def test_on_the_bound_is_inside_and_a_nan_never_hides_a_violation():
    positions = np.zeros((5, 3))
    positions[0, 1] = 2.0  # exactly on the bound: inside
    assert joint_limit_violations(tape(positions), kind_limits()) == ()
    positions[1, 1] = np.nan
    positions[3, 1] = 2.2
    (violation,) = joint_limit_violations(tape(positions), kind_limits())
    assert violation.samples == 1 and violation.t == pytest.approx(0.3)


def test_unstated_kind_is_still_checked_and_records_none():
    positions = np.zeros((3, 3))
    positions[2, 2] = 3.5
    (violation,) = joint_limit_violations(tape(positions), kind_limits())
    assert violation.joint == "wrist" and violation.kind is None


def test_quantities_pick_what_is_judged():
    positions = np.zeros((5, 3))
    positions[2, 1] = 2.5
    velocities = np.zeros((5, 3))
    velocities[3, 0] = -9.0
    limits = kind_limits().replace(max_velocity=[1.0, 1.0, 1.0])
    run = tape(positions, velocities=velocities)
    every = joint_limit_violations(run, limits)
    assert [(v.quantity, v.joint) for v in every] == [
        ("position", "elbow"),
        ("velocity", "shoulder"),
    ]
    assert every[1].kind is None and every[1].value == -9.0 and every[1].excess == 8.0
    only_positions = joint_limit_violations(run, limits, quantities=("position",))
    assert [v.quantity for v in only_positions] == ["position"]
    with pytest.raises(ValueError, match="unknown quantities"):
        joint_limit_violations(run, limits, quantities=("torque",))


def test_the_sentences_are_the_records_text():
    positions = np.zeros((6, 3))
    positions[2, 0] = 1.3
    positions[4, 1] = -2.6
    velocities = np.full((6, 3), 0.5)
    run = tape(
        positions, velocities=velocities, limits=kind_limits().replace(max_velocity=[0.4] * 3)
    )
    records = joint_limit_violations(run)
    assert run.limit_violations() == tuple(str(record) for record in records)
    assert run.limit_violations()[0] == "elbow: position reaches -2.6 < -2 at t=0.4 s"


def test_without_limits_it_says_so():
    with pytest.raises(ValueError, match="no limits given"):
        joint_limit_violations(tape(np.zeros((2, 3))))


def test_a_violation_record_is_validated():
    fields = dict(
        joint="a",
        quantity="position",
        side="max",
        kind=SOFTWARE,
        limit=1.0,
        value=1.5,
        excess=0.5,
        t=0.2,
        t_first=0.1,
        t_last=0.3,
        samples=3,
    )
    JointLimitViolation(**fields)
    for change, message in (
        ({"excess": 0.0}, "excess must be > 0"),
        ({"quantity": "velocity"}, "only a position bound has a kind"),
        ({"quantity": "velocity", "kind": None, "side": "min"}, "its only side is 'max'"),
        ({"kind": "hard"}, "not in"),
        ({"t": 0.5}, "between the first and the last"),
        ({"samples": 0}, "at least one sample"),
    ):
        with pytest.raises(ValueError, match=message):
            JointLimitViolation(**{**fields, **change})
