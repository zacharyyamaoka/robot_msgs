"""robot_msgs.space: a message holds a value OR a range, and the space is derived from it.

Zach's Oct 5 2026 ruling (py_dataclass_viewer): same class, not a second `PoseSpace` class.
A field is a `Value(…, low=, high=)` that still IS a float; `Space(msg).sample()` returns the
ordinary message class; `plain()` drops the notes at a runtime boundary (ROS, JSON).
"""

from __future__ import annotations

import json
import pickle
import random

import numpy as np
import pytest

from robot_msgs.geometry import Point, Pose, Quaternion
from robot_msgs.space import Basis, Normal, Space, Value, meta, plain
from robot_msgs.trajectory import JointTrajectory


def ranged_pose() -> Pose:
    return Pose(
        position=Point(
            x=Value(0.5, "m", low=0.4, high=0.6, source="bin centre ± 10 cm"),
            y=0.0,
            z=Value(0.3, "m", dist=Normal(0.3, 0.01), low=0.25, high=0.35),
        ),
        orientation=Quaternion(),
    )


# ── a Value is the number ────────────────────────────────────────────────────


def test_value_is_the_builtin_and_keeps_its_notes_inside_a_message():
    pose = ranged_pose()
    x = pose.position.x
    assert isinstance(x, float) and isinstance(x, Value)
    assert x == 0.5 and x * 2 == 1.0
    assert meta(x)["low"] == 0.4 and meta(x)["source"] == "bin centre ± 10 cm"
    # equality and hashing are on the numbers, so a ranged message equals its plain twin
    assert pose == plain(pose)
    assert hash(pose) == hash(plain(pose))


def test_a_value_outside_its_own_range_is_refused():
    with pytest.raises(ValueError, match="outside"):
        Point(x=Value(0.7, "m", low=0.4, high=0.6))


def test_non_finite_values_are_still_refused():
    with pytest.raises(ValueError, match="finite"):
        Point(x=Value(float("nan"), "m"))


# ── Space(msg).sample() returns the same class ───────────────────────────────


def test_sample_returns_the_same_message_class_drawn_inside_every_range():
    space = Space(ranged_pose())
    for seed in range(200):
        drawn = space.sample(random.Random(seed))
        assert type(drawn) is Pose
        assert 0.4 <= drawn.position.x <= 0.6
        assert 0.25 <= drawn.position.z <= 0.35
        assert drawn.position.y == 0.0  # no range → unchanged
        assert meta(drawn.position.x)["basis"] is Basis.SAMPLED
    assert space.sample(random.Random(7)) == space.sample(random.Random(7))
    assert len({space.sample(random.Random(s)).position.x for s in range(20)}) > 10


def test_sampled_messages_are_inside_their_space():
    space = Space(ranged_pose())
    assert all(space.contains(space.sample(random.Random(s))) for s in range(50))


# ── contains / violations: the acceptable range in a test ────────────────────


def test_contains_names_every_field_that_broke_its_range():
    space = Space(ranged_pose())
    good = Pose(position=Point(x=0.45, y=9.0, z=0.3))
    bad = Pose(position=Point(x=0.7, y=0.0, z=0.2))
    assert space.contains(good)
    assert not space.contains(bad)
    broken = space.violations(bad)
    assert [v.split(" ")[0] for v in broken] == ["position.x=0.7", "position.z=0.2"]
    assert "[0.4, 0.6]" in broken[0]


def test_contains_refuses_a_different_message_type():
    with pytest.raises(TypeError):
        Space(ranged_pose()).contains(Point())


# ── array fields: gym Box bounds, still the same class ───────────────────────


def joint_trajectory() -> JointTrajectory:
    return JointTrajectory(
        t=np.linspace(0, 1, 5),
        joint_names=("j1", "j2"),
        positions=np.zeros((5, 2)),
    )


def test_array_bounds_sample_a_trajectory_of_the_same_class():
    low, high = np.array([-0.1, -0.2]), np.array([0.1, 0.2])  # per joint, broadcast over time
    space = Space(joint_trajectory(), positions=(low, high))
    drawn = space.sample(random.Random(3))
    assert type(drawn) is JointTrajectory
    assert drawn.positions.shape == (5, 2)
    assert np.all(drawn.positions >= low) and np.all(drawn.positions <= high)
    assert np.ptp(drawn.positions) > 0
    assert space.contains(drawn)
    assert space.sample(random.Random(3)) == drawn


def test_array_bounds_report_the_first_offending_sample():
    space = Space(joint_trajectory(), positions=(-0.1, 0.1))
    positions = np.zeros((5, 2))
    positions[3, 1] = 0.5
    bad = joint_trajectory().replace(positions=positions)
    assert not space.contains(bad)
    (violation,) = space.violations(bad)
    assert violation.startswith("positions[3, 1]=0.5")


def test_array_bounds_must_name_an_array_field():
    with pytest.raises(ValueError, match="no array field"):
        Space(joint_trajectory(), nonsense=(0, 1))


# ── boundaries: the wire never sees the notes ────────────────────────────────


def test_plain_and_to_dict_drop_the_notes_but_pickle_keeps_them():
    pose = ranged_pose()
    bare = plain(pose)
    assert type(bare.position.x) is float
    encoded = pose.to_dict()
    assert type(encoded["position"]["x"]) is float
    assert Pose.from_dict(json.loads(json.dumps(encoded))) == pose
    restored = pickle.loads(pickle.dumps(pose))
    assert meta(restored.position.x)["high"] == 0.6
