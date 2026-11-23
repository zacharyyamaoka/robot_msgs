"""AllowedCollisionMatrix: MoveIt semantics, SRDF round trip, validation, modified copies."""

import numpy as np
import pytest

from robot_msgs.manipulation import (
    REASONS,
    AllowedCollisionMatrix,
    allowed_collision_matrix_from_srdf,
    allowed_collision_matrix_to_srdf,
)

NAMES = ("base", "shoulder", "upper", "fore", "bin")


def acm():
    return AllowedCollisionMatrix.from_pairs(
        NAMES,
        [("base", "shoulder", "Adjacent"), ("upper", "shoulder", "Adjacent"),
         ("base", "bin", "Default"), ("shoulder", "fore", "Never"), ("upper", "bin")],
        why="test",
    )


def test_cells_follow_moveit_true_means_allowed():
    m = acm()
    assert m.is_allowed("base", "shoulder") and m.is_allowed("shoulder", "base")
    assert not m.is_allowed("base", "fore")
    assert m.is_allowed("fore", "fore")  # the diagonal
    assert m.reason("shoulder", "upper") == "Adjacent"  # order-free; stored in entry order
    assert m.reason("upper", "bin") == "User"  # allowed, no recorded reason
    assert m.reason("base", "fore") is None  # checked
    assert ("base", "fore") in m.checked_pairs() and ("base", "shoulder") in m.allowed_pairs()
    assert len(m.checked_pairs()) + len(m.allowed_pairs()) == len(NAMES) * (len(NAMES) - 1) // 2


def test_counts_cover_every_pair_once():
    c = acm().counts()
    assert c == {"Adjacent": 2, "Default": 1, "Always": 0, "Never": 1, "User": 1, "Checked": 5}
    assert sum(c.values()) == 10


def test_srdf_round_trip_keeps_cells_and_reasons():
    m = acm()
    xml = allowed_collision_matrix_to_srdf(m, "arm")
    assert xml.count("<disable_collisions") == 5 and 'reason="Never"' in xml
    back = allowed_collision_matrix_from_srdf(xml, NAMES)
    assert np.array_equal(back.entry_values, m.entry_values)
    assert back.reason("upper", "bin") == "User" and back.reason("shoulder", "fore") == "Never"
    assert back.digest() == m.digest()


def test_srdf_unknown_reason_reads_as_user_and_names_default_to_first_seen():
    xml = '<robot name="r"><disable_collisions link1="a" link2="b" reason="Not Disabled"/></robot>'
    m = allowed_collision_matrix_from_srdf(xml)
    assert m.entry_names == ("a", "b") and m.reason("a", "b") == "User"


def test_digest_tracks_cells_not_reasons_or_why():
    m = acm()
    assert m.replace(why="other").digest() == m.digest()
    assert m.with_checked("upper", "bin").digest() != m.digest()


def test_modified_copies():
    m = acm()
    checked = m.with_checked("base", "shoulder")
    assert not checked.is_allowed("base", "shoulder") and checked.reason("base", "shoulder") is None
    assert m.is_allowed("base", "shoulder")  # the original is untouched
    grown = m.with_allowed("fore", "camera", "User")
    assert grown.entry_names[-1] == "camera" and grown.is_allowed("camera", "fore")
    assert not grown.is_allowed("camera", "base")  # a new body is checked against the rest


def test_default_entries_fall_back_like_moveit():
    m = acm().replace(default_entry_names=("octomap",), default_entry_values=[True])
    assert m.is_allowed("octomap", "fore") and not m.is_allowed("ghost", "fore")
    with pytest.raises(ValueError, match="both as entries and defaults"):
        acm().replace(default_entry_names=("bin",), default_entry_values=[True])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"entry_values": np.triu(np.ones((5, 5), bool))}, "not symmetric"),
        ({"entry_values": np.zeros((5, 5), bool)}, "diagonal"),
        ({"entry_names": NAMES[:4]}, "shape"),
        ({"entry_names": ("base", "base", "u", "f", "b")}, "twice"),
        ({"reasons": (("base", "fore", "Never"),)}, "checked, not allowed"),
        ({"reasons": (("base", "shoulder", "Sometimes"),)}, "not one of"),
    ],
)
def test_validation_names_the_problem(change, message):
    with pytest.raises(ValueError, match=message):
        acm().replace(**change)


def test_reasons_are_moveits_strings():
    assert REASONS == ("Adjacent", "Default", "Always", "Never", "User")
