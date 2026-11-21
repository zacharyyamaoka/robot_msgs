"""NamedPose: a robot's saved positions by name (MoveIt SRDF group_state), their digest, the
Cartesian kind's IK tie-break, and the SRDF / JSON files a description publishes."""

import json
import math
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest

from robot_msgs.geometry import PoseStamped
from robot_msgs.manipulation import (
    NamedPose,
    find_named_pose,
    named_poses_from_json,
    named_poses_from_srdf,
    named_poses_to_json,
    named_poses_to_srdf,
    nearest_solution,
)
from robot_msgs.sensor import JointState

READY = NamedPose.from_degrees(
    name="ready",
    group="arm",
    degrees={"revolute_3": -12.7, "revolute_6": -24.754, "revolute_11": 1.167, "revolute_14": 90.0},
    tolerance_deg=3.0,
    why="where the arm waits between picks",
)

# a two-link planar arm (shoulder, elbow) for the Cartesian kind: two IK branches per reachable point
L1, L2 = 0.30, 0.25


def two_link_ik(pose: NamedPose):
    """Both elbow branches that put the tip at the tool pose's (x, y)."""
    x, y = pose.tool_pose.pose.position.x, pose.tool_pose.pose.position.y
    c = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        return []
    out = []
    for elbow in (math.acos(c), -math.acos(c)):
        shoulder = math.atan2(y, x) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
        out.append((shoulder, elbow))
    return out


def tip(q):
    return (L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]), L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1]))


def over_bin(seed, x=0.35, y=0.20) -> NamedPose:
    return NamedPose(
        name="over_bin",
        group="arm",
        joint_names=("shoulder", "elbow"),
        positions=seed,
        tool_pose=PoseStamped.from_xyz_rpy(x, y, 0.0, 0.0, 0.0, 0.0, frame_id="base_link"),
        tip_link="tip",
        why="the tip over the bin",
    )


# ---------------------------------------------------------------------------------------------- the value


def test_written_in_degrees_stored_in_radians():
    assert READY.kind == "joint"
    assert READY.position_of("revolute_3") == math.radians(-12.7)
    assert READY.degrees()["revolute_3"] == pytest.approx(-12.7, abs=1e-12)
    assert READY.tolerance.tolist() == [math.radians(3.0)] * 4
    assert READY.as_dict() == dict(zip(READY.joint_names, READY.positions.tolist(), strict=True))
    state = READY.joint_state()
    assert isinstance(state, JointState) and state.name == READY.joint_names
    assert np.array_equal(state.position, READY.positions)


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"name": ""}, "non-empty"),
        ({"name": " ready"}, "surrounding spaces"),
        ({"group": "arm "}, "surrounding spaces"),
        ({"joint_names": (), "positions": []}, "no joints"),
        ({"joint_names": ("a", "a"), "positions": [0, 1]}, "twice"),
        ({"joint_names": ("a", "b"), "positions": [0, 1, 2]}, "shape"),
        ({"joint_names": ("a",), "positions": [math.nan]}, "finite"),
        ({"joint_names": ("a",), "positions": [0.0], "tolerance": [-1.0]}, ">= 0"),
        ({"joint_names": ("a",), "positions": [0.0], "tip_link": "tcp"}, "without a tool_pose"),
    ],
)
def test_refuses_what_cannot_be_a_saved_position(changes, words):
    fields = {"name": "ready", "group": "arm", "joint_names": ("a",), "positions": [0.0]} | changes
    with pytest.raises(ValueError, match=words):
        NamedPose(**fields)
    with pytest.raises(ValueError, match="tip_link"):
        NamedPose(name="x", group="arm", joint_names=("a",), positions=[0.0],
                  tool_pose=PoseStamped())


def test_deviation_is_state_minus_target_per_joint():
    state = {n: v + 0.01 for n, v in READY.as_dict().items()} | {"other_joint": 5.0}
    assert READY.deviation(state) == pytest.approx(dict.fromkeys(READY.joint_names, 0.01))
    with pytest.raises(KeyError, match="no value"):
        READY.deviation({"revolute_3": 0.0})


# ---------------------------------------------------------------------------------------------- the digest


def test_the_digest_names_where_the_pose_is_not_why_or_how_close():
    digest = READY.digest()
    assert len(digest) == 16 and int(digest, 16) >= 0
    # the same pose, joints listed in another order: the same place
    order = tuple(reversed(READY.joint_names))
    shuffled = READY.replace(joint_names=order, positions=[READY.position_of(n) for n in order])
    assert shuffled.digest() == digest
    # a new reason or tolerance moves nothing
    assert READY.replace(why="reworded", tolerance=None).digest() == digest
    # a moved joint, a renamed pose or another group is somewhere else
    moved = READY.positions.copy()
    moved[0] += 1e-9
    assert READY.replace(positions=moved).digest() != digest
    assert READY.replace(name="home").digest() != digest
    assert READY.replace(group="wrist").digest() != digest
    # -0.0 is 0.0
    zero = NamedPose(name="z", group="g", joint_names=("a",), positions=[0.0])
    assert zero.replace(positions=[-0.0]).digest() == zero.digest()


def test_the_digest_algorithm_is_pinned():
    """A record carries the digest of the pose it was measured at: changing the algorithm would call
    every record stale. This value is the algorithm; change it only on purpose."""
    pose = NamedPose(name="ready", group="arm", joint_names=("a", "b"), positions=[0.5, -0.25])
    assert pose.digest() == "becd1c70e5169665"


# ---------------------------------------------------------------------------------------------- lookup


def test_find_named_pose_by_name_and_group():
    home = NamedPose.from_degrees(name="home", group="arm", degrees={"revolute_3": 0.0})
    closed = NamedPose.from_degrees(name="closed", group="claw", degrees={"revolute_5": 0.0})
    closed_arm = NamedPose.from_degrees(name="closed", group="arm", degrees={"revolute_3": 1.0})
    poses = (READY, home, closed, closed_arm)
    assert find_named_pose(poses, "ready") is READY
    assert find_named_pose(poses, "closed", group="claw") is closed
    with pytest.raises(KeyError, match="pass group="):
        find_named_pose(poses, "closed")
    with pytest.raises(KeyError, match=r"there are: \['closed \(arm\)', 'closed \(claw\)', 'home \(arm\)'"):
        find_named_pose(poses, "fold")
    with pytest.raises(ValueError, match="given twice"):
        find_named_pose((READY, READY), "ready")


# ---------------------------------------------------------------------------------------------- Cartesian


def test_a_cartesian_pose_resolves_to_the_solution_nearest_its_seed():
    up, down = two_link_ik(over_bin([0.0, 0.0]))
    for seed, want in ((np.add(up, 0.1), up), (np.add(down, -0.1), down)):
        pose = over_bin(seed)
        assert pose.kind == "cartesian"
        resolved = pose.resolved(two_link_ik)
        assert resolved.kind == "joint" and resolved.name == "over_bin" and resolved.why == pose.why
        assert resolved.positions.tolist() == pytest.approx(list(want))
        assert tip(resolved.positions) == pytest.approx((0.35, 0.20))
        assert pose.joint_state(two_link_ik).position.tolist() == resolved.positions.tolist()


def test_a_cartesian_pose_never_passes_its_seed_off_as_the_target():
    pose = over_bin([0.1, 0.2])
    for read in (lambda: pose.position_of("elbow"), pose.as_dict, pose.degrees, pose.joint_state):
        with pytest.raises(ValueError, match="Cartesian"):
            read()
    with pytest.raises(ValueError, match="no solution"):
        over_bin([0.0, 0.0], x=2.0).resolved(two_link_ik)
    # the tool pose is part of where it is
    assert over_bin([0.1, 0.2]).digest() != over_bin([0.1, 0.2], x=0.36).digest()


def test_the_tie_break_is_the_largest_single_joint_move_first():
    seed = [0.0, 0.0, 0.0]
    flipped = [0.0, 0.0, 0.9]  # one joint swings 0.9 (a branch flip): Euclidean 0.9
    spread = [0.6, 0.6, 0.6]  # every joint moves 0.6: Euclidean 1.04, but no joint past 0.6
    assert nearest_solution(seed, [flipped, spread]).tolist() == spread
    assert nearest_solution(seed, []) is None
    with pytest.raises(ValueError, match="shape"):
        nearest_solution(seed, [[0.0, 0.0]])


# ---------------------------------------------------------------------------------------------- the files


def test_srdf_is_moveits_shape_and_reads_back_exactly():
    claw = NamedPose.from_degrees(name="closed", group="claw", degrees={"revolute_5": -47.5},
                                  why="jaws shut -- the claw's own stop")
    text = named_poses_to_srdf((READY, claw), "bam_fb_v2")
    root = ElementTree.fromstring(text)
    assert root.tag == "robot" and root.get("name") == "bam_fb_v2"
    assert [g.get("name") for g in root.findall("group")] == ["arm", "claw"]
    assert [j.get("name") for j in root.find("group").findall("joint")] == list(READY.joint_names)
    states = root.findall("group_state")
    assert [(s.get("name"), s.get("group")) for s in states] == [("ready", "arm"), ("closed", "claw")]
    assert "jaws shut - - the claw's own stop" in text and READY.digest() in text
    back = named_poses_from_srdf(text)
    assert [p.name for p in back] == ["ready", "closed"]
    assert back[0].positions.tolist() == READY.positions.tolist()  # exact, not approx
    assert back[0].digest() == READY.digest()  # an SRDF loses why and tolerance, never where


def test_srdf_needs_the_ik_for_a_cartesian_pose():
    pose = over_bin([0.1, 0.2])
    with pytest.raises(ValueError, match="pass an ik"):
        named_poses_to_srdf([pose], "two_link")
    back = named_poses_from_srdf(named_poses_to_srdf([pose], "two_link", ik=two_link_ik))
    assert tip(back[0].positions) == pytest.approx((0.35, 0.20))


def test_json_is_lossless_and_refuses_a_hand_edit():
    cartesian = over_bin([0.1, 0.2])
    text = named_poses_to_json((READY, cartesian), "bam_fb_v2", source="descriptions/bam_fb_v2.py")
    back = named_poses_from_json(text)
    assert back == (READY, cartesian)
    assert [p.digest() for p in back] == [READY.digest(), cartesian.digest()]
    edited = text.replace(repr(float(READY.positions[0])), repr(float(READY.positions[0]) + 0.1), 1)
    with pytest.raises(ValueError, match="edited after it was generated"):
        named_poses_from_json(edited)
    with pytest.raises(ValueError, match="schema"):
        named_poses_from_json('{"schema": "something-else", "poses": []}')


# ----------------------------------------------------------------------------------------- description


def test_description_defaults_empty_and_is_taken_by_from_degrees_and_resolved():
    assert READY.description == ""
    described = NamedPose.from_degrees(name="ready", group="arm", degrees={"j": 1.0},
                                       description="Parked above the bin; the arm waits here.")
    assert described.description == "Parked above the bin; the arm waits here."
    assert described.resolved().description == described.description


def test_description_leaves_the_digest_alone():
    assert READY.replace(description="where the arm waits").digest() == READY.digest()


def test_json_carries_a_description_and_omits_an_empty_one():
    described = READY.replace(description="Parked above the bin; the arm waits here.")
    text = named_poses_to_json((described,), "r")
    assert json.loads(text)["poses"][0]["pose"]["description"] == described.description
    assert named_poses_from_json(text)[0].description == described.description
    # empty: no key, so a file published before the field existed is unchanged
    plain = named_poses_to_json((READY,), "r")
    assert "description" not in json.loads(plain)["poses"][0]["pose"]
    assert named_poses_from_json(plain)[0].description == ""


def test_srdf_prints_a_description_only_when_there_is_one():
    assert "waits here" not in named_poses_to_srdf((READY,), "r")
    srdf = named_poses_to_srdf((READY.replace(description="The arm waits here."),), "r")
    assert "<!-- The arm waits here. -->" in srdf
