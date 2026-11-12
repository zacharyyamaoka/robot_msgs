"""GraspPose and WaypointAction: what a grasp detector and a pick recipe hand the generator."""

from __future__ import annotations

import json
import math

import pytest

from robot_msgs.geometry import Pose, Vector3
from robot_msgs.manipulation import LABELS, GraspPose, Waypoint, WaypointAction
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits

DOWN = (math.pi, 0.0, 0.0)


def pose(x, y, z):
    return Pose.from_xyz_rpy(x, y, z, *DOWN)


# bam_traj_gen's make_pick_and_place names (2026-11-11), each with the label it means
PICK_AND_PLACE = [
    ("start", "start"),
    ("above_pick", "approach"),
    ("pre_grasp", "pre_grasp"),
    ("start_grasp", "grasp"),
    ("closed_grasp", "close"),
    ("lift", "lift"),
    ("above_place", "carry"),
    ("pre_place", "pre_place"),
    ("start_place", "place"),
    ("open_grasp", "release"),
    ("retreat", "retreat"),
    ("ready", "ready"),
]


def pick_and_place(**changes) -> WaypointAction:
    values = dict(
        header=Header(frame_id="world"),
        tool_frame="tcp_world",
        waypoints=tuple(
            Waypoint(name=name, label=label, pose=pose(0.4, 0.0, 0.05 + 0.01 * i))
            for i, (name, label) in enumerate(PICK_AND_PLACE)
        ),
    )
    values.update(changes)
    return WaypointAction(**values)


def test_every_label_today_s_pick_and_place_needs_exists():
    assert {label for _, label in PICK_AND_PLACE} <= set(LABELS)
    assert "ready" in LABELS  # Zach's "the arm can move on" point


def test_waypoints_are_found_by_name_by_index_and_by_label():
    action = pick_and_place()
    assert action["closed_grasp"].label == "close"
    assert action[4] is action["closed_grasp"]
    assert action.index_of("lift") == 5
    assert [w.name for w in action.labelled("ready")] == ["ready"]
    assert action.pose_stamped("lift").header.frame_id == "world"
    with pytest.raises(KeyError, match="no waypoint named"):
        action["nope"]


def test_a_free_text_name_needs_no_label():
    action = WaypointAction(waypoints=(Waypoint(name="wiggle 3", pose=pose(0, 0, 0)),))
    assert action[0].label is None


def test_the_sync_target_defaults_to_the_one_close_waypoint():
    action = pick_and_place()
    assert action.sync_target_index() == 4
    assert pick_and_place(sync_target="lift").sync_target_index() == 5
    assert not action.is_sync_required


def test_a_moving_target_marks_the_action_as_needing_sync():
    waypoints = list(pick_and_place().waypoints)
    waypoints[4] = waypoints[4].replace(target_velocity=Vector3(0.1, 0, 0))
    assert pick_and_place(waypoints=tuple(waypoints)).is_sync_required


def test_a_grasp_pose_carries_widths_score_and_frame():
    grasp = GraspPose(
        pose=pose(0.4, 0, 0.02),
        width=0.01,
        pre_grasp_width=0.05,
        score=0.9,
        header=Header(frame_id="world"),
    )
    assert grasp.pose_stamped.header.frame_id == "world"
    assert GraspPose(pose=pose(0, 0, 0), width=0.0).pre_grasp_width is None


def test_round_trips_keep_labels_velocities_and_nested_limits():
    action = pick_and_place(
        arm_joints=("j1", "j2"),
        arm_limits=JointLimits(joint_names=("j1", "j2"), max_velocity=[1.0, 1.0]),
    )
    again = WaypointAction.from_dict(json.loads(json.dumps(action.to_dict())))
    assert again == action and again["closed_grasp"].label == "close"


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Waypoint(name="", pose=pose(0, 0, 0)), "empty"),
        (lambda: Waypoint(name="a", pose=pose(0, 0, 0), label="grab"), "label"),
        (lambda: Waypoint(name="a", pose=pose(0, 0, 0), gripper_width=-0.01), "gripper_width"),
        (lambda: Waypoint(name="a", pose=pose(0, 0, 0), duration=0.0), "duration"),
        (lambda: Waypoint(name="a", pose=pose(0, 0, 0), interpolation_points=1.5), "whole"),
        (lambda: WaypointAction(waypoints=()), "at least one"),
        (
            lambda: WaypointAction(waypoints=(Waypoint(name="a", pose=pose(0, 0, 0)),) * 2),
            "duplicate",
        ),
        (
            lambda: WaypointAction(
                waypoints=(Waypoint(name="a", pose=pose(0, 0, 0), blend_radius=0.01),)
            ),
            "last waypoint",
        ),
        (lambda: pick_and_place(sync_target="nope"), "sync_target"),
        (
            lambda: pick_and_place(
                arm_joints=("j1",), arm_limits=JointLimits(joint_names=("j2",), max_velocity=[1.0])
            ),
            "arm_limits",
        ),
        (lambda: GraspPose(pose=pose(0, 0, 0), width=0.05, pre_grasp_width=0.01), "smaller"),
        (lambda: GraspPose(pose=pose(0, 0, 0), width=math.nan), "width"),
    ],
)
def test_bad_requests_are_refused_at_construction(build, match):
    with pytest.raises((ValueError, TypeError), match=match):
        build()
