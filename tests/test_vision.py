"""vision messages: boxes, hypotheses, detections and detection arrays."""

import dataclasses
import json
import math

import numpy as np
import pytest

from robot_msgs._core import load_npz, save_npz
from robot_msgs.geometry import Point, Pose, Pose2D, Quaternion, Vector3
from robot_msgs.std import Header
from robot_msgs.vision import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)


def round_trip(msg, tmp_path):
    """to_dict → JSON → from_dict, and save_npz → load_npz: both must give back an equal value."""
    wire = json.loads(json.dumps(msg.to_dict()))
    assert wire["_type"] == msg.type_name
    assert type(msg).from_dict(wire) == msg
    save_npz(msg, tmp_path / "msg.npz")
    assert load_npz(tmp_path / "msg.npz") == msg


def hypothesis(class_id="cup", score=0.9, **kwargs):
    return ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id, score), **kwargs)


def detection2d(xyxy, class_id="cup", score=0.9, **kwargs):
    return Detection2D(
        results=(hypothesis(class_id, score),), bbox=BoundingBox2D.from_xyxy(*xyxy), **kwargs
    )


def detection3d(center, size, class_id="cup", score=0.9, orientation=None, **kwargs):
    pose = Pose(Point(*center), orientation if orientation is not None else Quaternion())
    return Detection3D(
        results=(hypothesis(class_id, score),), bbox=BoundingBox3D(pose, Vector3(*size)), **kwargs
    )


# ── BoundingBox2D ────────────────────────────────────────────────────────────


def test_box2d_from_xyxy_and_back():
    box = BoundingBox2D.from_xyxy(10, 20, 50, 100)
    assert (box.center.x, box.center.y, box.center.theta) == (30.0, 60.0, 0.0)
    assert (box.size_x, box.size_y) == (40.0, 80.0)
    assert box.to_xyxy() == (10.0, 20.0, 50.0, 100.0)
    assert box.area == 3200.0


def test_box2d_iou_against_hand_computed_boxes():
    a = BoundingBox2D.from_xyxy(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BoundingBox2D.from_xyxy(5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert a.iou(BoundingBox2D.from_xyxy(2, 2, 4, 4)) == pytest.approx(4 / 100)  # fully inside
    assert a.iou(BoundingBox2D.from_xyxy(0, 5, 10, 15)) == pytest.approx(50 / 150)
    assert a.iou(BoundingBox2D.from_xyxy(10, 0, 20, 10)) == 0.0  # touching edges
    assert a.iou(BoundingBox2D.from_xyxy(20, 20, 30, 30)) == 0.0  # apart
    b = BoundingBox2D.from_xyxy(5, 5, 15, 15)
    assert a.iou(b) == b.iou(a)


def test_box2d_iou_of_two_empty_boxes_is_zero():
    empty = BoundingBox2D.from_xyxy(1, 1, 1, 1)
    assert empty.iou(empty) == 0.0


def test_box2d_iou_and_xyxy_refuse_rotated_boxes():
    flat = BoundingBox2D.from_xyxy(0, 0, 10, 10)
    tilted = BoundingBox2D(Pose2D(5.0, 5.0, 0.3), 10.0, 10.0)
    with pytest.raises(ValueError, match="axis-aligned"):
        flat.iou(tilted)
    with pytest.raises(ValueError, match="axis-aligned"):
        tilted.iou(flat)
    with pytest.raises(ValueError, match="to_xyxy"):
        tilted.to_xyxy()


def test_box2d_contains_includes_edges_and_honours_rotation():
    box = BoundingBox2D.from_xyxy(0, 0, 10, 20)
    assert box.contains(5, 10) and box.contains(0, 0) and box.contains(10, 20)
    assert not box.contains(10.01, 10) and not box.contains(5, -0.01)
    # a 10×2 bar centred at the origin, turned 90°: now it is 2 wide and 10 tall
    bar = BoundingBox2D(Pose2D(0.0, 0.0, math.pi / 2), 10.0, 2.0)
    assert bar.contains(0.0, 4.9) and not bar.contains(4.9, 0.0)
    assert not bar.contains(1.1, 0.0) and bar.contains(0.9, 0.0)


@pytest.mark.parametrize(
    "size_x, size_y", [(-1.0, 1.0), (1.0, -0.5), (math.inf, 1.0), (math.nan, 1.0)]
)
def test_box2d_rejects_negative_or_non_finite_sizes(size_x, size_y):
    with pytest.raises(ValueError, match="BoundingBox2D.size_[xy]"):
        BoundingBox2D(Pose2D(), size_x, size_y)


def test_box2d_from_xyxy_rejects_inverted_corners():
    with pytest.raises(ValueError, match="BoundingBox2D.size_x"):
        BoundingBox2D.from_xyxy(5, 5, 4, 6)


def test_box2d_is_an_immutable_value():
    box = BoundingBox2D.from_xyxy(0, 0, 4, 2)
    assert box == BoundingBox2D.from_xyxy(0, 0, 4, 2)
    assert hash(box) == hash(BoundingBox2D.from_xyxy(0, 0, 4, 2))
    assert box != BoundingBox2D.from_xyxy(0, 0, 4, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.size_x = 1.0
    assert box.replace(size_x=8.0).size_x == 8.0 and box.size_x == 4.0


def test_box2d_round_trip(tmp_path):
    round_trip(BoundingBox2D(Pose2D(1.5, 2.5, 0.25), 3.0, 4.0), tmp_path)


# ── BoundingBox3D ────────────────────────────────────────────────────────────


def test_box3d_corners_of_an_axis_aligned_box():
    box = BoundingBox3D(Pose(Point(1.0, 2.0, 3.0)), Vector3(2.0, 4.0, 6.0))
    corners = box.corners()
    assert corners.shape == (8, 3)
    assert corners[0].tolist() == [0.0, 0.0, 0.0] and corners[7].tolist() == [2.0, 4.0, 6.0]
    assert corners[1].tolist() == [0.0, 0.0, 6.0]  # z is the fastest axis
    assert corners[2].tolist() == [0.0, 4.0, 0.0]
    assert corners[4].tolist() == [2.0, 0.0, 0.0]
    assert len({tuple(c) for c in corners.tolist()}) == 8
    assert np.allclose(corners.mean(axis=0), [1.0, 2.0, 3.0])


def test_box3d_corners_follow_the_orientation():
    yaw = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
    box = BoundingBox3D(Pose(Point(1.0, 2.0, 3.0), yaw), Vector3(2.0, 4.0, 6.0))
    corners = box.corners()
    # local x (length 2) now points along world y, local y (length 4) along world -x
    assert np.allclose(corners.min(axis=0), [-1.0, 1.0, 0.0])
    assert np.allclose(corners.max(axis=0), [3.0, 3.0, 6.0])
    assert np.allclose(corners[7], [1.0 - 2.0, 2.0 + 1.0, 6.0])  # local (+1, +2, +3)
    # rigid: distances between corners are unchanged
    flat = BoundingBox3D(Pose(Point(1.0, 2.0, 3.0)), Vector3(2.0, 4.0, 6.0)).corners()
    assert np.allclose(np.linalg.norm(corners[7] - corners[0]), np.linalg.norm(flat[7] - flat[0]))


def test_box3d_volume():
    assert BoundingBox3D(Pose(), Vector3(2.0, 3.0, 4.0)).volume == 24.0
    assert BoundingBox3D(Pose(), Vector3(2.0, 0.0, 4.0)).volume == 0.0


def test_box3d_contains_axis_aligned():
    box = BoundingBox3D(Pose(Point(1.0, 1.0, 1.0)), Vector3(2.0, 2.0, 2.0))
    points = [[1, 1, 1], [0, 0, 0], [2, 2, 2], [2.01, 1, 1], [1, 1, -0.01], [5, 5, 5]]
    result = box.contains(points)
    assert result.dtype == np.bool_ and result.shape == (6,)
    assert result.tolist() == [True, True, True, False, False, False]
    assert box.contains(np.zeros((0, 3))).shape == (0,)


def test_box3d_contains_rotated_box_uses_the_local_frame():
    yaw = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
    box = BoundingBox3D(Pose(Point(0.0, 0.0, 0.0), yaw), Vector3(4.0, 2.0, 2.0))
    # long axis now along world y
    assert box.contains([[0.0, 1.9, 0.0], [0.9, 0.0, 0.0]]).tolist() == [True, True]
    assert box.contains([[1.9, 0.0, 0.0], [0.0, 2.1, 0.0]]).tolist() == [False, False]


def test_box3d_contains_agrees_with_its_own_corners_and_centre():
    pose = Pose.from_xyz_rpy(0.5, -0.3, 1.0, 0.3, -0.4, 0.8)
    box = BoundingBox3D(pose, Vector3(1.0, 2.0, 0.5))
    shrunk = box.center.position.as_array() + 0.99 * (
        box.corners() - box.center.position.as_array()
    )
    assert box.contains(shrunk).all()
    grown = box.center.position.as_array() + 1.01 * (box.corners() - box.center.position.as_array())
    assert not box.contains(grown).any()
    assert box.contains([box.center.position.as_array()]).all()


def test_box3d_validation_and_shapes():
    with pytest.raises(ValueError, match="BoundingBox3D.size"):
        BoundingBox3D(Pose(), Vector3(1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="points"):
        BoundingBox3D(Pose(), Vector3(1.0, 1.0, 1.0)).contains(np.zeros((3, 2)))


def test_box3d_is_an_immutable_value_and_round_trips(tmp_path):
    box = BoundingBox3D(Pose.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3), Vector3(1.0, 2.0, 3.0))
    assert box == BoundingBox3D(Pose.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3), Vector3(1.0, 2.0, 3.0))
    assert box != box.replace(size=Vector3(1.0, 2.0, 3.5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.size = Vector3()
    round_trip(box, tmp_path)


# ── ObjectHypothesis(WithPose) ───────────────────────────────────────────────


def test_object_hypothesis_stores_class_and_score():
    result = ObjectHypothesis("bottle", 1)
    assert result.class_id == "bottle" and result.score == 1.0 and isinstance(result.score, float)
    assert ObjectHypothesis(7, 0.5).class_id == "7"
    with pytest.raises(ValueError, match="ObjectHypothesis.score"):
        ObjectHypothesis("x", math.nan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0


def test_hypothesis_with_pose_defaults_and_covariance_validation():
    result = ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("cup", 0.8))
    assert result.pose == Pose() and result.covariance is None
    covered = ObjectHypothesisWithPose(hypothesis=result.hypothesis, covariance=np.eye(6) * 0.01)
    assert covered.covariance.shape == (6, 6)
    with pytest.raises(ValueError):
        covered.covariance[0, 0] = 1.0
    with pytest.raises(ValueError, match="ObjectHypothesisWithPose.covariance must be"):
        ObjectHypothesisWithPose(hypothesis=result.hypothesis, covariance=np.eye(3))


def test_hypothesis_with_pose_value_semantics():
    a = hypothesis("cup", 0.8, covariance=np.eye(6))
    b = hypothesis("cup", 0.8, covariance=np.eye(6))
    assert a == b and hash(a) == hash(b)
    assert a != hypothesis("cup", 0.8)
    assert a != hypothesis("mug", 0.8, covariance=np.eye(6))


def test_hypotheses_round_trip(tmp_path):
    round_trip(ObjectHypothesis("cup", 0.75), tmp_path)
    round_trip(hypothesis(), tmp_path)
    round_trip(
        hypothesis("mug", 0.5, pose=Pose.from_xyz_rpy(1, 2, 3, 0, 0, 1), covariance=np.eye(6)),
        tmp_path,
    )


# ── Detection2D / Detection3D ────────────────────────────────────────────────


def test_detection_best_is_the_highest_score_and_ties_keep_the_first():
    detection = Detection2D(
        results=(hypothesis("a", 0.2), hypothesis("b", 0.7), hypothesis("c", 0.7)),
        bbox=BoundingBox2D.from_xyxy(0, 0, 1, 1),
    )
    assert detection.best.hypothesis.class_id == "b"
    assert (detection.class_id, detection.score) == ("b", 0.7)


def test_detection_without_results_has_no_best():
    for detection in (
        Detection2D(bbox=BoundingBox2D.from_xyxy(0, 0, 1, 1)),
        Detection3D(bbox=BoundingBox3D(Pose(), Vector3(1.0, 1.0, 1.0))),
    ):
        assert detection.best is None
        assert detection.class_id == "" and detection.score == 0.0


def test_detection_fields_follow_ros_and_are_keyword_only():
    names = [f.name for f in dataclasses.fields(Detection2D)]
    assert names == ["header", "results", "bbox", "id"]
    assert [f.name for f in dataclasses.fields(Detection3D)] == names
    with pytest.raises(TypeError):
        Detection2D(Header(), (), BoundingBox2D.from_xyxy(0, 0, 1, 1))


def test_detection_results_are_normalised_to_a_tuple():
    detection = Detection2D(results=[hypothesis()], bbox=BoundingBox2D.from_xyxy(0, 0, 1, 1), id=3)
    assert isinstance(detection.results, tuple) and detection.id == "3"


def test_detection_value_semantics():
    a = detection2d((0, 0, 4, 4), id="t1", header=Header(1.0, "cam"))
    b = detection2d((0, 0, 4, 4), id="t1", header=Header(1.0, "cam"))
    assert a == b and hash(a) == hash(b)
    assert a != detection2d((0, 0, 4, 5), id="t1", header=Header(1.0, "cam"))
    assert a != a.replace(id="t2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.id = "x"
    assert detection3d((0, 0, 0), (1, 1, 1)) == detection3d((0, 0, 0), (1, 1, 1))


def test_detections_round_trip(tmp_path):
    round_trip(detection2d((1, 2, 30, 40), id="a", header=Header(2.0, "cam")), tmp_path)
    round_trip(
        detection3d((1, 2, 3), (0.1, 0.2, 0.3), orientation=Quaternion.from_euler(0, 0, 1)),
        tmp_path,
    )
    round_trip(Detection2D(bbox=BoundingBox2D.from_xyxy(0, 0, 1, 1)), tmp_path)


# ── Detection2DArray ─────────────────────────────────────────────────────────


def make_array2d():
    return Detection2DArray(
        header=Header(9.0, "cam"),
        detections=(
            detection2d((0, 0, 10, 10), "cup", 0.9),
            detection2d((1, 1, 11, 11), "cup", 0.8),  # IoU 81/119 with the first
            detection2d((50, 50, 60, 70), "bottle", 0.6),
            detection2d((100, 100, 110, 110), "cup", 0.3),
        ),
    )


def test_array2d_sequence_protocol():
    array = make_array2d()
    assert len(array) == 4
    assert [d.score for d in array] == [0.9, 0.8, 0.6, 0.3]
    assert array[2].class_id == "bottle" and array[-1].score == 0.3
    part = array[1:3]
    assert isinstance(part, Detection2DArray) and len(part) == 2
    assert part.header == array.header and part[0] is array[1]
    assert len(array[:0]) == 0
    with pytest.raises(IndexError):
        array[4]


def test_array2d_as_arrays_is_the_batch_view():
    batch = make_array2d().as_arrays()
    assert batch["boxes_xyxy"].shape == (4, 4)
    assert batch["boxes_xyxy"][2].tolist() == [50.0, 50.0, 60.0, 70.0]
    assert batch["scores"].tolist() == [0.9, 0.8, 0.6, 0.3]
    assert batch["class_ids"].tolist() == ["cup", "cup", "bottle", "cup"]
    assert batch["class_ids"].dtype.kind == "U"
    for values in batch.values():
        assert not values.flags.writeable


def test_array2d_as_arrays_of_an_empty_array_has_the_right_shapes():
    batch = Detection2DArray().as_arrays()
    assert batch["boxes_xyxy"].shape == (0, 4)
    assert batch["scores"].shape == (0,) and batch["class_ids"].shape == (0,)


def test_array2d_as_arrays_uses_empty_class_and_zero_score_without_results():
    array = Detection2DArray(detections=(Detection2D(bbox=BoundingBox2D.from_xyxy(0, 0, 1, 1)),))
    batch = array.as_arrays()
    assert batch["class_ids"].tolist() == [""] and batch["scores"].tolist() == [0.0]


def test_array2d_filtered_by_score_and_class():
    array = make_array2d()
    assert [d.score for d in array.filtered(min_score=0.6)] == [0.9, 0.8, 0.6]
    assert [d.score for d in array.filtered(class_ids=["bottle"])] == [0.6]
    assert [d.score for d in array.filtered(0.5, {"cup"})] == [0.9, 0.8]
    assert len(array.filtered()) == 4
    assert len(array.filtered(min_score=0.95)) == 0
    assert array.filtered(0.5).header == array.header


def test_array2d_nms_keeps_the_highest_score_of_overlapping_boxes():
    array = make_array2d()
    kept = array.nms(0.5)
    assert [d.score for d in kept] == [0.9, 0.6, 0.3]
    assert len(array.nms(0.7)) == 4  # IoU 0.68 is below the threshold
    assert [d.score for d in array.nms(0.0)] == [0.9, 0.6, 0.3]  # any overlap suppresses
    assert kept.header == array.header


def test_array2d_nms_does_not_depend_on_input_order_for_who_wins():
    array = make_array2d()
    shuffled = Detection2DArray(detections=tuple(reversed(array.detections)))
    assert sorted(d.score for d in shuffled.nms(0.5)) == [0.3, 0.6, 0.9]
    # survivors keep the order they came in
    assert [d.score for d in shuffled.nms(0.5)] == [0.3, 0.6, 0.9]


def test_array2d_nms_suppression_is_chained_through_kept_boxes_only():
    # b overlaps a (suppressed); c overlaps only b. Since b is gone, c survives.
    a = detection2d((0, 0, 10, 10), score=0.9)
    b = detection2d((6, 0, 16, 10), score=0.8)  # IoU with a = 40/160 = 0.25
    c = detection2d((12, 0, 22, 10), score=0.7)  # IoU with b = 40/160, with a = 0
    kept = Detection2DArray(detections=(a, b, c)).nms(0.2)
    assert [d.score for d in kept] == [0.9, 0.7]


def test_array2d_nms_validates_threshold_and_handles_empty():
    assert len(Detection2DArray().nms(0.5)) == 0
    with pytest.raises(ValueError, match="iou_threshold"):
        make_array2d().nms(1.5)
    tilted = Detection2D(bbox=BoundingBox2D(Pose2D(1.0, 1.0, 0.2), 2.0, 2.0))
    with pytest.raises(ValueError, match="axis-aligned"):
        Detection2DArray(detections=(tilted, detection2d((0, 0, 1, 1)))).nms(0.5)


def test_array2d_value_semantics_and_round_trip(tmp_path):
    array = make_array2d()
    assert array == make_array2d() and hash(array) == hash(make_array2d())
    assert array != array[:3]
    assert isinstance(Detection2DArray(detections=list(array)).detections, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        array.detections = ()
    round_trip(array, tmp_path)
    round_trip(Detection2DArray(), tmp_path)


# ── Detection3DArray ─────────────────────────────────────────────────────────


def make_array3d():
    return Detection3DArray(
        header=Header(9.0, "base"),
        detections=(
            detection3d((0, 0, 0), (2, 2, 2), "cup", 0.9),
            detection3d((1, 0, 0), (2, 2, 2), "cup", 0.8),  # overlap 4, union 12 → IoU 1/3
            detection3d((10, 0, 0), (1, 2, 3), "box", 0.6),
        ),
    )


def test_array3d_sequence_protocol():
    array = make_array3d()
    assert len(array) == 3 and [d.class_id for d in array] == ["cup", "cup", "box"]
    assert array[2].score == 0.6
    part = array[:2]
    assert isinstance(part, Detection3DArray) and len(part) == 2 and part.header == array.header


def test_array3d_as_arrays_is_the_batch_view():
    batch = make_array3d().as_arrays()
    assert batch["centers"].tolist() == [[0, 0, 0], [1, 0, 0], [10, 0, 0]]
    assert batch["sizes"].tolist() == [[2, 2, 2], [2, 2, 2], [1, 2, 3]]
    assert batch["scores"].tolist() == [0.9, 0.8, 0.6]
    assert batch["class_ids"].tolist() == ["cup", "cup", "box"]
    empty = Detection3DArray().as_arrays()
    assert empty["centers"].shape == (0, 3) and empty["sizes"].shape == (0, 3)


def test_array3d_filtered_by_score_and_class():
    array = make_array3d()
    assert [d.score for d in array.filtered(min_score=0.7)] == [0.9, 0.8]
    assert [d.class_id for d in array.filtered(class_ids=["box"])] == ["box"]
    assert len(array.filtered(0.7, ["box"])) == 0


def test_array3d_nms_uses_volume_iou():
    array = make_array3d()
    assert [d.score for d in array.nms(0.3)] == [0.9, 0.6]  # 1/3 > 0.3 → second dropped
    assert [d.score for d in array.nms(0.4)] == [0.9, 0.8, 0.6]
    assert len(Detection3DArray().nms(0.5)) == 0


def test_array3d_nms_refuses_rotated_boxes_and_bad_thresholds():
    yaw = Quaternion.from_euler(0.0, 0.0, 0.5)
    array = Detection3DArray(detections=(detection3d((0, 0, 0), (1, 1, 1), orientation=yaw),))
    with pytest.raises(ValueError, match="axis-aligned"):
        array.nms(0.5)
    with pytest.raises(ValueError, match="iou_threshold"):
        make_array3d().nms(-0.1)


def test_array3d_value_semantics_and_round_trip(tmp_path):
    array = make_array3d()
    assert array == make_array3d() and hash(array) == hash(make_array3d())
    assert array != array[:2]
    round_trip(array, tmp_path)
    round_trip(Detection3DArray(), tmp_path)
