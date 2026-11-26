"""CollisionScene / CollisionElement: tags, validation, queries, modified copies, JSON round trip."""

import json

import numpy as np
import pytest

from robot_msgs._core import from_dict
from robot_msgs.geometry import Pose
from robot_msgs.manipulation import (
    COMPLIANCES,
    CONTACT_POLICIES,
    MOTIONS,
    SOURCES,
    AllowedCollisionMatrix,
    CollisionElement,
    CollisionScene,
)
from robot_msgs.shape import Plane, SolidPrimitive

IDENTITY = Pose.identity()


def box(element_id="crate", **kw):
    kw.setdefault("source", "scene")
    return CollisionElement(id=element_id, primitives=(SolidPrimitive.box(0.4, 0.3, 0.2),),
                            primitive_poses=(IDENTITY,), **kw)


def cell():
    floor = CollisionElement(id="floor", source="scene", planes=(Plane.from_point_normal((0, 0, 0), (0, 0, 1)),),
                             plane_poses=(IDENTITY,), label="concrete floor")
    bottle = CollisionElement(id="bottle", source="object", pose=Pose.from_xyz_rpy(0.2, 0, 0.7, 0, 0, 0),
                              primitives=(SolidPrimitive.cylinder(0.2, 0.04),), primitive_poses=(IDENTITY,),
                              motion="moving", compliance="deformable", margin=0.01)
    link = box("forearm", source="self", link="forearm", motion="moving")
    return CollisionScene(elements=(floor, box(), bottle, link), name="cell")


def test_the_simplest_scene_is_empty_and_valid():
    scene = CollisionScene.empty()
    assert len(scene) == 0 and scene.header.frame_id == "world"
    assert scene.counts()["elements"] == 0


def test_tags_have_zachs_two_axes_and_v1_contact_is_avoid():
    assert SOURCES == ("scene", "object", "self")
    assert MOTIONS == ("fixed", "moving") and COMPLIANCES == ("rigid", "deformable")
    assert CONTACT_POLICIES[0] == "avoid"
    e = box()
    assert e.tags == ("scene", "fixed", "rigid", "avoid") and e.margin == 0.0


@pytest.mark.parametrize("field,value", [("source", "camera"), ("motion", "drifting"),
                                         ("compliance", "soft"), ("contact", "ignore")])
def test_unknown_tag_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        box(**{field: value} if field != "source" else {"source": value})


@pytest.mark.parametrize("margin", [-0.001, float("nan"), float("inf")])
def test_margin_must_be_finite_and_non_negative(margin):
    with pytest.raises(ValueError, match="margin"):
        box(margin=margin)


def test_an_element_needs_geometry_and_one_pose_per_shape():
    with pytest.raises(ValueError, match="no geometry"):
        CollisionElement(id="ghost", source="scene")
    with pytest.raises(ValueError, match="primitive_poses"):
        CollisionElement(id="x", source="scene", primitives=(SolidPrimitive.sphere(0.1),), primitive_poses=())
    with pytest.raises(ValueError, match="surrounding"):
        box(" padded")


def test_ids_are_unique_in_a_scene():
    with pytest.raises(ValueError, match="twice"):
        CollisionScene(elements=(box(), box()))


def test_where_and_counts_group_by_tag():
    scene = cell()
    assert [e.id for e in scene.where(motion="fixed", compliance="rigid")] == ["floor", "crate"]
    assert [e.id for e in scene.where(source="object")] == ["bottle"]
    counts = scene.counts()
    assert counts["moving"] == 2 and counts["deformable"] == 1 and counts["self"] == 1
    assert counts["avoid"] == 4 and counts["shapes"] == 4
    with pytest.raises(ValueError):
        scene.where(motion="sometimes")


def test_shapes_are_posed_in_the_scene_frame():
    bottle = cell().element("bottle")
    (shape, pose), = bottle.shapes()
    assert shape.type == SolidPrimitive.CYLINDER
    np.testing.assert_allclose(pose[:3, 3], [0.2, 0.0, 0.7])


def test_modified_copies_never_mutate():
    scene = cell()
    padded = scene.with_margin(0.02, source="object")
    assert padded.element("bottle").margin == 0.02 and scene.element("bottle").margin == 0.01
    assert padded.element("crate").margin == 0.0
    moved = scene.with_element(scene.element("crate").moved(Pose.from_xyz_rpy(1, 0, 0, 0, 0, 0)))
    assert moved.ids == scene.ids and moved.element("crate").pose.position.x == 1.0
    assert scene.without("crate").ids == ("floor", "bottle", "forearm")
    with pytest.raises(KeyError):
        scene.without("nope")


def test_is_checked_reads_contact_policy_and_acm():
    scene = cell()
    assert scene.is_checked("floor", "forearm")
    allowed = scene.replace(acm=AllowedCollisionMatrix.from_pairs(("floor", "forearm"), [("floor", "forearm")]))
    assert not allowed.is_checked("floor", "forearm")
    touch = scene.with_element(scene.element("bottle").with_contact("touch"))
    assert not touch.is_checked("bottle", "forearm")


def test_digest_tracks_what_a_checker_reads_only():
    scene = cell()
    assert scene.digest() == scene.replace(name="renamed", why="notes").digest()
    assert scene.digest() != scene.with_margin(0.03, source="object").digest()
    assert scene.digest() != scene.with_element(scene.element("bottle").with_contact("touch")).digest()


def test_json_round_trip_keeps_every_tag_and_margin():
    scene = cell()
    back = from_dict(json.loads(json.dumps(scene.to_dict())))
    assert back == scene
    assert [e.tags for e in back.elements] == [e.tags for e in scene.elements]
    assert back.element("bottle").margin == 0.01
