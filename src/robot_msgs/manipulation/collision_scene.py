"""CollisionScene — everything a planner must not hit (or may touch), each element tagged with how.

Mirrors ``moveit_msgs/PlanningScene`` trimmed to its collision part: a list of ``CollisionObject``s
(here `CollisionElement`: ``id``, ``pose``, ``primitives`` + ``primitive_poses``, ``meshes`` +
``mesh_poses``, ``planes`` + ``plane_poses``, field names verbatim), the robot state the robot's own
geometry was posed at, and an optional `AllowedCollisionMatrix`. What MoveIt has no field for is the
point of this type, in Zach's words (Daily Note, 2026-11-26, "collision scene agent"):

    *"our model of the collision scene can continue to improve, thats the key type though... in the
    simplest case its basically empty, in v1 it just has some static objects in v2 we incorporate
    the camera occupancy map, dynamic objects etc."*

    *"we have fixed vs moving and deformable vs non deforable. A concrete floor is fixed and non
    deformable. Any collision is not acceptable. objects are moving and deformable. actually a
    success grasp is one where you do get some deformation!!!"*

So each element carries:

- `source` — where it came from: ``"scene"`` (the SceneDescription: floor, conveyor, bins),
  ``"object"`` (the things being handled, generally moving) or ``"self"`` (the robot's own links,
  from its RobotDescription; `link` names which).
- `motion` — ``"fixed"`` or ``"moving"`` (a cable, an object on a conveyor, a robot link).
- `compliance` — ``"rigid"`` or ``"deformable"`` (a mattress, a bag, a glove).
- `margin` (m) — the element is inflated by this much: a point closer than `margin` to its surface
  counts as touching it (coal / FCL ``security_margin``, MoveIt's link padding).
- `contact` — the policy for touching it, one of `CONTACT_POLICIES`. **v1 uses only ``"avoid"``**
  (*"as v1 we should say like any collision body assume its Avoid. eventually in the future we may
  update this though right, cause like sometimes we actively need contact"*). ``"allow"`` (contact is
  fine, the pair is not checked, as an ACM entry) and ``"touch"`` (contact is the goal: a grasp, a
  push) are reserved names so a later planner can read them without a schema change.

**Planes are half-spaces.** A `Plane` element is solid on the side opposite its normal: a floor with
normal +z fills everything below it. WHY: a two-sided infinitely thin plane (FCL's ``Plane``) lets a
discrete check step straight through a floor; MoveIt users model floors as thin boxes for the same
reason. coal's ``Halfspace`` is the matching checker shape.

**Not here yet (v2, a documented future field):** a camera occupancy map (MoveIt's
``PlanningScene.world.octomap``). It will be an additional optional field (an occupancy grid or
OctoMap message) with its own `source` ``"sensor"``; adding it is a minor version, so v1 consumers keep
working. Dynamic elements (an object's predicted motion) likewise arrive as additive fields.

The checker that answers "is this point / pose / configuration in collision with the scene, with
margin" and the viewer live in robot_collision (`robot_collision.scene`); this is the value they share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from robot_msgs._core import Msg, set_field, to_dict
from robot_msgs.geometry import Pose
from robot_msgs.manipulation.allowed_collision_matrix import AllowedCollisionMatrix
from robot_msgs.sensor import JointState
from robot_msgs.shape import Mesh, Plane, SolidPrimitive
from robot_msgs.std import Header

SOURCES = ("scene", "object", "self")
"""Where an element came from: the SceneDescription, the handled objects, the robot itself."""
MOTIONS = ("fixed", "moving")
COMPLIANCES = ("rigid", "deformable")
CONTACT_POLICIES = ("avoid", "allow", "touch")
"""``avoid``: any contact is a collision (v1's only value in use). ``allow``: contact is fine, not
checked. ``touch``: contact is the goal (a grasp, a push). Reserved so a planner can grow into them."""

DIGEST_LENGTH = 16  # hex characters of sha256, as AllowedCollisionMatrix.digest


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CollisionElement(Msg):
    """One thing in a `CollisionScene`: moveit_msgs/CollisionObject's geometry plus its tags.

    - `id`: unique in its scene (MoveIt ``CollisionObject.id``); also its ACM entry name.
    - `pose`: the element frame in the scene frame (MoveIt ``CollisionObject.pose``). Every shape
      pose below is in the element frame.
    - `primitives` / `primitive_poses`, `meshes` / `mesh_poses`, `planes` / `plane_poses`: the
      geometry, one pose per shape. At least one shape.
    - `source`, `motion`, `compliance`, `margin`, `contact`: see the module docstring.
    - `link`: for ``source="self"``, the robot link the geometry belongs to (its frame); for an
      object held by the robot, the link it is attached to. Empty otherwise.
    - `label`: what it is ("floor", "bottle", "cable"); free text for people and detectors' classes.
    - `why`: where the element came from (the description file, the detector, the fit).
    """

    id: str
    source: str
    pose: Pose = field(default_factory=Pose)
    primitives: tuple[SolidPrimitive, ...] = ()
    primitive_poses: tuple[Pose, ...] = ()
    meshes: tuple[Mesh, ...] = ()
    mesh_poses: tuple[Pose, ...] = ()
    planes: tuple[Plane, ...] = ()
    plane_poses: tuple[Pose, ...] = ()
    motion: str = "fixed"
    compliance: str = "rigid"
    margin: float = 0.0
    contact: str = "avoid"
    link: str = ""
    label: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        name = str(self.id)
        # WHY refuse instead of strip: ids are looked up byte for byte (and become ACM entries)
        if not name or name != name.strip():
            raise ValueError(f"CollisionElement.id must be non-empty without surrounding spaces: {name!r}")
        set_field(self, "id", name)
        for label, value, allowed in (("source", self.source, SOURCES), ("motion", self.motion, MOTIONS),
                                      ("compliance", self.compliance, COMPLIANCES),
                                      ("contact", self.contact, CONTACT_POLICIES)):
            if value not in allowed:
                raise ValueError(f"CollisionElement {name!r}: {label} {value!r} is not one of {allowed}")
        if not isinstance(self.pose, Pose):
            raise TypeError(f"CollisionElement {name!r}: pose must be a Pose, got {type(self.pose).__name__}")
        for shapes, poses, kind in (("primitives", "primitive_poses", SolidPrimitive),
                                    ("meshes", "mesh_poses", Mesh), ("planes", "plane_poses", Plane)):
            items, places = tuple(getattr(self, shapes)), tuple(getattr(self, poses))
            if any(not isinstance(s, kind) for s in items):
                raise TypeError(f"CollisionElement {name!r}: {shapes} must all be {kind.__name__}")
            if any(not isinstance(p, Pose) for p in places):
                raise TypeError(f"CollisionElement {name!r}: {poses} must all be Pose")
            if len(items) != len(places):
                raise ValueError(f"CollisionElement {name!r} has {len(items)} {shapes} and {len(places)} {poses}")
            set_field(self, shapes, items)
            set_field(self, poses, places)
        if not (self.primitives or self.meshes or self.planes):
            raise ValueError(f"CollisionElement {name!r} has no geometry (primitives, meshes or planes)")
        margin = float(self.margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"CollisionElement {name!r}: margin must be finite and >= 0 m, got {self.margin!r}")
        set_field(self, "margin", margin)
        for label in ("link", "label", "why"):
            set_field(self, label, str(getattr(self, label)))

    # ── reading ───────────────────────────────────────────────────────────────────────────────

    @property
    def shape_count(self) -> int:
        return len(self.primitives) + len(self.meshes) + len(self.planes)

    def shapes(self) -> tuple[tuple[SolidPrimitive | Mesh | Plane, np.ndarray], ...]:
        """Every shape with its 4x4 pose in the SCENE frame (``pose @ shape_pose``), primitives
        first, then meshes, then planes."""
        frame = self.pose.to_matrix()
        out = []
        for items, places in ((self.primitives, self.primitive_poses), (self.meshes, self.mesh_poses),
                              (self.planes, self.plane_poses)):
            out.extend((shape, frame @ place.to_matrix()) for shape, place in zip(items, places, strict=True))
        return tuple(out)

    @property
    def tags(self) -> tuple[str, str, str, str]:
        """``(source, motion, compliance, contact)``: what a scorer or viewer groups by."""
        return (self.source, self.motion, self.compliance, self.contact)

    # ── modified copies ───────────────────────────────────────────────────────────────────────

    def with_margin(self, margin: float) -> CollisionElement:
        return self.replace(margin=margin)

    def with_contact(self, contact: str) -> CollisionElement:
        return self.replace(contact=contact)

    def moved(self, pose: Pose) -> CollisionElement:
        """The same element with its frame at `pose` (in the scene frame)."""
        return self.replace(pose=pose)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CollisionScene(Msg):
    """Every collision element of a scene, in one frame (`header.frame_id`, usually ``"world"``).

    - `elements`: scene, object and self elements; ids unique. Empty is valid: the simplest scene
      (*"in the simplest case its basically empty"*).
    - `robot_state`: the joint configuration the ``self`` elements were posed at (MoveIt
      ``PlanningScene.robot_state``); None when the scene has no self elements or they are static.
    - `acm`: optional `AllowedCollisionMatrix` over element ids (and robot links): pairs it allows
      are never checked. Without one, every pair whose elements are both ``avoid`` is checked.
    - `name`, `why`: what the scene is and how it was built (fidelity level, inputs, generator).
    """

    elements: tuple[CollisionElement, ...] = ()
    header: Header = field(default_factory=lambda: Header(frame_id="world"))
    robot_state: JointState | None = None
    acm: AllowedCollisionMatrix | None = None
    name: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        if any(not isinstance(e, CollisionElement) for e in elements):
            raise TypeError("CollisionScene.elements must all be CollisionElement")
        ids = [e.id for e in elements]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"CollisionScene names elements twice: {duplicates}")
        set_field(self, "elements", elements)
        if self.acm is not None and not isinstance(self.acm, AllowedCollisionMatrix):
            raise TypeError("CollisionScene.acm must be an AllowedCollisionMatrix or None")
        if self.robot_state is not None and not isinstance(self.robot_state, JointState):
            raise TypeError("CollisionScene.robot_state must be a JointState or None")
        set_field(self, "name", str(self.name))
        set_field(self, "why", str(self.why))

    @classmethod
    def empty(cls, frame_id: str = "world", *, name: str = "empty", why: str = "") -> CollisionScene:
        """The simplest collision scene: nothing in it, every point is free."""
        return cls(header=Header(frame_id=frame_id), name=name, why=why)

    # ── reading ───────────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.elements)

    def element(self, element_id: str) -> CollisionElement:
        for e in self.elements:
            if e.id == element_id:
                return e
        raise KeyError(f"CollisionScene has no element {element_id!r}; it has {list(self.ids)}")

    def where(self, *, source: str | None = None, motion: str | None = None, compliance: str | None = None,
              contact: str | None = None) -> tuple[CollisionElement, ...]:
        """The elements matching every tag given: ``scene.where(motion="fixed", compliance="rigid")``."""
        for label, value, allowed in (("source", source, SOURCES), ("motion", motion, MOTIONS),
                                      ("compliance", compliance, COMPLIANCES), ("contact", contact, CONTACT_POLICIES)):
            if value is not None and value not in allowed:
                raise ValueError(f"CollisionScene.where: {label} {value!r} is not one of {allowed}")
        return tuple(e for e in self.elements
                     if (source is None or e.source == source) and (motion is None or e.motion == motion)
                     and (compliance is None or e.compliance == compliance)
                     and (contact is None or e.contact == contact))

    def counts(self) -> dict[str, int]:
        """How many elements carry each tag value, plus ``"elements"`` and ``"shapes"``."""
        out = dict.fromkeys((*SOURCES, *MOTIONS, *COMPLIANCES, *CONTACT_POLICIES), 0)
        for e in self.elements:
            for tag in e.tags:
                out[tag] += 1
        out["elements"] = len(self.elements)
        out["shapes"] = sum(e.shape_count for e in self.elements)
        return out

    def is_checked(self, a: str, b: str) -> bool:
        """Whether contact between elements (or ACM entries) `a` and `b` must be checked: both
        ``avoid`` (v1), and not allowed by the `acm`. An id the scene does not hold counts as
        ``avoid`` (a robot link named only in the ACM)."""
        known = {e.id: e for e in self.elements}
        for name in (a, b):
            if name in known and known[name].contact != "avoid":
                return False
        return not (self.acm is not None and self.acm.is_allowed(a, b))

    def digest(self) -> str:
        """Short hash of everything a checker reads (geometry, poses, tags, margins, ACM cells),
        not of `name`, `why`, labels or the header stamp. Same digest, same collision answers."""
        payload = []
        for e in self.elements:
            d = to_dict(e)
            for key in ("label", "why"):
                d.pop(key, None)
            payload.append(d)
        h = hashlib.sha256(json.dumps([self.header.frame_id, payload], sort_keys=True).encode())
        h.update((self.acm.digest() if self.acm is not None else "").encode())
        return h.hexdigest()[:DIGEST_LENGTH]

    # ── modified copies ───────────────────────────────────────────────────────────────────────

    def with_element(self, element: CollisionElement) -> CollisionScene:
        """A copy holding `element`: replaces the element with its id, or appends it."""
        if element.id in self.ids:
            return self.replace(elements=tuple(element if e.id == element.id else e for e in self.elements))
        return self.replace(elements=self.elements + (element,))

    def with_elements(self, elements: Iterable[CollisionElement]) -> CollisionScene:
        scene = self
        for element in elements:
            scene = scene.with_element(element)
        return scene

    def without(self, *ids: str) -> CollisionScene:
        """A copy without the named elements (an unknown id is an error, never ignored)."""
        missing = sorted(set(ids) - set(self.ids))
        if missing:
            raise KeyError(f"CollisionScene.without: no elements {missing}")
        return self.replace(elements=tuple(e for e in self.elements if e.id not in ids))

    def with_margin(self, margin: float, **where: str) -> CollisionScene:
        """A copy where every element matching `where` (see `where`) has `margin`;
        ``scene.with_margin(0.02, source="object")``."""
        chosen = {e.id for e in self.where(**where)}
        return self.replace(elements=tuple(e.with_margin(margin) if e.id in chosen else e for e in self.elements))
