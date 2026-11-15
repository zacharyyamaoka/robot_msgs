"""NamedPose — a robot's saved position, by name: MoveIt's SRDF ``<group_state>``, with its reason.

Zach, 2026-11-15: *"It's the idea of like save positions. So each robot description can basically
have a number of saved positions that could be like in joint space or Cartesian space. And then
ready position is just a saved position in space. And so all the code should not be like
referencing an exact position, but should rather be like accessing that saved position by name, so
that in the future you can always just change it easily and it continues to work."*

Prior art: MoveIt's SRDF ``<group_state name="ready" group="arm">`` and
``MoveGroupInterface.setNamedTarget("ready")`` / ``getNamedTargetValues``; Zach's own ROS stack
(2025, bam_descriptions ``RobotParam._init_saved_poses``: ZERO / HOME / READY each as a joint
state AND the tool pose it gives, captured from the live arm by ``bam_calibration``'s
``save_curr_pose.py``, which also printed the SRDF ``<group_state>`` to paste into moveit_config).

The functions here are the one way to look a pose up (`find_named_pose`), the stock file it lowers
to (`named_poses_to_srdf` / `named_poses_from_srdf`) and the lossless file a description publishes
(`named_poses_to_json` / `named_poses_from_json`, each pose with its `NamedPose.digest`).
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import quoteattr

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.geometry import PoseStamped
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.sensor import JointState

NamedPoseKind = Literal["joint", "cartesian"]
NAMED_POSES_SCHEMA = "robot_msgs/named-poses/1"
DIGEST_LENGTH = 16  # hex characters of sha256: 64 bits, plenty to tell one saved pose from another

InverseKinematics = Callable[["NamedPose"], Iterable[Sequence[float]]]
"""Every joint solution (in the pose's ``joint_names`` order) that puts ``tip_link`` at
``tool_pose``. It receives the whole `NamedPose`: tool pose, tip link, joint names, seed."""


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class NamedPose(Msg):
    """A saved position of a robot's joint group, looked up by name (``"ready"``, ``"home"``).

    - `name`, `group`: the SRDF ``<group_state name=… group=…>``; a name is unique in its group.
    - `joint_names`, `positions` (rad, m for a prismatic joint; the URDF's joint frame): the target
      of a joint-space pose. For a Cartesian pose they are the IK seed, never the target.
    - `tolerance` (same units, one per joint): how far a realisation may sit from the target and
      still count as this pose; `None` = not stated (the reader's gate decides).
    - `tool_pose` + `tip_link`: a Cartesian pose, `tip_link`'s pose in the tool pose's frame. It
      becomes joint values through an IK the caller passes (`joint_state`, `resolved`), which
      keeps the solution nearest the seed (`nearest_solution`).
    - `why`: the reason for these numbers, kept beside them.

    `digest()` names WHERE the pose is (names and numbers, not `why` or `tolerance`): a record
    measured at this pose carries the digest, and a different digest means the pose moved since.

    WHY a type and not constants (Zach, 2026-11-15, module docstring): a literal copied into each
    consumer cannot move with the pose. `WaypointLabel` ``"ready"`` says what a waypoint is for; a
    `NamedPose` ``"ready"`` says where that is.
    """

    name: str
    group: str
    joint_names: tuple[str, ...]
    positions: Array
    tolerance: Array | None = None
    tool_pose: PoseStamped | None = None
    tip_link: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        for label in ("name", "group"):
            value = str(getattr(self, label))
            # WHY refuse instead of strip: a name is looked up byte for byte; never edit one silently
            if not value or value != value.strip():
                raise ValueError(
                    f"NamedPose.{label} must be non-empty without surrounding spaces, got {value!r}"
                )
            set_field(self, label, value)
        names = tuple(str(n) for n in self.joint_names)
        if not names:
            raise ValueError(f"NamedPose {self.name!r} has no joints")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"NamedPose {self.name!r} names a joint twice: {duplicates}")
        set_field(self, "joint_names", names)
        positions = frozen_array(self.positions)
        if positions.shape != (len(names),):
            raise ValueError(
                f"NamedPose {self.name!r}.positions has shape {positions.shape}, "
                f"joint_names has {len(names)}"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"NamedPose {self.name!r}.positions must be finite: {positions}")
        set_field(self, "positions", positions)
        tolerance = optional_array(self.tolerance)
        if tolerance is not None:
            if tolerance.shape != (len(names),):
                raise ValueError(
                    f"NamedPose {self.name!r}.tolerance has shape {tolerance.shape}, "
                    f"joint_names has {len(names)}"
                )
            if not (np.all(np.isfinite(tolerance)) and np.all(tolerance >= 0.0)):
                raise ValueError(f"NamedPose {self.name!r}.tolerance must be finite and >= 0")
        set_field(self, "tolerance", tolerance)
        if self.tool_pose is not None:
            expect_type(f"NamedPose {self.name!r}.tool_pose", self.tool_pose, PoseStamped)
            if not str(self.tip_link):
                raise ValueError(f"NamedPose {self.name!r} has a tool_pose but no tip_link")
        elif str(self.tip_link):
            raise ValueError(f"NamedPose {self.name!r} names tip_link {self.tip_link!r} "
                             "without a tool_pose")
        set_field(self, "tip_link", str(self.tip_link))
        set_field(self, "why", str(self.why))

    @classmethod
    def from_degrees(
        cls,
        *,
        name: str,
        group: str,
        degrees: Mapping[str, float],
        tolerance_deg: float | Mapping[str, float] | None = None,
        why: str = "",
    ) -> NamedPose:
        """A joint-space pose written in degrees (what people read), stored in radians (SI).
        `tolerance_deg` is one number for every joint or one per joint."""
        names = tuple(degrees)
        tolerance = None
        if tolerance_deg is not None:
            per_joint = (tolerance_deg if isinstance(tolerance_deg, Mapping)
                         else dict.fromkeys(names, tolerance_deg))
            tolerance = [math.radians(float(per_joint[n])) for n in names]
        return cls(name=name, group=group, joint_names=names,
                   positions=[math.radians(float(degrees[n])) for n in names],
                   tolerance=tolerance, why=why)

    @property
    def kind(self) -> NamedPoseKind:
        """``"joint"`` (the positions are the target) or ``"cartesian"`` (a tool pose whose IK the
        positions seed)."""
        return "joint" if self.tool_pose is None else "cartesian"

    def _refuse_seed(self, what: str) -> None:
        if self.tool_pose is not None:
            raise ValueError(
                f"NamedPose {self.name!r} is Cartesian: its positions are the IK seed, not the "
                f"target; {what} the resolved pose (resolved(ik)) instead"
            )

    def index_of(self, joint: str) -> int:
        try:
            return self.joint_names.index(joint)
        except ValueError:
            raise KeyError(
                f"NamedPose {self.name!r} has no joint {joint!r}; joints: {self.joint_names}"
            ) from None

    def position_of(self, joint: str) -> float:
        """One joint's target (rad)."""
        self._refuse_seed("read")
        return float(self.positions[self.index_of(joint)])

    def as_dict(self) -> dict[str, float]:
        """Joint name -> target (rad)."""
        self._refuse_seed("read")
        return {n: float(p) for n, p in zip(self.joint_names, self.positions, strict=True)}

    def degrees(self) -> dict[str, float]:
        """Joint name -> target in degrees, for showing a pose to people."""
        return {n: math.degrees(p) for n, p in self.as_dict().items()}

    def joint_state(self, ik: InverseKinematics | None = None) -> JointState:
        """The pose as joint values: its positions or (Cartesian) the IK solution nearest the seed."""
        return JointState(name=self.joint_names, position=self.resolved(ik).positions)

    def resolved(self, ik: InverseKinematics | None = None) -> NamedPose:
        """This pose in joint space. A joint pose comes back as it is; a Cartesian one needs `ik`
        and comes back as a joint pose (same name, group, tolerance, why) at the solution
        `nearest_solution` keeps."""
        if self.tool_pose is None:
            return self
        if ik is None:
            raise ValueError(f"NamedPose {self.name!r} is Cartesian ({self.tip_link} in "
                             f"{self.tool_pose.header.frame_id!r}): pass an ik")
        solution = nearest_solution(self.positions, ik(self))
        if solution is None:
            raise ValueError(f"NamedPose {self.name!r}: the IK found no solution for "
                             f"{self.tip_link} at {self.tool_pose!r}")
        return NamedPose(name=self.name, group=self.group, joint_names=self.joint_names,
                         positions=solution, tolerance=self.tolerance, why=self.why)

    def deviation(self, state: JointState | Mapping[str, float]) -> dict[str, float]:
        """``state - target`` for each of this pose's joints (rad, unwrapped: say which turn you
        mean). A joint missing from `state` raises."""
        values = state.as_dict() if isinstance(state, JointState) else dict(state)
        missing = [n for n in self.joint_names if n not in values]
        if missing:
            raise KeyError(f"NamedPose {self.name!r}: the state has no value for {missing}")
        return {n: float(values[n]) - target for n, target in self.as_dict().items()}

    def digest(self) -> str:
        """16 hex of sha256 over where the pose is: name, group, kind and every joint's value
        (sorted by joint name), plus a Cartesian pose's tool pose and tip link.

        Not `why` (rewording the reason moves nothing) and not `tolerance` (how close counts is the
        reader's judgement, made against the pose as it is now). WHY a digest: a record measured
        at "ready" carries it, so a reader tells "measured at THIS ready" from "at an older one"
        without comparing numbers."""
        payload: dict = {
            "name": self.name,
            "group": self.group,
            "kind": self.kind,
            # + 0.0 folds -0.0 into 0.0: the same position must give the same digest
            "joints": sorted(
                (n, float(p) + 0.0) for n, p in zip(self.joint_names, self.positions, strict=True)
            ),
        }
        if self.tool_pose is not None:
            payload["tool_pose"] = self.tool_pose.to_dict()
            payload["tip_link"] = self.tip_link
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_LENGTH]


def nearest_solution(
    seed: Sequence[float], solutions: Iterable[Sequence[float]]
) -> np.ndarray | None:
    """The joint-space tie-break between IK solutions: the one whose largest single-joint move from
    `seed` is smallest, then the smallest overall (Euclidean) move, then the IK's own order.
    `None` when there is none.

    WHY the largest single-joint move first: a branch flip (elbow up vs down, a wrist turned half
    round) is one joint swinging far, which a sum over all joints can hide behind small moves."""
    seed = np.asarray(seed, dtype=float)
    best, best_key = None, None
    for solution in solutions:
        candidate = np.asarray(solution, dtype=float)
        if candidate.shape != seed.shape:
            raise ValueError(f"an IK solution has shape {candidate.shape}, the seed {seed.shape}")
        move = np.abs(candidate - seed)
        key = (float(move.max()), float(np.linalg.norm(move)))
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    return best


def check_named_poses(poses: Iterable[NamedPose]) -> tuple[NamedPose, ...]:
    """`poses` as a tuple, refusing anything else and a (group, name) given twice."""
    out = tuple(poses)
    seen: set[tuple[str, str]] = set()
    for pose in out:
        expect_type("a named pose", pose, NamedPose)
        key = (pose.group, pose.name)
        if key in seen:
            raise ValueError(f"named pose {pose.name!r} is given twice for group {pose.group!r}")
        seen.add(key)
    return out


def find_named_pose(poses: Iterable[NamedPose], name: str, group: str | None = None) -> NamedPose:
    """The pose called `name` (in `group`, when two groups share the name): MoveIt's
    ``setNamedTarget`` lookup. An unknown name raises KeyError listing the names there are."""
    poses = check_named_poses(poses)
    found = [p for p in poses if p.name == name and (group is None or p.group == group)]
    if len(found) == 1:
        return found[0]
    if found:
        raise KeyError(f"named pose {name!r} is in groups {[p.group for p in found]}: pass group=")
    have = sorted(f"{p.name} ({p.group})" for p in poses)
    where = f" in group {group!r}" if group else ""
    raise KeyError(f"no named pose {name!r}{where}; there are: {have}")


def _comment(text: str) -> str:
    """`text` on one line, safe inside an XML comment (no "--", no trailing "-")."""
    return " ".join(str(text).replace("--", "- -").split()).rstrip("-")


def named_poses_to_srdf(
    poses: Iterable[NamedPose], robot: str, *, ik: InverseKinematics | None = None
) -> str:
    """The poses as an SRDF (MoveIt's semantic description): one ``<group>`` per group (the joints
    its poses name) and one ``<group_state>`` per pose. Values print exactly (``repr``), so
    `named_poses_from_srdf` reads back the same floats. A Cartesian pose needs `ik` (an SRDF holds
    joint values only). The reason and digest go in a comment above each state, and each joint's
    degrees beside it: SRDF has no field for them."""
    poses = tuple(p.resolved(ik) for p in check_named_poses(poses))
    groups: dict[str, list[str]] = {}
    for pose in poses:
        joints = groups.setdefault(pose.group, [])
        joints.extend(n for n in pose.joint_names if n not in joints)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<!-- generated by robot_msgs.manipulation.named_poses_to_srdf: "
        "one group_state per NamedPose -->",
        f"<robot name={quoteattr(robot)}>",
    ]
    for group, joints in groups.items():
        lines.append(f"  <group name={quoteattr(group)}>")
        lines += [f"    <joint name={quoteattr(joint)}/>" for joint in joints]
        lines.append("  </group>")
    for pose in poses:
        lines.append(f"  <!-- {_comment(pose.name)} ({_comment(pose.group)}), "
                     f"digest {pose.digest()}: {_comment(pose.why)} -->")
        lines.append(f"  <group_state name={quoteattr(pose.name)} group={quoteattr(pose.group)}>")
        for joint, value in zip(pose.joint_names, pose.positions, strict=True):
            lines.append(f'    <joint name={quoteattr(joint)} value="{float(value)!r}"/>'
                         f"  <!-- {math.degrees(value):.4f} deg -->")
        lines.append("  </group_state>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def named_poses_from_srdf(text: str) -> tuple[NamedPose, ...]:
    """Every ``<group_state>`` of an SRDF as a joint-space `NamedPose` (an SRDF has no reason,
    tolerance or tool pose; those live in the description that wrote it)."""
    root = ElementTree.fromstring(text)
    out = []
    for state in root.iter("group_state"):
        joints = [(j.get("name"), j.get("value")) for j in state.iter("joint")]
        try:
            positions = [float(value) for _, value in joints]
        except (TypeError, ValueError):
            raise ValueError(f"group_state {state.get('name')!r}: a joint value is missing "
                             "or is not one number") from None
        out.append(NamedPose(name=state.get("name") or "", group=state.get("group") or "",
                             joint_names=tuple(name or "" for name, _ in joints),
                             positions=positions))
    return check_named_poses(out)


def named_poses_to_json(poses: Iterable[NamedPose], robot: str, *, source: str = "") -> str:
    """The lossless file a description publishes: ``{"schema", "robot", "source", "poses": [{name,
    group, kind, digest, pose}]}``, ``pose`` being the message's own `to_dict`. The digest beside
    each pose lets a reader with only a JSON parser (the rig's monitor) compare a record's digest
    with the description's."""
    doc = {
        "schema": NAMED_POSES_SCHEMA,
        "robot": str(robot),
        "source": str(source),
        "poses": [
            {"name": p.name, "group": p.group, "kind": p.kind, "digest": p.digest(),
             "pose": p.to_dict()}
            for p in check_named_poses(poses)
        ],
    }
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def named_poses_from_json(text: str) -> tuple[NamedPose, ...]:
    """Read what `named_poses_to_json` wrote. Each stated digest must be the pose's own: a file
    edited by hand (or by another generator) is refused; regenerate it from the description."""
    doc = json.loads(text)
    if doc.get("schema") != NAMED_POSES_SCHEMA:
        raise ValueError(f"not a {NAMED_POSES_SCHEMA} file (schema {doc.get('schema')!r})")
    out = []
    for entry in doc.get("poses") or []:
        pose = NamedPose.from_dict(entry["pose"])
        if entry.get("digest") != pose.digest():
            raise ValueError(
                f"named pose {pose.name!r}: the file says digest {entry.get('digest')!r}, the "
                f"pose's own is {pose.digest()!r} - it was edited after it was generated; "
                "regenerate it from the description"
            )
        out.append(pose)
    return check_named_poses(out)
