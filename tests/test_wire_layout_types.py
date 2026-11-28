"""Every message field has a wire layout: numbers, strings, arrays, tuples of them, str-keyed mappings, nested messages.

WHY (2026-11-28): `GroundTruth.meta: Mapping[str, Any]` (4137538) passed this suite but made every like_keras attached
core fail to encode `ServoState`, because like_keras's v0 wire layout (its CONTRACT §7.1, `like_keras/codec.py::_node`)
has no layout for `Any`. robot_msgs cannot import like_keras (it sits below it), so this test states the same rule.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping

import numpy as np

import robot_msgs  # noqa: F401  (registers every message)
from robot_msgs._core import _REGISTRY, Msg


def _carried(annotation) -> bool:
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in args:
        rest = [arg for arg in args if arg is not type(None)]
        return len(rest) == 1 and _carried(rest[0])
    origin = typing.get_origin(annotation)
    if annotation in (float, int, bool, str, np.ndarray) or origin is typing.Literal:
        return True
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return _carried(args[0])
    if origin is Mapping and args and args[0] is str:
        return _carried(args[1])
    return isinstance(annotation, type) and issubclass(annotation, Msg)


# like_keras's V0_TYPES (like_keras/src/like_keras/codec.py): the messages that cross its wire today. Keep in step;
# a message added there joins this list. Their nested messages are walked too (ServoState -> GroundTruth).
WIRE_TYPES = (
    "std/Header",
    "std/LogRecord",
    "std/DiagnosticRecord",
    "std/Metrics",
    "sensor/JointState",
    "control/ServoState",
    "control/ServoCommand",
    "trajectory/JointTrajectory",
)


def _nested(annotation):
    if isinstance(annotation, type) and issubclass(annotation, Msg):
        yield annotation
    for arg in typing.get_args(annotation):
        yield from _nested(arg)


def test_every_wire_message_field_has_a_wire_layout():
    bad, seen, todo = [], set(), [_REGISTRY[name] for name in WIRE_TYPES]
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            if not _carried(hints[field.name]):
                bad.append(f"{cls.__name__}.{field.name}: {hints[field.name]}")
            todo.extend(_nested(hints[field.name]))
    assert "GroundTruth" in {cls.__name__ for cls in seen}
    assert not bad, "fields with no wire layout:\n" + "\n".join(bad)
