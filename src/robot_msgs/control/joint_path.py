"""Joint paths — a joint's name is its slash path in the cell, so a command can be addressed at any level.

``right_arm/shoulder`` is one joint's cell-unique name. An arm's own `JointTable` names its joints bare
(``shoulder``); `JointTable.mounted("right_arm")` prefixes them when the arm goes into a cell, and
`JointTable.combine` joins the mounted arms into the cell's one table. So one arm description serves as left or
right, and only the mount says which.

Every level takes the SAME joint-named `ServoCommand` / `ServoState`. A level is not a new API, only a view
that filters rows by name prefix: `select(command, "right_arm")` is the right arm's part of a cell command,
`select(command, ["right_arm/shoulder", "left_arm/shoulder"])` a set, `select(command, "right_arm/shoulder")`
one. That is tview's all / a set / one (``A>``, ``11,12>``, ``11>``) with names instead of CAN ids.
`strip_prefix` turns a cell-named view into the arm's own bare names and `add_prefix` puts them back, so an
arm-level controller written against ``shoulder`` runs unchanged inside a cell.

WHY (Zach, Daily Note 2026-11-28): "cell.right_arm.shoulder_joint.command() ... address at the different
levels you want" and "commands should be addressed by human readable name". The default he was offered and
did not overrule: names are cell-unique paths, and the cell adds the arm's prefix when the arm mounts.
WHY one message for every level and not a ``command()`` per level: N levels each with their own meaning are N
APIs to keep in sync; a prefix filter over one message is one. Prior art: ros2_control names its interfaces
the same way (``joint1/position``).

Matching respects segments: ``arm`` is under nothing but ``arm`` and ``arm/...``, never ``arm2/x``. The empty
path ``""`` is the root: everything is under it. No live handles, backends or bus access live here; that is a
later layer over these messages.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from typing import Any, TypeVar

import numpy as np

from robot_msgs.control.servo_command import ServoCommand
from robot_msgs.control.servo_state import ServoState

PATH_SEPARATOR = "/"
"""Between the levels of a joint path: ``"right_arm/shoulder"``."""

ROOT = ""
"""The empty path: the whole cell. Every name is under it."""

M = TypeVar("M", ServoCommand, ServoState)


def split_path(path: str) -> tuple[str, ...]:
    """``"right_arm/shoulder"`` -> ``("right_arm", "shoulder")``; ``""`` (the root) -> ``()``.

    Raises on an empty segment (``"a//b"``, ``"/a"``, ``"a/"``): a stray slash would otherwise make two
    spellings of one joint.
    """
    path = str(path)
    if path == ROOT:
        return ()
    parts = tuple(path.split(PATH_SEPARATOR))
    if any(not part for part in parts):
        raise ValueError(f"joint path {path!r} has an empty segment")
    return parts


def join_path(*parts: str) -> str:
    """``join_path("right_arm", "shoulder")`` -> ``"right_arm/shoulder"``.

    Each part may itself be a path (``join_path("cell/right_arm", "shoulder")``); root parts (``""``) drop
    out, so ``join_path("", "shoulder") == "shoulder"``.
    """
    segments: list[str] = []
    for part in parts:
        segments.extend(split_path(part))
    return PATH_SEPARATOR.join(segments)


def is_under(name: str, prefix: str) -> bool:
    """True when joint `name` is `prefix` itself or inside it, segment by segment.

    ``is_under("arm/x", "arm")`` and ``is_under("arm", "arm")`` are True; ``is_under("arm2/x", "arm")`` is
    False. Everything is under the root ``""``.
    """
    prefix_parts = split_path(prefix)
    return split_path(name)[: len(prefix_parts)] == prefix_parts


def strip_path(name: str, prefix: str) -> str:
    """`name` relative to `prefix`: ``strip_path("right_arm/shoulder", "right_arm") == "shoulder"``.

    Raises when `name` is not strictly inside `prefix` (a name equal to the prefix would strip to the root,
    which is not a joint).
    """
    prefix_parts = split_path(prefix)
    parts = split_path(name)
    if parts[: len(prefix_parts)] != prefix_parts or len(parts) == len(prefix_parts):
        raise ValueError(f"joint {name!r} is not inside {prefix!r}")
    return PATH_SEPARATOR.join(parts[len(prefix_parts) :])


def matching(names: Iterable[str], paths: str | Iterable[str]) -> tuple[str, ...]:
    """The names under any of `paths`, in `names`' order.

    `paths` is one path (a level, ``"right_arm"``, or one joint) or several (a set). Every path must match
    at least one name: a path that matches nothing is a typo or a stale name, and silently commanding nobody
    is the failure this refuses.
    """
    wanted = (paths,) if isinstance(paths, str) else tuple(str(p) for p in paths)
    names = tuple(names)
    unmatched = [p for p in wanted if not any(is_under(n, p) for n in names)]
    if unmatched:
        raise KeyError(f"no joint under {unmatched}; joints: {list(names)}")
    return tuple(n for n in names if any(is_under(n, p) for p in wanted))


def select(message: M, paths: str | Iterable[str], *, strip: bool = False) -> M:
    """The rows of `message` whose joint is under any of `paths` — all / a set / one, by name.

    Every column (targets, gains, caps, measurements, `mode`, and a state's joint-named simulator `truth`) is cut
    to the same rows and keeps its values, NaN included (NaN still means "board default"); stamps, `seq` and
    the header pass through. Row order is the message's own. `strip=True` (one path only) also drops that path, giving the level's own names:
    ``select(cell_cmd, "right_arm", strip=True).name == ("shoulder", ...)``.
    """
    _check_type(message)
    names = matching(message.name, paths)
    out = _take(message, names)
    truth = getattr(message, "truth", None)
    if truth is not None:
        # The simulator's truth is joint-named and may cover other joints than the sample: cut it to the same
        # paths, leniently (a level with no truth rows keeps an empty truth, not an error).
        wanted = (paths,) if isinstance(paths, str) else tuple(paths)
        out = out.replace(
            truth=_take(truth, tuple(n for n in truth.name if any(is_under(n, p) for p in wanted)))
        )
    if strip:
        if not isinstance(paths, str):
            raise ValueError("select(strip=True) takes one path, not a set")
        out = strip_prefix(out, paths)
    return out


def strip_prefix(message: M, prefix: str) -> M:
    """`message` with `prefix` removed from every name; every name must be inside it. Values unchanged."""
    _check_type(message)
    return _renamed(message, lambda n: strip_path(n, prefix))


def add_prefix(message: M, prefix: str) -> M:
    """`strip_prefix`'s inverse: ``"shoulder"`` -> ``"right_arm/shoulder"``. Values unchanged."""
    _check_type(message)
    if not split_path(prefix):
        raise ValueError("add_prefix needs a non-empty prefix")
    return _renamed(message, lambda n: join_path(prefix, n))


def _check_type(message: Any) -> None:
    if not isinstance(message, (ServoCommand, ServoState)):
        raise TypeError(
            f"joint paths filter a ServoCommand or a ServoState, got {type(message).__name__}"
        )


def _renamed(message: Any, rename: Any) -> Any:
    out = message.replace(name=tuple(rename(n) for n in message.name))
    truth = getattr(message, "truth", None)
    if truth is not None:
        out = out.replace(truth=truth.replace(name=tuple(rename(n) for n in truth.name)))
    return out


def _take(message: Any, names: tuple[str, ...]) -> Any:
    rows = np.array([message.name.index(n) for n in names], dtype=np.intp)
    # WHY by field type and not a column list: these messages (and GroundTruth) keep every per-servo column as a 1-D array the
    # length of `name` (their __post_init__ enforces it), so a column added later is cut too instead of
    # silently keeping the full length and failing validation.
    changes: dict[str, Any] = {"name": names}
    for f in fields(message):
        value = getattr(message, f.name)
        if isinstance(value, np.ndarray):
            changes[f.name] = value[rows]
    mode = getattr(message, "mode", None)
    if mode is not None:
        changes["mode"] = tuple(mode[i] for i in rows)
    return message.replace(**changes)


__all__ = [
    "PATH_SEPARATOR",
    "ROOT",
    "add_prefix",
    "is_under",
    "join_path",
    "matching",
    "select",
    "split_path",
    "strip_path",
    "strip_prefix",
]
