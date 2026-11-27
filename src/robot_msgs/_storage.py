"""Files: `msg.save(path)` / `Type.load(path)` for one message, `robot_msgs.save` / `load` for many.

One storage format, two encodings, picked by the file's suffix:

- ``.json`` — human-readable (M1): the `to_dict` tree, arrays inline as lists, indented.
- ``.npz``  — compact (M2): the same tree as JSON in ``__msg__``, arrays as raw npz entries, no
  pickle. It is `save_npz`'s format, so `load_npz` reads a single message saved this way.
- anything else — ``.npz`` is appended, exactly as `numpy.savez` does: ``state.save("arm")``
  writes ``arm.npz`` and ``RobotState.load("arm")`` reads it.

WHY append ``.npz`` rather than a folder (HF ``save_pretrained``) or a by-content pick (Zach,
2026-11-26, "state.save("arm")"): one file per name, renamed atomically, and the rule is numpy's
own, so nobody has to remember a new one. Lossless for arrays of any size; ask for ``.json`` when
you want to read the file.

A bundle (a list, tuple or dict of messages, nested as deep as you like) is the same tree with
its container recorded, ``{"_bundle": "dict", "items": {...}}``, so a container round-trips as the
container it was. WHY no RobotMemory type (Zach, 2026-11-26: "save([state, action, etc...])"):
any message saves itself; grouping is the caller's list or dict, not a new type to keep in step.
A history is a list of messages; for long ones the trajectory types (`ServoStateTrajectory`,
`JointTrajectory`) store the same samples column-wise.

Writes are atomic: a temporary file beside the target, flushed and fsynced, then `os.replace`.
A crash or power cut mid-save leaves the previous file whole.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_msgs._core import _REGISTRY, Msg, from_dict, to_dict

SUFFIXES = (".json", ".npz")
_MSG_KEY = "__msg__"  # WHY this name: save_npz already writes it, so its files load here too
_BUNDLE = "_bundle"


def resolve(path: str | os.PathLike[str]) -> Path:
    """The file a save or load of `path` uses: `path` if it ends in ``.json`` or ``.npz``, else
    `path` + ``.npz`` (numpy.savez's rule)."""
    path = Path(os.fspath(path)).expanduser()
    return path if path.suffix in SUFFIXES else path.with_name(path.name + ".npz")


# ── trees ────────────────────────────────────────────────────────────────────


def _encode(objs: Any, arrays: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if isinstance(objs, Msg):
        return to_dict(objs, arrays=arrays)
    if isinstance(objs, Mapping):
        for key in objs:
            if not isinstance(key, str):
                raise TypeError(f"a bundle's keys are strings; got {key!r}")
        items: Any = {key: _encode(value, arrays) for key, value in objs.items()}
        return {_BUNDLE: "dict", "items": items}
    if isinstance(objs, (list, tuple)):
        kind = "list" if isinstance(objs, list) else "tuple"
        return {_BUNDLE: kind, "items": [_encode(value, arrays) for value in objs]}
    raise TypeError(
        f"save takes a message, or a list, tuple or dict of messages; got {type(objs).__name__}"
    )


def _decode(tree: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None) -> Any:
    kind = tree.get(_BUNDLE)
    if kind is None:
        return from_dict(tree, arrays=arrays)
    items = tree["items"]
    if kind == "dict":
        return {key: _decode(value, arrays) for key, value in items.items()}
    decoded = [_decode(value, arrays) for value in items]
    return decoded if kind == "list" else tuple(decoded)


# ── bytes on disk ────────────────────────────────────────────────────────────


def _payload(objs: Any, target: Path) -> bytes:
    if target.suffix == ".json":
        return (json.dumps(_encode(objs, None), indent=2) + "\n").encode()
    arrays: dict[str, np.ndarray] = {}
    tree = _encode(objs, arrays)
    buffer = io.BytesIO()
    np.savez(buffer, **{_MSG_KEY: np.array(json.dumps(tree))}, **arrays)
    return buffer.getvalue()


def _write_atomically(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    # WHY fsync the folder too: the rename itself must survive a power cut (best effort; some
    # filesystems refuse a directory fsync).
    try:
        directory = os.open(target.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def _read(target: Path) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if not target.exists():
        raise FileNotFoundError(f"no saved message at {target}")
    if target.suffix == ".json":
        return json.loads(target.read_text()), None
    with np.load(target, allow_pickle=False) as archive:
        tree = json.loads(str(archive[_MSG_KEY]))
        arrays = {key: archive[key] for key in archive.files if key != _MSG_KEY}
    return tree, arrays


# ── the API ──────────────────────────────────────────────────────────────────


def save(objs: Any, path: str | os.PathLike[str]) -> Path:
    """Save a message, or a list / tuple / dict of messages, to `path`; returns the file written.

    ``save({"state": state, "action": action}, "step.npz")`` · ``save(states, "history.json")``.
    The encoding follows the suffix (module docstring); the write is atomic.
    """
    target = resolve(path)
    payload = _payload(objs, target)  # WHY before touching the disk: a bad input writes nothing
    _write_atomically(target, payload)
    return target


def load(path: str | os.PathLike[str]) -> Any:
    """What `save` wrote: the message, or the list / tuple / dict of messages, as it was."""
    tree, arrays = _read(resolve(path))
    return _decode(tree, arrays)


def load_message(cls: type[Msg], path: str | os.PathLike[str]) -> Msg:
    """`Type.load(path)`: one message, checked to be a `cls` before it is decoded."""
    target = resolve(path)
    tree, arrays = _read(target)
    if _BUNDLE in tree:
        raise TypeError(
            f"{target} holds a bundle ({tree[_BUNDLE]} of messages), not one message; "
            "read it with robot_msgs.load"
        )
    found = _REGISTRY.get(tree.get("_type", ""))
    if found is None:
        raise TypeError(f"{target} names no known message type ({tree.get('_type')!r})")
    if not issubclass(found, cls):
        raise TypeError(f"{target} holds a {found.type_name}, not a {cls.type_name}")
    return from_dict(tree, arrays=arrays)
