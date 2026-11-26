"""Find one item by marker id (and dictionary) in a tuple of fiducial messages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def find(items: Sequence[T], marker_id: int, dictionary: str | None, owner: str) -> T:
    """The one item with `marker_id` (and `dictionary`, which may be left out when only one
    dictionary has that id). KeyError when none, ValueError when it is ambiguous."""
    found = [
        item
        for item in items
        if item.id == marker_id and (dictionary is None or item.dictionary == dictionary)
    ]
    if not found:
        where = f"{dictionary} " if dictionary else ""
        raise KeyError(f"{owner} has no marker {where}{marker_id}")
    if len(found) > 1:
        dictionaries = sorted({item.dictionary for item in found})
        hint = "pass dictionary=" if len(dictionaries) > 1 and dictionary is None else "dedupe"
        raise ValueError(
            f"{owner} has {len(found)} entries for marker {marker_id} ({dictionaries}); {hint}"
        )
    return found[0]


def keys_of(items: Sequence[Any]) -> tuple[tuple[str, int], ...]:
    return tuple((item.dictionary, item.id) for item in items)
