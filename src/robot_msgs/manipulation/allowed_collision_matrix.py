"""AllowedCollisionMatrix — which body pairs a collision checker may skip, and why.

Mirrors ``moveit_msgs/AllowedCollisionMatrix`` (``entry_names``, ``entry_values``,
``default_entry_names``, ``default_entry_values``) and carries the reason MoveIt's setup assistant
writes into the SRDF (``<disable_collisions link1=… link2=… reason="Never"/>``) beside each
allowed pair.

Zach, 2026-11-23: *"I will have a robot description I want to just like pass it into this, and get a
good ACM out. ACM should be a robot_msg... where things that are never in collision of course
don't we don't check."* The generator lives in robot_collision; this is the value it hands every
planner, checker and viewer.

Meaning of one cell, as in MoveIt: ``entry_values[i, j] == True`` means contact between
``entry_names[i]`` and ``entry_names[j]`` is ALLOWED, so the pair is never checked. A pair that must
be checked is ``False``. The reasons (`REASONS`) are MoveIt's own strings:

- ``Adjacent`` — joined by a joint (through links without geometry), touching by construction;
- ``Default`` — already touching at the robot's default configuration;
- ``Always`` — touching in (nearly) every sampled configuration: a modelling artefact;
- ``Never`` — touching in no sampled configuration: checking it is wasted work;
- ``User`` — allowed by hand.

The functions here are the stock file it lowers to (`allowed_collision_matrix_to_srdf`,
`allowed_collision_matrix_from_srdf`); JSON and npz come from `Msg` like every other type.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, set_field

REASONS = ("Adjacent", "Default", "Always", "Never", "User")
"""MoveIt setup assistant's ``DisabledReason`` strings, in the order it decides them (User last)."""

DIGEST_LENGTH = 16  # hex characters of sha256, as NamedPose.digest

PairLike = Sequence[str]  # (a, b) or (a, b, reason)


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class AllowedCollisionMatrix(Msg):
    """Pairwise allowed-contact matrix over named bodies (robot links and scene objects).

    - `entry_names`: every body the matrix speaks for, each once.
    - `entry_values` ``(n, n)`` bool, symmetric: ``True`` = contact allowed, the pair is skipped.
      The diagonal is ``True`` (a body never collides with itself).
    - `reasons`: ``(a, b, reason)`` for allowed pairs, ``a`` before ``b`` in `entry_names` order;
      a reason must be one of `REASONS`. An allowed pair without a reason is fine (``User`` is
      assumed by `reason`).
    - `default_entry_names` / `default_entry_values`: MoveIt's fallback for a body that is not an
      entry — contact with it is allowed when its default value is ``True``.
    - `why`: how this matrix was made (generator, sample count, seed), kept beside it.

    `digest()` names WHICH pairs are checked (names and cells, not reasons or `why`): a planner
    config built against this matrix carries the digest, and a different digest means it changed.
    """

    entry_names: tuple[str, ...]
    entry_values: Array
    reasons: tuple[tuple[str, str, str], ...] = ()
    default_entry_names: tuple[str, ...] = ()
    default_entry_values: Array | None = None
    why: str = ""

    def __post_init__(self) -> None:
        names = _names("entry_names", self.entry_names)
        set_field(self, "entry_names", names)
        values = frozen_array(self.entry_values, bool)
        n = len(names)
        if values.shape != (n, n):
            raise ValueError(f"AllowedCollisionMatrix.entry_values has shape {values.shape}, "
                             f"entry_names has {n}")
        if not np.array_equal(values, values.T):
            i, j = map(int, np.argwhere(values != values.T)[0])
            raise ValueError(f"AllowedCollisionMatrix.entry_values is not symmetric at "
                             f"({names[i]!r}, {names[j]!r})")
        if n and not bool(np.all(np.diag(values))):
            raise ValueError("AllowedCollisionMatrix.entry_values must be True on the diagonal")
        set_field(self, "entry_values", values)

        index = {name: k for k, name in enumerate(names)}
        reasons: dict[tuple[int, int], str] = {}
        for item in self.reasons:
            a, b, reason = (str(v) for v in item)
            if reason not in REASONS:
                raise ValueError(f"AllowedCollisionMatrix reason {reason!r} for ({a!r}, {b!r}) "
                                 f"is not one of {REASONS}")
            if a not in index or b not in index or a == b:
                raise ValueError(f"AllowedCollisionMatrix reason names an unknown pair ({a!r}, {b!r})")
            i, j = sorted((index[a], index[b]))
            if not values[i, j]:
                raise ValueError(f"AllowedCollisionMatrix gives reason {reason!r} to ({a!r}, {b!r}), "
                                 "which is checked, not allowed")
            if reasons.setdefault((i, j), reason) != reason:
                raise ValueError(f"AllowedCollisionMatrix gives ({a!r}, {b!r}) two reasons")
        set_field(self, "reasons",
                  tuple((names[i], names[j], reasons[i, j]) for i, j in sorted(reasons)))

        defaults = _names("default_entry_names", self.default_entry_names, allow_empty=True)
        clash = sorted(set(defaults) & set(names))
        if clash:
            raise ValueError(f"AllowedCollisionMatrix names {clash} both as entries and defaults")
        set_field(self, "default_entry_names", defaults)
        default_values = (np.zeros(len(defaults), bool) if self.default_entry_values is None
                          else self.default_entry_values)
        default_values = frozen_array(default_values, bool)
        if default_values.shape != (len(defaults),):
            raise ValueError(f"AllowedCollisionMatrix.default_entry_values has shape "
                             f"{default_values.shape}, default_entry_names has {len(defaults)}")
        set_field(self, "default_entry_values", default_values)
        set_field(self, "why", str(self.why))

    # ── construction ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_pairs(cls, entry_names: Iterable[str], allowed: Iterable[PairLike] = (), *,
                   why: str = "") -> AllowedCollisionMatrix:
        """A matrix over `entry_names` where only the `allowed` pairs are skipped.
        Each pair is ``(a, b)`` or ``(a, b, reason)``; the first reason given for a pair wins."""
        names = tuple(entry_names)
        index = {name: k for k, name in enumerate(names)}
        values = np.eye(len(names), dtype=bool)
        reasons: dict[tuple[int, int], tuple[str, str, str]] = {}
        for pair in allowed:
            a, b = str(pair[0]), str(pair[1])
            if a not in index or b not in index:
                raise ValueError(f"from_pairs: ({a!r}, {b!r}) names a body not in entry_names")
            i, j = sorted((index[a], index[b]))
            values[i, j] = values[j, i] = True
            if len(pair) > 2 and pair[2] and i != j:
                reasons.setdefault((i, j), (names[i], names[j], str(pair[2])))
        return cls(entry_names=names, entry_values=values, reasons=tuple(reasons.values()), why=why)

    # ── reading ───────────────────────────────────────────────────────────────────────────────

    def is_allowed(self, a: str, b: str) -> bool:
        """True when contact between `a` and `b` may be ignored (MoveIt ``getAllowedCollision``,
        falling back to the default entries). A body the matrix does not know is checked."""
        index = self._index
        if a in index and b in index:
            return bool(self.entry_values[index[a], index[b]])
        return self._default(a) or self._default(b)

    def reason(self, a: str, b: str) -> str | None:
        """Why the pair is allowed (``"User"`` when no reason was recorded), None when checked."""
        if not self.is_allowed(a, b):
            return None
        index = self._index
        if a in index and b in index:
            key = tuple(sorted((index[a], index[b]), key=int))
            for x, y, why in self.reasons:
                if (index[x], index[y]) == key:
                    return why
        return "User"

    def checked_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every entry pair a checker must test, ``a`` before ``b`` in entry order."""
        i, j = np.nonzero(np.triu(~self.entry_values, 1))
        return tuple((self.entry_names[a], self.entry_names[b]) for a, b in zip(i, j, strict=True))

    def allowed_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every entry pair a checker skips, ``a`` before ``b`` in entry order."""
        i, j = np.nonzero(np.triu(self.entry_values, 1))
        return tuple((self.entry_names[a], self.entry_names[b]) for a, b in zip(i, j, strict=True))

    def counts(self) -> dict[str, int]:
        """How many entry pairs each reason covers, plus ``"Checked"`` and ``"User"`` (allowed with
        no recorded reason folded in): the five MoveIt buckets and the pairs left to check."""
        out = dict.fromkeys((*REASONS, "Checked"), 0)
        for _a, _b, why in self.reasons:
            out[why] += 1
        allowed = len(self.allowed_pairs())
        out["User"] += allowed - len(self.reasons)
        out["Checked"] = len(self.checked_pairs())
        return out

    def digest(self) -> str:
        """Short hash of the names and cells (what a checker does), not of reasons or `why`."""
        h = hashlib.sha256()
        h.update("\x1f".join(self.entry_names).encode())
        h.update(np.packbits(self.entry_values).tobytes())
        h.update("\x1f".join(self.default_entry_names).encode())
        h.update(np.packbits(self.default_entry_values).tobytes())
        return h.hexdigest()[:DIGEST_LENGTH]

    # ── modified copies ───────────────────────────────────────────────────────────────────────

    def with_allowed(self, a: str, b: str, reason: str = "User") -> AllowedCollisionMatrix:
        """A copy where the pair is skipped, with `reason` (bodies new to the matrix are added)."""
        acm = self.with_entries((a, b))
        return acm._set(a, b, True, reason)

    def with_checked(self, a: str, b: str) -> AllowedCollisionMatrix:
        """A copy where the pair must be checked (bodies new to the matrix are added)."""
        acm = self.with_entries((a, b))
        return acm._set(a, b, False, None)

    def with_entries(self, names: Iterable[str]) -> AllowedCollisionMatrix:
        """A copy that also speaks for `names`; a new body is checked against everything."""
        new = tuple(n for n in dict.fromkeys(str(n) for n in names) if n not in self._index)
        if not new:
            return self
        n, k = len(self.entry_names), len(new)
        values = np.eye(n + k, dtype=bool)
        values[:n, :n] = self.entry_values
        keep = [i for i, name in enumerate(self.default_entry_names) if name not in new]
        return self.replace(
            entry_names=self.entry_names + new, entry_values=values,
            default_entry_names=tuple(self.default_entry_names[i] for i in keep),
            default_entry_values=self.default_entry_values[keep])

    # ── internals ─────────────────────────────────────────────────────────────────────────────

    @property
    def _index(self) -> dict[str, int]:
        return {name: k for k, name in enumerate(self.entry_names)}

    def _default(self, name: str) -> bool:
        if name in self.default_entry_names:
            return bool(self.default_entry_values[self.default_entry_names.index(name)])
        return False

    def _set(self, a: str, b: str, allowed: bool, reason: str | None) -> AllowedCollisionMatrix:
        index = self._index
        i, j = index[a], index[b]
        if i == j:
            raise ValueError(f"a body is always allowed to touch itself ({a!r})")
        values = self.entry_values.copy()
        values[i, j] = values[j, i] = allowed
        pair = {a, b}
        reasons = tuple(r for r in self.reasons if {r[0], r[1]} != pair)
        if allowed and reason:
            reasons += ((a, b, reason),)
        return self.replace(entry_values=values, reasons=reasons)


def _names(label: str, value: Iterable[str], allow_empty: bool = False) -> tuple[str, ...]:
    names = tuple(str(n) for n in value)
    for name in names:
        # WHY refuse instead of strip: names are looked up byte for byte; never edit one silently
        if not name or name != name.strip():
            raise ValueError(f"AllowedCollisionMatrix.{label} has a blank or padded name {name!r}")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"AllowedCollisionMatrix.{label} names {duplicates} twice")
    if not names and not allow_empty:
        return ()
    return names


def allowed_collision_matrix_to_srdf(acm: AllowedCollisionMatrix, robot_name: str = "robot") -> str:
    """The SRDF MoveIt reads: one ``<disable_collisions>`` per allowed pair, in entry order, with its
    reason. Default entries have no SRDF form and are left out."""
    lines = [f"<robot name={quoteattr(robot_name)}>"]
    for a, b in acm.allowed_pairs():
        lines.append(f"  <disable_collisions link1={quoteattr(a)} link2={quoteattr(b)} "
                     f"reason={quoteattr(acm.reason(a, b) or 'User')}/>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def allowed_collision_matrix_from_srdf(srdf: str, entry_names: Iterable[str] | None = None, *,
                                       why: str = "") -> AllowedCollisionMatrix:
    """Read every ``<disable_collisions>`` of an SRDF. `entry_names` sets the bodies (and their order);
    without it they are the links the SRDF mentions, in first-seen order. A reason outside `REASONS`
    (an older SRDF's ``"Not Disabled"`` aside) is read as ``User``, as MoveIt does."""
    root = ElementTree.fromstring(srdf)
    pairs = []
    for element in root.iter("disable_collisions"):
        a, b = element.get("link1", ""), element.get("link2", "")
        reason = element.get("reason", "User")
        pairs.append((a, b, reason if reason in REASONS else "User"))
    names = (tuple(entry_names) if entry_names is not None
             else tuple(dict.fromkeys(n for a, b, _ in pairs for n in (a, b))))
    known = set(names)
    pairs = [p for p in pairs if p[0] in known and p[1] in known]
    return AllowedCollisionMatrix.from_pairs(names, pairs, why=why)
