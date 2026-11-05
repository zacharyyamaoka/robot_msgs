"""Header — when a message was true, and in which frame. Mirrors std_msgs/Header."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg, set_field


@dataclass(frozen=True, eq=False, repr=False)
class Header(Msg):
    """`stamp` is seconds since the Unix epoch (or since sim start); `frame_id` names the frame.

    WHY float seconds, not ROS's (sec, nanosec): it is what every trajectory and controller in
    this codebase already uses. float64 resolves 0.24 µs at today's epoch; `stamp_ns` and
    `from_ns` exist for the lossless edge (bags, MCAP).

    `meta` is context ROS has no field for (an origin, units, where the numbers came from, what
    a cleaner changed) as an immutable ``str -> str`` mapping.
    WHY on the header, not a free-form field on each message: the header is already the one place
    a message says what its numbers are relative to (`frame_id`); meta extends that slot, so every
    stamped message and every trajectory gets it once, and everything that keeps the header
    (slicing, `.at()`, `resample`, serialisation) keeps it too. WHY strings only: the values are
    read by people and other tools; a number would be `str()`-ed silently, so it is refused and
    the caller picks the spelling (``repr(origin)``). Dropped on ROS export: std_msgs/Header has
    only stamp and frame_id (docs/DESIGN.md, "Header.meta").
    """

    stamp: float = 0.0
    frame_id: str = ""
    meta: Mapping[str, str] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not math.isfinite(stamp):
            raise ValueError(f"Header.stamp must be finite, got {stamp}")
        set_field(self, "stamp", stamp)
        set_field(self, "frame_id", str(self.frame_id))
        for key, value in self.meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"Header.meta maps str to str; got {key!r}: {value!r} "
                    "(spell numbers yourself, e.g. repr(x), so nothing is coerced silently)"
                )
        set_field(self, "meta", FrozenDict(self.meta))

    @classmethod
    def now(cls, frame_id: str = "") -> Header:
        return cls(stamp=time.time(), frame_id=frame_id)

    @classmethod
    def from_ns(cls, stamp_ns: int, frame_id: str = "") -> Header:
        # int / int is correctly rounded; `stamp_ns / 1e9` would round twice.
        return cls(stamp=int(stamp_ns) / 1_000_000_000, frame_id=frame_id)

    @property
    def stamp_ns(self) -> int:
        sec, nanosec = self.to_sec_nanosec()
        return sec * 1_000_000_000 + nanosec

    def to_sec_nanosec(self) -> tuple[int, int]:
        """ROS builtin_interfaces/Time split.

        WHY split before scaling: ``stamp * 1e9`` near today's epoch is ~1.76e18, where float64
        steps by 256 ns. ``stamp - floor(stamp)`` is exact, so only the fraction is rounded.
        """
        sec = math.floor(self.stamp)
        nanosec = round((self.stamp - sec) * 1e9)
        if nanosec == 1_000_000_000:
            sec, nanosec = sec + 1, 0
        return sec, nanosec

    @classmethod
    def from_sec_nanosec(cls, sec: int, nanosec: int, frame_id: str = "") -> Header:
        return cls.from_ns(sec * 1_000_000_000 + nanosec, frame_id)

    def time_shifted(self, seconds: float) -> Header:
        return self.replace(stamp=self.stamp + seconds)
