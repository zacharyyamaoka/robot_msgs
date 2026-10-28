"""Header — when a message was true, and in which frame. Mirrors std_msgs/Header."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from robot_msgs._core import Msg, set_field


@dataclass(frozen=True, eq=False, repr=False)
class Header(Msg):
    """`stamp` is seconds since the Unix epoch (or since sim start); `frame_id` names the frame.

    WHY float seconds, not ROS's (sec, nanosec): it is what every trajectory and controller in
    this codebase already uses. float64 resolves 0.24 µs at today's epoch; `stamp_ns` and
    `from_ns` exist for the lossless edge (bags, MCAP).
    """

    stamp: float = 0.0
    frame_id: str = ""

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not math.isfinite(stamp):
            raise ValueError(f"Header.stamp must be finite, got {stamp}")
        set_field(self, "stamp", stamp)
        set_field(self, "frame_id", str(self.frame_id))

    @classmethod
    def now(cls, frame_id: str = "") -> Header:
        return cls(stamp=time.time(), frame_id=frame_id)

    @classmethod
    def from_ns(cls, stamp_ns: int, frame_id: str = "") -> Header:
        return cls(stamp=stamp_ns / 1e9, frame_id=frame_id)

    @property
    def stamp_ns(self) -> int:
        return round(self.stamp * 1e9)

    def to_sec_nanosec(self) -> tuple[int, int]:
        """ROS builtin_interfaces/Time split."""
        sec, nanosec = divmod(self.stamp_ns, 1_000_000_000)
        return sec, nanosec

    @classmethod
    def from_sec_nanosec(cls, sec: int, nanosec: int, frame_id: str = "") -> Header:
        return cls.from_ns(sec * 1_000_000_000 + nanosec, frame_id)

    def shifted(self, seconds: float) -> Header:
        return self.replace(stamp=self.stamp + seconds)
