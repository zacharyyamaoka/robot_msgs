"""LogRecord — one line a component logged: who, at which step, how loud, and what it said."""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Any

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std.header import Header


def _whole_number(label: str, value: Any) -> int:
    """`value` as a Python int >= 0; a bool, a float or a negative number is refused."""
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a whole number, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"{label} must be a whole number, got {value!r}") from None
    if number < 0:
        raise ValueError(f"{label} must be >= 0, got {number}")
    return number


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class LogRecord(Msg):
    """One log line from a running component, as it crosses a process or lands in a recording.

    - `header.stamp`: wall time of the record, seconds since the Unix epoch.
    - `path`: the component that logged, as a dotted path (``"pick_cell.arm.jtc"``).
    - `tick`: that component's step count when it logged, so a line sits next to the step it
      describes even when the wall clock and the step clock disagree (a lockstep replay).
    - `level`: the stdlib `logging` number: 10 debug, 20 info, 30 warning, 40 error, 50 critical.
      ROS's rcl_interfaces/Log uses the same numbers.
    - `message`: the text, already formatted.

    WHY the stdlib numbers, not names: a Python author writes ``self.log.info(...)`` and the record
    keeps exactly what `logging` used, custom levels included; `level_name` spells it back.
    The field order is the cross-process layout (like_keras's codec reads it), so it never changes.
    """

    header: Header
    path: str
    tick: int
    level: int
    message: str

    def __post_init__(self) -> None:
        expect_type("LogRecord.header", self.header, Header)
        set_field(self, "path", str(self.path))
        set_field(self, "tick", _whole_number("LogRecord.tick", self.tick))
        set_field(self, "level", _whole_number("LogRecord.level", self.level))
        set_field(self, "message", str(self.message))

    @property
    def level_name(self) -> str:
        """The stdlib's name for `level`: ``"INFO"``, ``"WARNING"``, ``"Level 25"`` if unnamed."""
        return logging.getLevelName(self.level)
