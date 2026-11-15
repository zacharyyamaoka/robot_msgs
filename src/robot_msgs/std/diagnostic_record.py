"""DiagnosticRecord — how one component is doing now: a level, a reason and its lifecycle mode.

A flat subset of io-diagnostics' DiagnosticStatus v0 (/home/bam/io-diagnostics, contract/record.py;
spec docs/diagnostics-record-v0.md), same field names and meanings, so a RobotDx surface reads it
without a mapping table.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.vector3 import expect_type
from robot_msgs.std.header import Header

LEVELS = ("unknown", "ok", "warn", "error")
"""What a producer may report. STALE is not here: an observer derives it (`stale_after_s`)."""

MODES = ("initialization", "operational", "maintenance", "software_update", "offline")
"""Readiness, not severity: io-diagnostics' modes, DroneCAN's names."""

# WHY these two numbers (io-diagnostics STALE_FACTOR / STALE_FLOOR_S): 3x is DroneCAN's 1 s period
# against its 3 s offline timeout; 5 s is the floor rqt, Lichtblick and the ROS aggregator use.
STALE_FACTOR = 3.0
STALE_FLOOR_S = 5.0

# PascalCase, optionally dotted ("MoteusFault.33"), at most 64 characters: io-diagnostics' `_CODE`.
_CODE = re.compile(r"^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)*$")


def _seconds(label: str, value: Any, *, zero_allowed: bool) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero_allowed):
        bound = ">= 0" if zero_allowed else "> 0"
        raise ValueError(f"{label} must be a finite number of seconds {bound}, got {number}")
    return number


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class DiagnosticRecord(Msg):
    """One evaluation of one component's health, sent periodically.

    - `header.stamp`: source time of the evaluation, seconds since the Unix epoch.
    - `path`: the component, as a dotted path (``"pick_cell.arm.jtc"``). Machines key on it.
    - `level`: ``"unknown"`` (the zero value: a record nobody evaluated is never green), ``"ok"``,
      ``"warn"`` or ``"error"``. ``"stale"`` is refused: an observer derives it when no record
      arrived for `stale_after_s`, so a producer that went silent cannot claim otherwise.
    - `code`: the stable machine reason, PascalCase, optionally dotted (``"InputStale"``,
      ``"MoteusFault.33"``), at most 64 characters. Required for ``"warn"`` and ``"error"``.
    - `message`: one sentence for a person, a rendering of the code.
    - `mode`: the lifecycle, not the severity: ``"initialization"``, ``"operational"``,
      ``"maintenance"``, ``"software_update"`` or ``"offline"``.
    - `expected_period_s`: the producer's promise of a fresh record at least this often.
    - `uptime_s`: seconds since the reporter started; going backwards means it restarted.
    - `stale_inputs`: the input ports whose data went stale, by name.

    WHY flat (no `values`, `measurements` or `caused_by`): every field is a scalar or a list of
    strings, so the record crosses a process as one fixed struct; numbers travel as `Metrics`.
    WHY `code` may be empty for ``"unknown"``: the zero value has to construct. io-diagnostics'
    `validate()` also asks an unknown record for a code (``"PortBusy"``: not evaluated, and why);
    give one whenever you know it.
    The field order is the cross-process layout (like_keras's codec reads it), so it never changes.
    """

    header: Header
    path: str
    level: str = "unknown"
    code: str = ""
    message: str = ""
    mode: str = "initialization"
    expected_period_s: float = 1.0
    uptime_s: float = 0.0
    stale_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expect_type("DiagnosticRecord.header", self.header, Header)
        set_field(self, "path", str(self.path))
        level = str(self.level)
        if level == "stale":
            raise ValueError(
                "DiagnosticRecord.level 'stale' is never sent: an observer derives it when no "
                "record arrived for stale_after_s"
            )
        if level not in LEVELS:
            raise ValueError(f"DiagnosticRecord.level must be one of {LEVELS}, got {level!r}")
        set_field(self, "level", level)
        code = str(self.code)
        if code and (len(code) > 64 or not _CODE.match(code)):
            raise ValueError(
                f"DiagnosticRecord.code {code!r} must be PascalCase, optionally dotted "
                "('MoteusFault.33'), at most 64 characters"
            )
        if not code and level in ("warn", "error"):
            raise ValueError(f"DiagnosticRecord.code is required when level is {level!r}")
        set_field(self, "code", code)
        set_field(self, "message", str(self.message))
        mode = str(self.mode)
        if mode not in MODES:
            raise ValueError(f"DiagnosticRecord.mode must be one of {MODES}, got {mode!r}")
        set_field(self, "mode", mode)
        set_field(
            self,
            "expected_period_s",
            _seconds(
                "DiagnosticRecord.expected_period_s", self.expected_period_s, zero_allowed=False
            ),
        )
        set_field(
            self,
            "uptime_s",
            _seconds("DiagnosticRecord.uptime_s", self.uptime_s, zero_allowed=True),
        )
        if isinstance(self.stale_inputs, str):
            raise TypeError(
                f"DiagnosticRecord.stale_inputs is a tuple of port names; got the string "
                f"{self.stale_inputs!r} (write ({self.stale_inputs!r},))"
            )
        set_field(self, "stale_inputs", tuple(str(name) for name in self.stale_inputs))

    @property
    def stale_after_s(self) -> float:
        """Seconds of silence after which an observer shows this component as stale:
        ``max(3 × expected_period_s, 5 s)``, io-diagnostics' rule."""
        return max(STALE_FACTOR * self.expected_period_s, STALE_FLOOR_S)
