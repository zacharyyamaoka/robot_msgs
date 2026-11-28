"""ServoCommand — one command to a set of named servos, stamped when it was sent.

The shape is the servo command most smart-servo boards share: a position and velocity target, a
feed-forward effort and per-servo stiffness and damping for the board's own PD loop. That is
moteus's position mode (with kp/kd as scales of the board's gains) and the "MIT mode" of Damiao,
RobStride, CubeMars and Unitree motors. The driver converts SI gains into its board's units.

Two things a trajectory cannot carry ride beside the targets (robot_msgs 0.8, the Joint Command Controller's
contract barrier, Zach 2026-11-27: "You cannot send stops via the JTC"):

- `mode[i]` is what servo `i` is told to do: `position` (the targets above), `stop` (driver off; on moteus it
  also clears a fault and the mode-11 watchdog latch) or `brake` (windings shorted: passive damping). A
  `stop` or `brake` servo's targets are ignored. `mode=None` means every servo is in `position`.
- NaN in `stiffness`, `damping` or `max_effort` means "this servo's board default" (its configured gains or
  cap), per servo. WHY per servo: the arrays are all-or-none, so one joint at zero gains (pure effort) used to
  force every other board's gains into the controller's config, and an uncapped joint went out as +inf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.control.servo_state import ServoState
from robot_msgs.std import Header

_OPTIONAL = ("velocity", "feedforward_effort", "stiffness", "damping", "max_effort")
_NON_NEGATIVE = ("stiffness", "damping", "max_effort")

SERVO_MODES: tuple[str, ...] = ("position", "stop", "brake")
"""What `ServoCommand.mode[i]` may say."""


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoCommand(Msg):
    """Targets for servo `name[i]`, in SI.

    - `position` rad or m; NaN means "no position target" (velocity or effort only, as moteus).
    - `velocity` rad/s or m/s; `feedforward_effort` N·m or N, added to the board's PD output.
    - `stiffness` N·m/rad (or N/m) and `damping` N·m·s/rad (or N·s/m) for the board's PD loop;
      `None` keeps the board's configured gains. `max_effort` caps the board's output. NaN in any of the three
      keeps that one servo's board default.
    - `mode[i]` (`SERVO_MODES`): `position`, `stop` or `brake`; `None` = all `position`.
    - `header.stamp` is when the command was sent (s). `state_stamp` / `state_seq` name the
      `ServoState` it was computed from, so `latency` (sensor-to-command) is measurable from a log
      alone. `None` means open loop: nothing was read.
    """

    name: tuple[str, ...]
    position: Array
    velocity: Array | None = None
    feedforward_effort: Array | None = None
    stiffness: Array | None = None
    damping: Array | None = None
    max_effort: Array | None = None
    state_stamp: float | None = None
    state_seq: int | None = None
    header: Header = field(default_factory=Header)
    mode: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.name)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"ServoCommand.name has duplicate names: {duplicates}")
        set_field(self, "name", names)
        set_field(self, "position", frozen_array(self.position))
        for field_name in _OPTIONAL:
            set_field(self, field_name, optional_array(getattr(self, field_name)))
        if self.state_stamp is not None:
            set_field(self, "state_stamp", float(self.state_stamp))
        if self.state_seq is not None:
            set_field(self, "state_seq", int(self.state_seq))
        for field_name in ("position", *_OPTIONAL):
            values = getattr(self, field_name)
            if values is None:
                continue
            if values.ndim != 1:
                raise ValueError(f"ServoCommand.{field_name} must be 1-D, got shape {values.shape}")
            if len(values) != len(names):
                raise ValueError(
                    f"ServoCommand.{field_name} has {len(values)} values, name has {len(names)}"
                )
        for field_name in _NON_NEGATIVE:
            values = getattr(self, field_name)
            if values is not None and not np.all((values >= 0) | np.isnan(values)):
                raise ValueError(
                    f"ServoCommand.{field_name} must be >= 0 (NaN = board default), got {values}"
                )
        if self.mode is not None:
            modes = tuple(str(m) for m in self.mode)
            if len(modes) != len(names):
                raise ValueError(f"ServoCommand.mode has {len(modes)} values, name has {len(names)}")
            unknown = sorted({m for m in modes if m not in SERVO_MODES})
            if unknown:
                raise ValueError(f"ServoCommand.mode: {unknown} not in {SERVO_MODES}")
            set_field(self, "mode", modes)

    def __len__(self) -> int:
        return len(self.name)

    def mode_of(self, name: str) -> str:
        """Servo `name`'s mode: `position`, `stop` or `brake`."""
        return "position" if self.mode is None else self.mode[self.name.index(name)]

    @property
    def latency(self) -> float | None:
        """Seconds from the state sample to sending this command; `None` when open loop."""
        return None if self.state_stamp is None else self.header.stamp - self.state_stamp

    @classmethod
    def from_state(cls, state: ServoState, *, stamp: float) -> ServoCommand:
        """Hold at the measured position with zero velocity, sent at `stamp`, board gains kept."""
        return cls(
            name=state.name,
            position=state.position,
            velocity=np.zeros(len(state)),
            state_stamp=state.header.stamp,
            state_seq=state.seq,
            header=state.header.replace(stamp=stamp),
        )

    def compare(self, measured: ServoCommand) -> Any:
        """`measured` against this one as the reference → a `robot_msgs.compare.Comparison`.

        Values only (stamps ignored), as one-sample streams; see `robot_msgs.compare.compare`.
        """
        from robot_msgs.compare import compare

        return compare(self, measured)
