"""Transports — which device each bus of one robot copy is reached through: ``"fdcanusb:272813EF"`` -> its by-id tty.

A `JointTable` names each servo's bus by a key; this maps the key to the concrete device (a tty, ``can0``, a pi3hat
channel). It is a `RobotConfig` component (``components={"joints": table, "transports": Transports(...)}``), so a real
backend opens exactly what the config names and never guesses a port (moteus autodetect can grab the wrong adapter).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg, set_field


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Transports(Msg):
    """Bus key -> device. Keys and devices are non-empty strings; `device(bus)` raises naming what is configured."""

    buses: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        buses = FrozenDict((str(bus), str(device)) for bus, device in dict(self.buses).items())
        empty = sorted(bus for bus, device in buses.items() if not bus or not device)
        if empty:
            raise ValueError(f"Transports: bus keys and devices are non-empty, got {empty}")
        set_field(self, "buses", buses)

    def __len__(self) -> int:
        return len(self.buses)

    def device(self, bus: str) -> str:
        try:
            return self.buses[bus]
        except KeyError:
            raise KeyError(f"no transport for bus {bus!r}; configured: {dict(self.buses)}") from None


__all__ = ["Transports"]
