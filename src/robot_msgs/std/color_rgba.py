"""ColorRGBA — a colour with transparency. Mirrors std_msgs/ColorRGBA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_msgs._core import Msg, set_field
from robot_msgs.geometry.vector3 import array_of, finite_float

_CHANNELS = ("r", "g", "b", "a")


@dataclass(frozen=True, eq=False, repr=False)
class ColorRGBA(Msg):
    """Red, green, blue and alpha, each a float in 0..1 (alpha 1 is opaque).

    `from_hex` / `to_hex` read and write ``#rrggbb`` and ``#rrggbbaa`` (8 bits a channel).
    std_msgs/ColorRGBA is float32, so a ROS round trip keeps 8-bit colours exactly but not
    arbitrary float64 values.
    """

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 1.0

    def __post_init__(self) -> None:
        for name in _CHANNELS:
            value = finite_float("ColorRGBA", name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ColorRGBA.{name} must be in 0..1, got {value}")
            set_field(self, name, value)

    def as_array(self) -> np.ndarray:
        """``(4,)`` = r g b a."""
        return np.array([self.r, self.g, self.b, self.a])

    @classmethod
    def from_array(cls, array: Any) -> ColorRGBA:
        r, g, b, a = array_of("ColorRGBA.from_array", array, 4).tolist()
        return cls(r, g, b, a)

    @classmethod
    def from_hex(cls, text: str) -> ColorRGBA:
        """``#rrggbb``, ``#rrggbbaa`` or the CSS shorthand ``#rgb``; the ``#`` is optional."""
        digits = text[1:] if text.startswith("#") else text
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        if len(digits) not in (6, 8) or not all(c in "0123456789abcdefABCDEF" for c in digits):
            raise ValueError(f"ColorRGBA.from_hex needs a hex colour like #rrggbb, got {text!r}")
        channels = [int(digits[i : i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
        return cls(*channels)

    def to_hex(self, alpha: bool | None = None) -> str:
        """``#rrggbb``, plus the alpha pair when `alpha` is true or (by default) when not opaque."""
        values = [round(channel * 255) for channel in (self.r, self.g, self.b, self.a)]
        if alpha is None:
            alpha = self.a != 1.0
        return "#" + "".join(f"{v:02x}" for v in (values if alpha else values[:3]))
