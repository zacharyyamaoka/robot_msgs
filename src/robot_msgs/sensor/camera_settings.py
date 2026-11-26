"""CameraSettings — the capture settings a camera ran with: exposure, gain, white balance, gamma.

ROS has no message for this (drivers expose them as parameters); the field set is V4L2 / UVC's
controls and GenICam's SFNC names, the ones every camera driver already maps to. Asked for by the
ArUco camera-calibration track (2026-11-26): its tuner's output, and what every measurement it
scores must carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot_msgs._core import Msg, set_field
from robot_msgs.std import Header


def _optional(name: str, value: object, *, low: float | None = None, positive: bool = False):
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    if (
        not math.isfinite(number)
        or (positive and number <= 0.0)
        or (low is not None and number < low)
    ):
        bound = "> 0" if positive else f">= {low}"
        raise ValueError(f"CameraSettings.{name} must be finite and {bound}, got {number}")
    return number


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CameraSettings(Msg):
    """Settings in effect for the frames stamped from `header.stamp` on, `header.frame_id` the
    camera. Every field is `None` when not set or not known.

    - `exposure_time`: s (SI, like the rest of robot_msgs; 8 ms is ``0.008``). GenICam
      ``ExposureTime`` is µs and V4L2 ``exposure_time_absolute`` 100 µs: convert at the driver.
    - `gain`: dB. `gamma`: the transfer exponent (1.0 linear).
    - `white_balance`: colour temperature, K.
    - `auto_exposure` / `auto_white_balance`: whether the camera's own loop is on (then the
      numeric field is what it last chose, if the driver reports it).
    - `focus`: the driver's focus position (V4L2 ``focus_absolute``), unitless and camera
      specific — record the lens and driver in `header.meta` to compare across cameras.
    - `frame_rate`: Hz.

    Vendor knobs without a field here ride in `header.meta`.
    """

    exposure_time: float | None = None
    gain: float | None = None
    white_balance: float | None = None
    gamma: float | None = None
    auto_exposure: bool | None = None
    auto_white_balance: bool | None = None
    focus: float | None = None
    frame_rate: float | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        set_field(
            self, "exposure_time", _optional("exposure_time", self.exposure_time, positive=True)
        )
        set_field(self, "gain", _optional("gain", self.gain))
        set_field(
            self, "white_balance", _optional("white_balance", self.white_balance, positive=True)
        )
        set_field(self, "gamma", _optional("gamma", self.gamma, positive=True))
        set_field(self, "focus", _optional("focus", self.focus))
        set_field(self, "frame_rate", _optional("frame_rate", self.frame_rate, positive=True))
        for name in ("auto_exposure", "auto_white_balance"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"CameraSettings.{name} must be a bool or None, got {value!r}")

    @property
    def max_exposure_for_frame_rate(self) -> float | None:
        """The longest exposure the frame rate allows (``1 / frame_rate``), s."""
        return None if self.frame_rate is None else 1.0 / self.frame_rate
