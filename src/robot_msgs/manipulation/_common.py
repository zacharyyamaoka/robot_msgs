"""Shared checks for the manipulation messages."""

from __future__ import annotations

import math


def non_negative(
    owner: str, name: str, value: float | None, *, allow_none: bool = True
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{owner}.{name} is required")
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{owner}.{name} must be finite and >= 0, got {value!r}")
    return value


def positive(owner: str, name: str, value: float, *, allow_inf: bool = False) -> float:
    value = float(value)
    if math.isnan(value) or value <= 0.0 or (math.isinf(value) and not allow_inf):
        raise ValueError(f"{owner}.{name} must be positive, got {value!r}")
    return value
