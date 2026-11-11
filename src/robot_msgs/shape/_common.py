"""Small helpers the shape types share."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_msgs.geometry import Point, Vector3


def points_of(label: str, value: Any) -> tuple[np.ndarray, bool]:
    """`value` as a float64 ``(N, 3)`` array, and whether it was a single point.

    A `Point`, a `Vector3` or a ``(3,)`` sequence is one point; an ``(N, 3)`` array is N.
    """
    if isinstance(value, (Point, Vector3)):
        return value.as_array()[None, :], True
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        return array[None, :], True
    if array.ndim == 2 and array.shape[1] == 3:
        return array, False
    raise ValueError(f"{label} needs a point or points of shape (N, 3), got shape {array.shape}")


def vector_of(label: str, value: Any) -> np.ndarray:
    """A `Point`, `Vector3` or ``(3,)`` sequence as a float64 ``(3,)`` array."""
    array, single = points_of(label, value)
    if not single:
        raise ValueError(f"{label} needs a single point of shape (3,), got shape {array.shape}")
    return array[0]


def positive_mass(label: str, mass: float) -> float:
    number = float(mass)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} needs a positive finite mass, got {mass}")
    return number
