"""The `plot` entry point: one generic function, one implementation per base class."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

from matplotlib.figure import Figure


@singledispatch
def plot(msg: Any, *, ax: Any = None, **options: Any) -> Figure:
    """Draw any standard message and return the matplotlib figure.

    Implementations are registered on the base classes (``Trajectory``, ``Pose``, ``Image`` …),
    so a user subclass falls back to its parent's plot. Single-panel types draw into ``ax`` when
    one is given; multi-panel types ignore it and make their own figure. Options are per type;
    see the docstring of each ``_plot_*`` in this package.
    """
    raise TypeError(f"robot_msgs.viz has no plot for {type(msg).__name__}")
