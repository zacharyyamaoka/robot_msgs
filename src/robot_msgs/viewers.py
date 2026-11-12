"""viewers — the visualisers robot_msgs knows, and how to add one.

Each `Viewer` names one tool (matplotlib, Rerun, Viser ...), what it draws (2-D, 3-D or both),
the extra that installs it, and three functions that import it lazily:

- `supports(type)`: whether the tool has a view for that message type;
- `show(msg, name)`: open the view now (a window, a viewer, a server);
- `save(msg, path)`: write the view to a file (a PNG, an .rrd, a .viser recording).

`msg.show(viewer="viser")` and robotmsgs.com both go through this registry, so a viewer added
with `register()` shows up on every type page that it supports, with no site edit.
Importing this module imports none of the tools.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Viewer:
    """One visualiser. `module` is imported only when one of the functions is called."""

    name: str
    title: str
    dims: str
    extra: str
    suffix: str
    module: str
    show_code: str
    url: str
    _supports: Callable[[Any, type], bool]
    _show: Callable[[Any, Any, str], Any]
    _save: Callable[[Any, Any, str], str]

    def load(self) -> Any:
        try:
            return importlib.import_module(self.module)
        except ImportError as error:
            raise ImportError(
                f"the {self.title} viewer needs: pip install 'robot-msgs[{self.extra}]'"
            ) from error

    def available(self) -> bool:
        try:
            self.load()
        except ImportError:
            return False
        return True

    def supports(self, message_type: type) -> bool:
        return self._supports(self.load(), message_type)

    def show(self, msg: Any, name: str | None = None) -> Any:
        return self._show(self.load(), msg, name or type(msg).__name__)

    def save(self, msg: Any, path: str) -> str:
        return self._save(self.load(), msg, path)


VIEWERS: dict[str, Viewer] = {}


def register(viewer: Viewer) -> Viewer:
    """Add (or replace) a viewer; returns it."""
    VIEWERS[viewer.name] = viewer
    return viewer


def get(name: str) -> Viewer:
    try:
        return VIEWERS[name]
    except KeyError:
        raise KeyError(f"no viewer {name!r}; viewers: {sorted(VIEWERS)}") from None


def _plot_supports(module: Any, message_type: type) -> bool:
    from robot_msgs.viz._dispatch import plot

    return plot.dispatch(message_type) is not plot.dispatch(object)


def _plot_show(module: Any, msg: Any, name: str) -> Any:
    import matplotlib.pyplot as plt

    figure = module.plot(msg)
    plt.show()
    return figure


def _plot_save(module: Any, msg: Any, path: str) -> str:
    import matplotlib.pyplot as plt

    figure = module.plot(msg)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


register(
    Viewer(
        name="matplotlib",
        title="Matplotlib",
        dims="2D",
        extra="viz",
        suffix=".png",
        module="robot_msgs.viz",
        show_code="msg.plot()",
        url="https://matplotlib.org",
        _supports=_plot_supports,
        _show=_plot_show,
        _save=_plot_save,
    )
)
register(
    Viewer(
        name="rerun",
        title="Rerun",
        dims="2D + 3D",
        extra="rerun",
        suffix=".rrd",
        module="robot_msgs.viz.rerun",
        show_code='msg.show(viewer="rerun")',
        url="https://rerun.io",
        _supports=lambda module, message_type: True,
        _show=lambda module, msg, name: module.show(**{name: msg}),
        _save=lambda module, msg, path: module.save(path, **{type(msg).__name__: msg}),
    )
)
register(
    Viewer(
        name="viser",
        title="Viser",
        dims="3D",
        extra="viser",
        suffix=".viser",
        module="robot_msgs.viz.viser",
        show_code='msg.show(viewer="viser")',
        url="https://viser.studio",
        _supports=lambda module, message_type: module.supports(message_type),
        _show=lambda module, msg, name: module.show(**{name: msg}),
        _save=lambda module, msg, path: module.save(path, **{type(msg).__name__: msg}),
    )
)

__all__ = ["VIEWERS", "Viewer", "get", "register"]
