"""The docs are generated from the code (robotmsgs.com reads this package), so the code must carry them.

WHY (Zach, 2026-11-11, "Docs Generated From Code — Prior Art"): a type page with no docstring prints
"No docstring yet."; this test turns that into a red build instead of a quiet gap on the site.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import robot_msgs
from robot_msgs._core import Msg


def public_message_types() -> list[type]:
    found: dict[str, type] = {}
    for module_info in pkgutil.walk_packages(robot_msgs.__path__, "robot_msgs."):
        parts = module_info.name.split(".")
        if "viz" in parts or any(part.startswith("_") for part in parts):
            continue
        module = importlib.import_module(module_info.name)
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, Msg)
                and value is not Msg
                and not name.startswith("_")
                and value.__module__.startswith("robot_msgs.")
            ):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return [found[key] for key in sorted(found)]


def test_the_package_has_types_to_document():
    assert len(public_message_types()) >= 40


def test_every_public_message_type_has_its_own_docstring():
    undocumented = []
    for message_type in public_message_types():
        # WHY own __doc__: @dataclass writes "Name(field: type, ...)" when the class has none,
        # and inspect.getdoc would inherit the parent's.
        doc = message_type.__dict__.get("__doc__") or ""
        if not doc.strip() or doc.startswith(f"{message_type.__name__}("):
            undocumented.append(f"{message_type.__module__}.{message_type.__qualname__}")
    assert not undocumented, f"add a docstring to: {undocumented}"


def test_the_readme_opens_with_the_landing_copy():
    """robotmsgs.com's home page renders the README's opening paragraph, so it must exist."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert readme.startswith("# robot_msgs\n")
    lead = readme.split("\n\n", 2)[1]
    assert "frozen Python dataclasses" in lead
