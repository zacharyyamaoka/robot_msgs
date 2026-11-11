"""Everything the site knows about a message type, read from the robot_msgs package itself.

Nothing here is written per type: enumeration, fields, functions, relations, examples, plots and
the ROS 2 mapping all come from introspecting the installed package and its test catalog.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")  # a build never opens a window

import robot_msgs  # noqa: E402
from robot_msgs.trajectory import Trajectory, point_class, sample_fields  # noqa: E402

ROBOT_MSGS_DIR = Path(robot_msgs.__file__).resolve().parents[2]
CATALOG = ROBOT_MSGS_DIR / "tests" / "catalog.py"
TOUR = ROBOT_MSGS_DIR / "examples" / "tour.py"
DESIGN = ROBOT_MSGS_DIR / "docs" / "DESIGN.md"
README = ROBOT_MSGS_DIR / "README.md"

# WHY this order (Zach): the sidebar reads std, geometry, sensor, vision, trajectory.
# shape sits beside geometry (solids of a scene); compare comes last (it reads the others).
PACKAGE_ORDER = ("std", "geometry", "shape", "sensor", "vision", "trajectory", "compare")


def domains_in(types) -> list[str]:
    """Every domain that has types, preferred order first, then any new one alphabetically.

    WHY: a hardcoded list silently dropped new robot_msgs domains (shape, compare) from the
    index while their pages were still built; a new domain must show up without a site edit.
    """
    present = {t.domain for t in types}
    return [d for d in PACKAGE_ORDER if d in present] + sorted(present - set(PACKAGE_ORDER))

# Dunder methods that are plumbing, not API. Anything else a robot_msgs class defines
# (__getitem__, __matmul__, __add__ ...) is an operator worth showing.
INTERNAL_DUNDERS = frozenset(
    {
        "__init__", "__post_init__", "__eq__", "__ne__", "__hash__", "__repr__", "__str__",
        "__setattr__", "__delattr__", "__getattribute__", "__getstate__", "__setstate__",
        "__reduce__", "__reduce_ex__", "__init_subclass__", "__class_getitem__", "__replace__",
        "__copy__", "__deepcopy__", "__sizeof__", "__dir__", "__format__", "__new__",
        "__subclasshook__", "__lt__", "__le__", "__gt__", "__ge__", "__class__",
        "__set_name__", "__getattr__",
    }
)  # fmt: skip


@dataclass
class FieldInfo:
    name: str
    annotation: str
    default: str
    per_sample: bool


@dataclass
class FnInfo:
    name: str
    signature: str
    summary: str
    group: str  # constructors | properties | methods | operators
    owner: str  # class that defines it
    inherited: bool


@dataclass
class TypeInfo:
    key: str  # "geometry/Pose"
    domain: str
    name: str
    cls: type
    summary: str
    has_docstring: bool
    doc_rest: str
    import_line: str
    ros_type: str | None
    example: Any
    example_repr: str
    example_json: str
    plot_png: bytes | None
    plot_size: tuple[int, int] | None
    fields: list[FieldInfo] = field(default_factory=list)
    functions: list[FnInfo] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)  # keys referenced by this type's fields
    used_by: list[str] = field(default_factory=list)  # keys whose fields reference this type
    bases: list[str] = field(default_factory=list)
    subclasses: list[str] = field(default_factory=list)
    point_type: str | None = None  # what indexing a trajectory returns
    point_of: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"/types/{self.domain}/{self.name}/"


# ---------------------------------------------------------------------------------- docstrings


def split_doc(cls: type) -> tuple[str, bool, str]:
    """(summary, real docstring?, rest). A dataclass with no docstring gets its signature as one."""
    doc = inspect.getdoc(cls) or ""
    if not doc or doc.startswith(f"{cls.__name__}("):
        return "", False, ""
    first, _, rest = doc.partition("\n\n")
    return " ".join(first.split()), True, rest.strip()


def first_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.splitlines()[0].strip() if doc else ""


# ------------------------------------------------------------------------------------- fields


def default_text(f: dataclasses.Field) -> str:
    if f.default is not dataclasses.MISSING:
        return repr(f.default)
    factory = f.default_factory
    if factory is dataclasses.MISSING:
        return "required"
    if factory is dict:
        return "{}"
    if factory is tuple:
        return "()"
    if inspect.isclass(factory):
        return f"{factory.__name__}()"
    try:
        return repr(factory())
    except Exception:  # noqa: BLE001  a factory we cannot call is shown as what it is
        return "<factory>"


def fields_of(cls: type) -> list[FieldInfo]:
    per_sample = {f.name for f in sample_fields(cls)} if issubclass(cls, Trajectory) else set()
    return [
        FieldInfo(f.name, str(f.type), default_text(f), f.name in per_sample)
        for f in dataclasses.fields(cls)
    ]


# ----------------------------------------------------------------------------------- functions


class Raw:
    """An annotation kept as written: signature printing would repr (and quote) a string."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def as_written(sig: inspect.Signature) -> inspect.Signature:
    def raw(annotation: Any) -> Any:
        return Raw(annotation) if isinstance(annotation, str) else annotation

    return sig.replace(
        parameters=[p.replace(annotation=raw(p.annotation)) for p in sig.parameters.values()],
        return_annotation=raw(sig.return_annotation),
    )


def render_signature(name: str, sig: inspect.Signature) -> str:
    sig = as_written(sig)
    text = f"{name}{sig}"
    if len(text) <= 72:
        return text
    params = [str(p) for p in sig.parameters.values()]
    ret = f" -> {sig.return_annotation!r}" if sig.return_annotation is not sig.empty else ""
    return f"{name}(\n" + "".join(f"    {p},\n" for p in params) + f"){ret}"


def drop_first(sig: inspect.Signature) -> inspect.Signature:
    return sig.replace(parameters=list(sig.parameters.values())[1:])


def owner_of(cls: type, name: str) -> type | None:
    return next((base for base in cls.__mro__ if name in base.__dict__), None)


def function_infos(cls: type) -> list[FnInfo]:
    field_names = {f.name for f in dataclasses.fields(cls)}
    found: dict[str, FnInfo] = {}
    for name in dir(cls):
        dunder = name.startswith("__") and name.endswith("__")
        if (name.startswith("_") and not dunder) or name in field_names:
            continue
        if dunder and name in INTERNAL_DUNDERS:
            continue
        owner = owner_of(cls, name)
        if owner is None or not owner.__module__.startswith("robot_msgs."):
            continue
        if dunder and owner is robot_msgs.Msg:
            continue
        raw = inspect.getattr_static(cls, name)
        try:
            info = describe(cls, owner, name, raw)
        except (TypeError, ValueError):  # a callable with no introspectable signature
            continue
        if info:
            found[name] = info
    order = {"constructors": 0, "properties": 1, "methods": 2, "operators": 3}
    return sorted(found.values(), key=lambda f: (order[f.group], f.inherited, f.name))


def describe(cls: type, owner: type, name: str, raw: Any) -> FnInfo | None:
    dunder = name.startswith("__")
    inherited = owner is not cls
    if isinstance(raw, (classmethod, staticmethod)):
        sig = inspect.signature(getattr(cls, name))
        return FnInfo(
            name, render_signature(name, sig), first_line(raw.__func__), "constructors",
            owner.__name__, inherited,
        )  # fmt: skip
    if isinstance(raw, (property, functools.cached_property)):
        fget = raw.fget if isinstance(raw, property) else raw.func
        ret = inspect.signature(fget).return_annotation
        shown = name if ret is inspect.Signature.empty else f"{name}: {ret if isinstance(ret, str) else inspect.formatannotation(ret)}"
        return FnInfo(name, shown, first_line(fget), "properties", owner.__name__, inherited)
    if inspect.isfunction(raw):
        sig = drop_first(inspect.signature(raw))
        group = "operators" if dunder else "methods"
        return FnInfo(
            name, render_signature(name, sig), first_line(raw), group, owner.__name__, inherited
        )  # fmt: skip
    return None


# ------------------------------------------------------------------------------ the catalogue


def load_catalog() -> Any:
    spec = importlib.util.spec_from_file_location("robot_msgs_test_catalog", CATALOG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def abbreviate(value: Any, limit: int = 16) -> Any:
    """Shorten long encoded arrays for display, with an explicit marker (never silently)."""
    if isinstance(value, dict):
        if {"dtype", "shape", "data"} <= value.keys() and len(value["data"]) > limit:
            kept = value["data"][:limit] + [f"… {len(value['data']) - limit} more values"]
            return {**value, "data": kept}
        return {k: abbreviate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [abbreviate(v, limit) for v in value]
    return value


def format_json(value: Any, indent: int = 0) -> str:
    """Two-space JSON where lists of scalars stay on one line, so arrays stay readable."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = [f'{pad}  {json.dumps(k)}: {format_json(v, indent + 1)}' for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in value) + "]"
        rows = [f"{pad}  {format_json(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(rows) + f"\n{pad}]"
    return json.dumps(value, ensure_ascii=False)


def render_plot(example: Any) -> tuple[bytes, tuple[int, int]] | None:
    """The robot_msgs.viz figure for an example as (WebP bytes, size), or None without a plot."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from robot_msgs.viz import plot

    try:
        figure = plot(example)
    except TypeError:
        return None
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    out = io.BytesIO()
    picture = Image.open(buffer)
    picture.save(out, "WEBP", quality=80, method=6)
    return out.getvalue(), picture.size


def ros_type_of(example: Any) -> str | None:
    from robot_msgs.ros import ros_type

    try:
        return ros_type(example)
    except Exception:  # noqa: BLE001  every "no ROS equivalent" failure reads the same on the page
        return None


def package_types(domain: str, keys: list[str]) -> list[str]:
    """Keys of one package in the order its `__all__` lists them (alphabetical if not listed)."""
    listed = list(getattr(getattr(robot_msgs, domain), "__all__", []))
    rank = {name: i for i, name in enumerate(listed)}
    return sorted(
        (k for k in keys if k.split("/")[0] == domain),
        key=lambda k: (rank.get(k.split("/")[1], len(rank)), k),
    )


def collect() -> list[TypeInfo]:
    registry = {
        key: cls
        for key, cls in robot_msgs.message_types().items()
        if cls.__module__.startswith("robot_msgs.")
    }
    catalog = load_catalog()
    examples: dict[type, Any] = {}
    for item in catalog.EXAMPLES:
        examples.setdefault(type(item), item)

    infos: dict[str, TypeInfo] = {}
    for key, cls in registry.items():
        domain, name = key.split("/")
        example = examples[cls]  # the catalog contract guarantees one per type
        summary, has_doc, rest = split_doc(cls)
        data = abbreviate(example.to_dict())
        plot = render_plot(example)
        infos[key] = TypeInfo(
            key=key, domain=domain, name=name, cls=cls, summary=summary, has_docstring=has_doc,
            doc_rest=rest, import_line=f"from robot_msgs.{domain} import {name}",
            ros_type=ros_type_of(example), example=example, example_repr=repr(example),
            example_json=format_json(data), plot_png=plot[0] if plot else None,
            plot_size=plot[1] if plot else None,
            fields=fields_of(cls), functions=function_infos(cls),
        )  # fmt: skip

    by_name = {i.name: i.key for i in infos.values()}
    by_class = {i.cls: i.key for i in infos.values()}
    for info in infos.values():
        words = {w for f in info.fields for w in re.findall(r"[A-Za-z_]\w*", f.annotation)}
        info.uses = sorted(by_name[w] for w in words if w in by_name and by_name[w] != info.key)
        info.bases = [by_class[b] for b in info.cls.__mro__[1:] if b in by_class]
        if issubclass(info.cls, Trajectory):
            try:
                point = by_class.get(point_class(info.cls))
            except Exception:  # noqa: BLE001  a trajectory with no registered point type
                point = None
            info.point_type = point if point != info.key else None
    for info in infos.values():
        for other in infos.values():
            if info.key in other.uses:
                info.used_by.append(other.key)
            if info.key in other.bases[:1]:
                info.subclasses.append(other.key)
            if other.point_type == info.key:
                info.point_of.append(other.key)
        info.used_by.sort()

    order = {d: i for i, d in enumerate(PACKAGE_ORDER)}
    ordered: list[TypeInfo] = []
    for domain in sorted({i.domain for i in infos.values()}, key=lambda d: order.get(d, 99)):
        ordered += [infos[k] for k in package_types(domain, list(infos))]
    return ordered


# ------------------------------------------------------------------------------- page content


def tour() -> tuple[str, str]:
    """examples/tour.py without its docstring, and its real stdout."""
    import sys

    source = TOUR.read_text()
    code = re.sub(r'^""".*?"""\n\n', "", source, count=1, flags=re.S).rstrip()
    result = subprocess.run(
        [sys.executable, str(TOUR)], check=True, capture_output=True, text=True
    )
    return code, result.stdout.rstrip()


@dataclass(frozen=True)
class Intro:
    """The home page's words, all read from the package (WHY: hand-written copy here went stale)."""

    tagline: str  # pyproject description
    eyebrow: str  # Python floor · license · version
    lead: str  # README: the paragraph under the title (Markdown inline)
    pillars: list[tuple[str, str]]  # README: the first bullet list, "**Title.** body"


def intro() -> Intro:
    from importlib.metadata import metadata

    meta = metadata("robot-msgs")
    blocks = README.read_text().split("\n\n")
    lead = " ".join(blocks[1].split())
    bullets = re.split(r"\n- ", "\n" + blocks[2].strip())[1:]
    pillars = []
    for bullet in bullets:
        text = " ".join(bullet.split())
        match = re.match(r"\*\*(.+?)\*\*\s*(.*)", text)
        if match:
            pillars.append((match.group(1).rstrip("."), match.group(2)))
    python = (meta.get("Requires-Python") or "").replace(">=", "≥ ")
    license_name = meta.get("License-Expression") or meta.get("License") or ""
    eyebrow = " · ".join(part for part in (f"Python {python}".strip(), license_name, f"v{meta['Version']}") if part)
    return Intro(tagline=meta["Summary"], eyebrow=eyebrow, lead=lead, pillars=pillars)


def conventions() -> list[tuple[str, str]]:
    """The Conventions table of docs/DESIGN.md as (thing, convention) rows."""
    text = DESIGN.read_text()
    section = text.split("## Conventions", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if line.startswith("|") and not set(line) <= set("|- "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and cells[0] != "Thing":
                rows.append((cells[0], cells[1]))
    return rows


def source_stamp() -> str:
    """'robot_msgs 0.1.0 @ 5d4cab0' for the footer; the commit part is best effort."""
    version = f"robot_msgs {robot_msgs.__version__}"
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROBOT_MSGS_DIR), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return version
    return f"{version} @ {sha}"
