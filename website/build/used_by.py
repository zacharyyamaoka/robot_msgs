"""The "Used by packages" section of each type page: which robot-sdk packages consume or produce this type.

Read from robot_sdk's generated graph (`site/data/graph.json`, schema `robot-sdk-graph/1`, written by robot_sdk's
`tools/sdk_graph` from the packages' code and linted against each package's `robot-sdk.toml`).

WHY fetched from GitHub, not copied in (2026-11-24): one source of truth with no vendored copy to go stale, and CI has
no robot_sdk checkout to copy from. `ROBOT_SDK_GRAPH=/path/to/graph.json` (or another URL) overrides it, e.g. to
preview a graph before it is pushed. If it can't be read the build goes on without the section and says so.

WHY Zach (Daily Note 2026-11-24): "in like robot_msgs I feel like there should be a section under each thing almost
like links to the packages that consume that message."
"""

from __future__ import annotations

import functools
import html
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

GRAPH_URL = "https://raw.githubusercontent.com/zacharyyamaoka/robot_sdk/main/site/data/graph.json"
SDK_URL = "https://github.com/zacharyyamaoka/robot_sdk"
KIND_TITLE = {"consumes": "Takes it", "produces": "Returns it"}


def esc(text: object) -> str:
    return html.escape(str(text))


def load() -> dict[str, Any] | None:
    source = os.environ.get("ROBOT_SDK_GRAPH", GRAPH_URL)
    try:
        if source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=20) as response:  # noqa: S310  a fixed https URL
                graph = json.load(response)
        else:
            graph = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"used_by: no robot-sdk graph from {source} ({error}); type pages go without 'Used by packages'",
              file=sys.stderr)  # fmt: skip
        return None
    if graph.get("schema") != "robot-sdk-graph/1":
        print(f"used_by: {source} is {graph.get('schema')!r}, not robot-sdk-graph/1; skipped", file=sys.stderr)
        return None
    return graph


@functools.cache
def graph() -> dict[str, Any] | None:
    """The graph, fetched once per build."""
    return load()


def source_link(graph: dict[str, Any], package: dict[str, Any], where: str) -> str | None:
    """The cited file:line on GitHub, at the commit the graph was generated from."""
    path, _, line = where.rpartition(":")
    if not path or not line.isdigit():
        return None
    sha = str(graph.get("sources", {}).get(package["id"], "main")).removesuffix("-dirty")
    return f"{package['repo_url']}/blob/{sha}/{path}#L{line}"


def type_id(domain: str, name: str) -> str:
    return f"robot_msgs.{domain}.{name}"


def section(graph: dict[str, Any] | None, domain: str, name: str) -> str:
    """HTML for one type page, or "" when there is no graph."""
    if graph is None:
        return ""
    tid = type_id(domain, name)
    packages = {p["id"]: p for p in graph["packages"]}
    order = {p["id"]: i for i, p in enumerate(graph["packages"])}
    rows = [u for u in graph["uses"] if u["type"] == tid and u.get("verified", True)]
    api = [u for u in rows if u["kind"] in KIND_TITLE]
    imports = sorted({u["package"] for u in rows if u["kind"] == "imports"} - {u["package"] for u in api},
                     key=order.get)  # fmt: skip

    blocks = []
    for pkg_id in sorted({u["package"] for u in api}, key=order.get):
        pkg = packages[pkg_id]
        mine = [u for u in api if u["package"] == pkg_id]
        # WHY declared symbols first: they are the contract the package states in its robot-sdk.toml; the rest are
        # other public signatures the scanner found, folded so a package with 20 hits stays one glance.
        lines = []
        for kind, title in KIND_TITLE.items():
            of_kind = sorted((u for u in mine if u["kind"] == kind), key=lambda u: (not u["declared"], u["symbol"]))
            if not of_kind:
                continue
            shown = [u for u in of_kind if u["declared"]] or of_kind[:3]
            rest = [u for u in of_kind if u not in shown]
            chips = " ".join(symbol_chip(graph, pkg, u) for u in shown)
            more = ""
            if rest:
                more = (f'<details class="ub-more"><summary>{len(rest)} more</summary><p class="chips">'
                        + " ".join(symbol_chip(graph, pkg, u) for u in rest) + "</p></details>")  # fmt: skip
            lines.append(f'<div class="ub-kind"><span class="ub-k">{title}</span><p class="chips">{chips}</p>{more}</div>')
        badge = f'<span class="badge">{esc(pkg["layer"])}</span>'
        blocks.append(
            f'<div class="ub-pkg"><h3><a href="{esc(pkg["repo_url"])}"><code>{esc(pkg_id)}</code></a>{badge}</h3>'
            f'<p class="muted small">{esc(pkg["one_line"])}</p>{"".join(lines)}</div>'
        )
    if imports:
        names = ", ".join(f'<a href="{esc(packages[p]["repo_url"])}"><code>{esc(p)}</code></a>' for p in imports)
        blocks.append(f'<p class="muted small">Also imported, but not on the public API, by {names}.</p>')
    if not blocks:
        blocks.append('<p class="muted">No robot-sdk package names this type in its code yet.</p>')
    generated = esc(str(graph.get("generated", ""))[:10])
    return (
        '<section aria-labelledby="used-by-packages">'
        '<h2 id="used-by-packages"><a class="anchor" href="#used-by-packages" aria-label="Link to Used by packages">'
        'Used by packages</a></h2>'
        f'<p class="muted">The <a href="{SDK_URL}">robot-sdk</a> packages whose public API takes or returns a '
        f'<code>{esc(name)}</code>, read from their code (<code>robot-sdk</code> graph of {generated}). '
        f'Each symbol links to the line that names the type.</p>{"".join(blocks)}</section>'
    )


def symbol_chip(graph: dict[str, Any], pkg: dict[str, Any], use: dict[str, Any]) -> str:
    short = use["symbol"].removeprefix(pkg["id"] + ".")
    href = source_link(graph, pkg, use["where"])
    title = esc(f'{use["symbol"]} ({use["where"]})' + (", declared in robot-sdk.toml" if use["declared"] else ""))
    if not href:
        return f'<code class="chip" title="{title}">{esc(short)}</code>'
    return f'<a class="chip" href="{esc(href)}" title="{title}">{esc(short)}</a>'
