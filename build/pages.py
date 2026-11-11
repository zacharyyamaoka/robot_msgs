"""HTML for every page of robotmsgs.com. Pure functions from model data to strings."""

from __future__ import annotations

import html
import re

from highlight import highlight
import robot_msgs
from model import FnInfo, TypeInfo, domains_in

GITHUB = "https://github.com/zacharyyamaoka/robot_msgs"
PYPI = "https://pypi.org/project/robot-msgs/"
GROUP_TITLES = {
    "constructors": "Constructors",
    "properties": "Properties",
    "methods": "Methods",
    "operators": "Operators",
}
GROUP_HINTS = {
    "constructors": "Classmethods and staticmethods: ways to build one.",
    "properties": "Read-only views computed from the fields.",
    "methods": "Return new values; messages are never mutated.",
    "operators": "Python operators and protocols the type supports.",
}


def esc(text: object) -> str:
    return html.escape(str(text))


def inline(text: str) -> str:
    """Escape text and turn its ``code`` / `code` spans (reST or Markdown) into <code>."""
    escaped = esc(text)
    escaped = re.sub(r"``([^`]+)``", r"<code>\1</code>", escaped)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def install_note() -> str:
    # WHY generated from the package: a hand-written "release coming" note outlived 0.1.0 by a day.
    return f'Version {esc(robot_msgs.__version__)} on <a href="{PYPI}" rel="noopener">PyPI</a>.'


def plural(count: int) -> str:
    return f"{count} type" + ("" if count == 1 else "s")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------------- components


def code_block(code: str, language: str = "python", *, wrap: bool = False, label: str = "") -> str:
    cls = "code wrap" if wrap else "code"
    return (
        f'<div class="{cls}">'
        f'<button class="copy" type="button" aria-label="Copy {esc(label or "code")}">Copy</button>'
        f'<pre><code>{highlight(code, language)}</code></pre></div>'
    )


def heading(level: int, text: str, anchor: str) -> str:
    return (
        f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}" '
        f'aria-label="Link to {esc(text)}">{esc(text)}</a></h{level}>'
    )


def link(info: TypeInfo, *, text: str | None = None) -> str:
    return f'<a class="chip" href="{info.path}">{esc(text or info.name)}</a>'


def shell(
    *,
    title: str,
    description: str,
    path: str,
    body: str,
    asset_version: str,
    types: list[TypeInfo],
    section: str = "",
    current: str = "",
    sidebar: bool = True,
    toc: list[tuple[str, str, int]] | None = None,
    stamp: str = "",
    robots: str = "",
) -> str:
    nav_items = [("Home", "/", "home"), ("Docs", "/docs/", "docs"), ("Types", "/types/", "types")]
    nav_links = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if section == key else ""}>{label}</a>'
        for label, href, key in nav_items
    ) + f'<a href="{GITHUB}" rel="noopener">GitHub<span class="ext" aria-hidden="true">↗</span></a>'

    side = sidebar_html(types, current, section, full=sidebar, nav_links=nav_links)
    toc_html = ""
    if toc:
        items = "".join(
            f'<li class="l{level}"><a href="#{anchor}">{esc(label)}</a></li>'
            for anchor, label, level in toc
        )
        toc_html = (
            f'<aside class="toc" aria-label="On this page"><div class="toc-in">'
            f'<h2 class="toc-h">On this page</h2><ul>{items}</ul></div></aside>'
        )
    layout = "docs" if sidebar else "wide"
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="dark light">{f'<meta name="robots" content="{robots}">' if robots else ""}
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">
<script>try{{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<link rel="stylesheet" href="/assets/site.css?v={asset_version}">
</head>
<body class="{layout}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-in">
    <button class="icon-btn menu-btn" type="button" aria-label="Open menu" aria-expanded="false" aria-controls="sidebar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
    </button>
    <a class="brand" href="/"><svg width="20" height="20" viewBox="0 0 32 32" aria-hidden="true"><rect x="3" y="3" width="26" height="26" rx="6" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M10 21V11l6 5 6-5v10" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg><span>robot_msgs</span></a>
    <nav class="topnav" aria-label="Main">{nav_links}</nav>
    <div class="spacer"></div>
    <button class="icon-btn theme-btn" type="button" aria-label="Toggle light and dark theme">
      <svg class="i-sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      <svg class="i-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>
    </button>
  </div>
</header>
<div class="shell">
{side}
<main id="main" class="main" tabindex="-1">
{body}
<footer class="foot"><span>{esc(stamp)}</span><span><a href="{GITHUB}" rel="noopener">GitHub</a> · MIT license · generated from the package, not written per type</span></footer>
</main>
{toc_html}
</div>
<div class="scrim" hidden></div>
<script src="/assets/site.js?v={asset_version}" defer></script>
</body>
</html>
"""


def sidebar_html(
    types: list[TypeInfo], current: str, section: str, *, full: bool, nav_links: str
) -> str:
    drawer = f'<nav class="drawer-nav" aria-label="Site">{nav_links}</nav>'
    if not full:
        return f'<aside id="sidebar" class="sidebar lite">{drawer}</aside>'
    overview = [("All types", "/types/", "types-index"), ("Conventions", "/docs/", "docs")]
    groups = [
        '<div class="sb-group"><h2 class="sb-h">Overview</h2><ul>'
        + "".join(
            f'<li><a href="{href}"{" aria-current=\"page\"" if current == key else ""}>{label}</a></li>'
            for label, href, key in overview
        )
        + "</ul></div>"
    ]
    for domain in domains_in(types):
        members = [t for t in types if t.domain == domain]
        if not members:
            continue
        items = "".join(
            f'<li data-name="{esc(t.name.lower())}"><a href="{t.path}"'
            f'{" aria-current=\"page\"" if current == t.key else ""}>{esc(t.name)}</a></li>'
            for t in members
        )
        groups.append(
            f'<div class="sb-group" data-group><h2 class="sb-h">{esc(domain)}</h2><ul>{items}</ul></div>'
        )
    return (
        f'<aside id="sidebar" class="sidebar">{drawer}'
        '<div class="search"><input id="search" type="search" placeholder="Search types…" '
        'aria-label="Filter message types" autocomplete="off" spellcheck="false"><kbd>/</kbd></div>'
        '<nav class="sb-nav" aria-label="Message types">'
        + "".join(groups)
        + '<p class="sb-empty" hidden>No type matches.</p></nav></aside>'
    )


# ------------------------------------------------------------------------------------- pages


def home(types: list[TypeInfo], tour_code: str, tour_output: str, **ctx) -> str:
    domains = domains_in(types)
    packages = "".join(
        f'<a class="pkg" href="/types/#{d}"><span class="pkg-name">{d}</span>'
        f'<span class="pkg-n">{plural(sum(t.domain == d for t in types))}</span></a>'
        for d in domains
    )
    body = f"""<div class="home">
<section class="hero" aria-labelledby="hero-h">
  <p class="eyebrow">Python · numpy only · MIT</p>
  <h1 id="hero-h">robot_msgs <span class="dash">—</span> the standard vocabulary for robot data</h1>
  <p class="lead">ROS's <code>geometry_msgs</code>, <code>sensor_msgs</code>, <code>trajectory_msgs</code> and <code>vision_msgs</code> as frozen Python dataclasses, <em>with the functions that move between them</em>: offset a pose, interpolate a trajectory, project a point, turn a depth image into a cloud.</p>
  <div class="cta">
    <a class="btn primary" href="/types/">Browse {len(types)} types</a>
    <a class="btn" href="{GITHUB}" rel="noopener">View on GitHub</a>
  </div>
  <div class="install">
    {code_block("pip install robot-msgs", "shell", label="install command")}
    <p class="note">{install_note()}</p>
  </div>
</section>

<section class="pillars" aria-label="Principles">
  <div class="pillar"><h2>ROS's shape, not ROS</h2><p>Same packages and field names, so the ROS docs are our docs. No ROS install, no codegen. ROS 2 messages and bags are an export, not the programming model.</p></div>
  <div class="pillar"><h2>One entity, many views</h2><p>A trajectory is one value with batch arrays, single points, slices and interpolation, not 500 point messages. Views are free; messages are immutable.</p></div>
  <div class="pillar"><h2>numpy only</h2><p>One runtime dependency. Plots and ROS export are optional extras that the core never imports.</p></div>
</section>

<section class="tour" aria-labelledby="tour-h">
  <h2 id="tour-h">A tour in one screen</h2>
  <p class="muted">This is <code>examples/tour.py</code>, run for real; the output below is what it printed.</p>
  {code_block(tour_code, label="tour example")}
  <p class="out-label">Output</p>
  {code_block(tour_output, "text", wrap=True, label="tour output")}
</section>

<section class="packages" aria-labelledby="pk-h">
  <h2 id="pk-h">Packages</h2>
  <div class="pkgs">{packages}</div>
  <p class="muted">Every message has <code>==</code> and <code>hash</code> by value, a short <code>repr</code>, <code>replace(...)</code>, <code>to_dict()</code> / <code>from_dict()</code>, <code>save_npz</code> / <code>load_npz</code> and pickling. <a href="/types/">See all types</a>.</p>
</section>
</div>"""
    return shell(
        title="robot_msgs — the standard vocabulary for robot data",
        description="ROS's standard message types as frozen Python dataclasses, with the functions that move between them. numpy only.",
        path="/", body=body, section="home", sidebar=False, types=types, **ctx,
    )  # fmt: skip


def types_index(types: list[TypeInfo], **ctx) -> str:
    toc = [("all-types", "All types", 2)]
    sections = []
    for domain in domains_in(types):
        members = [t for t in types if t.domain == domain]
        if not members:
            continue
        toc.append((domain, domain, 3))
        cards = "".join(
            f'<a class="card" href="{t.path}"><span class="card-name">{esc(t.name)}</span>'
            f'<span class="card-sum">{inline(card_summary(t))}</span></a>'
            for t in members
        )
        sections.append(
            f'<section aria-labelledby="{domain}">{heading(2, domain, domain)}'
            f'<p class="muted"><code>robot_msgs.{domain}</code> · {plural(len(members))}</p>'
            f'<div class="grid">{cards}</div></section>'
        )
    body = f"""<article class="prose">
<p class="crumbs">Types</p>
<h1 id="all-types">All types</h1>
<p class="lead">{len(types)} message types in {len(sections)} packages. Every page is generated from the package: fields, functions, a real example and the plot <code>robot_msgs.viz</code> draws for it.</p>
{"".join(sections)}
</article>"""
    return shell(
        title="All types · robot_msgs",
        description=f"All {len(types)} robot_msgs message types, grouped by package.",
        path="/types/", body=body, section="types", current="types-index", toc=toc, types=types, **ctx,
    )  # fmt: skip


def card_summary(info: TypeInfo) -> str:
    if info.summary:
        return info.summary
    return f"ROS 2: {info.ros_type}" if info.ros_type else "No docstring yet."


def docs_page(types: list[TypeInfo], conventions: list[tuple[str, str]], **ctx) -> str:
    conv_rows = "".join(
        f"<tr><th scope='row'>{inline(k)}</th><td>{inline(v)}</td></tr>" for k, v in conventions
    )
    pkg_rows = "".join(
        f'<tr><th scope="row"><code>robot_msgs.{d}</code></th><td>'
        + " ".join(link(t) for t in types if t.domain == d)
        + "</td></tr>"
        for d in domains_in(types)
    )
    ros_rows = "".join(
        f'<tr><th scope="row">{link(t)}</th><td><code>{esc(t.ros_type)}</code></td></tr>'
        for t in types
        if t.ros_type
    )
    toc = [
        ("overview", "Overview", 2),
        ("conventions", "Conventions", 2),
        ("packages", "Packages", 2),
        ("ros-mapping", "ROS 2 mapping", 2),
    ]
    body = f"""<article class="prose">
<p class="crumbs">Docs</p>
{heading(1, "Conventions", "overview")}
<p class="lead">Everything in robot_msgs follows a few fixed rules, written down so nothing is inferred. The types themselves are documented one by one under <a href="/types/">Types</a>; the full design contract is <a href="{GITHUB}/blob/main/docs/DESIGN.md" rel="noopener">docs/DESIGN.md</a>.</p>
{code_block("pip install robot-msgs", "shell", label="install command")}
<p class="note">{install_note()}</p>
{heading(2, "Conventions", "conventions")}
<div class="table-wrap"><table class="kv"><thead><tr><th>Thing</th><th>Convention</th></tr></thead><tbody>{conv_rows}</tbody></table></div>
{heading(2, "Packages", "packages")}
<p class="muted">Import types from their package, like ROS packages: <code>from robot_msgs.geometry import Pose</code>.</p>
<div class="table-wrap"><table class="kv"><thead><tr><th>Import</th><th>Types</th></tr></thead><tbody>{pkg_rows}</tbody></table></div>
{heading(2, "ROS 2 mapping", "ros-mapping")}
<p class="muted">What <code>robot_msgs.ros.ros_type(msg)</code> returns for each type (ROS 2 Jazzy, via rosbags).</p>
<div class="table-wrap"><table class="kv"><thead><tr><th>robot_msgs</th><th>ROS 2</th></tr></thead><tbody>{ros_rows}</tbody></table></div>
</article>"""
    return shell(
        title="Docs · robot_msgs",
        description="robot_msgs conventions, packages and the ROS 2 mapping.",
        path="/docs/", body=body, section="docs", current="docs", toc=toc, types=types, **ctx,
    )  # fmt: skip


def not_found(types: list[TypeInfo], **ctx) -> str:
    body = """<div class="home">
<section class="hero">
  <p class="eyebrow">404</p>
  <h1>That page is not in the vocabulary</h1>
  <p class="lead">Nothing lives at this address. Try the list of types, or go back to the start.</p>
  <div class="cta"><a class="btn primary" href="/types/">Browse types</a><a class="btn" href="/">Home</a></div>
</section></div>"""
    return shell(
        title="Not found · robot_msgs", description="Page not found.", path="/404.html",
        body=body, sidebar=False, types=types, robots="noindex", **ctx,
    )  # fmt: skip


# ------------------------------------------------------------------------------- type pages


def fn_html(fn: FnInfo, types_by_name: dict[str, TypeInfo]) -> str:
    anchor = f"fn-{slug(fn.name.strip('_'))}" if not fn.name.startswith("__") else f"op-{slug(fn.name)}"
    badge = ""
    if fn.inherited:
        owner = types_by_name.get(fn.owner)
        label = f"from {esc(fn.owner)}"
        badge = (
            f'<a class="badge inh" href="{owner.path}">{label}</a>'
            if owner
            else f'<span class="badge inh">{label}</span>'
        )
    sig = fn.signature
    summary = inline(fn.summary) if fn.summary else '<span class="faint">No docstring.</span>'
    return (
        f'<div class="fn" id="{anchor}"><div class="fn-head">'
        f'<h4><a class="anchor" href="#{anchor}"><code>{esc(fn.name)}</code></a></h4>{badge}</div>'
        f'<pre class="sig"><code>{highlight(sig)}</code></pre><p class="fn-doc">{summary}</p></div>'
    )


def related_html(info: TypeInfo, by_key: dict[str, TypeInfo]) -> str:
    def row(label: str, keys: list[str], hint: str) -> str:
        if not keys:
            return ""
        chips = " ".join(link(by_key[k]) for k in keys)
        return f'<div class="rel"><h3>{label}</h3><p class="muted">{hint}</p><p class="chips">{chips}</p></div>'

    rows = [
        row("Uses", info.uses, "Types its fields refer to."),
        row("Used by", info.used_by, "Types whose fields refer to it."),
        row("Extends", info.bases[:1], "The base class it specialises."),
        row("Extended by", info.subclasses, "Types that specialise it."),
        row("Sample type", [info.point_type] if info.point_type else [], "What <code>traj[i]</code> returns."),
        row("Sample of", info.point_of, "The trajectory whose samples this is."),
    ]
    inner = "".join(r for r in rows if r)
    return inner or '<p class="muted">No other type refers to it, and it refers to none.</p>'


def type_page(info: TypeInfo, types: list[TypeInfo], plot_file: str | None, **ctx) -> str:
    by_key = {t.key: t for t in types}
    by_name = {t.name: t for t in types}
    toc: list[tuple[str, str, int]] = []
    sections: list[str] = []

    if info.plot_png and plot_file:
        toc.append(("preview", "Preview", 2))
        sections.append(
            f'<section aria-labelledby="preview">{heading(2, "Preview", "preview")}'
            f'<p class="muted">The figure <code>robot_msgs.viz.plot(example)</code> draws for the example below.</p>'
            f'<figure class="figure"><img src="{plot_file}" alt="Plot of an example {esc(info.name)}" '
            f'loading="lazy" decoding="async" width="{info.plot_size[0]}" height="{info.plot_size[1]}"></figure></section>'
        )

    toc.append(("fields", "Fields", 2))
    rows = []
    for f in info.fields:
        badge = ' <span class="badge sample" title="First axis is time: one value per sample">per sample</span>' if f.per_sample else ""
        rows.append(
            f'<tr><th scope="row"><code>{esc(f.name)}</code>{badge}</th>'
            f'<td><code>{highlight(f.annotation)}</code></td><td><code>{esc(f.default)}</code></td></tr>'
        )
    sections.append(
        f'<section aria-labelledby="fields">{heading(2, "Fields", "fields")}'
        f'<div class="table-wrap"><table class="fields"><thead><tr><th>Name</th><th>Type</th><th>Default</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )

    toc.append(("example", "Example", 2))
    sections.append(
        f'<section aria-labelledby="example">{heading(2, "Example", "example")}'
        f'<p class="muted">One instance from the package\'s own test catalog; the output is its real <code>repr()</code>.</p>'
        f'{code_block(info.example_repr, wrap=True, label="example repr")}'
        f'<details class="json"><summary>to_dict() as JSON</summary>'
        f'{code_block(info.example_json, "json", label="example JSON")}'
        f'<p class="muted small">Long arrays are cut after 16 values and say so with “…”.</p></details></section>'
    )

    toc.append(("functions", "Functions", 2))
    fn_sections = []
    for group in ("constructors", "properties", "methods", "operators"):
        members = [fn for fn in info.functions if fn.group == group]
        if not members:
            continue
        toc.append((group, GROUP_TITLES[group], 3))
        fn_sections.append(
            f'<div class="fn-group">{heading(3, GROUP_TITLES[group], group)}'
            f'<p class="muted">{GROUP_HINTS[group]}</p>'
            + "".join(fn_html(fn, by_name) for fn in members)
            + "</div>"
        )
    sections.append(
        f'<section aria-labelledby="functions">{heading(2, "Functions", "functions")}{"".join(fn_sections)}</section>'
    )

    toc.append(("related", "Related types", 2))
    sections.append(
        f'<section aria-labelledby="related">{heading(2, "Related types", "related")}{related_html(info, by_key)}</section>'
    )

    ros = (
        f'<span class="meta-k">ROS 2</span><code>{esc(info.ros_type)}</code>'
        if info.ros_type
        else '<span class="meta-k">ROS 2</span><span class="faint">no ROS equivalent</span>'
    )
    summary = (
        f'<p class="lead">{inline(info.summary)}</p>'
        if info.summary
        else '<p class="lead faint">No docstring in robot_msgs yet.</p>'
    )
    rest = (
        f'<details class="doc-rest"><summary>Full docstring</summary><pre class="plain">{esc(info.doc_rest)}</pre></details>'
        if info.doc_rest
        else ""
    )
    body = f"""<article class="prose">
<nav class="crumbs" aria-label="Breadcrumb"><a href="/types/">Types</a> <span aria-hidden="true">/</span> <a href="/types/#{info.domain}">{info.domain}</a> <span aria-hidden="true">/</span> <span aria-current="page">{esc(info.name)}</span></nav>
<h1 id="top">{esc(info.name)}</h1>
{summary}{rest}
<div class="meta">
  <div class="meta-row"><span class="meta-k">Import</span><div class="meta-v">{code_block(info.import_line, label="import line")}</div></div>
  <div class="meta-row">{ros}</div>
</div>
{"".join(sections)}
</article>"""
    return shell(
        title=f"{info.name} · robot_msgs.{info.domain}",
        description=info.summary or f"robot_msgs.{info.domain}.{info.name}: fields, example and functions.",
        path=info.path, body=body, section="types", current=info.key, toc=toc, types=types, **ctx,
    )  # fmt: skip
