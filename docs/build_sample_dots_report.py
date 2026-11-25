"""Build the sample-dots review report (robot_msgs 0f8fb8f) from the real captures in reports/media/sample-dots-2026-11-25/.

    /home/bam/robot_msgs/.venv/bin/python /home/bam/robot_msgs/docs/build_sample_dots_report.py

Writes /home/bam/robot_msgs/reports/media/sample-dots-2026-11-25.html (media inlined, so it lives in the gitignored
reports/media/). Every image is a screenshot of Rerun 0.40's own web viewer on the real ur5e workspace-dev-001 tape,
except 00 (Zach's screenshot, the before) and 06-09 (a two-series prototype recording, the no-code path).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "reports" / "media" / "sample-dots-2026-11-25"
OUT = REPO / "reports" / "media" / "sample-dots-2026-11-25.html"


def uri(name: str) -> str:
    path = MEDIA / name
    kind = {"jpg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}[path.suffix[1:]]
    return f"data:image/{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


def gif(frames: list[str], out: str, ms: int) -> str:
    """A looping GIF of real captures (no synthetic frames): the hero, with the stills beside it."""
    images = [Image.open(MEDIA / name).convert("RGB") for name in frames]
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=ms, loop=0)
    (MEDIA / out).write_bytes(buffer.getvalue())
    return out


def figure(name: str, caption: str) -> str:
    return f'<figure><img src="{uri(name)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


hero = gif(["04-3d-dots-on.jpg", "05-3d-dots-off.jpg"], "hero-3d-toggle.gif", 1300)
nocode = gif(["06-nocode-select.jpg", "07-nocode-menu.jpg", "08-nocode-added.jpg"], "nocode-add.gif", 1500)

PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rerun sample dots</title>
<style>
:root {{ --bg: #ffffff; --ink: #1d232b; --muted: #5d6773; --line: #dfe3e8; --card: #f6f8fa; --accent: #2563a0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111418; --ink: #e6e9ed; --muted: #9aa4af; --line: #2a3038;
  --card: #181c21; --accent: #6aa7e0; }} }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 20px 16px 60px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }} h2 {{ font-size: 17px; margin: 32px 0 8px; }}
.verdict {{ font-size: 16px; padding: 12px 14px; border-left: 4px solid var(--accent); background: var(--card); }}
.muted {{ color: var(--muted); }} code {{ background: var(--card); padding: 1px 4px; border-radius: 4px; }}
figure {{ margin: 0; }} img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }}
figcaption {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
ul {{ padding-left: 20px; }} li {{ margin: 3px 0; }}
</style></head><body><main>
<h1>Rerun: a dot on every sample</h1>
<p class="verdict"><b>Yes. Every joint, jerk and TCP plot, and the 3D tool path, now shows a dot per sample, and it is
live on the Datasets page.</b> In 3D the dots are their own entity, so one eye click turns them on or off. Plots have no
single switch in Rerun 0.40: they're per series (Selection › Visualizers). <span class="muted">robot_msgs
<code>0f8fb8f</code>; trajectory_tracking_agent <code>6dc5191</code> (54 recordings rebuilt, Pages deployed, live file
byte-identical).</span></p>

<h2>Hero: the 3D toggle, on your real dev-001 recording</h2>
<div class="grid">{figure(hero, "Blueprint › 3D › world/ur5e/tool/samples › eye: dots on ↔ off (two real captures, looping)")}</div>

<h2>Before / after</h2>
<div class="grid">
{figure("00-before-zach.webp", "Before (your screenshot): each plot and the 3D path is one continuous line")}
{figure("01-dashboard-after.jpg", "After, the same tape at full zoom: the dots read as the line (100 Hz is ~1.4 px apart)")}
{figure("02-zoomed-0.7-2.1s.jpg", "After, zoomed to 0.7–2.1 s (your window): the dots separate; one legend entry per series")}
{figure("03-tcp-closeup.png", "TCP position maximised over 0.7–2.1 s: one dot per 10 ms sample")}
</div>

<h2>Toggling them</h2>
<ul>
<li><b>3D</b>: blueprint panel (left; it opens collapsed, so toggle it from the top-right icons), search
<code>samples</code>, click its eye. Proved above.</li>
<li><b>One plot</b>: select the series → Selection panel › Visualizers → SeriesPoints <b>×</b> removes the dots; <b>+</b> ›
SeriesPoints puts them back. That works on <i>any</i> recording, even old ones with no dots baked in:</li>
</ul>
<div class="grid">
{figure(nocode, "No code: select a series, Visualizers +, SeriesPoints (prototype recording, Rerun 0.40 web viewer)")}
{figure("09-nocode-group-none.jpg", "But a parent entity shows Visualizers: none, so there is no group or view-wide switch")}
</div>

<h2>What changed</h2>
<ul>
<li><code>robot_msgs/viz/rerun.py</code>: each plotted series also logs Rerun's stock <code>SeriesPoints</code>
(same name and colour as its line, circle, 1.5 pt); the tool path gets <code>/world/&lt;name&gt;/tool/samples</code>, every
sample as a <code>Points3D</code> (the polylines are thinned to 256 vertices, so the dots are the only full-rate 3D
view). Only on the <code>robot=</code> / <code>log_joint_tabs</code> path. <code>log</code> without <code>robot=</code>
is frozen by the golden-digest test and is unchanged.</li>
<li>Test <code>test_every_plotted_series_and_the_tool_path_show_a_dot_per_sample</code> passes, and fails against the old
viewer. robot_msgs suite: 1238 passed. README documents the toggle.</li>
<li>Datasets page: <code>tools/build_coverage.py build</code> rewrote all 54 .rrd (check: ok), <code>tta verify</code>
OK, <code>blackbox check</code> in the format. Built in a separate worktree so the peer's uncommitted
<code>site/comparison</code> files were never touched.</li>
</ul>

<h2>Decided for you</h2>
<ul>
<li>Dots are <b>on by default</b> and small: the line still looks like a line at full zoom, and you see samples as soon
as you zoom in. Undo: <code>git -C /home/bam/robot_msgs revert 0f8fb8f</code>, then rebuild the Datasets page.</li>
<li>Rebuilt and published the 54 recordings (+87 MB to the public repo's history, the same as the two viewer rebuilds
earlier today). Undo: <code>git -C /home/bam/trajectory_tracking_agent revert 6dc5191</code> and push.</li>
<li>That push also published the peer's already-committed <code>344cfd3</code> (Model × Dataset pivot), which was one
commit ahead of origin.</li>
</ul>

<h2>Deliberately not done</h2>
<ul>
<li>No view-wide plot toggle: Rerun 0.40 can only show or hide visualizers per entity. A second set of "dots" entities
would still need one eye click per plot (24+ plots), so it isn't worth the extra layout.</li>
<li>The cards (% of limit) and the limit lines keep plain lines.</li>
</ul>
<p class="muted">GitHub Pages caches for ~10 min; if app.rerun.io still shows plain lines, hard-refresh.</p>
</main></body></html>
"""

OUT.write_text(PAGE)
print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
