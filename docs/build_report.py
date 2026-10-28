"""Build the robot_msgs v1 review report from real runs (tests, wheel, examples, gallery).

    uv run --directory /home/bam/robot_msgs --isolated --extra test python docs/build_report.py

Writes /home/bam/robot_msgs/reports/media/robot-msgs-v1-2026-10-28.html (media-heavy, so it
lives in the gitignored reports/media/ beside its gallery folder).
"""

from __future__ import annotations

import base64
import html
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "reports" / "media"
GALLERY = MEDIA / "robot-msgs-v1-gallery"
OUT = MEDIA / "robot-msgs-v1-2026-10-28.html"
ENV = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"} | {"MPLBACKEND": "Agg"}


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=REPO, env=ENV, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def test_counts() -> tuple[Counter, int, int, str]:
    with tempfile.TemporaryDirectory() as scratch:
        junit = Path(scratch) / "junit.xml"
        tail = run(sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}").strip().splitlines()[-1]
        root = ET.parse(junit).getroot()
    per_file: Counter = Counter()
    failures = skipped = 0
    for case in root.iter("testcase"):
        per_file[case.get("classname", "").split(".")[-1]] += 1
        failures += any(child.tag in ("failure", "error") for child in case)
        skipped += any(child.tag == "skipped" for child in case)
    return per_file, failures, skipped, tail


def wheel_requires() -> tuple[str, list[str]]:
    with tempfile.TemporaryDirectory() as scratch:
        run("uv", "build", "--wheel", "-o", scratch)
        wheel = next(Path(scratch).glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            metadata = next(n for n in archive.namelist() if n.endswith("METADATA"))
            lines = archive.read(metadata).decode().splitlines()
    return wheel.name, [line for line in lines if line.startswith("Requires-Dist")]


def import_probe() -> str:
    probe = (
        "import sys, robot_msgs; "
        "print(sorted({m.split('.')[0] for m in sys.modules if not m.startswith('_')} "
        "- set(sys.stdlib_module_names)))"
    )
    env = {"PYTHONPATH": str(REPO / "src")}
    result = subprocess.run([sys.executable, "-I", "-c", probe], env=env, capture_output=True, text=True)
    return result.stdout.strip()


def data_uri(path: Path) -> str:
    kind = {".png": "image/png", ".gif": "image/gif", ".mp4": "video/mp4", ".svg": "image/svg+xml"}
    return f"data:{kind[path.suffix]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def code(text: str) -> str:
    return f"<pre><code>{html.escape(text.rstrip())}</code></pre>"


def tour_block() -> str:
    source = (REPO / "examples" / "tour.py").read_text()
    body = source.split('"""', 2)[2].strip()
    output = run(sys.executable, "examples/tour.py")
    return code(body) + "<p class='label'>Real output</p>" + code(output)


def pick_output() -> str:
    return code(run(sys.executable, "examples/ur5e_pick.py"))


def ros_output() -> str:
    bag = Path(tempfile.mkdtemp()) / "pick_bag"
    source = (REPO / "examples" / "ros_export.py").read_text()
    body = source.split('"""', 2)[2].strip()
    return code(body) + "<p class='label'>Real output</p>" + code(run(sys.executable, "examples/ros_export.py", str(bag)))


def gallery() -> tuple[str, str]:
    GALLERY.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "examples/gallery.py", str(GALLERY))
    hero_mp4, hero_gif = GALLERY / "hero.mp4", GALLERY / "hero.gif"
    hero = (
        f"<video autoplay muted loop playsinline controls poster='' src='{data_uri(hero_mp4)}'>"
        f"<img src='{data_uri(hero_gif)}' alt='hero animation'></video>"
        if hero_mp4.exists()
        else f"<img src='{data_uri(hero_gif)}' alt='hero animation'>"
    )
    figures = []
    for png in sorted(GALLERY.glob("*.png")):
        title = re.sub(r"^\d+[-_]", "", png.stem).replace("_", " ").replace("-", " ")
        figures.append(
            f"<figure><img loading='lazy' src='{data_uri(png)}' alt='{html.escape(title)}'>"
            f"<figcaption><code>plot()</code> · {html.escape(title)}</figcaption></figure>"
        )
    return hero, "\n".join(figures)


def main() -> None:
    per_file, failures, skipped, tail = test_counts()
    wheel, requires = wheel_requires()
    probe = import_probe()
    hero, figures = gallery()
    template = (REPO / "docs" / "report_template.html").read_text()
    rows = "".join(
        f"<tr><td><code>{html.escape(name)}.py</code></td><td class='num'>{count}</td></tr>"
        for name, count in sorted(per_file.items())
    )
    page = (
        template.replace("{{HERO}}", hero)
        .replace("{{TOUR}}", tour_block())
        .replace("{{PICK_OUTPUT}}", pick_output())
        .replace("{{ROS_OUTPUT}}", ros_output())
        .replace("{{GALLERY}}", figures)
        .replace("{{TEST_ROWS}}", rows)
        .replace("{{TEST_TOTAL}}", str(sum(per_file.values())))
        .replace("{{TEST_TAIL}}", html.escape(tail))
        .replace("{{TEST_FAILURES}}", str(failures))
        .replace("{{TEST_SKIPPED}}", str(skipped))
        .replace("{{WHEEL}}", html.escape(wheel))
        .replace("{{REQUIRES}}", html.escape("\n".join(requires)))
        .replace("{{PROBE}}", html.escape(probe))
        .replace("{{COMMIT}}", run("git", "rev-parse", "--short", "HEAD").strip())
    )
    OUT.write_text(page)
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.2f} MB  tests: {tail}")


if __name__ == "__main__":
    main()
