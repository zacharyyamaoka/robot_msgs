"""Build robotmsgs.com into dist/: every page is generated from the robot_msgs package.

    env -u VIRTUAL_ENV uv run --isolated --with-editable '/home/bam/robot_msgs[viz,ros]' python build/build.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402
import pages  # noqa: E402

DIST = ROOT / "dist"
SITE = ROOT / "site"
ORIGIN = "https://robotmsgs.com"


def write(path: str, content: str | bytes) -> None:
    target = DIST / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SITE, DIST / "assets")
    digest = hashlib.sha256()
    for name in ("site.css", "site.js"):
        digest.update((SITE / name).read_bytes())
    ctx = {"asset_version": digest.hexdigest()[:8], "stamp": model.source_stamp()}

    types = model.collect()
    tour_code, tour_output = model.tour()

    write("index.html", pages.home(types, tour_code, tour_output, **ctx))
    write("types/index.html", pages.types_index(types, **ctx))
    write("docs/index.html", pages.docs_page(types, model.conventions(), **ctx))
    write("404.html", pages.not_found(types, **ctx))
    urls = ["/", "/docs/", "/types/"]
    for info in types:
        plot_file = None
        if info.plot_png:
            plot_file = f"/assets/plots/{info.domain}-{info.name}.webp"
            write(plot_file, info.plot_png)
        write(f"{info.path}index.html", pages.type_page(info, types, plot_file, **ctx))
        urls.append(info.path)

    write("robots.txt", f"User-agent: *\nAllow: /\nSitemap: {ORIGIN}/sitemap.xml\n")
    entries = "".join(f"<url><loc>{ORIGIN}{u}</loc></url>" for u in urls)
    write(
        "sitemap.xml",
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>\n',
    )
    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"built {len(types)} type pages, {len(urls) + 1} pages total, dist {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
