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


def viser_viewer() -> str:
    """Viser's own client page, loading the scene named by ``?scene=`` before it boots.

    WHY one shared page, not `as_html()` per type: the client is ~0.8 MB; embedded per type that
    is ~50 MB of copies. The client reads ``window.__VISER_EMBED_DATA__`` (viser infra/_infra.py).
    """
    import viser

    client = (Path(viser.__file__).parent / "client" / "build" / "index.html").read_text()
    loader = (
        "<script>(function(){var q=new URLSearchParams(location.search);var s=q.get('scene');"
        "if(!s)return;var x=new XMLHttpRequest();x.open('GET',s,false);x.send();"
        "if(x.status===200){window.__VISER_EMBED_DATA__=x.responseText.trim();}"
        "window.__VISER_EMBED_CONFIG__={darkMode:q.get('dark')==='1'};})();</script>"
    )
    head_end = client.index("</head>")
    return client[:head_end] + loader + client[head_end:]


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
    write("assets/viser/viewer.html", viser_viewer())
    for info in types:
        files: dict[str, str] = {}
        for view in info.views:
            if view.data is not None:
                files[view.name] = f"/assets/viz/{view.name}/{info.domain}-{info.name}{view.suffix}"
                write(files[view.name], view.data)
        write(f"{info.path}index.html", pages.type_page(info, types, files, **ctx))
        urls.append(info.path)
    # WHY CORS on the view files: the Rerun tab embeds Rerun's hosted viewer (app.rerun.io), which
    # fetches the .rrd from this origin.
    write("_headers", "/assets/viz/*\n  Access-Control-Allow-Origin: *\n")

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
