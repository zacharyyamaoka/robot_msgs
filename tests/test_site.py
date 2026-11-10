"""robotmsgs.com: every message type has a complete page, every link resolves, pages stay small.

Run inside the build environment (`pnpm run test`), where robot_msgs is importable. The catalog is
robot_msgs's own contract test list, so a type added there fails here until the site renders it.
"""

import html
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
sys.path.insert(0, str(ROOT / "build"))

import build as site_build  # noqa: E402
import model  # noqa: E402

MAX_PAGE_BYTES = 1_500_000


def plain(path):
    """A page's visible text: tags dropped, entities decoded."""
    return html.unescape(re.sub(r"<[^>]+>", "", Path(path).read_text()))


def pages():
    return sorted(DIST.rglob("*.html"))


class Site(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        site_build.main()  # always test a fresh build, never a stale dist
        cls.catalog = model.load_catalog()
        cls.types = {type(e): e for e in reversed(cls.catalog.EXAMPLES)}

    def page_for(self, cls):
        domain = cls.__module__.split(".")[1]
        return DIST / "types" / domain / cls.__name__ / "index.html"

    def test_every_catalog_type_has_a_page(self):
        for cls in self.types:
            self.assertTrue(self.page_for(cls).is_file(), f"no page for {cls.__name__}")

    def test_every_type_page_has_fields_and_functions(self):
        for cls in self.types:
            text = self.page_for(cls).read_text()
            self.assertIn('id="fields"', text, cls.__name__)
            self.assertIn('id="functions"', text, cls.__name__)
            self.assertRegex(text, r'<tr><th scope="row"><code>', cls.__name__)
            self.assertIn('class="fn"', text, f"{cls.__name__} lists no function")
            self.assertIn(
                f"from robot_msgs.{cls.__module__.split('.')[1]} import {cls.__name__}",
                plain(self.page_for(cls)),
            )

    def test_fields_table_lists_every_dataclass_field(self):
        import dataclasses

        for cls in self.types:
            text = self.page_for(cls).read_text()
            for f in dataclasses.fields(cls):
                self.assertIn(f"<code>{f.name}</code>", text, f"{cls.__name__}.{f.name}")

    def test_per_sample_badge_on_trajectory_fields(self):
        text = plain(DIST / "types/trajectory/JointTrajectory/index.html")
        self.assertEqual(text.count("per sample"), 4)  # positions, velocities, accelerations, effort
        self.assertNotIn("per sample", plain(DIST / "types/geometry/Pose/index.html"))

    def test_example_is_the_real_repr(self):
        for cls, example in self.types.items():
            self.assertIn(repr(example), plain(self.page_for(cls)), cls.__name__)

    def test_plot_and_ros_sections(self):
        self.assertIn('id="preview"', (DIST / "types/trajectory/JointTrajectory/index.html").read_text())
        pose = (DIST / "types/geometry/Pose/index.html").read_text()
        self.assertIn("geometry_msgs/msg/Pose", pose)
        self.assertIn("no ROS equivalent", (DIST / "types/trajectory/JointLimits/index.html").read_text())

    def test_related_types_both_directions(self):
        pose = (DIST / "types/geometry/Pose/index.html").read_text()
        self.assertRegex(pose, r'<h3>Uses</h3>.*href="/types/geometry/Point/"')
        self.assertRegex(pose, r'<h3>Used by</h3>.*href="/types/geometry/PoseStamped/"')

    def test_home_has_install_line_and_tour(self):
        text = plain(DIST / "index.html")
        self.assertIn("pip install robot-msgs", text)
        self.assertIn("PyPI release coming", text)
        code, output = model.tour()
        for line in output.splitlines():
            self.assertIn(line.strip(), text)
        self.assertIn("pregrasp = grasp.shifted", text)

    def test_index_lists_every_type(self):
        text = (DIST / "types/index.html").read_text()
        for cls in self.types:
            self.assertIn(f'href="/types/{cls.__module__.split(".")[1]}/{cls.__name__}/"', text)

    def test_sidebar_order(self):
        text = (DIST / "types/geometry/Pose/index.html").read_text()
        order = [text.index(f'<h2 class="sb-h">{d}</h2>') for d in model.PACKAGE_ORDER]
        self.assertEqual(order, sorted(order))
        self.assertIn('aria-current="page">Pose</a>', text)

    def test_every_internal_link_resolves(self):
        for page in pages():
            text = page.read_text()
            for href in re.findall(r'(?:href|src)="(/[^"]*)"', text):
                path = href.split("#")[0].split("?")[0]
                target = DIST / path.lstrip("/")
                if path.endswith("/") or path == "":
                    target = target / "index.html"
                self.assertTrue(target.is_file(), f"{page.relative_to(DIST)}: {href} -> missing")

    def test_in_page_anchors_resolve(self):
        for page in pages():
            text = page.read_text()
            ids = set(re.findall(r'\bid="([^"]+)"', text))
            for anchor in re.findall(r'href="#([^"]+)"', text):
                self.assertIn(anchor, ids, f"{page.relative_to(DIST)}: #{anchor}")

    def test_cross_page_anchors_resolve(self):
        for page in pages():
            for href in re.findall(r'href="(/[^"#]*)#([^"]+)"', page.read_text()):
                path, anchor = href
                target = DIST / path.lstrip("/") / "index.html" if path.endswith("/") else DIST / path.lstrip("/")
                self.assertIn(f'id="{anchor}"', target.read_text(), f"{page.relative_to(DIST)}: {path}#{anchor}")

    def test_ids_are_unique_per_page(self):
        for page in pages():
            ids = re.findall(r'\bid="([^"]+)"', page.read_text())
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in {page.relative_to(DIST)}")

    def test_no_page_over_limit(self):
        for page in pages():
            self.assertLess(page.stat().st_size, MAX_PAGE_BYTES, str(page))

    def test_pages_have_landmarks_and_one_h1(self):
        for page in pages():
            text = page.read_text()
            self.assertEqual(text.count("<h1"), 1, page.name)
            for landmark in ("<main", "<header", "<nav"):
                self.assertIn(landmark, text, f"{page.relative_to(DIST)} lacks {landmark}")

    def test_external_links_are_https(self):
        for page in pages():
            for url in re.findall(r'href="(http[^"]*)"', page.read_text()):
                self.assertTrue(url.startswith("https://"), url)

    def test_404_and_sitemap_exist(self):
        self.assertTrue((DIST / "404.html").is_file())
        sitemap = (DIST / "sitemap.xml").read_text()
        self.assertEqual(sitemap.count("<url>"), len(self.types) + 3)


if __name__ == "__main__":
    unittest.main()
