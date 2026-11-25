"""build/used_by.py: the "Used by packages" section from robot_sdk's graph.json (a tiny fixture, no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))

import used_by  # noqa: E402

GRAPH = {
    "schema": "robot-sdk-graph/1",
    "generated": "2026-11-24T21:00:00-07:00",
    "sources": {"robot_controllers": "7d22dc5-dirty"},
    "packages": [{"id": "robot_controllers", "repo_url": "https://github.com/z/robot_controllers", "layer": "leaf",
                  "one_line": "Controllers"}, {"id": "sim2real", "repo_url": "https://github.com/z/sim2real",
                  "layer": "outer", "one_line": "Twins"}],
    "uses": [
        {"package": "robot_controllers", "type": "robot_msgs.sensor.JointState", "kind": "consumes",
         "symbol": "robot_controllers.core.Controller.tick", "where": "src/robot_controllers/core.py:62",
         "public": True, "declared": True, "verified": True},
        {"package": "robot_controllers", "type": "robot_msgs.sensor.JointState", "kind": "consumes",
         "symbol": "robot_controllers.x.ghost", "where": "robot-sdk.toml:9",
         "public": True, "declared": True, "verified": False},
        {"package": "sim2real", "type": "robot_msgs.sensor.JointState", "kind": "imports",
         "symbol": "sim2real.rig", "where": "src/sim2real/rig.py:3", "public": False, "declared": False,
         "verified": True},
    ],
}  # fmt: skip


class UsedBy(unittest.TestCase):
    def test_links_each_symbol_to_its_line_at_the_scanned_commit(self):
        html = used_by.section(GRAPH, "sensor", "JointState")
        self.assertIn('id="used-by-packages"', html)
        self.assertIn("https://github.com/z/robot_controllers/blob/7d22dc5/src/robot_controllers/core.py#L62", html)
        self.assertIn(">core.Controller.tick<", html)
        self.assertIn("not on the public API, by", html)  # imports-only packages are named, not listed as API

    def test_declared_but_unverified_rows_are_never_shown(self):
        self.assertNotIn("ghost", used_by.section(GRAPH, "sensor", "JointState"))

    def test_unused_type_says_so_and_no_graph_means_no_section(self):
        self.assertIn("No robot-sdk package", used_by.section(GRAPH, "geometry", "Pose"))
        self.assertEqual(used_by.section(None, "sensor", "JointState"), "")


if __name__ == "__main__":
    unittest.main()
