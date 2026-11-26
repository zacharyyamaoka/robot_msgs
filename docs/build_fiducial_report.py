"""Build the robot_msgs.fiducial review report from real OpenCV runs.

    /home/bam/robot_msgs/.venv/bin/python /home/bam/robot_msgs/docs/build_fiducial_report.py

Writes /home/bam/robot_msgs/reports/media/fiducial-msgs-2026-11-26.html (inlines images).
"""

from __future__ import annotations

import base64
import html
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
OUT = REPO / "reports" / "media" / "fiducial-msgs-2026-11-26.html"

from test_fiducial import CAMERA, LENGTH, T_CAMERA_MARKER, _detect, _render_marker  # noqa: E402

from robot_msgs.fiducial import FiducialBoard, FiducialDetection, FiducialPose  # noqa: E402

LABELS = ("TL", "TR", "BR", "BL")


def png(image: np.ndarray) -> str:
    ok, data = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(data.tobytes()).decode()


def corner_figure() -> tuple[str, float]:
    marker_id, pixels = _detect(_render_marker(7, T_CAMERA_MARKER))
    detection = FiducialDetection(id=marker_id, dictionary="DICT_4X4_50", corners=pixels)
    predicted = FiducialPose(id=7, dictionary="DICT_4X4_50", pose=T_CAMERA_MARKER).project(
        CAMERA, LENGTH
    )
    image = cv2.cvtColor(_render_marker(7, T_CAMERA_MARKER), cv2.COLOR_GRAY2BGR)
    for label, seen, ours in zip(LABELS, detection.corners, predicted, strict=True):
        cv2.circle(image, tuple(int(v) for v in ours), 9, (0, 160, 0), 2)
        cv2.drawMarker(image, tuple(int(v) for v in seen), (0, 0, 230), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(image, label, (int(seen[0]) + 12, int(seen[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)  # fmt: skip
    centre = tuple(int(v) for v in detection.center)
    cv2.drawMarker(image, centre, (200, 0, 200), cv2.MARKER_TILTED_CROSS, 16, 2)
    x0, y0 = (detection.corners.min(axis=0) - 60).astype(int)
    x1, y1 = (detection.corners.max(axis=0) + 60).astype(int)
    crop = image[max(y0, 0) : y1, max(x0, 0) : x1]
    return png(cv2.resize(crop, None, fx=2, fy=2)), float(np.abs(detection.corners - predicted).max())


def board_figure() -> str:
    board = FiducialBoard(layout="charuco", dictionary="DICT_4X4_50", cols=5, rows=4,
                          marker_length=0.017, square_length=0.023)  # fmt: skip
    oracle = cv2.aruco.CharucoBoard(
        (5, 4), 0.023, 0.017, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    scale = 4000  # px per metre
    image = cv2.cvtColor(oracle.generateImage((int(0.115 * scale), int(0.092 * scale))),
                         cv2.COLOR_GRAY2BGR)  # fmt: skip
    for mount in board.mounts():
        corners = (mount.corners_in_parent()[:, :2] * scale).astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 160, 0), 2)
        cv2.circle(image, tuple(corners[0]), 5, (0, 0, 230), -1)  # top-left first
        cv2.putText(image, str(mount.id), tuple(corners[3] + [2, 14]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 0, 0), 1)  # fmt: skip
    for point in board.chessboard_corners()[:, :2] * scale:
        cv2.circle(image, tuple(point.astype(int)), 4, (200, 0, 200), -1)
    return png(image)


def main() -> None:
    corners_png, worst = corner_figure()
    board_png = board_figure()
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.strip().splitlines()[-1]  # fmt: skip
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()  # fmt: skip
    rows = [
        ("FiducialDetection", "measured", "id, dictionary, corners (4,2) px TL TR BR BL, marker_length?, corner_covariance (4,2,2)?, decision_margin?, hamming?", "all three"),
        ("FiducialPose", "measured", "T_camera_marker + reprojection_error, alt_pose + alt_reprojection_error (the other IPPE solution), covariance?; ambiguity, flipped(), from_candidates()", "all three"),
        ("FiducialArray", "measured", "image header (stamp = join to JointState, frame = camera) + detections + poses; as_arrays() → ids, corners, solved, T_camera_marker (k,2,4,4), reprojection_error (k,2)", "tracker obs, BA input"),
        ("FiducialMount", "described", "id, dictionary, marker_length, pose = T_parent_marker on a link / board / world, covariance?; corners_in_parent(), transformed()", "tracker prior, BA output"),
        ("FiducialMap", "described", "all mounts, unique (dictionary, id); get, in_frame, merged, as_arrays()", "tracker prior"),
        ("FiducialBoard", "described", "grid / ChArUco spec (OpenCV, legacy pattern) → mounts(), chessboard_corners()", "calibration, rig boards, Studio"),
        ("FiducialTrack(Array)", "estimated", "filtered pose + 6×6 covariance, observed vs predicted, last_seen, nis; observed()/predicted(), as_arrays()", "tracker output"),
    ]  # fmt: skip
    table = "".join(
        f"<tr><td><code>{n}</code></td><td>{layer}</td><td>{html.escape(f)}</td><td>{who}</td></tr>"
        for n, layer, f, who in rows
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fiducial messages</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mute:#666;--line:#ddd;--ok:#1a7f37}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e8ea;--mute:#9a9aa0;--line:#333;--ok:#4cc26b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}}
h1{{font-size:22px}} .ok{{color:var(--ok);font-weight:600}} td,th{{border-top:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}}
table{{border-collapse:collapse;width:100%;font-size:13px}} img{{max-width:100%;border:1px solid var(--line)}} .mute{{color:var(--mute)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
</style></head><body>
<h1>robot_msgs.fiducial — ArUco / AprilTag / ChArUco messages</h1>
<p><b>Landed on robot_msgs <code>main</code> @ <code>{sha}</code> (0.6.0, pushed, not tagged for PyPI).</b>
Requirements came from the three ArUco agents running today (Kalman tracker, bundle adjustment, camera calibration) and from six older marker types (Aruco Studio, bam_rack_twin, bam_cad_descriptions, rig boards).
Suite: <span class="ok">{html.escape(tests)}</span>.</p>
<div class="grid">
<figure><img src="{corners_png}"><figcaption><b>Corner order + marker frame vs OpenCV.</b> OpenCV rendered a DICT_4X4_50 #7 at a tilted pose and its <code>ArucoDetector</code> found it.
Red crosses are the detected corners and green circles are where <code>FiducialPose.project</code> puts our TL/TR/BR/BL. Worst gap is {worst:.2f} px. The magenta cross is <code>detection.center</code>.</figcaption></figure>
<figure><img src="{board_png}"><figcaption><b>ChArUco board vs OpenCV.</b> The board image is <code>cv2.aruco.CharucoBoard.generateImage</code>. Green outlines are <code>FiducialBoard.mounts()</code>, red dots are each marker's top-left corner, and magenta dots are <code>chessboard_corners()</code>. Tests match <code>getObjPoints</code> to 1e-6 m, including the legacy pattern.</figcaption></figure>
</div>
<h2>Types</h2><table><tr><th>Type</th><th>Layer</th><th>Fields</th><th>Asked for by</th></tr>{table}</table>
<h2>Decided for you</h2><ul>
<li><b>Prefix <code>Fiducial*</code></b>, not <code>Marker*</code>: it covers ArUco and AprilTag (as fiducial_msgs does). Undo: rename in one commit.</li>
<li><b>Both IPPE solutions</b> on every pose. All three agents asked for this. A filter with an FK prior picks the right one.</li>
<li><b>Covariance is local-tangent</b> <code>[R<sup>T</sup>δp, rotvec(R<sup>T</sup>R<sub>true</sub>)]</code>, not ROS's fixed-axis. It is what the tracker and BA estimate in, and <code>local_to_parent_covariance</code> converts to ROS's. Checked by a 40k-sample Monte-Carlo draw.</li>
<li><b>No CaptureSettings type</b>: exposure, gain and white balance ride in <code>Header.meta</code> for now. The calibration agent was told it can ask for a typed one.</li>
<li><b>OpenCV is a test-only oracle</b> (added to the <code>[test]</code> extra), so the runtime stays numpy + dataclass-values.</li>
<li><b>Mutation check</b>: swapping TL/TR in <code>marker_object_points</code> turns 7 tests red, and breaking the legacy parity turns 2 red.</li></ul>
<h2>Not done</h2><ul class="mute">
<li>ROS conversion (aruco_msgs / fiducial_msgs aren't in rosbags' stock typestore), plotting/Rerun views, ChArUco interpolated-corner detections, a stamped PoseWithCovariance for camera extrinsics.</li>
<li>No PyPI release: tagging <code>v0.6.0</code> publishes the package, so that is your call.</li>
<li>Older marker types (bam_rack_twin, bam_cad_descriptions, Aruco Studio) are not migrated.</li></ul>
<p class="mute">Spec: robot_msgs docs/DESIGN.md § Fiducial markers. Built by docs/build_fiducial_report.py.</p>
</body></html>""")
    print(OUT)


if __name__ == "__main__":
    main()
