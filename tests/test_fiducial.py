"""robot_msgs.fiducial: OpenCV is the independent oracle for corner order, the marker frame and
board geometry; a Monte-Carlo draw is the oracle for the covariance conversion."""

import numpy as np
import pytest

from robot_msgs.fiducial import (
    FiducialArray,
    FiducialBoard,
    FiducialDetection,
    FiducialMap,
    FiducialMount,
    FiducialPose,
    FiducialTrack,
    FiducialTrackArray,
    local_to_parent_covariance,
    marker_object_points,
    parent_to_local_covariance,
)
from robot_msgs.geometry import Pose, Quaternion
from robot_msgs.sensor import CameraInfo
from robot_msgs.std import Header

try:
    import cv2
except ImportError:  # WHY not importorskip: only the OpenCV-oracle tests need it
    cv2 = None
needs_cv2 = pytest.mark.skipif(cv2 is None, reason="OpenCV is the oracle (test extra)")

CAMERA = CameraInfo.from_intrinsics(900.0, 900.0, 640.0, 360.0, 1280, 720)
K = np.asarray(CAMERA.k)
LENGTH = 0.04
# a marker 0.5 m ahead, tilted so perspective is visible, upright as printed
T_CAMERA_MARKER = Pose.from_xyz_rpy(0.03, -0.02, 0.5, np.pi + 0.35, 0.25, 0.1)


def _dictionary(name: str = "DICT_4X4_50"):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _render_marker(marker_id: int, pose: Pose) -> np.ndarray:
    """A camera image of one printed marker at `pose`, by OpenCV alone (no robot_msgs maths)."""
    side_px = (
        600  # marker plus a one-module white quiet zone on each side (4x4 + border = 6 modules)
    )
    module = side_px // 8
    printed = np.full((side_px, side_px), 255, np.uint8)
    inner = cv2.aruco.generateImageMarker(_dictionary(), marker_id, 6 * module)
    printed[module : 7 * module, module : 7 * module] = inner
    # printed pixel -> marker-frame metres: the black square spans modules 1..7, x right, y up
    scale = LENGTH / (6 * module)
    source = np.float32(
        [[module, module], [7 * module, module], [7 * module, 7 * module], [module, 7 * module]]
    )
    h = LENGTH / 2
    object_points = np.float32([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    rvec, _ = cv2.Rodrigues(pose.orientation.to_matrix())
    tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
    target, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    homography = cv2.getPerspectiveTransform(source, target.reshape(4, 2).astype(np.float32))
    assert scale > 0
    return cv2.warpPerspective(printed, homography, (1280, 720), borderValue=255)


def _detect(image: np.ndarray):
    detector = cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    assert ids is not None and len(ids) == 1
    return int(np.ravel(ids)[0]), corners[0].reshape(4, 2).astype(float)


# ── the conventions, against OpenCV ─────────────────────────────────────────────


@needs_cv2
def test_corner_order_and_marker_frame_match_opencv_detection():
    marker_id, pixels = _detect(_render_marker(7, T_CAMERA_MARKER))
    detection = FiducialDetection(id=marker_id, dictionary="DICT_4X4_50", corners=pixels)
    predicted = FiducialPose(id=7, dictionary="DICT_4X4_50", pose=T_CAMERA_MARKER).project(
        CAMERA, LENGTH
    )
    assert marker_id == 7
    np.testing.assert_allclose(detection.corners, predicted, atol=1.0)  # px, detector subpixel


@needs_cv2
def test_ippe_square_on_marker_object_points_recovers_the_pose():
    _, pixels = _detect(_render_marker(7, T_CAMERA_MARKER))
    count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        marker_object_points(LENGTH), pixels, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    candidates = [
        Pose.from_matrix(np.block([[cv2.Rodrigues(r)[0], t.reshape(3, 1)], [np.zeros((1, 3)), 1]]))
        for r, t in zip(rvecs, tvecs, strict=True)
    ]
    solved = FiducialPose.from_candidates(7, "DICT_4X4_50", candidates, errors.ravel())
    assert count == 2 and solved.alt_pose is not None
    assert solved.reprojection_error <= solved.alt_reprojection_error
    assert abs(solved.pose.position.z - 0.5) < 0.01
    assert solved.pose.orientation.angle_to(T_CAMERA_MARKER.orientation) < np.radians(3)
    assert 0.0 <= solved.ambiguity <= 1.0


@needs_cv2
def test_detection_center_is_the_projected_marker_centre():
    _, pixels = _detect(_render_marker(3, T_CAMERA_MARKER))
    detection = FiducialDetection(id=3, dictionary="DICT_4X4_50", corners=pixels)
    centre = CAMERA.project(np.array([T_CAMERA_MARKER.position.as_array()]))[0]
    np.testing.assert_allclose(detection.center, centre, atol=1.0)
    assert detection.area_px > 0  # facing the camera: corners run clockwise in the image


@needs_cv2
@pytest.mark.parametrize(
    "cols, rows, legacy", [(4, 3, False), (3, 4, False), (3, 4, True), (5, 6, True)]
)
def test_charuco_board_matches_opencv(cols, rows, legacy):
    board = FiducialBoard(
        layout="charuco",
        dictionary="DICT_4X4_50",
        cols=cols,
        rows=rows,
        marker_length=0.02,
        square_length=0.03,
        legacy=legacy,
        first_id=4,
    )
    oracle = cv2.aruco.CharucoBoard(
        (cols, rows), 0.03, 0.02, _dictionary(), ids=np.array(board.ids)
    )
    oracle.setLegacyPattern(legacy)
    mounts = board.mounts()
    assert [m.id for m in mounts] == oracle.getIds().ravel().tolist()
    for mount, points in zip(mounts, oracle.getObjPoints(), strict=True):
        np.testing.assert_allclose(mount.corners_in_parent(), points.reshape(4, 3), atol=1e-6)
    np.testing.assert_allclose(
        board.chessboard_corners(), oracle.getChessboardCorners().reshape(-1, 3), atol=1e-6
    )


@needs_cv2
def test_grid_board_matches_opencv():
    board = FiducialBoard(
        layout="grid", dictionary="DICT_4X4_50", cols=3, rows=2, marker_length=0.04, separation=0.01
    )
    oracle = cv2.aruco.GridBoard((3, 2), 0.04, 0.01, _dictionary())
    for mount, points in zip(board.mounts(), oracle.getObjPoints(), strict=True):
        np.testing.assert_allclose(mount.corners_in_parent(), points.reshape(4, 3), atol=1e-6)
    assert board.size == pytest.approx((0.14, 0.09))


# ── covariance convention ───────────────────────────────────────────────────────


def test_local_to_parent_covariance_matches_a_monte_carlo_draw():
    rng = np.random.default_rng(0)
    pose = Pose.from_xyz_rpy(0.2, -0.1, 0.5, 0.4, -0.3, 1.1)
    sigma = np.diag([2e-3, 1e-3, 4e-3, 1e-3, 3e-3, 2e-3]) ** 2
    local = rng.multivariate_normal(np.zeros(6), sigma, size=40_000)
    rotation = pose.orientation.to_matrix()
    parent = []
    for delta in local:
        true_rotation = rotation @ Quaternion.from_rotvec(delta[3:]).to_matrix()
        dp = rotation @ delta[:3]  # p_true - p_est, in parent axes
        dtheta = Quaternion.from_matrix(true_rotation @ rotation.T).to_rotvec()  # about parent axes
        parent.append(np.concatenate([dp, dtheta]))
    empirical = np.cov(np.array(parent).T)
    converted = local_to_parent_covariance(pose, sigma)
    np.testing.assert_allclose(converted, empirical, atol=3e-7)
    np.testing.assert_allclose(parent_to_local_covariance(pose, converted), sigma, atol=1e-15)


def test_a_local_covariance_survives_a_change_of_parent():
    mount = FiducialMount(
        id=1,
        dictionary="DICT_4X4_50",
        marker_length=0.02,
        pose=Pose.from_xyz_rpy(0.1, 0, 0, 0, 0, 0.5),
        covariance=np.eye(6) * 1e-6,
        header=Header(frame_id="link_6"),
    )
    moved = mount.transformed(Pose.from_xyz_rpy(0, 0, 0.3, 0.2, 0, 0), "base_link")
    assert moved.frame_id == "base_link"
    np.testing.assert_array_equal(moved.covariance, mount.covariance)
    np.testing.assert_allclose(
        moved.corners_in_parent()[0],
        Pose.from_xyz_rpy(0, 0, 0.3, 0.2, 0, 0).transform_point(mount.corners_in_parent()[0]),
    )


# ── the messages ────────────────────────────────────────────────────────────────


def _detection(marker_id, dictionary="DICT_4X4_50"):
    return FiducialDetection(
        id=marker_id, dictionary=dictionary, corners=[[0, 0], [10, 0], [10, 10], [0, 10]]
    )


def test_array_batch_view_marks_unsolved_and_missing_alternatives():
    two = FiducialPose.from_candidates(
        1, "DICT_4X4_50", [Pose.identity(), T_CAMERA_MARKER], [0.8, 0.3]
    )
    one = FiducialPose(id=2, dictionary="DICT_4X4_50", pose=Pose.identity(), reprojection_error=0.2)
    frame = FiducialArray(
        detections=(_detection(1), _detection(2), _detection(3)), poses=(two, one)
    )
    arrays = frame.as_arrays()
    assert arrays["corners"].shape == (3, 4, 2) and arrays["T_camera_marker"].shape == (3, 2, 4, 4)
    assert arrays["solved"].tolist() == [True, True, False]
    np.testing.assert_allclose(arrays["T_camera_marker"][0, 0], T_CAMERA_MARKER.to_matrix())
    assert (
        np.isnan(arrays["T_camera_marker"][1, 1]).all()
        and np.isnan(arrays["T_camera_marker"][2]).all()
    )
    np.testing.assert_allclose(arrays["reprojection_error"][0], [0.3, 0.8])
    assert frame.pose_of(1).ambiguity == pytest.approx(0.375)
    assert frame.pose_of(1).flipped().pose == Pose.identity()
    assert len(frame.filtered(ids=[1, 3])) == 2 and len(frame.filtered(ids=[1, 3]).poses) == 1


def test_lookup_needs_the_dictionary_only_when_an_id_is_shared():
    frame = FiducialArray(
        detections=(_detection(5), _detection(5, "DICT_APRILTAG_36h11"), _detection(6))
    )
    assert frame.detection_of(6).id == 6
    assert frame.detection_of(5, "DICT_APRILTAG_36h11").dictionary == "DICT_APRILTAG_36h11"
    with pytest.raises(ValueError, match="pass dictionary="):
        frame.detection_of(5)
    with pytest.raises(KeyError):
        frame.detection_of(9)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: FiducialArray(poses=(FiducialPose(id=1, dictionary="D", pose=Pose()),)),
            "not detected",
        ),
        (
            lambda: FiducialDetection(id=-1, dictionary="D", corners=np.zeros((4, 2))),
            "non-negative",
        ),
        (lambda: FiducialDetection(id=1, dictionary="", corners=np.zeros((4, 2))), "family"),
        (lambda: FiducialDetection(id=1, dictionary="D", corners=np.zeros((3, 2))), r"\(4, 2\)"),
        (lambda: FiducialMount(id=1, dictionary="D", marker_length=0.0), "positive length"),
        (
            lambda: FiducialMount(
                id=1, dictionary="D", marker_length=0.02, covariance=np.arange(36.0).reshape(6, 6)
            ),
            "symmetric",
        ),
        (
            lambda: FiducialMap(
                mounts=(FiducialMount(id=1, dictionary="D", marker_length=0.02),) * 2
            ),
            "twice",
        ),
        (lambda: FiducialTrack(id=1, dictionary="D", observed=True), "last_seen"),
        (
            lambda: FiducialPose(id=1, dictionary="D", pose=Pose(), alt_reprojection_error=0.1),
            "alt_pose",
        ),
        (
            lambda: FiducialBoard(
                layout="charuco",
                dictionary="D",
                cols=3,
                rows=3,
                marker_length=0.03,
                square_length=0.03,
            ),
            "under square_length",
        ),
        (
            lambda: FiducialBoard(
                layout="grid", dictionary="D", cols=3, rows=3, marker_length=0.03
            ),
            "separation",
        ),
    ],
)
def test_bad_messages_fail_at_construction(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_tracks_split_into_observed_and_predicted():
    tracks = FiducialTrackArray(
        header=Header(stamp=10.0, frame_id="base_link"),
        tracks=(
            FiducialTrack(
                id=1, dictionary="D", observed=True, last_seen=10.0, nis=2.1, covariance=np.eye(6)
            ),
            FiducialTrack(id=2, dictionary="D", last_seen=9.5),
            FiducialTrack(id=3, dictionary="D"),
        ),
    )
    assert [t.id for t in tracks.observed()] == [1] and [t.id for t in tracks.predicted()] == [2, 3]
    assert tracks.get(2).age(10.0) == pytest.approx(0.5) and tracks.get(3).age(10.0) == np.inf
    arrays = tracks.as_arrays()
    assert arrays["observed"].tolist() == [True, False, False]
    assert np.isnan(arrays["covariance"][1]).all() and np.isnan(arrays["nis"][1:]).all()


def test_map_batch_view_and_frames():
    board = FiducialBoard(
        layout="grid",
        dictionary="DICT_4X4_50",
        cols=2,
        rows=1,
        marker_length=0.02,
        separation=0.005,
        header=Header(frame_id="board"),
        first_id=10,
    ).mounts()
    sticker = FiducialMap(
        mounts=(
            FiducialMount(
                id=0,
                dictionary="DICT_4X4_50",
                marker_length=0.0275,
                header=Header(frame_id="link_6"),
            ),
        )
    )
    both = board.merged(sticker)
    assert both.frames == ("board", "link_6") and ("DICT_4X4_50", 10) in both
    assert both.as_arrays()["T_parent_marker"].shape == (3, 4, 4)
    assert both.get(11).pose.position.x == pytest.approx(0.025 + 0.01)  # pitch + half a marker
    with pytest.raises(ValueError, match="twice"):
        both.merged(sticker)
