"""The recorded streams and dataset nouns draw their own data: matplotlib lines are the arrays,
and Rerun's own reader (`rerun rrd print`) finds the frames, series and clouds we logged."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from catalog import (  # noqa: E402
    CLOUD,
    DATASET_COMPARISON,
    IMAGE_TRAJ,
    IMU_TRAJ,
    NAMES,
    RUN,
    SERVO_COMMANDS,
    SERVO_STATES,
    WRENCH_TRAJ,
)

from robot_msgs.compare import compare  # noqa: E402
from robot_msgs.geometry import Transform, Vector3  # noqa: E402
from robot_msgs.viz import plot  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def lines(fig, prefix: str) -> list:
    return [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if (line.get_gid() or "").startswith(prefix)
    ]


def test_servo_state_rows_are_one_line_per_servo_and_faults_are_steps():
    fig = plot(SERVO_STATES)
    for row in ("positions", "temperature", "voltage"):
        drawn = lines(fig, row + ":")
        assert [line.get_gid() for line in drawn] == [f"{row}:{name}" for name in NAMES]
        for column, line in enumerate(drawn):
            np.testing.assert_array_equal(line.get_ydata(), getattr(SERVO_STATES, row)[:, column])
    assert all(line.get_drawstyle() == "steps-post" for line in lines(fig, "fault:"))
    assert [t.get_text() for t in fig.legends[0].get_texts()] == list(NAMES)


def test_servo_command_rows_and_the_latency_channel():
    fig = plot(SERVO_COMMANDS)
    assert len(lines(fig, "stiffness:")) == 3
    (latency,) = lines(fig, "latency:")
    np.testing.assert_array_equal(latency.get_ydata(), SERVO_COMMANDS.channels["latency"])
    assert lines(fig, "damping:") == []  # not set, not drawn


def test_imu_and_wrench_rows_are_xyz():
    fig = plot(IMU_TRAJ)
    np.testing.assert_allclose(lines(fig, "rpy:yaw")[0].get_ydata(), 0.1 * np.arange(6))
    np.testing.assert_array_equal(
        lines(fig, "linear_acceleration:z")[0].get_ydata(), IMU_TRAJ.linear_accelerations[:, 2]
    )
    fig = plot(WRENCH_TRAJ)
    np.testing.assert_array_equal(lines(fig, "force:z")[0].get_ydata(), WRENCH_TRAJ.forces[:, 2])
    assert lines(fig, "saturated:")[0].get_drawstyle() == "steps-post"


def test_an_image_stream_is_a_filmstrip_and_its_error_one_signed_scale():
    fig = plot(IMAGE_TRAJ)
    shown = [ax for ax in fig.axes if (ax.get_gid() or "").startswith("frame:")]
    assert [ax.get_gid() for ax in shown] == ["frame:0", "frame:1", "frame:2"]
    np.testing.assert_array_equal(shown[1].images[0].get_array(), IMAGE_TRAJ.frames[1])
    error = compare(IMAGE_TRAJ, IMAGE_TRAJ.replace(frames=255 - IMAGE_TRAJ.frames)).error
    fig = plot(error)
    images = [ax.images[0] for ax in fig.axes if (ax.get_gid() or "").startswith("frame:")]
    assert len({image.get_clim() for image in images}) == 1  # one scale for every frame
    low, high = images[0].get_clim()
    assert low == -high and images[0].get_cmap().name == "RdBu_r"
    assert len(lines(fig, "mae:")) == 1


# ── Rerun ────────────────────────────────────────────────────────────────────


def test_rerun_logs_frames_on_the_clock_and_a_cloud_error_as_points(tmp_path):
    pytest.importorskip("rerun")
    from test_viz_rerun import chunks

    from robot_msgs.viz import rerun as viz_rerun

    shifted = CLOUD.transformed(Transform(Vector3(0.01, 0.0, 0.0)))
    found = chunks(
        Path(
            viz_rerun.save(
                str(tmp_path / "s.rrd"),
                camera=IMAGE_TRAJ,
                camera_error=compare(IMAGE_TRAJ, IMAGE_TRAJ.replace(frames=IMAGE_TRAJ.frames // 2)),
                scan_error=compare(CLOUD, shifted),
                servo=SERVO_STATES,
                imu=IMU_TRAJ,
                run=RUN,
                sweep=DATASET_COMPARISON,
            )
        )
    )
    rows, columns = found["/image/camera"]
    assert "Image:buffer" in columns and rows >= len(IMAGE_TRAJ)
    assert "Image:buffer" in found["/image/camera_error.error"][1]
    assert "/error/channels/mae/0/camera_error" in found  # the per-frame series, not one per pixel
    assert not any(entity.startswith("/frames/") for entity in found)
    assert "Points3D:positions" in found["/world/error/scan_error"][1]
    assert "/positions/j2/servo" in found  # servo columns are named by joint
    assert "Transform3D:quaternion" in found["/world/imu/frame"][1]
    assert "/positions/j1/run.sim.joints" in found and "/positions/j1/run.real.joints" in found
    assert "TextDocument:text" in found["/notes/sweep"][1]
