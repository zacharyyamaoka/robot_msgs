"""The base Trajectory: one struct-of-arrays entity, many views, pure time operations."""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from robot_msgs._core import Array, load_npz, save_npz
from robot_msgs.std import Header
from robot_msgs.trajectory.trajectory import Trajectory, TrajectoryPoint, concatenate, sample


def series(n=5):
    t = np.linspace(0.0, 1.0, n)
    return Trajectory(
        t=t,
        channels={"temperature": 20.0 + 10.0 * t, "mode": np.arange(n) % 2},
        header=Header(stamp=100.0, frame_id="lab"),
    )


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class GripperTrajectory(Trajectory):
    """A new domain: declare sample() fields, get everything else."""

    width: Array = sample()
    grasping: Array | None = sample(default=None, interp="hold", dtype=bool)
    gripper_name: str = "2f85"


def gripper():
    return GripperTrajectory(t=[0.0, 1.0, 2.0], width=[0.08, 0.04, 0.0], grasping=[False, False, True])


# ── construction ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "t, message",
    [([0.0, 1.0, 1.0], "strictly increasing"), ([[0.0, 1.0]], "1-D"), ([0.0, np.nan], "finite")],
)
def test_bad_time_axis_fails_at_construction(t, message):
    with pytest.raises(ValueError, match=message):
        Trajectory(t=t)


def test_channel_length_must_match_time():
    with pytest.raises(ValueError, match="has 2 samples, t has 3"):
        Trajectory(t=[0, 1, 2], channels={"x": [1.0, 2.0]})


def test_channel_cannot_shadow_a_field():
    with pytest.raises(ValueError, match="shadow"):
        GripperTrajectory(t=[0.0], width=[0.1], channels={"width": [0.2]})


def test_frozen():
    traj = series()
    with pytest.raises(FrozenInstanceError):
        traj.t = np.zeros(5)
    with pytest.raises(ValueError):
        traj.t[0] = 9.0
    with pytest.raises(TypeError):
        traj.channels["new"] = np.zeros(5)


# ── views ───────────────────────────────────────────────────────────────────


def test_string_index_is_the_batch_view():
    traj = series()
    assert traj["temperature"] is traj.channels["temperature"]
    assert traj["t"] is traj.t
    with pytest.raises(KeyError, match="not a field or channel"):
        traj["missing"]


def test_int_index_is_a_point_and_iteration_yields_every_point():
    traj = series()
    point = traj[2]
    assert isinstance(point, TrajectoryPoint)
    assert point.t == 0.5 and point["temperature"] == 25.0 and point.time_from_start == 0.5
    assert traj[-1].t == 1.0
    assert [p.t for p in traj] == traj.t.tolist()
    with pytest.raises(IndexError):
        traj[5]


def test_slice_is_a_trajectory_over_the_same_memory():
    big = Trajectory(t=np.arange(1_000_000) * 1e-3, channels={"x": np.zeros((1_000_000, 6))})
    part = big[1000:2000]
    assert len(part) == 1000 and type(part) is Trajectory
    assert np.shares_memory(part.t, big.t)
    assert np.shares_memory(part.channels["x"], big.channels["x"])


def test_mask_and_index_arrays_select_samples():
    traj = series()
    assert traj[traj.t > 0.5].t.tolist() == [0.75, 1.0]
    assert traj[[0, 4]].t.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        traj[[4, 0]]


def test_points_round_trip():
    traj = series()
    assert Trajectory.from_points(traj.to_points(), header=traj.header) == traj


# ── interpolation ───────────────────────────────────────────────────────────


def test_at_returns_samples_bit_exact_and_interpolates_between():
    traj = series()
    for index in range(len(traj)):
        assert traj.at(traj.t[index])["temperature"] == traj.channels["temperature"][index]
    mid = traj.at(0.125)
    assert mid["temperature"] == pytest.approx(21.25)
    assert mid["mode"] == 0  # integers hold the previous sample, never blend


def test_out_of_range_raises_unless_clamped():
    traj = series()
    with pytest.raises(ValueError, match="outside"):
        traj.at(1.5)
    assert traj.at(1.5, extrapolate="clamp")["temperature"] == 30.0


def test_resample_on_a_fixed_grid():
    traj = series().resample(dt=0.1)
    assert len(traj) == 11 and traj.is_uniform()
    assert traj.dt == pytest.approx(0.1)
    np.testing.assert_allclose(traj["temperature"], 20.0 + 10.0 * traj.t)


def test_dt_refuses_to_guess_on_a_non_uniform_grid():
    with pytest.raises(ValueError, match="not uniformly spaced"):
        Trajectory(t=[0.0, 0.1, 0.3]).dt


def test_generated_point_type_and_per_field_interpolation():
    traj = gripper()
    point = traj[1]
    assert type(point).__name__ == "GripperTrajectoryPoint"
    assert point.width == 0.04 and point.grasping is False
    halfway = traj.at(1.5)
    assert halfway.width == pytest.approx(0.02)
    assert halfway.grasping is False  # "hold" — the gripper closes at t=2, not before
    assert traj.at(2.0).grasping is True


# ── time operations ─────────────────────────────────────────────────────────


def test_window_shift_and_rebase():
    traj = series().shifted(2.0)
    assert traj.start_time == 2.0
    window = traj.window(2.2, 2.8)
    assert window.t.tolist() == [2.25, 2.5, 2.75]
    rebased = window.rebased()
    assert rebased.t[0] == 0.0
    np.testing.assert_allclose(rebased.absolute_times, window.absolute_times)


def test_channels_add_and_remove_without_mutation():
    traj = series()
    more = traj.with_channel("kp", np.full(5, 40.0))
    assert "kp" in more.channels and "kp" not in traj.channels
    assert more.without_channel("kp") == traj


def test_concatenate_aligns_pieces_through_their_stamps():
    first = Trajectory(t=[0.0, 1.0], channels={"x": [0.0, 1.0]}, header=Header(10.0, "lab"))
    second = Trajectory(t=[0.0, 1.0], channels={"x": [2.0, 3.0]}, header=Header(12.0, "lab"))
    joined = concatenate([first, second])
    assert joined.t.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert joined.channels["x"].tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="different frames"):
        concatenate([first, second.replace(header=Header(12.0, "world"))])


# ── value semantics and files ───────────────────────────────────────────────


def test_dict_and_npz_round_trips(tmp_path):
    for traj in (series(), gripper()):
        assert type(traj).from_dict(traj.to_dict()) == traj
        save_npz(traj, tmp_path / "traj.npz")
        assert load_npz(tmp_path / "traj.npz") == traj


def test_repr_is_a_summary():
    text = repr(series(1000))
    assert text.startswith("Trajectory(N=1000, t=[0 … 1] s")
    assert "float64[1000]" in text and len(text) < 200
