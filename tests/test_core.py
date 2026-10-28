"""The Msg base: read-only arrays, value equality, compact repr, lossless round-trips."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from robot_msgs._core import (
    Array,
    Msg,
    allclose,
    frozen_array,
    load_npz,
    message_types,
    optional_array,
    save_npz,
    set_field,
)
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Sample(Msg):
    values: Array
    labels: tuple[str, ...] = ()
    extra: Array | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        set_field(self, "values", frozen_array(self.values))
        set_field(self, "extra", optional_array(self.extra))
        set_field(self, "labels", tuple(self.labels))


def test_construction_copies_and_freezes_the_callers_array():
    mine = np.arange(4.0)
    msg = Sample(values=mine)
    mine[0] = 99.0
    assert msg.values[0] == 0.0
    with pytest.raises(ValueError):
        msg.values[0] = 1.0


def test_views_of_frozen_arrays_are_not_copied_again():
    msg = Sample(values=np.arange(10.0))
    view = msg.values[2:5]
    assert frozen_array(view) is view
    assert np.shares_memory(Sample(values=view).values, msg.values)


def test_a_callers_read_only_array_is_still_copied():
    theirs = np.arange(3.0)
    theirs.setflags(write=False)
    msg = Sample(values=theirs)
    assert not np.shares_memory(msg.values, theirs)  # they could flip it writeable again


def test_equality_and_hash_are_by_value():
    a = Sample(values=[1.0, 2.0], labels=("x",))
    b = Sample(values=np.array([1.0, 2.0]), labels=["x"])
    assert a == b and hash(a) == hash(b)
    assert a != Sample(values=[1.0, 2.5], labels=("x",))
    assert Sample(values=[np.nan, -0.0]) == Sample(values=[np.nan, 0.0])
    assert hash(Sample(values=[np.nan, -0.0])) == hash(Sample(values=[np.nan, 0.0]))


def test_repr_summarises_big_arrays_and_skips_unset_fields():
    assert repr(Sample(values=np.zeros((500, 6)))) == "Sample(values=float64[500×6])"
    assert "header" not in repr(Sample(values=[1.0]))
    assert "frame_id='map'" in repr(Sample(values=[1.0], header=Header(frame_id="map")))


def test_replace_revalidates_and_never_mutates():
    a = Sample(values=[1.0])
    b = a.replace(values=[2.0, 3.0])
    assert a.values.tolist() == [1.0] and b.values.tolist() == [2.0, 3.0]


def test_dict_round_trip_is_lossless_and_names_the_type():
    msg = Sample(values=np.ones((2, 3)), labels=("a", "b"), header=Header(1.5, "world"))
    data = msg.to_dict()
    assert data["_type"] == "test_core/Sample"
    assert Msg.from_dict.__func__(Sample, data) == msg
    assert Sample.from_dict(data) == msg


def test_npz_round_trip(tmp_path):
    msg = Sample(values=np.random.default_rng(0).normal(size=(50, 7)), extra=[1, 2])
    save_npz(msg, tmp_path / "m.npz")
    assert load_npz(tmp_path / "m.npz") == msg


def test_allclose_tolerates_float_noise_only():
    a = Sample(values=[1.0, 2.0])
    assert allclose(a, Sample(values=[1.0 + 1e-13, 2.0]))
    assert not allclose(a, Sample(values=[1.1, 2.0]))


def test_header_time_helpers():
    h = Header.from_sec_nanosec(1_700_000_000, 123_456_789, "base")
    sec, nanosec = h.to_sec_nanosec()
    assert sec == 1_700_000_000 and abs(nanosec - 123_456_789) < 500
    assert abs(h.stamp_ns - 1_700_000_000_123_456_789) < 500  # float64 keeps ~0.24 µs here
    with pytest.raises(ValueError):
        Header(stamp=float("nan"))


def test_types_register_under_domain_names():
    assert message_types()["std/Header"] is Header
