"""LogRecord, DiagnosticRecord and Metrics: what a running component reports about itself.

The golden tests pin each type's fields (names, order, annotated types) and its exact JSON. WHY:
like_keras derives the cross-process layout from the dataclass fields in order (its contract §7.1),
so reordering or retyping a field is a wire break, and these tests are the place it shows.
"""

import dataclasses
import json
import logging
import typing

import numpy as np
import pytest

from robot_msgs import from_dict
from robot_msgs._core import Array
from robot_msgs.std import DiagnosticRecord, Header, LogRecord, Metrics

STAMP = Header(stamp=1_761_600_000.5)


def layout(cls: type) -> list[tuple[str, object]]:
    hints = typing.get_type_hints(cls)
    return [(f.name, hints[f.name]) for f in dataclasses.fields(cls)]


# ── golden: the fields, in order, with their types ──────────────────────────


def test_log_record_layout_is_the_contract():
    assert layout(LogRecord) == [
        ("header", Header),
        ("path", str),
        ("tick", int),
        ("level", int),
        ("message", str),
    ]


def test_diagnostic_record_layout_is_the_contract():
    assert layout(DiagnosticRecord) == [
        ("header", Header),
        ("path", str),
        ("level", str),
        ("code", str),
        ("message", str),
        ("mode", str),
        ("expected_period_s", float),
        ("uptime_s", float),
        ("stale_inputs", tuple[str, ...]),
    ]
    defaults = {f.name: f.default for f in dataclasses.fields(DiagnosticRecord)}
    assert defaults["level"] == "unknown" and defaults["mode"] == "initialization"
    assert (defaults["code"], defaults["message"], defaults["stale_inputs"]) == ("", "", ())
    assert (defaults["expected_period_s"], defaults["uptime_s"]) == (1.0, 0.0)


def test_metrics_layout_is_the_contract():
    assert layout(Metrics) == [
        ("header", Header),
        ("path", str),
        ("names", tuple[str, ...]),
        ("values", Array),
    ]


def test_every_field_is_keyword_only_and_header_and_path_are_required():
    for cls in (LogRecord, DiagnosticRecord, Metrics):
        assert all(f.kw_only for f in dataclasses.fields(cls)), cls
        required = [
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        assert required[:2] == ["header", "path"], cls


# ── golden: the exact JSON ───────────────────────────────────────────────────

HEADER_JSON = {"_type": "std/Header", "stamp": 1761600000.5, "frame_id": "", "meta": {}}


def test_log_record_json_is_golden():
    record = LogRecord(header=STAMP, path="cell.arm", tick=7, level=30, message="slow: 12 ms")
    assert record.to_dict() == {
        "_type": "std/LogRecord",
        "header": HEADER_JSON,
        "path": "cell.arm",
        "tick": 7,
        "level": 30,
        "message": "slow: 12 ms",
    }
    assert from_dict(json.loads(json.dumps(record.to_dict()))) == record


def test_diagnostic_record_json_is_golden():
    record = DiagnosticRecord(header=STAMP, path="cell.arm")
    assert record.to_dict() == {
        "_type": "std/DiagnosticRecord",
        "header": HEADER_JSON,
        "path": "cell.arm",
        "level": "unknown",
        "code": "",
        "message": "",
        "mode": "initialization",
        "expected_period_s": 1.0,
        "uptime_s": 0.0,
        "stale_inputs": [],
    }
    assert from_dict(json.loads(json.dumps(record.to_dict()))) == record


def test_metrics_json_is_golden():
    record = Metrics(header=STAMP, path="cell.arm", names=("steps", "step_p99_s"), values=[3, 0.5])
    assert record.to_dict() == {
        "_type": "std/Metrics",
        "header": HEADER_JSON,
        "path": "cell.arm",
        "names": ["steps", "step_p99_s"],
        "values": {"dtype": "<f8", "shape": [2], "data": [3.0, 0.5]},
    }
    assert from_dict(json.loads(json.dumps(record.to_dict()))) == record


# ── LogRecord ────────────────────────────────────────────────────────────────


def test_log_record_keeps_the_stdlib_level_and_names_it():
    record = LogRecord(header=STAMP, path="a", tick=np.int64(3), level=logging.WARNING, message="m")
    assert type(record.tick) is int and record.tick == 3
    assert record.level_name == "WARNING"
    assert LogRecord(header=STAMP, path="a", tick=0, level=25, message="").level_name == "Level 25"


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"tick": -1}, ValueError, "LogRecord.tick must be >= 0"),
        ({"tick": 1.5}, TypeError, "LogRecord.tick must be a whole number"),
        ({"level": True}, TypeError, "LogRecord.level must be a whole number"),
        ({"level": "INFO"}, TypeError, "LogRecord.level must be a whole number"),
        ({"header": 12.0}, TypeError, "LogRecord.header must be a Header"),
    ],
)
def test_log_record_refuses(changes, error, match):
    fields = {"header": STAMP, "path": "a", "tick": 0, "level": 20, "message": "m"}
    with pytest.raises(error, match=match):
        LogRecord(**{**fields, **changes})


# ── DiagnosticRecord ─────────────────────────────────────────────────────────


def test_the_zero_value_is_unknown_never_green():
    record = DiagnosticRecord(header=STAMP, path="cell.arm")
    assert (record.level, record.mode, record.code) == ("unknown", "initialization", "")


def test_stale_after_follows_io_diagnostics():
    # max(3 x expected period, 5 s floor)
    assert DiagnosticRecord(header=STAMP, path="a").stale_after_s == 5.0
    assert DiagnosticRecord(header=STAMP, path="a", expected_period_s=2.5).stale_after_s == 7.5


def test_an_unknown_record_may_say_why():
    record = DiagnosticRecord(header=STAMP, path="a", code="PortBusy")
    assert record.level == "unknown" and record.code == "PortBusy"


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"level": "stale"}, ValueError, "'stale' is never sent"),
        ({"level": "WARN"}, ValueError, "level must be one of"),
        ({"level": "warn"}, ValueError, "code is required when level is 'warn'"),
        ({"level": "error"}, ValueError, "code is required when level is 'error'"),
        ({"level": "error", "code": "input stale"}, ValueError, "must be PascalCase"),
        ({"code": "inputStale"}, ValueError, "must be PascalCase"),
        ({"code": "A" * 65}, ValueError, "at most 64 characters"),
        ({"mode": "running"}, ValueError, "mode must be one of"),
        ({"expected_period_s": 0.0}, ValueError, "expected_period_s must be a finite number"),
        ({"expected_period_s": float("inf")}, ValueError, "expected_period_s must be a finite"),
        ({"uptime_s": -1.0}, ValueError, "uptime_s must be a finite number of seconds >= 0"),
        ({"stale_inputs": "state"}, TypeError, "stale_inputs is a tuple of port names"),
        ({"header": None}, TypeError, "DiagnosticRecord.header must be a Header"),
    ],
)
def test_diagnostic_record_refuses(changes, error, match):
    with pytest.raises(error, match=match):
        DiagnosticRecord(**{"header": STAMP, "path": "a", **changes})


def test_codes_take_the_io_diagnostics_shape():
    for code in ("InputStale", "Overrun", "MoteusFault.33", "Exception", "A" * 64):
        assert DiagnosticRecord(header=STAMP, path="a", level="error", code=code).code == code


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_metrics_by_name_and_back():
    record = Metrics.from_mapping(
        {"steps": 1000, "overruns": 2, "step_p99_s": float("nan")}, header=STAMP, path="cell.arm"
    )
    assert record.names == ("steps", "overruns", "step_p99_s")
    assert record.values.dtype == np.float64 and not record.values.flags.writeable
    assert len(record) == 3 and record["overruns"] == 2.0 and type(record["steps"]) is float
    assert np.isnan(record["step_p99_s"])  # not measured this period, never a made-up 0
    assert list(record.as_dict()) == ["steps", "overruns", "step_p99_s"]
    with pytest.raises(KeyError, match="no metric 'drops'"):
        record["drops"]


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"names": ("a", "a"), "values": [1, 2]}, ValueError, "duplicate names: \\['a'\\]"),
        ({"names": ("a", "b"), "values": [1.0]}, ValueError, "one number per name: 2 names"),
        ({"names": ("a",), "values": [[1.0]]}, ValueError, "one number per name"),
        ({"names": "ab", "values": [1.0, 2.0]}, TypeError, "names is a tuple of names"),
    ],
)
def test_metrics_refuses(changes, error, match):
    with pytest.raises(error, match=match):
        Metrics(**{"header": STAMP, "path": "a", **changes})
