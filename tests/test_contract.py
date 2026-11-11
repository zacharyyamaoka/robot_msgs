"""Contract every message type keeps: frozen value semantics, lossless round-trips, numpy only."""

import copy
import dataclasses
import json
import pickle
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from catalog import EXAMPLES

import robot_msgs
from robot_msgs._core import load_npz, save_npz

LIBRARY_TYPES = {
    name: cls
    for name, cls in robot_msgs.message_types().items()
    if cls.__module__.startswith("robot_msgs.")
}
IDS = [type(example).__name__ for example in EXAMPLES]


def test_every_type_has_an_example():
    covered = {type(example).type_name for example in EXAMPLES}
    assert set(LIBRARY_TYPES) - covered == set()


@pytest.mark.parametrize("cls", LIBRARY_TYPES.values(), ids=list(LIBRARY_TYPES))
def test_every_type_is_a_frozen_value_with_the_shared_eq_and_repr(cls):
    params = cls.__dataclass_params__
    assert params.frozen and not params.eq and not params.repr, cls


@pytest.mark.parametrize("message", EXAMPLES, ids=IDS)
def test_round_trips_are_lossless(message, tmp_path):
    assert pickle.loads(pickle.dumps(message)) == message
    assert copy.deepcopy(message) == message
    assert type(message).from_dict(json.loads(json.dumps(message.to_dict()))) == message
    save_npz(message, tmp_path / "message.npz")
    assert load_npz(tmp_path / "message.npz") == message


@pytest.mark.parametrize("message", EXAMPLES, ids=IDS)
def test_values_are_hashable_immutable_and_print_short(message):
    assert hash(message) == hash(copy.deepcopy(message))
    field = dataclasses.fields(message)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(message, field.name, getattr(message, field.name))
    holds_big_array = any(
        isinstance(value := getattr(message, f.name), np.ndarray) and value.size > 8
        for f in dataclasses.fields(message)
    )
    if holds_big_array:
        assert len(repr(message)) < 400  # arrays print as float64[500×6], never their values


def test_runtime_dependencies_are_numpy_and_dataclass_values():
    # WHY dataclass-values: the stdlib-only Value/Space layer shared with py_dataclass_viewer
    # (Zach, 2026-11-10)
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["dependencies"] == ["numpy>=1.26", "dataclass-values>=0.1.0"]
    probe = (
        "import sys, robot_msgs; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'scipy', 'matplotlib', 'rosbags', 'pandas', 'torch', 'transforms3d'}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(robot_msgs.__file__).parents[1])},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
