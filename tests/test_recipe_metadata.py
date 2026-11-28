"""Recipe metadata on the primitives (2026-11-27): `call` (the helper call as written), `label` (`step`), `outline`.

The metadata is for logs and viewers; the last test is the one that matters: it never changes what runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_msgs.control import (
    Command,
    Condition,
    Delay,
    InCollision,
    JointCommand,
    JointCommandController,
    JointCommandSequence,
    Reached,
    ServoState,
)
from robot_msgs.control.helpers import (
    at_once,
    call_text,
    find_hardstop,
    find_hardstop_and_hold,
    hold,
    move_by,
    move_to,
    outline,
    step,
    stop_all,
    wait,
    wait_until,
)


def test_every_helper_stamps_its_call_as_written() -> None:
    primitives = find_hardstop_and_hold(
        "elbow",
        direction=-1,
        max_vel=0.3,
        max_effort=3.0,
        max_travel=1.2,
        repeats=3,
        define=math.radians(-135),
    )
    assert len(primitives) == 11
    # the outermost helper wins (find_hardstop built them), keywords stay keywords, floats to 6 significant digits
    assert {p.call for p in primitives} == {
        'find_hardstop_and_hold("elbow", direction=-1, max_vel=0.3, max_effort=3.0, max_travel=1.2, repeats=3, '
        "define=-2.35619)"
    }
    assert hold("shoulder", 12.0)[0].call == 'hold("shoulder", 12.0)'
    assert stop_all()[0].call == "stop_all()"
    assert wait(0.5)[0].call == "wait(0.5)"


def test_call_text_leaves_out_the_name_and_arguments_at_their_default() -> None:
    # how a translator calls it: everything passed, most of it the default
    primitives = find_hardstop(
        "elbow",
        1,
        0.3,
        40.0,
        0.5,
        timeout=None,
        repeats=1,
        backoff=0.035,
        probe_speed=None,
        define=1.65,
        tolerance=0.001,
        feedforward_effort=0.0,
        name="FindHardstop",
    )
    assert primitives[0].call == 'find_hardstop("elbow", 1, 0.3, 40.0, 0.5, define=1.65)'
    assert primitives[0].name == "FindHardstop"
    # a trailing positional argument at its default is dropped, one before a written argument is kept
    assert hold("elbow", 40.0, 0.0, name="Hold")[0].call == 'hold("elbow", 40.0)'
    assert (
        move_to("elbow", 1.0, 0.5, 8.0, 0.0, tolerance=0.02)[0].call
        == 'move_to("elbow", 1.0, 0.5, 8.0, tolerance=0.02)'
    )
    assert (
        call_text(move_by, ("elbow", 0.1, 0.5, 8.0, 1.5), {})
        == 'move_by("elbow", 0.1, 0.5, 8.0, 1.5)'
    )


def test_predicates_and_nested_helpers_read_as_written() -> None:
    waiting = wait_until(
        InCollision("elbow"), Reached(joints=("elbow",), tolerance=0.01), timeout=2.0
    )
    assert (
        waiting[0].call
        == 'wait_until(InCollision("elbow"), Reached(("elbow",), 0.01), timeout=2.0)'
    )
    together = at_once(move_to("elbow", -1.0, 1.0, 8.0), move_to("shoulder", 0.0, 1.0, 8.0))
    assert {p.call for p in together} == {
        'at_once(move_to("elbow", -1.0, 1.0, 8.0), move_to("shoulder", 0.0, 1.0, 8.0))'
    }
    assert [type(p) for p in together] == [Command, Command, Condition]


def test_step_labels_a_stretch_and_keeps_calls_and_names() -> None:
    bare = [
        *find_hardstop("elbow", -1, 0.3, 3.0, 1.2, define=-2.356),
        *move_by("elbow", 0.035, 0.3, 3.0),
    ]
    labelled = step(
        "Home the elbow",
        find_hardstop("elbow", -1, 0.3, 3.0, 1.2, define=-2.356),
        move_by("elbow", 0.035, 0.3, 3.0),
    )
    assert [p.label for p in labelled] == ["Home the elbow"] * len(bare)
    assert [(p.call, p.name) for p in labelled] == [(p.call, p.name) for p in bare]
    assert [p.replace(label="") for p in labelled] == bare
    assert step("Wait", Delay(0.2))[0] == Delay(0.2, label="Wait")
    with pytest.raises(ValueError, match="needs a label"):
        step("", wait(0.1))
    with pytest.raises(TypeError, match="not a Command, Delay or Condition"):
        step("x", [JointCommand.limp("elbow")])


def test_outline_reads_the_recipe_back_as_label_call_primitives() -> None:
    recipe = JointCommandSequence(name="home elbow, then park", steps=[
        step("Hold the shoulder still", hold("shoulder", max_effort=12.0)),
        step("Home the elbow", find_hardstop("elbow", -1, 0.3, 3.0, 1.2, repeats=2, define=-2.356),
             move_by("elbow", 0.035, 0.3, 3.0)),
        move_to("elbow", -1.0, 1.0, 8.0),
        Condition((Reached(joints=("elbow",), tolerance=0.02),), name="settle"),
        Delay(0.1), Delay(0.1),
        stop_all(),
    ])  # fmt: skip
    rows = outline(recipe)
    assert [
        (row["label"], row["call"].split("(")[0], row["first"], row["last"]) for row in rows
    ] == [
        ("Hold the shoulder still", "hold", 0, 0),
        ("Home the elbow", "find_hardstop", 1, 7),
        ("Home the elbow", "move_by", 8, 9),
        ("", "move_to", 10, 11),
        ("", "", 12, 12),  # a primitive written directly is its own row
        ("", "", 13, 13),
        ("", "", 14, 14),
        ("", "stop_all", 15, 15),
    ]
    assert rows[1]["kinds"] == [
        "Command",
        "Condition",
        "Command",
        "Condition",
        "Command",
        "Condition",
        "Command",
    ]
    assert rows[4]["name"] == "settle"
    assert outline(recipe.primitives) == rows
    # the metadata survives the dict round trip a log or another process reads
    again = JointCommandSequence.from_dict(recipe.to_dict())
    assert outline(again) == rows


def _run(recipe: JointCommandSequence, ticks: int = 400) -> list[tuple]:
    """Drive a one-joint ideal servo (it goes where it is told) and keep every wire command."""
    controller = JointCommandController(("elbow",))
    controller.submit(recipe, client="test")
    position, sent = 0.0, []
    for _ in range(ticks):
        state = ServoState(name=("elbow",), position=[position], velocity=[0.0], effort=[0.0])
        tick = controller.tick(state, 0.01)
        command = tick.command
        sent.append(
            (
                tick.modes,
                tuple(np.asarray(command.position).tolist()),
                tuple(np.asarray(command.max_effort).tolist()),
            )
        )
        target = float(np.asarray(command.position)[0])
        if math.isfinite(target):
            position = target
    return sent


def test_metadata_never_changes_what_runs() -> None:
    def body() -> list:
        return [
            *move_to("elbow", 0.5, 1.0, 8.0),
            *wait(0.2),
            *move_by("elbow", -0.2, 1.0, 8.0),
            *hold("elbow", 4.0),
        ]

    bare = JointCommandSequence(steps=[p.replace(call="") for p in body()])
    labelled = JointCommandSequence(steps=[step("Out and back", body())])
    assert {p.call for p in bare.primitives} == {""}
    assert {p.label for p in labelled.primitives} == {"Out and back"}
    assert all(p.call for p in labelled.primitives)
    wire_bare, wire_labelled = _run(bare), _run(labelled)
    assert len(wire_bare) == len(wire_labelled) == 400
    for a, b in zip(wire_bare, wire_labelled, strict=True):
        assert a[0] == b[0]
        np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
        np.testing.assert_array_equal(np.asarray(a[2]), np.asarray(b[2]))
