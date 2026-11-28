"""`robot_msgs.control.recipe`: short ways to WRITE a recipe (2026-11-28).

The test that matters is the first: one reference recipe written five ways is the same primitives as the one
written with today's helpers, once `call` (the step as written, the only thing a style changes) is blanked.
"""

from __future__ import annotations

import inspect
import math

import pytest

from robot_msgs.control import (
    Command,
    Condition,
    JointCommand,
    JointCommandSequence,
    helpers,
    recipe,
)
from robot_msgs.control.recipe import (
    ALL,
    VERBS,
    VOCABULARY,
    Group,
    Joint,
    deg,
    find_hardstop,
    hold,
    move_by,
    move_to,
    outline,
    script,
    signature,
    step,
    stop,
    wait,
)

NAME = "home the elbow, then park"
HOLD = "Hold the shoulder against gravity"
FIND = "Find the elbow's lower stop and call it -135°"
PARK = "Park 20° off the stop"


def baseline() -> JointCommandSequence:
    """The reference recipe, written with today's helpers and their positional numbers."""
    return JointCommandSequence(
        name=NAME,
        steps=[
            helpers.step(HOLD, helpers.hold("shoulder", 12.0, 5.0)),
            helpers.step(
                FIND,
                helpers.find_hardstop_and_hold(
                    "elbow", -1, 0.5, 3.0, 1.0, define=math.radians(-135)
                ),
            ),
            helpers.step(
                PARK, helpers.move_by("elbow", math.radians(20), 0.5, 8.0), helpers.wait(0.3)
            ),
            helpers.stop("shoulder", "elbow"),
        ],
    )


def named_joints_and_short_keywords() -> list:
    J1, J2 = "shoulder", "elbow"
    return [
        step(HOLD, hold(J1, max_tau=12, ff=5)),
        step(FIND, find_hardstop(J2, dir=-1, vel=0.5, max_tau=3, travel=1.0, define=deg(-135))),
        step(PARK, move_by(J2, deg(20), vel=0.5, max_tau=8), wait(0.3)),
        stop(J1, J2),
    ]


def joint_objects() -> list:
    J1 = Joint("shoulder", max_tau=12, ff=5)
    J2 = Joint("elbow", vel=0.5, max_tau=3)
    Arm = Group(J1, J2, name="Arm")
    return [
        step(HOLD, J1.hold()),
        step(FIND, J2.find_hardstop(dir=-1, travel=1.0, define=deg(-135))),
        step(PARK, J2.move_by(deg(20), max_tau=8), wait(0.3)),
        Arm.stop(),
    ]


def verbs_over_joints_and_groups() -> list:
    J1 = Joint("shoulder", max_tau=12, ff=5)
    J2 = Joint("elbow", vel=0.5, max_tau=3)
    Arm = Group(J1, J2, name="Arm")
    return [
        step(HOLD, hold(J1)),
        step(FIND, find_hardstop(J2, dir=-1, travel=1.0, define=deg(-135))),
        step(PARK, move_by(J2, deg(20), max_tau=8), wait(0.3)),
        stop(Arm),
    ]


def indexed_addressing() -> list:
    arm = Group(
        Joint("shoulder", max_tau=12, ff=5, id=11),
        Joint("elbow", vel=0.5, max_tau=3, id=12),
        name="arm",
    )
    return [
        step(HOLD, arm["shoulder"].hold()),
        step(FIND, arm[12].find_hardstop(dir=-1, travel=1.0, define=deg(-135))),
        step(PARK, arm["elbow"].move_by(deg(20), max_tau=8), wait(0.3)),
        arm[ALL].stop(),
    ]


def text_lines() -> list:
    return script(
        """
        Hold the shoulder against gravity:
            shoulder> hold max_tau=12 ff=5
        Find the elbow's lower stop and call it -135°:
            elbow> find_hardstop dir=-1 vel=0.5 max_tau=3 travel=1.0 define=-135deg
        Park 20° off the stop:
            elbow> move_by 20deg vel=0.5 max_tau=8      # the first value may drop its keyword
            wait 0.3
        A> stop
        """,
        joints=["shoulder", "elbow"],
    )


STYLES = {
    "named joints + short keywords": named_joints_and_short_keywords,
    "joint objects": joint_objects,
    "verbs over joints and groups": verbs_over_joints_and_groups,
    "indexed addressing": indexed_addressing,
    "text lines": text_lines,
}


def blank(primitives) -> list:
    return [p.replace(call="") for p in primitives]


@pytest.mark.parametrize("style", STYLES)
def test_every_style_is_the_same_primitives_as_the_baseline(style: str) -> None:
    written = JointCommandSequence(name=NAME, steps=STYLES[style]())
    reference = baseline()
    assert len(written.primitives) == len(reference.primitives) == 8
    assert blank(written.primitives) == blank(reference.primitives)
    # and it IS the same message: only the call column differs
    assert written != reference
    assert [p.label for p in written.primitives] == [p.label for p in reference.primitives]


def test_call_is_the_step_as_written_in_each_style() -> None:
    def calls(steps) -> list[str]:
        return [row["call"] for row in outline(JointCommandSequence(name=NAME, steps=steps))]

    assert calls(named_joints_and_short_keywords()) == [
        'hold("shoulder", max_tau=12, ff=5)',
        'find_hardstop("elbow", dir=-1, vel=0.5, max_tau=3, travel=1.0, define=-2.35619)',
        'move_by("elbow", 0.349066, vel=0.5, max_tau=8)',
        "wait(0.3)",
        'stop("shoulder", "elbow")',
    ]
    assert calls(joint_objects()) == [
        "shoulder.hold()",
        "elbow.find_hardstop(dir=-1, travel=1.0, define=-2.35619)",
        "elbow.move_by(0.349066, max_tau=8)",
        "wait(0.3)",
        "Arm.stop()",
    ]
    assert calls(verbs_over_joints_and_groups()) == [
        "hold(shoulder)",
        "find_hardstop(elbow, dir=-1, travel=1.0, define=-2.35619)",
        "move_by(elbow, 0.349066, max_tau=8)",
        "wait(0.3)",
        "stop(Arm)",
    ]
    assert calls(indexed_addressing()) == [
        'arm["shoulder"].hold()',
        "arm[12].find_hardstop(dir=-1, travel=1.0, define=-2.35619)",
        'arm["elbow"].move_by(0.349066, max_tau=8)',
        "wait(0.3)",
        "arm[ALL].stop()",
    ]
    # a text line is its own call, without its comment
    assert calls(text_lines()) == [
        "shoulder> hold max_tau=12 ff=5",
        "elbow> find_hardstop dir=-1 vel=0.5 max_tau=3 travel=1.0 define=-135deg",
        "elbow> move_by 20deg vel=0.5 max_tau=8",
        "wait 0.3",
        "A> stop",
    ]
    # the labels are helpers.step's, on every style
    assert [row["label"] for row in outline(text_lines())] == [HOLD, FIND, PARK, PARK, ""]


def test_a_joint_declares_its_defaults_once_and_a_step_overrides_them() -> None:
    elbow = Joint("elbow", vel=0.5, max_tau=3, ff=0.2)
    command = elbow.move_by(deg(20))[0].commands[0]
    assert (command.speed, command.max_effort, command.feedforward_effort) == (0.5, 3.0, 0.2)
    command = elbow.move_by(deg(20), vel=1.5, max_tau=8, ff=0.0)[0].commands[0]
    assert (command.speed, command.max_effort, command.feedforward_effort) == (1.5, 8.0, 0.0)
    # the value may be written by its keyword, and the two read differently in `call` only
    assert blank(elbow.move_by(by=deg(20))) == blank(elbow.move_by(deg(20)))
    assert elbow.move_by(by=0.5)[0].call == "elbow.move_by(by=0.5)"
    assert blank(move_to(elbow, pos=1.0)) == blank(helpers.move_to("elbow", 1.0, 0.5, 3.0, 0.2))
    # a keyword the joint has no default for falls to the helper's own
    assert elbow.move_to(1.0)[1].until[0].tolerance == helpers.ONE_DEGREE
    assert elbow.move_to(1.0, tol=0.1, timeout=4.0)[1] == helpers.move_to(
        "elbow", 1.0, 0.5, 3.0, 0.2, tolerance=0.1, timeout=4.0
    )[1].replace(call="elbow.move_to(1.0, tol=0.1, timeout=4.0)")
    # name= is the step name, and stays out of the call
    named = elbow.hold(name="rest")[0]
    assert (named.name, named.call) == ("rest", "elbow.hold()")


def test_a_missing_value_names_the_joint_and_the_keyword() -> None:
    with pytest.raises(ValueError, match=r"hold shoulder: needs max_tau= \(effort cap, N·m\)"):
        hold("shoulder")
    with pytest.raises(
        ValueError, match=r'move_by elbow: needs vel=.*Joint\("elbow", vel=\.\.\.\)'
    ):
        Joint("elbow", max_tau=3).move_by(deg(20))
    with pytest.raises(ValueError, match=r"move_by elbow: needs by= \(distance"):
        Joint("elbow", vel=0.5, max_tau=3).move_by()
    with pytest.raises(ValueError, match="find_hardstop elbow: needs dir="):
        Joint("elbow", vel=0.5, max_tau=3).find_hardstop(travel=1.0)
    with pytest.raises(ValueError, match=r"find_hardstop elbow: give travel= \(rad\) or timeout="):
        Joint("elbow", vel=0.5, max_tau=3).find_hardstop(dir=1)
    # a joint's default speed is not a velocity command's value: its sign is the direction
    with pytest.raises(ValueError, match="velocity elbow: needs vel="):
        Joint("elbow", vel=0.5, max_tau=3).velocity()
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_effort'"):
        hold("shoulder", max_effort=12)


def test_every_verb_maps_one_to_one_onto_its_helper() -> None:
    elbow = Joint("elbow", vel=0.5, max_tau=3)
    assert blank(elbow.limp()) == blank(helpers.limp("elbow"))
    assert blank(elbow.brake()) == blank(helpers.brake("elbow"))
    assert blank(elbow.velocity(-0.2)) == blank(helpers.velocity("elbow", -0.2, 3.0))
    assert blank(elbow.effort(1.5)) == blank(helpers.effort("elbow", 1.5, 3.0))
    assert blank(recipe.effort("elbow", tau=1.5)) == blank(helpers.effort("elbow", 1.5))
    assert blank(elbow.define(0.0)) == blank(helpers.define("elbow", 0.0, 3.0))
    assert blank(recipe.define("elbow", pos=0.0)) == blank(helpers.define("elbow", 0.0))
    assert blank(
        elbow.find_hardstop(dir=1, timeout=4.0, repeats=3, probe_vel=0.1, contact_tau=2.0)
    ) == blank(
        helpers.find_hardstop(
            "elbow", 1, 0.5, 3.0, timeout=4.0, repeats=3, probe_speed=0.1, contact_effort=2.0
        )
    )
    assert blank(elbow.verify_hardstop(repeats=3, dir=-1, travel=1.0, expected=-2.3)) == blank(
        helpers.verify_hard_stop("elbow", 3, -1, 0.5, 3.0, 1.0, expected=-2.3)
    )
    # one vocabulary: every verb is a function, a method on Joint and Group, and documented by `signature`
    for verb, spec in VERBS.items():
        function, method = getattr(recipe, verb), getattr(Joint, verb)
        keywords = set(spec.names) | {"name"}
        assert set(inspect.signature(method).parameters) - {"self"} == keywords, verb
        assert set(inspect.signature(function).parameters) - {"target", "targets"} == keywords, verb
        assert getattr(Group, verb) is method
        assert [row[0] for row in signature(verb)] == list(spec.names)
        assert set(spec.names) <= set(VOCABULARY)
    assert signature("move_by")[0] == (
        "by",
        "distance from where the joint starts, rad (signed)",
        "required",
    )
    assert signature("hold") == [
        ("max_tau", "effort cap, N·m", "the Joint's max_tau, else required"),
        ("ff", "constant feedforward torque, N·m", "the Joint's ff, else 0.0"),
    ]


def test_a_group_broadcast_is_one_command() -> None:
    J1 = Joint("shoulder", max_tau=12, ff=5)
    J2 = Joint("elbow", vel=0.5, max_tau=3)
    Arm = Group(J1, J2, name="Arm")
    for verb in ("stop", "limp", "brake"):
        (command,) = getattr(Arm, verb)()
        assert isinstance(command, Command) and command.joints == ("shoulder", "elbow")
        assert [command.replace(call="")] == blank(getattr(helpers, verb)("shoulder", "elbow"))
        assert command.call == f"Arm.{verb}()"
    # hold: ONE Command, each joint under its own defaults; a keyword on the step applies to every joint
    (command,) = Arm.hold()
    assert command.commands == (
        JointCommand.hold("shoulder", max_effort=12.0, feedforward_effort=5.0),
        JointCommand.hold("elbow", max_effort=3.0),
    )
    assert (command.name, command.call) == ("hold", "Arm.hold()")
    assert [c.max_effort for c in Arm.hold(max_tau=2)[0].commands] == [2.0, 2.0]
    assert blank(hold(Arm)) == blank(hold(J1, J2)) == blank(Arm.hold())
    assert hold(J1, J2)[0].call == "hold(shoulder, elbow)"
    # a motion verb on a group runs the joints together and waits for all: helpers.at_once
    together = Arm.move_by(deg(10), vel=1.0)
    assert blank(together) == blank(
        helpers.at_once(
            helpers.move_by("shoulder", deg(10), 1.0, 12.0, 5.0),
            helpers.move_by("elbow", deg(10), 1.0, 3.0),
        )
    )
    assert [type(p) for p in together] == [Command, Command, Condition]
    assert {p.call for p in together} == {"Arm.move_by(0.174533, vel=1.0)"}
    assert blank(move_by([J1, J2], deg(10), vel=1.0)) == blank(together)
    assert (
        move_by([J1, J2], deg(10), vel=1.0)[0].call
        == "move_by([shoulder, elbow], 0.174533, vel=1.0)"
    )
    # a hard-stop find is one joint at a time
    with pytest.raises(ValueError, match="find_hardstop takes one joint.*got shoulder, elbow"):
        Arm.find_hardstop(dir=-1, vel=0.5, travel=1.0)


def test_a_group_addresses_all_a_set_or_one() -> None:
    J1 = Joint("shoulder", max_tau=12, id=11)
    J2 = Joint("elbow", max_tau=3, id=12)
    J3 = Joint("wrist", max_tau=1, id=13)
    Arm = Group(J1, J2, J3, name="Arm")
    assert (
        Arm.names == ("shoulder", "elbow", "wrist") and len(Arm) == 3 and list(Arm) == [J1, J2, J3]
    )
    # all
    assert Arm[ALL].joints == Arm.all.joints == Arm.joints
    assert (Arm[ALL].stop()[0].call, Arm.all.stop()[0].call) == (
        "Arm[ALL].stop()",
        "Arm.all.stop()",
    )
    # a set: by Joint, by name, by id
    for picked, text in (
        (Arm[J1, J2], "Arm[shoulder, elbow]"),
        (Arm["shoulder", "elbow"], 'Arm["shoulder", "elbow"]'),
        (Arm[11, 12], "Arm[11, 12]"),
    ):
        (command,) = picked.stop()
        assert command.joints == ("shoulder", "elbow")
        assert command.call == f"{text}.stop()"
    # one: it reads as addressed, and expands as the joint itself does
    assert blank(Arm["elbow"].hold()) == blank(Arm[12].hold()) == blank(J2.hold())
    assert Arm[13].hold()[0].call == "Arm[13].hold()"
    # a selection is a Group: address inside it again
    assert Arm[11, 12]["elbow"].joints == (J2,)
    # groups nest, and a verb over several targets names each joint once
    Both = Group(Group(J1, J2, name="Upper"), "gripper", name="Both")
    assert Both.names == ("shoulder", "elbow", "gripper")
    assert stop(Arm, J1, "gripper")[0].joints == ("shoulder", "elbow", "wrist", "gripper")
    with pytest.raises(LookupError, match=r'Arm has no joint "hand"; it holds shoulder \(id 11\)'):
        Arm["hand"]
    with pytest.raises(LookupError, match="Arm has no id 14"):
        Arm[14]
    with pytest.raises(LookupError, match="ALL stands alone"):
        Arm[ALL, "elbow"]
    with pytest.raises(ValueError, match="two joints share an id"):
        Group(Joint("a", id=1), Joint("b", id=1))
    with pytest.raises(TypeError, match=r"an id \(11\) is addressed through its Group: Arm\[11\]"):
        stop(11)
    # with no group, ALL is every joint the controller drives
    assert stop(ALL)[0].commands == (JointCommand.stop(),)
    assert stop(ALL)[0].call == "stop(ALL)"


def test_text_lines_address_all_a_set_or_one_by_name_or_id() -> None:
    Arm = Group(
        Joint("shoulder", vel=1.0, max_tau=12, id=11), Joint("elbow", vel=0.5, max_tau=3, id=12)
    )
    assert blank(script("A> stop", joints=Arm)) == blank(Arm.stop())
    assert blank(script("11,12> hold", joints=Arm)) == blank(Arm.hold())
    assert blank(script("shoulder, elbow> hold", joints=Arm)) == blank(Arm.hold())
    assert blank(script("12> move_to pos=90deg", joints=Arm)) == blank(Arm[12].move_to(deg(90)))
    assert blank(script("A> move_by 10deg", joints=Arm)) == blank(Arm.move_by(deg(10)))
    assert blank(script("elbow> velocity -0.2", joints=Arm)) == blank(Arm["elbow"].velocity(-0.2))
    assert blank(script("elbow> find_hardstop dir=1 timeout=4 verify=true", joints=Arm)) == blank(
        Arm["elbow"].find_hardstop(dir=1, timeout=4, verify=True)
    )
    # without joints=: bare names with no defaults, and A is every joint the controller drives
    assert blank(script("elbow> hold max_tau=3")) == blank(helpers.hold("elbow", 3.0))
    assert script("A> stop")[0].commands == (JointCommand.stop(),)
    # comments, blank lines, and an unlabelled step after a labelled stretch
    primitives = script(
        """
        # home, then let go
        Home:
            elbow> find_hardstop dir=-1 travel=1.0   # the lower stop

            elbow> move_by 2deg
        A> limp
        """,
        joints=Arm,
    )
    assert [(row["label"], row["call"]) for row in outline(primitives)] == [
        ("Home", "elbow> find_hardstop dir=-1 travel=1.0"),
        ("Home", "elbow> move_by 2deg"),
        ("", "A> limp"),
    ]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("\nelbow> mov_by 20deg", "script line 2: no verb `mov_by`; the verbs are limp, stop"),
        (
            "elbow> move_by 20deg speed=1",
            "line 1: move_by has no keyword `speed`; it takes by, vel",
        ),
        ("elbow> move_by 20deg 0.5", r"line 1: `0.5` needs its keyword \(move_by takes by, vel"),
        ("elbow> stop 1", r"line 1: `1` needs its keyword \(stop takes no keywords\)"),
        ("elbow> move_by fast", "line 1: `fast` is not a number: write 0.5, -1 or 20deg"),
        ("elbow> hold max_tau=3deg", "line 1: `max_tau` is not an angle"),
        ("elbow> move_by 1 by=2", "line 1: `by` is given twice"),
        ("move_by 20deg", "line 1: move_by needs a target before `>`: `A> move_by`"),
        ("elbow>", "line 1: no verb after `>`"),
        ("elbow,> stop", "line 1: a target is missing"),
        ("A,elbow> stop", r"line 1: `A` \(all\) stands alone"),
        ("wrist> stop", 'line 1: A has no joint "wrist"; it holds shoulder, elbow'),
        ("elbow> hold", r"line 1: hold elbow: needs max_tau= \(effort cap, N·m\)"),
        (
            "A> find_hardstop dir=1 vel=1 max_tau=1 travel=1",
            "line 1: find_hardstop takes one joint",
        ),
        ("elbow> wait 0.3", "line 1: wait takes no target"),
        ("wait", "line 1: wait takes the seconds"),
        ("Home:\nA> stop", 'line 1: the label "Home" has no steps'),
        ("Home:\n  A> stop\n  Again:\n    A> stop", "line 3: a label inside a labelled stretch"),
        ("# nothing here", "script: no steps"),
    ],
)
def test_a_text_error_names_the_line_and_what_is_wrong(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        script(text, joints=["shoulder", "elbow"])


def test_an_id_needs_the_group_that_knows_it() -> None:
    with pytest.raises(ValueError, match=r"script line 1: an id \(11\) needs joints="):
        script("11> stop")
    with pytest.raises(ValueError, match="script line 1: A has no id 11"):
        script("11> stop", joints=["shoulder", "elbow"])


def test_deg_is_radians_at_the_call_site() -> None:
    assert deg(180) == math.pi
    assert script("elbow> move_by 20deg vel=30deg max_tau=1")[0].commands[
        0
    ] == JointCommand.move_by("elbow", math.radians(20), speed=math.radians(30), max_effort=1.0)
