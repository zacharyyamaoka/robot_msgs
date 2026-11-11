"""ServoState / ServoCommand: the controller ↔ servo seam carries the stamps timing metrics need."""

import numpy as np
import pytest

from robot_msgs import from_dict, to_dict
from robot_msgs.control import ServoCommand, ServoState
from robot_msgs.sensor import JointState
from robot_msgs.std import Header

NAMES = ("shoulder", "elbow")


def state(**changes):
    values = dict(
        name=NAMES,
        position=[0.1, -0.2],
        velocity=[0.0, 0.5],
        fault=[0, 0],
        seq=41,
        header=Header(stamp=100.000),
    )
    values.update(changes)
    return ServoState(**values)


def test_latency_is_send_time_minus_the_sample_it_came_from():
    command = ServoCommand.from_state(state(), stamp=100.0035)
    assert command.state_seq == 41
    assert command.latency == pytest.approx(0.0035, abs=1e-9)


def test_open_loop_commands_have_no_latency():
    assert ServoCommand(name=NAMES, position=[0.0, 0.0]).latency is None


def test_from_state_holds_the_measured_position_at_zero_velocity_with_board_gains():
    command = ServoCommand.from_state(state(), stamp=100.002)
    np.testing.assert_array_equal(command.position, [0.1, -0.2])
    np.testing.assert_array_equal(command.velocity, [0.0, 0.0])
    assert command.stiffness is None and command.damping is None


def test_nan_position_means_no_position_target():
    command = ServoCommand(name=NAMES, position=[np.nan, 0.0], velocity=[1.0, 0.0])
    assert np.isnan(command.position[0])


@pytest.mark.parametrize("field", ["stiffness", "damping", "max_effort"])
@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test_gains_and_limits_must_be_non_negative(field, bad):
    with pytest.raises(ValueError, match=field):
        ServoCommand(name=NAMES, position=[0.0, 0.0], **{field: [1.0, bad]})


@pytest.mark.parametrize("cls", [ServoState, ServoCommand])
def test_lengths_and_names_are_checked_at_construction(cls):
    with pytest.raises(ValueError, match="has 3 values, name has 2"):
        cls(name=NAMES, position=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        cls(name=("a", "a"), position=[0.0, 0.0])


def test_fault_codes_are_integers_and_flag_a_fault():
    assert not state().faulted
    faulted = state(fault=[0, 33])
    assert faulted.faulted and faulted.fault.dtype == np.int64
    assert not state(fault=None).faulted


def test_the_kinematic_part_is_a_joint_state_with_the_same_stamp():
    joints = state(effort=[1.0, 2.0]).joint_state()
    assert isinstance(joints, JointState)
    assert joints.header.stamp == 100.0
    np.testing.assert_array_equal(joints.effort, [1.0, 2.0])


def test_round_trip_keeps_the_stamps_that_link_command_to_state():
    command = ServoCommand.from_state(state(), stamp=100.004).replace(stiffness=[20.0, 10.0])
    back = from_dict(to_dict(command))
    assert back == command
    assert back.latency == pytest.approx(0.004)
