"""Specs that hold two pieces of Zach's feedback of 2026-11-28 on the HardwareInterface redesign.

fb-9442d16b, "Ok I completely agree with this current direction": the HardwareInterface is the one standard type, at every
level of the robot (an arm, a whole cell), and switching sim -> real is switching the backend config, nothing else.

fb-0c803047, "Ok I like P4 How you are composing things together... in the long run that is going to be much much cleaner
and simple than if we like write the values back into RobotDescription": the description (nominal, shared by every copy),
the calibrated parameters and the RobotConfig (one copy's table and transports) are composed at connect time; nothing is
written back into the description, so two copies share one description.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from test_robot_config import LoopbackConfig

from robot_msgs.control import Arm, HardwareInterface, JointTable, Servo, ServoCommand
from robot_msgs.robot import RobotConfig

ARM_TABLE = JointTable(joints={"shoulder": Servo(can_id=12), "elbow": Servo(can_id=13)})


def copy(name: str, bus: str, prefix: str | None = None) -> RobotConfig:
    table = JointTable(bus=bus, joints=ARM_TABLE.joints)
    table = table.mounted(prefix) if prefix else table
    return RobotConfig.for_hardware(name, table, {bus: f"/dev/fake/{bus}"})


@dataclasses.dataclass  # WHY mutable on purpose: a write-back into the description would show up here
class Description:
    name: str = "two_link"
    gravity_nm: float = 5.0


def test_fb_9442d16b_one_standard_type_at_every_level_and_the_backend_config_is_the_only_switch():
    rd = Description()
    arm = HardwareInterface.from_robot_description(rd, LoopbackConfig(), config=copy("arm", "can0"))
    cell_table = JointTable.combine(copy("l", "can0", "left_arm").joints, copy("r", "can1", "right_arm").joints)
    cell_config = RobotConfig.for_hardware("cell", cell_table, {"can0": "/dev/fake/can0", "can1": "/dev/fake/can1"})
    cell = HardwareInterface.from_robot_description(rd, LoopbackConfig(), config=cell_config)
    assert type(arm) is HardwareInterface and type(cell) is HardwareInterface  # no Arm/Cell types: one type, any level
    assert Arm is HardwareInterface  # the old name is only an alias of it

    def run(hw: HardwareInterface):
        hw.write(ServoCommand(name=hw.joint_names, position=np.full(len(hw.joint_names), 0.3)))
        return hw.read()

    sim = run(HardwareInterface.from_robot_description(rd, LoopbackConfig(simulated=True), config=copy("arm", "can0")))
    real = run(HardwareInterface.from_robot_description(rd, LoopbackConfig(simulated=False), config=copy("arm", "can0")))
    np.testing.assert_allclose(sim.position, real.position)  # the same script, only the backend config switched
    assert sim.truth is not None and real.truth is None  # and truth never reaches a script from a real backend


def test_fb_0c803047_description_parameters_and_config_compose_and_nothing_is_written_back():
    rd = Description()
    left = HardwareInterface.from_robot_description(
        rd, LoopbackConfig(), config=copy("left_copy", "can0"), parameters={"gravity_nm": 5.2}
    )
    right = HardwareInterface.from_robot_description(
        rd, LoopbackConfig(), config=copy("right_copy", "can1"), parameters={"gravity_nm": 4.9}
    )
    nominal = HardwareInterface.from_robot_description(rd, LoopbackConfig(), config=copy("bench_copy", "can2"))
    assert rd == Description()  # never written back: one nominal description shared by every copy
    assert left.description is rd and right.description is rd
    assert (left.read().truth.meta["gravity_nm"], right.read().truth.meta["gravity_nm"]) == (5.2, 4.9)  # each copy's own
    assert nominal.read().truth.meta["gravity_nm"] == 5.0  # parameters None: the description's nominal values
    assert left.config.name == "left_copy" and right.config.name == "right_copy"  # where each copy is plugged in
    assert left.config.joints.key("shoulder") != right.config.joints.key("shoulder")
