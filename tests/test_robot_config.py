"""RobotConfig + BackendConfig + HardwareInterface.from_robot_description: generate, connect, calibrate's connect step.

WHY (Zach, Daily Note 2026-11-28): "So you get like a HardwareInterface to a Particular Deployment ... we are connecting to
something real, we need to be concrete not abstract." The description is nominal, the twin's values are this copy's
calibration, the RobotConfig is this copy's table and transports; a frozen BackendConfig picks the backend.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robot_msgs.control import (
    BackendConfig,
    GroundTruth,
    HardwareBackend,
    HardwareInterface,
    JointTable,
    Servo,
    ServoCommand,
    ServoState,
)
from robot_msgs.robot import RobotConfig
from robot_msgs.std import Header

TABLE = JointTable(bus="fdcanusb:272813EF", joints={"pendulum": Servo(can_id=11, sign=-1.0, offset_rad=0.5)})
CONFIG = RobotConfig.for_hardware("pendulum_can11", TABLE, {"fdcanusb:272813EF": "/dev/serial/by-id/usb-mjbots_fdcanusb_272813EF-if00"})


@dataclasses.dataclass(frozen=True)
class Description:
    """A stand-in robot description: nominal values only."""

    name: str = "pendulum"
    gravity_nm: float = 5.0


class Loopback:
    """Servo-keyed: the next read is the last command's position; truth when simulated."""

    def __init__(self, config: RobotConfig, gravity_nm: float, *, simulated: bool) -> None:
        self.keys = tuple(config.joints.addresses())
        self.names = config.joints.names
        self.gravity_nm, self.simulated, self.period_s = gravity_nm, simulated, 0.01
        self.q = np.zeros(len(self.keys))
        self.closed = False

    def read(self) -> ServoState:
        truth = GroundTruth(name=self.names, position=self.q.copy(), meta={"gravity_nm": self.gravity_nm})
        return ServoState(name=self.keys, position=self.q.copy(), header=Header(stamp=0.0), truth=truth)

    def write(self, command, defines=None):
        self.q = np.array(command.position, dtype=float)

    def close(self):
        self.closed = True


@dataclasses.dataclass(frozen=True)
class LoopbackConfig:
    simulated: bool = True

    def build(self, rd, config, parameters=None):
        gravity = rd.gravity_nm if parameters is None else parameters["gravity_nm"]
        return Loopback(config, gravity, simulated=self.simulated)


def test_robot_config_holds_the_table_and_transports():
    assert CONFIG.buses == ("fdcanusb:272813EF",)
    assert CONFIG.transport("fdcanusb:272813EF").endswith("272813EF-if00")
    with pytest.raises(KeyError, match="no transport for bus 'can0'"):
        CONFIG.transport("can0")
    homed = CONFIG.with_zero({"pendulum": 0.1})
    assert homed.joints.joints["pendulum"].offset_rad == 0.1 and CONFIG.joints.joints["pendulum"].offset_rad == 0.5


def test_robot_config_refuses_what_is_not_this_copy():
    from robot_msgs.control import Transports

    with pytest.raises(TypeError, match="robot_msgs message"):
        RobotConfig.for_hardware("x", {"pendulum": Servo(can_id=11)})
    with pytest.raises(TypeError, match="must be a JointTable"):
        RobotConfig(name="x", components={"joints": Transports()}).joints
    with pytest.raises(ValueError, match="which no joint's servo is on"):
        RobotConfig.for_hardware("x", TABLE, {"can0": "can0"})
    with pytest.raises(ValueError, match="non-empty"):
        Transports(buses={"fdcanusb:272813EF": ""})
    sim_only = RobotConfig.for_hardware("sim only", TABLE)  # no transports: a sim-only config is allowed
    assert len(sim_only.transports) == 0
    with pytest.raises(KeyError, match="no component 'joints'"):
        RobotConfig(name="empty").joints


def test_robot_config_round_trips_as_a_message():
    from robot_msgs._core import from_dict

    assert from_dict(CONFIG.to_dict()) == CONFIG


def test_from_robot_description_connects_one_copy_nominal_and_calibrated():
    assert isinstance(LoopbackConfig(), BackendConfig)
    rd = Description()
    nominal = HardwareInterface.from_robot_description(rd, LoopbackConfig(), config=CONFIG)
    fitted = HardwareInterface.from_robot_description(rd, LoopbackConfig(), config=CONFIG, parameters={"gravity_nm": 5.2})
    assert nominal.read().truth.meta["gravity_nm"] == 5.0  # parameters None: the description's nominal values
    assert fitted.read().truth.meta["gravity_nm"] == 5.2  # the twin's values, composed, never written into rd
    assert rd.gravity_nm == 5.0
    assert (fitted.name, fitted.backend_name, fitted.config, fitted.description) == ("pendulum_can11", "LoopbackConfig", CONFIG, rd)
    assert fitted.parameters == {"gravity_nm": 5.2} and isinstance(fitted.backend_config, LoopbackConfig)
    # the table applies: joint = sign * servo + offset
    fitted.write(ServoCommand(name=("pendulum",), position=np.array([0.2])))
    assert fitted.backend.q[0] == pytest.approx(-(0.2 - 0.5))
    assert fitted.read().position[0] == pytest.approx(0.2)
    fitted.close()
    assert fitted.backend.closed


def test_from_robot_description_takes_rd_robot_config_or_refuses():
    @dataclasses.dataclass(frozen=True)
    class WithConfig(Description):
        def robot_config(self):
            return CONFIG

    hw = HardwareInterface.from_robot_description(WithConfig(), LoopbackConfig())
    assert hw.config is CONFIG
    with pytest.raises(TypeError, match="needs config=<RobotConfig>"):
        HardwareInterface.from_robot_description(Description(), LoopbackConfig())
    with pytest.raises(TypeError, match="must be a robot_msgs.robot.RobotConfig"):
        HardwareInterface.from_robot_description(Description(), LoopbackConfig(), config=TABLE)
    assert RobotConfig.for_hardware("x", TABLE).name == "x"
    with pytest.raises(TypeError, match="takes a BackendConfig"):
        HardwareInterface.from_robot_description(Description(), "sim", config=CONFIG)


def test_a_build_that_is_no_backend_is_refused_and_closed():
    closed = []

    class Half:
        def close(self):
            closed.append(True)

    @dataclasses.dataclass(frozen=True)
    class Broken:
        def build(self, rd, config, parameters=None):
            return Half()

    with pytest.raises(TypeError, match="not an HardwareBackend"):
        HardwareInterface.from_robot_description(Description(), Broken(), config=CONFIG)
    assert closed == [True]


def test_a_real_backend_never_hands_truth_up_through_from_robot_description():
    hw = HardwareInterface.from_robot_description(Description(), LoopbackConfig(simulated=False), config=CONFIG)
    assert isinstance(hw.backend, HardwareBackend) and not hw.simulated
    assert hw.read().truth is None


def test_plain_construction_reports_no_connection():
    hw = HardwareInterface(TABLE, backend=Loopback(CONFIG, 5.0, simulated=True))
    assert (hw.description, hw.backend_config, hw.config, hw.parameters) == (None, None, None, None)
