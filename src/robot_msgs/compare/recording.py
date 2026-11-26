"""Recording — one source's streams for one run: what the target, sim or real robot logged."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robot_msgs._core import FrozenDict, Msg, set_field
from robot_msgs.std import Header


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Recording(Msg):
    """Every stream one source recorded in one run, by stream name.

    ``streams = {"servo_state": ServoStateTrajectory, "servo_command": ServoCommandTrajectory,
    "imu": ImuTrajectory, "camera": ImageTrajectory, "joints": JointTrajectory, "scan":
    PointCloud}``: any message `compare` knows, usually a trajectory on the recording's clock.
    Two recordings of a `Run` are compared stream by stream, matched by name, so sim and real
    use the same names for the same signal. `header` says where the recording came from (its
    `meta`: rig, firmware, sim build); each stream keeps its own stamps.
    """

    streams: Mapping[str, Msg] = field(default_factory=dict)
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        streams = {}
        for name, stream in self.streams.items():
            if not isinstance(stream, Msg):
                raise TypeError(
                    f"Recording.streams[{name!r}] must be a robot_msgs message, "
                    f"got {type(stream).__name__}"
                )
            streams[str(name)] = stream
        set_field(self, "streams", FrozenDict(streams))

    def __getitem__(self, name: str) -> Msg:
        return self.streams[name]

    def __len__(self) -> int:
        return len(self.streams)
