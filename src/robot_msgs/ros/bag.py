"""Read and write rosbag2 bags of robot_msgs values (mcap by default, so Foxglove opens them)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from rosbags.rosbag2 import Reader, StoragePlugin, Writer

from robot_msgs._core import Msg
from robot_msgs.ros.convert import _FROM_ROS, _stamp_ns, from_ros, to_ros, typestore

_STORAGE = {"mcap": StoragePlugin.MCAP, "sqlite3": StoragePlugin.SQLITE3}


def write_bag(
    path: str | Path, messages: Iterable[tuple[str, Msg]], *, storage: str = "mcap"
) -> None:
    """Write `(topic, message)` pairs to a new rosbag2 directory at `path`.

    One connection per topic; a topic carries one ROS type. Log time is the message's header
    stamp, or for a message without a header (a bare `Pose`) the number of messages written
    before it, in nanoseconds: a running counter, so such messages keep their order.
    A value that is already a rosbags message (to write a `PoseTrajectory` as MultiDOF, say) is
    written as it is.
    """
    if storage not in _STORAGE:
        raise ValueError(f"storage must be one of {sorted(_STORAGE)}, got {storage!r}")
    store = typestore()
    connections: dict[str, Any] = {}
    with Writer(path, version=9, storage_plugin=_STORAGE[storage]) as writer:
        for count, (topic, msg) in enumerate(messages):
            ros_msg = msg if hasattr(msg, "__msgtype__") else to_ros(msg)
            msgtype = ros_msg.__msgtype__
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msgtype, typestore=store)
            elif connections[topic].msgtype != msgtype:
                raise ValueError(
                    f"topic {topic!r} carries {connections[topic].msgtype}, got a {msgtype}"
                )
            header = getattr(ros_msg, "header", None)
            log_time = _stamp_ns(header.stamp) if header is not None else count
            writer.write(connections[topic], log_time, store.serialize_cdr(ros_msg, msgtype))


def read_bag(path: str | Path) -> Iterator[tuple[str, Msg]]:
    """Yield `(topic, message)` in log-time order. A topic whose ROS type has no robot_msgs
    counterpart raises `TypeError` naming it."""
    store = typestore()
    with Reader(path) as reader:
        for connection, _, raw in reader.messages():
            if connection.msgtype not in _FROM_ROS:
                raise TypeError(
                    f"topic {connection.topic!r}: no robot_msgs type for {connection.msgtype}"
                )
            yield connection.topic, from_ros(store.deserialize_cdr(raw, connection.msgtype))
