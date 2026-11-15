"""std — the header every stamped message carries, the colour, and what a running component
reports about itself: its log lines, its health and its numbers."""

from robot_msgs.std.header import Header

# isort: split
# WHY after Header: color_rgba imports robot_msgs.geometry, whose pose_stamped imports Header
# from this package, so Header must already exist when that import runs. The three records below
# import robot_msgs.geometry too, for the same type check every message uses.
from robot_msgs.std.color_rgba import ColorRGBA
from robot_msgs.std.diagnostic_record import DiagnosticRecord
from robot_msgs.std.log_record import LogRecord
from robot_msgs.std.metrics import Metrics

__all__ = ["ColorRGBA", "DiagnosticRecord", "Header", "LogRecord", "Metrics"]
