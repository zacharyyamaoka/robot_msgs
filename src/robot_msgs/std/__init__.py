"""std — the header every stamped message carries, and the colour."""

from robot_msgs.std.header import Header

# isort: split
# WHY after Header: color_rgba imports robot_msgs.geometry, whose pose_stamped imports Header
# from this package, so Header must already exist when that import runs.
from robot_msgs.std.color_rgba import ColorRGBA

__all__ = ["ColorRGBA", "Header"]
