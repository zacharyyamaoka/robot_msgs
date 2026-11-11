"""robot_msgs.viz — easy visualisation of every standard message type.

    from robot_msgs.viz import plot, animate

    plot(joint_trajectory).savefig("joints.png")
    animate(joint_trajectory, tool_trajectory, path="pick.mp4")

Needs matplotlib (``pip install 'robot-msgs[viz]'``). `import robot_msgs` never imports this
package, so the core stays numpy-only.
"""

try:
    import matplotlib  # noqa: F401
except ImportError as error:
    raise ImportError("robot_msgs.viz needs matplotlib: pip install 'robot-msgs[viz]'") from error

from robot_msgs.viz import images, shapes, spatial, trajectory  # noqa: F401  (each registers its plots)
from robot_msgs.viz._dispatch import plot
from robot_msgs.viz.movie import animate

__all__ = ["animate", "plot"]
