"""shape — solids and surfaces for any 3-D scene: primitives, triangle meshes, planes.

Mirrors ROS 2 ``shape_msgs`` (`SolidPrimitive`, `Mesh`, `Plane`), plus `MeshResource`, the
file-reference mesh URDF and visualization_msgs use. Material (colour, texture) stays with the
robot description: ROS has no message for it; the colour itself is `robot_msgs.std.ColorRGBA`.
"""

from robot_msgs.shape.mesh import Mesh
from robot_msgs.shape.mesh_resource import MeshResource
from robot_msgs.shape.plane import Plane
from robot_msgs.shape.solid_primitive import SolidPrimitive

__all__ = ["Mesh", "MeshResource", "Plane", "SolidPrimitive"]
