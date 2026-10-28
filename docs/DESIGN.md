# robot_msgs — design contract (v1)

A cohesive domain model for robot data: the vocabulary that flows between robot components,
plus the functions that move between its parts. The parts are defined by each other, like a
dictionary — a trajectory is poses over time, a pose is a point and an orientation in a frame,
a point cloud comes from a depth image and its camera info — so the relationships ship with the
types. It is the stable bottom tier: everything above it (like_keras/Clank, bam_traj_gen,
traj_compare, sim, drivers) can depend on it, and it depends on nothing but numpy.

## What we keep from ROS, and what we drop

| Keep | Drop |
|---|---|
| The domain model: `geometry_msgs`, `sensor_msgs`, `trajectory_msgs`, `vision_msgs` — decades of agreement on what the nouns are | Behaviourless structs. Pose offsetting, interpolation, projection and FK-shaped batch maths were separate libraries (tf, image_geometry, …); here they are methods on the type |
| Field names verbatim, so ROS docs are our docs and conversion is mechanical | Array-of-structs as the only storage: a `JointTrajectory` is not 500 point messages |
| Every stamped message carries its time and frame (`Header`) | Code generation, a build system and a ROS install just to have a `Pose` |
| Explicit frames (`frame_id`, `child_frame_id`) | Verbose construction (`msg.pose.pose.position.x = …`) and mutable messages |
| Wire formats as an *export* (`robot_msgs.ros`) | The wire format as the programming model |

## Scope

- **In:** the standard vocabulary and the standard functions between its parts — offsets
  (`pose.shifted(dz=-0.1, local=True)`), composition, interpolation, resampling, projection,
  depth → cloud, joint trajectory → joint state, limit checks.
- **Out:** business logic and non-standard types. The moment a type needs your application's
  concepts (a bin, an order, a grasp score policy) it belongs in your code. See *Extending*.

## Principles (each one is a rule, with its reason)

1. **Copy ROS's shape, not ROS.** Types and field names follow the ROS 2 message of the same
   name verbatim (`JointState.name/position/velocity/effort`, `CameraInfo.k/d/r/p`,
   `Header.stamp/frame_id`) so the ROS docs are our docs and conversion is mechanical.
   No ROS install, no rclpy, no codegen.
2. **Behaviour lives with its type.** ROS messages are dead structs; here `pose.py` holds `Pose`
   *and* `compose()`, `inverse()`, `to_matrix()`. Helper methods are the point of the package.
3. **Immutable values.** Every message is `@dataclass(frozen=True, eq=False, repr=False)` and
   subclasses `robot_msgs._core.Msg`. Methods never mutate; they return new values
   (`replace()` re-validates). WHY: a message handed to three consumers can't be changed under
   any of them, so logs, replays and tests are reproducible.
4. **Numbers live in read-only numpy arrays.** `frozen_array()` copies once at construction and
   marks the array read-only; views of it (slices, rows, columns) are then free and still
   read-only. WHY: Zach wants "the most compact, efficient storage, with many views of it".
   Tuple-backed storage (bam_traj_gen) makes every numpy "view" a copy.
5. **Validate at construction.** Shapes, lengths, monotonic time and name counts fail in
   `__post_init__` with a `ValueError` naming the field, never later in a consumer.
6. **One type per file, named for the type** (`geometry/pose_stamped.py` → `PoseStamped`), plus
   any functions that operate on it. Each domain package re-exports its types in `__init__`.
7. **Conventions are fixed and written down** (below), never inferred.
8. **The function decides how specific the type is.** A function that works on anything with
   samples over time takes a `Trajectory` (`resample`, `window`, `plot`); one that needs joint
   angles takes a `JointTrajectory`; one that needs poses takes a `PoseTrajectory`. The type is
   validation: passing a `JointTrajectory` where a `PoseTrajectory` is required is an error, and
   that is the point.
9. **Methods that return a modified copy are named for the result**: `shifted`, `rotated`,
   `normalized`, `rebased`, `time_shifted`, `time_scaled`, `transformed`, `with_*`. Spatial
   words (`shifted`, `rotated`) move things in space; `time_*` moves them in time.
10. **Runtime dependency: numpy only.** matplotlib (`[viz]`), rosbags (`[ros]`), scipy (tests,
   as an independent oracle) are extras. A boundary test enforces this.

## Conventions

| Thing | Convention |
|---|---|
| Units | SI: metres, radians, seconds, newtons, N·m |
| Quaternion | `(x, y, z, w)` as fields and as arrays; Hamilton product; MuJoCo/viser `wxyz` only via `xyzw_to_wxyz` / `wxyz_to_xyzw` |
| Euler | ROS RPY: fixed axes X then Y then Z, `R = Rz(yaw)·Ry(pitch)·Rx(roll)` |
| `a.compose(b)` / `a @ b` | apply `b` in `a`'s frame: `T_world_tool = T_world_base @ T_base_tool` |
| Time | `Header.stamp`: float seconds. Trajectory `t`: float seconds relative to `header.stamp` |
| Absent optional array | `None` (ROS uses an empty array; converters map between them) |
| Matrices | `(3, 3)` rotation, `(4, 4)` homogeneous, row-major numpy |

## Mechanics every message follows

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from robot_msgs._core import Array, Msg, frozen_array, optional_array, set_field
from robot_msgs.std import Header

@dataclass(frozen=True, eq=False, repr=False)       # always these three flags
class JointState(Msg):
    name: tuple[str, ...]
    position: Array
    velocity: Array | None = None
    effort: Array | None = None
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        set_field(self, "name", tuple(str(n) for n in self.name))
        set_field(self, "position", frozen_array(self.position))   # copies once, read-only
        ...validate shapes, raise ValueError("JointState.velocity has 5 values, name has 6")
```

- Small value types with only scalar fields (`Point`, `Quaternion`, `Vector3`) take positional
  arguments like ROS (`Point(1, 2, 3)`) and store Python floats. Array-holding and trajectory
  types are `kw_only=True`.
- `Msg` gives every message: array-aware `==` and `hash`, a compact `repr` (big arrays print as
  `float64[500×6]`), `replace(**changes)`, `to_dict()` / `from_dict()` (JSON-safe, lossless,
  `"_type": "geometry/Pose"`), and `save_npz` / `load_npz` for compact files.
- `robot_msgs.allclose(a, b)` compares messages approximately.

## Packages and their v1 types

| Package | Types (one file each) | ROS source |
|---|---|---|
| `robot_msgs.std` | `Header` | std_msgs |
| `robot_msgs.geometry` | `Vector3`, `Point`, `Quaternion`, `Pose`, `Pose2D`, `Transform`, `Twist`, `Wrench`, `Accel`, `PoseStamped`, `TransformStamped`; batch math in `rotations.py` | geometry_msgs |
| `robot_msgs.sensor` | `JointState`, `Image`, `CameraInfo`, `PointCloud`, `Imu` | sensor_msgs |
| `robot_msgs.vision` | `BoundingBox2D`, `BoundingBox3D`, `ObjectHypothesis`, `ObjectHypothesisWithPose`, `Detection2D`, `Detection2DArray`, `Detection3D`, `Detection3DArray` | vision_msgs |
| `robot_msgs.trajectory` | `Trajectory`, `TrajectoryPoint`, `JointTrajectory`, `JointTrajectoryPoint`, `PoseTrajectory`, `PoseTrajectoryPoint`, `JointLimits`, `concatenate` | trajectory_msgs, nav_msgs/Path |
| `robot_msgs.viz` | `plot(msg)` for every type above (matplotlib, `[viz]` extra) | — |
| `robot_msgs.ros` | `to_ros(msg)` / `from_ros(ros_msg)` via rosbags' typestore (`[ros]` extra) | — |

## Geometry API (frozen contract — other modules import these names)

- `Vector3(x=0.0, y=0.0, z=0.0)` and `Point(x=0.0, y=0.0, z=0.0)`: `as_array() -> (3,)`,
  `from_array(a)`, `+ - ` same type, `* / ` scalar, unary `-`, `norm()`, `dot()`, `cross()`
  (Vector3), `distance_to()` (Point), `normalized()` (Vector3).
- `Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)`: `identity()`, `as_array()` (xyzw), `from_array(a)`,
  `from_wxyz(a)`, `to_wxyz()`, `from_euler(roll, pitch, yaw)`, `to_euler() -> (r, p, y)`,
  `from_matrix(R)`, `to_matrix()`, `from_axis_angle(axis, angle)`, `from_rotvec(v)`,
  `to_rotvec()`, `multiply(other)` and `__mul__`, `conjugate()`, `inverse()`, `normalized()`,
  `rotate(v)` (Vector3, Point or array → same kind), `slerp(other, alpha)`, `angle_to(other)`.
- `Pose(position=Point(), orientation=Quaternion())` and
  `Transform(translation=Vector3(), rotation=Quaternion())`: `identity()`,
  `from_xyz_rpy(x, y, z, roll, pitch, yaw)`, `from_matrix(M)`, `to_matrix()`, `compose(other)` and
  `__matmul__`, `inverse()`, `transform_point(p)`, `as_array() -> (7,)` = x y z qx qy qz qw,
  `from_array(a)`, `Pose.as_transform()`, `Transform.as_pose()`.
- `PoseStamped(header=Header(), pose=Pose())`, `TransformStamped(header=Header(),
  child_frame_id="", transform=Transform())`, `Twist(linear=Vector3(), angular=Vector3())`,
  `Wrench(force=Vector3(), torque=Vector3())`, `Accel(linear, angular)`, `Pose2D(x, y, theta)`.

## Trajectory — one entity, many views

A trajectory is samples at strictly increasing times, in any domain. The base class holds the
time axis and the machinery; subclasses only declare their per-sample fields.

```python
traj = JointTrajectory(t=t, joint_names=names, positions=q, velocities=qd)

traj.positions          # (N, J) read-only array — the batch view, no copy
traj["positions"]       # same, by name (also works for channels)
traj[10]                # JointTrajectoryPoint at sample 10 (row views, no copy)
for point in traj: ...  # per-step view
traj[100:200]           # JointTrajectory, sliced as views — O(1)
traj.at(1.25)           # interpolated point (linear; slerp for orientations)
traj.resample(dt=0.004) # on a fixed controller grid
traj.window(1.0, 2.0)   # the part between two times
traj.joint("elbow")     # one joint, still a JointTrajectory
traj.channels["kp"]     # any extra per-sample data you layer on top
```

- **Per-sample fields** are declared with `sample()`; their first axis is time and they are
  sliced, iterated and interpolated together. `sample(interp="slerp")` for quaternions,
  `"hold"` for discrete values. Per-trajectory constants are ordinary fields
  (`joint_names`, `limits`, `child_frame_id`).
- **`channels`** carries extra per-sample arrays any trajectory can grow without a new type
  (`kp`, `tau_ff`, `mode`, `temp_c`). Float channels interpolate linearly, others hold.
- **New domains subclass `Trajectory`** and declare `sample()` fields; they get indexing,
  iteration, interpolation, resampling, plotting and serialisation for free, plus a generated
  point type.
- `t` is relative to `header.stamp`; `rebased()` moves `t[0]` to 0 and folds the offset into the
  stamp. Out-of-range interpolation raises unless `extrapolate="clamp"`.

## Extending without forking

1. **Extra per-sample data → `channels`.** Gains, feed-forward torque, controller mode,
   temperature, a waypoint's name: `traj.with_channel("kp", kp)`. No new type needed.
2. **A new kind of trajectory → subclass `Trajectory` in your own file** and declare its
   `sample()` fields. Indexing, iteration, interpolation, resampling, plotting and serialisation
   come for free:

   ```python
   @dataclass(frozen=True, eq=False, repr=False, kw_only=True)
   class GripperTrajectory(Trajectory):
       width: Array = sample()                                    # metres
       grasping: Array | None = sample(default=None, interp="hold", dtype=bool)
   ```
3. **Business types → your own dataclasses that hold robot_msgs values** (composition). A
   `GraspCandidate(target: Detection3D, approach: Pose, quality: float)` lives in your
   application, not here.
4. **A new standard type** (it exists in a ROS package and crosses a wire between domains) →
   add it here, one file, with a catalog example; the contract tests then cover it.
