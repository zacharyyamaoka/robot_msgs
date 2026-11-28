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
   `robot_msgs.adopt(array)` is the one public way around the copy, for memory nobody writes
   again (numpy views over a received buffer): it keeps a read-only array as it is, freezes a
   writeable one that owns its memory in place, and refuses a writeable view or a view of a
   writeable array. WHY public (like_keras, 2026-11-15): without it every receive across a process
   copies.
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
10. **Runtime dependencies: numpy and the stdlib-only `dataclass-values`** (`Value`/`Space`, shared with py_dataclass_viewer). matplotlib (`[viz]`), rosbags (`[ros]`), scipy (tests,
   as an independent oracle) are extras. A boundary test enforces this.

## Conventions

| Thing | Convention |
|---|---|
| Units | SI: metres, radians, seconds, newtons, N·m |
| Quaternion | `(x, y, z, w)` as fields and as arrays; Hamilton product; MuJoCo/viser `wxyz` only via `xyzw_to_wxyz` / `wxyz_to_xyzw` |
| Euler | ROS RPY: fixed axes X then Y then Z, `R = Rz(yaw)·Ry(pitch)·Rx(roll)` |
| `a.compose(b)` / `a @ b` | apply `b` in `a`'s frame: `T_world_tool = T_world_base @ T_base_tool` |
| Time | `Header.stamp`: float seconds. Trajectory `t`: float seconds relative to `header.stamp` |
| Context beyond a frame | `Header.meta`, str → str (origin, units, provenance); dropped on ROS export |
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
| `robot_msgs.std` | `Header`, `ColorRGBA`; what a running component reports about itself: `LogRecord` (stdlib logging levels), `DiagnosticRecord` (a flat subset of io-diagnostics' DiagnosticStatus v0: `unknown` is the zero value, `stale` is derived, never sent), `Metrics` (parallel `names` / `values`) | std_msgs; rcl_interfaces/Log; diagnostic_msgs |
| `robot_msgs.geometry` | `Vector3`, `Point`, `Quaternion`, `Pose`, `Pose2D`, `Transform`, `Twist`, `Wrench`, `Accel`, `Inertia`, `PoseStamped`, `TransformStamped`; batch math in `rotations.py` | geometry_msgs |
| `robot_msgs.shape` | `SolidPrimitive`, `Mesh`, `Plane`, `MeshResource` — solids and surfaces for any 3-D scene (see *Shapes*); `Material` is not here | shape_msgs (`MeshResource`: URDF `<mesh>`) |
| `robot_msgs.sensor` | `JointState`, `Image`, `CameraInfo`, `CameraSettings` (exposure s, gain dB, white balance K, gamma, auto flags, focus, frame rate; no ROS message), `PointCloud`, `Imu` | sensor_msgs |
| `robot_msgs.control` | `ServoState`, `ServoCommand` — the seam a controller loops across: read a stamped state, send a command that names the state it came from (`latency`); since 0.8 a per-servo `mode` (`position` / `stop` / `brake`, `SERVO_MODES`) and NaN gain/cap = that servo's board default (a `ServoCommandTrajectory` keeps the targets, not the mode); routines: `JointCommand` + the primitives `Command` / `Delay` / `Condition` (hooks `Record`, `Reduce`, `FailWhen`), `JointCommandSequence`, helper functions in `control.helpers`, run by `JointCommandController` (0.8; hold-don't-drop, stop wins, one writer per joint) or the 0.3 `JointCommandSequenceController` | — (moteus position mode; the "MIT mode" of Damiao/RobStride/CubeMars/Unitree) |
| `robot_msgs.manipulation` | `GraspPose`, `Waypoint` (+ `WaypointLabel`), `WaypointAction`, `CartesianLimits` — what a grasp detector hands the trajectory generator, and the request the generator takes in (see *Manipulation*); `NamedPose` — a robot's saved positions by name (see *Named poses*); `AllowedCollisionMatrix` — which body pairs a collision checker skips, each with MoveIt's reason (generated by robot_collision); `CollisionScene` of `CollisionElement`s — everything a planner must not hit, each element tagged `source` scene/object/self, `motion` fixed/moving, `compliance` rigid/deformable, with a `margin` (m) and a `contact` policy (v1: `avoid` only; `allow`/`touch` reserved); planes are half-spaces; a camera occupancy map is the documented v2 field (see *Collision scene*) | moveit_msgs (`Grasp`, `MotionPlanRequest`, `AllowedCollisionMatrix`, `PlanningScene`, `CollisionObject`); SRDF `<group_state>`, `<disable_collisions>` |
| `robot_msgs.behavior` | `BTState` — a behaviour tree's state: `tree_id`, `node_status` (path → IDLE/RUNNING/SUCCESS/FAILURE), `blackboard` (JSON values, frozen), `tick`, `clock_s` | — (BT.CPP NodeStatus; robot_bt's trace paths) |
| `robot_msgs.robot` | `RobotState`, `RobotConfig` — the whole robot as `components: name → message` (see *Files*) | — |
| `robot_msgs.vision` | `BoundingBox2D`, `BoundingBox3D`, `ObjectHypothesis`, `ObjectHypothesisWithPose`, `Detection2D`, `Detection2DArray`, `Detection3D`, `Detection3DArray` | vision_msgs |
| `robot_msgs.fiducial` | `FiducialDetection`, `FiducialPose`, `FiducialArray` (measured, per image); `FiducialMount`, `FiducialMap`, `FiducialBoard` (described); `FiducialTrack`, `FiducialTrackArray` (estimated); `marker_object_points`, `local_to_parent_covariance` (see *Fiducial markers*) | aruco_msgs, fiducial_msgs, apriltag_msgs; OpenCV `cv2.aruco` |
| `robot_msgs.trajectory` | `Trajectory`, `TrajectoryPoint`, `JointTrajectory`, `JointTrajectoryPoint`, `PoseTrajectory`, `PoseTrajectoryPoint`, `JointLimits` (with each joint's `position_kind`), `JointLimitViolation` + `joint_limit_violations`, `concatenate`; the recorded streams (each + its point): `ServoStateTrajectory` (`from_servo_states`, `state_at`, `joint_trajectory()`), `ServoCommandTrajectory` (`from_servo_commands`, `command_at`, channel `latency`), `ImuTrajectory` (`from_imus`, `imu_at`), `WrenchTrajectory` (`from_wrenches(t, wrenches)`, `wrench_at`), `ImageTrajectory` (`from_images`, `image_at`; frames held, dtype kept) | trajectory_msgs, nav_msgs/Path; URDF `<limit>`; a stream of sensor_msgs/Imu, Image, geometry_msgs/WrenchStamped |
| `robot_msgs.compare` | `compare(reference, measured)` → `Comparison` (error + `ErrorStats` per metric) for every trajectory type, single messages and point clouds, `align`, `error_stats`; also `.compare(other)` on every comparable type; the dataset nouns `Recording`, `Run`, `Dataset` → `compare_run` / `compare_dataset` → `RunComparison`, `DatasetComparison` — numpy only | control_msgs/JointTrajectoryControllerState (error sign) |
| `robot_msgs.coverage` | `binned(points, axes, reachable)` → `CoverageResult` (% of reachable cells visited), `occupancy`, `marginal`, `bin_index` (numpy.histogramdd's bins), `signed`, `random_plan`; continuous edges or `("levels", …)` axes. Which axes, edges and reachable cells a device type has is the caller's policy (sim2real), never here | — |
| `robot_msgs.viz` | `plot(msg)` for every type above but `MeshResource` (a file reference; matplotlib, `[viz]` extra) | — |
| `robot_msgs.ros` | `to_ros(msg)` / `from_ros(ros_msg)` / `ros_type()` against ROS 2 Jazzy via rosbags' `typestore()` (vision_msgs 4.1.1 registered); `write_bag` / `read_bag` (mcap or sqlite3) — `[ros]` extra | — |

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
- `Inertia(m=0.0, com=Vector3(), ixx, ixy, ixz, iyy, iyz, izz)` (geometry_msgs/Inertia, tensor
  about the centre of mass): `to_matrix()` / `from_matrix(m, com, I)` (symmetric only),
  `rotated(q)` (`R·I·Rᵀ`), `is_physical()`, `as_array() -> (10,)`.
- `rotations.euler_to_matrix(r, p, y)` / `matrix_to_euler(R)`: ROS RPY straight to and from
  rotation matrices, no quaternion detour (a URDF `rpy` is printed verbatim); yaw = 0 at gimbal
  lock.

## Trajectory — one entity, many views

A trajectory is samples at strictly increasing times, in any domain. The base class holds the
time axis and the machinery; subclasses only declare their per-sample fields.

```python
traj = JointTrajectory(t=t, joint_names=names, positions=q, velocities=qd)

traj.positions  # (N, J) read-only array — the batch view, no copy
traj["positions"]  # same, by name (also works for channels)
traj[10]  # JointTrajectoryPoint at sample 10 (row views, no copy)
for point in traj:
    ...  # per-step view
traj[100:200]  # JointTrajectory, sliced as views — O(1)
traj.at(1.25)  # interpolated point (linear; slerp for orientations)
traj.resample(dt=0.004)  # on a fixed controller grid
traj.window(1.0, 2.0)  # the part between two times
traj.joint("elbow")  # one joint, still a JointTrajectory
traj.channels["kp"]  # any extra per-sample data you layer on top
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
- **Every stream a robot records is one** (2026-11-25, for sim-to-real: record every sensor and
  actuator stream, align, compare each): `ServoStateTrajectory` / `ServoCommandTrajectory` (what
  servos report and are sent; `fault` held, `seq` and `latency` as channels), `ImuTrajectory`,
  `WrenchTrajectory`, `ImageTrajectory` (frames held, never blended; `traj[i]` is still a point,
  `traj[i].image` / `image_at(t)` the `Image`, stamped with its own capture time). Each has a
  `from_<messages>` constructor (stamps → `t`, first stamp → `header.stamp`). A JointState log is
  `JointTrajectory.from_joint_states`; no new type.
- `t` is relative to `header.stamp`; `rebased()` moves `t[0]` to 0 and folds the offset into the
  stamp. Out-of-range interpolation raises unless `extrapolate="clamp"`.
- Recordings whose stamps repeat or step back go through `from_samples(..., duplicate_times=...)`,
  which says what it changed (below).

## Header.meta — context ROS has no field for

ROS's only per-message context is `std_msgs/Header`: a stamp and a `frame_id`. There is no free
metadata, so a consumer that needs to say *what the numbers are relative to* beyond a frame name
(the origin a joint angle is measured from, its units, which recorder wrote it, what a cleaner
changed) ends up encoding it into `frame_id` (bam_rack_twin did, 2026-11-04).

**Decision:** `Header.meta: Mapping[str, str]`, default empty, frozen (`FrozenDict`), copied at
construction.

- **WHY on the header, not on each message:** the header is already the slot that says what a
  message is relative to; extending it gives every stamped message and every trajectory the same
  place once, and everything that keeps the header keeps meta for free: slicing, masks,
  `window`, `resample`, `rebased`, `joint()`, `with_*`, `to_dict` / JSON, `save_npz`, pickling.
- **Points carry their trajectory's header** (`traj[i].header`, `traj.at(t).header`), so a point
  taken out keeps its frame, origin and units. ROS's points have no header; ours are views, and
  `t` is meaningless without the stamp it counts from. `from_points` takes the points' shared
  header back (and refuses points whose headers differ, unless you pass `header=`).
- **WHY strings only:** meta is read by people and by other tools. A float would be `str()`-ed
  silently and its spelling chosen for you, so non-strings are refused and the caller writes
  `repr(x)` (render stored values truthfully). Structured values go in as JSON strings.
- **`concatenate` refuses pieces with different meta**, as it refuses different frames: two
  trajectories measured from different origins are not one trajectory.
- **Dropped on ROS export** (`to_ros`), always, without `drop_extras`: std_msgs/Header has nowhere
  to put it and every stamped message exports through one header converter. `from_ros` gives an
  empty meta. robot_msgs' own files are the lossless form.
- **Not for per-sample data** (that is `channels`) or per-trajectory numbers a function computes
  with (those are typed fields, like `limits`). Meta is description, never input.

## Recorded stamps — `from_samples`

`t` must strictly increase, and the constructor keeps raising otherwise: interpolation, `dt`,
`window` and `searchsorted` all depend on it. Recordings break it (a loop logs one stamp twice, a
clock steps back), so there is one explicit door:

```python
traj, repair = JointTrajectory.from_samples(
    t=t, joint_names=names, positions=q, duplicate_times="drop_later"  # | "average" | "error"
)
repair.dropped  # (17, 230): indices into the samples as given; repair.dropped_t their stamps
str(repair)  # "drop_later: dropped 2 of 1000 samples (repeated or backwards stamps) at ..."
traj.header.meta["time_repair.dropped"]  # the same receipt, inside the trajectory
```

- `duplicate_times` has **no default**: the caller decides what a repeated stamp means.
  `"drop_later"` keeps a sample only if its stamp is after every earlier kept one (first of a
  repeat stays, backwards stamps go, values never altered). `"average"` merges samples at exactly
  equal stamps (floats averaged, quaternions on the short arc, held/discrete values must agree or
  it raises) and drops backwards stamps, which have no partner. `"error"` is the constructor.
- **WHY a receipt in two places** (the returned `TimeRepair` and `header.meta["time_repair.*"]`):
  silent repair is not allowed. A cleaned recording that does not say so is indistinguishable
  from a clean one; the meta copy means the saved file says so too. Clean input changes nothing
  and stamps nothing.

## Joint limits — physical and software

Zach, 2026-11-11: *"lets intro a new concept into RobotDescription we can have like physical limit or
software limit. Physical limits actually prevent further rotation. software limits don't."* and *"the
way to enforce that check is just afterwards to check the position read out and make sure it stayed
within the limits the entire time. So no need to like 'enforce' it in mujoco"*, and *"the URDF and the
mujoco limits should be the same"*.

**Decision:** `JointLimits.position_kind`, one entry per joint: `"physical"`, `"software"` or `None`.

- **`physical`** is a hard stop. Nothing gets past it, so a simulator enforces it (MuJoCo: the joint's
  `range` with `limited="true"`), and a plan that goes through it is impossible.
- **`software`** is a bound the plan and the controller must keep and the hardware does not enforce
  (UR's ±360°, a cable wrap). A simulator lets the joint through (`range` kept, `limited="false"`),
  and the run is judged afterwards by `joint_limit_violations`.
- **`None`** is "not stated": every limit written before the concept, and the field's default, so no
  existing value changes. A simulator keeps its own default (MuJoCo enforces a `range`).
- **One kind per joint, both bounds.** WHY: MuJoCo's `limited` is per joint, and the URDF and the
  simulator must carry the same two numbers; a per-bound kind could not be lowered without changing
  one of them. ROS's other model, a hard `<limit>` with a soft `<safety_controller>` range nested
  inside it, is the known extension for a joint that has both (the bench five-bar's 112.5° tour cap
  inside its 114.4° stop); no joint described so far needs it.
- **The evidence stays with the description.** `position_kind` holds the plain word; a description
  may hand in a `dataclass_values` `StrValue` with its source, and the message keeps `str(kind)`.

**The check:** `joint_limit_violations(trajectory, limits=None, *, quantities=...)` returns one
`JointLimitViolation` per joint, quantity and side the run goes past: the bound and its kind, the worst
sample (`value`, `excess`, `t`), the first and last sample past it, and how many there were. Rules:

1. **A sample on the bound is inside; a NaN is never past one.** No buffer: a bound is the number.
2. **No interpolation.** The record says what was sampled, so a run sampled too coarsely to show an
   excursion has none; sample finer or check the plan.
3. **Every kind is checked.** A software bound is only ever checked here; a physical one is past only
   in a world that does not enforce it (a kinematic plan through a stop) or by a physics engine's
   penetration into it, and its record says `physical`, so a report can tell "drove into the stop"
   from "left the software range".
4. **One rule, two views.** `JointTrajectory.limit_violations()` is the records' text (the sentence it
   always printed); the two can never disagree.

## Comparing trajectories — `robot_msgs.compare`

Trajectories first (rules 1–17), then every stream a robot records, single messages, point
clouds and datasets of runs (rules 18–23).

Zach, 2026-11-10: *"literally this package is saying we are comparing two of these data types of
robot msgs. The whole issue with ROS msgs is that they don't have enough of the helpers at arm's
reach."* So comparison is a robot_msgs helper, built like `viz`: its own subpackage, one generic
function dispatched on the type, and a one-line method (`traj.compare(other)`) so the type files
stay the types. Viewers (traj_compare's HTML report, Rerun) live outside and read the values.

```python
result = target.compare(real)               # == compare(target, real)
result.error                                # JointTrajectory: target - real, on real's samples
result.metrics["positions/j2"].rms          # ErrorStats: n rms mean median std min max max_abs p95_abs final
tip = lambda traj: PoseTrajectory.from_matrices(traj.t, fk(traj.positions), child_frame_id="L2 tip")
tip(target).compare(tip(real)).metrics["lag"].median    # cartesian: one PoseTrajectory per point
```

Rules, each with its reason:

1. **Pairwise.** `compare(reference, measured)`; N runs, or sim vs real with no target, is a loop.
2. **`error = reference - measured`**, the ROS 2 `JointTrajectoryControllerState` convention, so a
   positive error is the correction still to apply and a positive `lag` means the run is behind.
3. **The error is the same type as its inputs**, so every view, plot and serialiser already works
   on it. Derived forms ride along as channels: a pose error carries `rotation_vector` (world
   axes; never a difference of Euler angles), `rotation_angle`, `lag`, `cross_track`, `contour` and
   `lag_time`.
4. **Joint ↔ cartesian is not compare's job.** Build a `PoseTrajectory` per point you care about
   (TCP, L2 tip, coupler) with your FK, name it in `child_frame_id`, and compare those.
5. **One clock, no extrapolation.** Both runs are placed on `header.stamp + t`; measured samples
   outside the reference are dropped (`dropped`), and `coverage` says how much of the reference
   was measured. A known latency is an explicit `time_offset_s`, never guessed.
6. **Errors on the measured run's own samples**, never on a shared resampled grid.
7. **Every decision travels with the numbers.** `Comparison` holds `time_offset_s`, `dropped` and
   `coverage` beside `error` and `metrics`, and round-trips through JSON/npz like any message.
8. **Metric names are stable strings** (`positions/<joint>`, `position`, `rotation`, `x`…`rz`,
   `lag`, `cross_track`, `contour`, `lag_time`, `channels/<name>`), so adding a metric never
   changes a type. `rows()` gives them as tidy rows for tables.
9. **A verdict speaks ROS 2.** `check(reference, measured, Tolerances(...))` returns an `Outcome`
   whose `error_code` is `control_msgs/FollowJointTrajectory`'s (`SUCCESSFUL` 0,
   `PATH_TOLERANCE_VIOLATED` -4, `GOAL_TOLERANCE_VIOLATED` -5), with `JointTolerance` /
   `PoseTolerance` limits where 0 and -1 mean "not checked" (a `Tolerances` already is the
   controller default set; merging a per-goal tolerance onto it is the caller's job). The path
   phase is the measured samples inside the reference; the goal phase starts at the reference's end
   (inclusive) and is judged against its final point held, the controller's own convention and the
   one place a held reference is legitimate, never used for a path statistic.
   `stopped_velocity_tolerance` is the default goal velocity limit on every joint. A velocity or
   acceleration limit with no data raises; a default that cannot be checked (a measured run with
   no velocities) is named in `Outcome.unchecked`, never a silent pass. `ToleranceMonitor` is the
   same rule fed one tick at a time (what a controller runs); both go through one per-sample
   function, so online and offline can never disagree.
10. **N runs split into bias and spread (ISO 9283).** `aggregate(reference, runs)` resamples every
    run onto the reference instants they all cover, takes the barycentre, and returns the bias
    (`accuracy` = `compare(reference, barycentre)`) and the spread (`repeatability`, per instant
    RP = l̄ + 3·S_l, S_l with ddof=1). Per-sample error arrays are not averaged across runs (evo's
    `merge_results`): they only mean something on identical timelines, and a mean of signed
    errors hides the spread.
11. **Latency is estimated, then passed explicitly.** `estimate_time_offset` cross-correlates the
    velocities for a coarse lag and refines by minimising the position RMS; its `offset_s` has
    `align`'s sign. `compare` never applies it on its own (rule 5).
12. **Contour is exact, cross-track is first-order.** `lag` / `cross_track` split the error along
    the reference's tangent (no search, leaks lag into cross-track on curves). `contour` is the
    distance to the nearest point of the reference polyline, `lag_time` the seconds behind that
    point (`+` = behind), searched only within `contour_window_s` of the sample's reference time so
    a retraced route cannot snap to its other pass.
13. **Goal, settling and overshoot (`goal`).** `GoalStats` per joint / pose, MATLAB `stepinfo`'s
    conventions: settling time is NaN when the run never stays inside the band (never "the end of
    the data"); overshoot is past the final value along the reference's net move, as a percent of
    that move. Same held-final rule as rule 9, shared code.
14. **Drift is relative (`relative_error`).** evo's RPE over windows of seconds or metres of the
    reference's path: a constant offset cancels, a drift does not.
15. **Shape without time is `frechet`.** Eiter–Mannila discrete Fréchet over the position curves,
    no alignment; a run on the right path at the wrong speed scores near 0 here and badly in
    `compare`.
16. **Every `ErrorStats` carries `iae` / `ise` / `itae`** (trapezoids over intervals observed at
    both ends; NaN without a time axis, so older serialised data still loads).
17. **The time base travels with the error.** Measured `speed_scaling` and `segment_index`
    channels (`TIME_BASE_CHANNELS`) are copied verbatim as `time_base/<name>` and never
    differenced or summarised: a speed-scaled run's timing error is expected, and has to be read
    against the scaling.
18. **Every recorded stream compares by the same rules** (one clock, measured samples,
    `reference - measured`, error of the same type). Metric names:

    | Stream | Metrics |
    |---|---|
    | `ServoStateTrajectory` | `positions/<j>`, `velocities/<j>`, `effort/<j>`, `temperature/<j>`, `voltage/<j>` (fields both carry); `fault_mismatch`, `fault_mismatch/<j>` |
    | `ServoCommandTrajectory` | `positions/<j>`, `velocities/<j>`, `feedforward_effort/<j>`, `stiffness/<j>`, `damping/<j>`, `max_effort/<j>`; `channels/latency` |
    | `ImuTrajectory` | `rotation`, `rx ry rz`, `angular_velocity`, `linear_acceleration` (magnitudes), `angular_velocities/x\|y\|z`, `linear_accelerations/x\|y\|z` |
    | `WrenchTrajectory` | `force`, `torque` (magnitudes), `forces/x\|y\|z`, `torques/x\|y\|z` |
    | `ImageTrajectory` | `pixels` (every pixel of every frame), `mae`, `rmse`, `psnr` (per frame, over t) |
    | `PointCloud` | `distance`, `distance_reverse`, `chamfer` |

    Joint-keyed streams match servos by name (`match_joints`). A fault code is not a quantity:
    the measured codes are copied, and `fault_mismatch` is the `ErrorStats` of a 0/1 indicator
    (its `mean` is the fraction of samples whose codes differ; channel `fault_mismatch` (N, J)
    shows where). The IMU's orientation error is the pose's (`rotation_error`, shared code).
    `channels/<name>` follow every stream's own metrics, as before.
19. **Camera frames are held, not blended.** The reference frame compared with a measured frame
    is the latest one at or before it. Frames must have the same encoding and shape (a mismatch
    raises: rgb8 against bgr8 is a channel swap, not an error to measure). The error is float32
    (`32FC1` / `32FC3` / `32FC4` by channel count) in the input's pixel units, except 16UC1 depth
    (mm), whose error is in metres as 32FC1 says. `pixels` has no time axis (its integrals are
    NaN); `psnr` is for integer inputs, ``10·log10(MAX² / MSE)`` with MAX the dtype's largest
    value, and a frame equal to its reference (infinite PSNR) is not counted.
20. **One instant is one instant.** A measured stamp within `time_tolerance` of a reference
    sample is that sample (`align` snaps it). Without this, two streams on an epoch clock land a
    few ulps apart and a held field (a frame, a fault code, a mode) takes the previous sample.
21. **Two single messages compare by value.** `JointState`, `ServoState`, `ServoCommand`,
    `Imu`, `Wrench` and `Image` are wrapped as one-sample streams at `t = 0` (stamps and
    `time_offset_s` ignored), so `Comparison.error` stays a trajectory and every name above holds.
22. **Point clouds compare by nearest neighbour, with no clock.** `error` is the measured cloud
    with channel `distance` (each point to the nearest reference point, m) — the one place
    `Comparison.error` is not a trajectory, so code that needs a series checks
    `isinstance(error, Trajectory)`. `distance_reverse` is the other direction and `chamfer` both
    sets of distances as one signal (its `mean` is the point-weighted symmetric mean, its `max`
    the Hausdorff distance). Both clouds must be in one frame. Exact brute force in numpy
    (10 k × 10 k points: under a second on one core); downsample larger clouds first.
23. **Many runs are a `Dataset`.** A `Run` holds one `Recording` per source (`target`, `sim`,
    `real`: streams by name, the same names on every side), `tags`, `pairs` and `time_offset_s`.
    `pairs=None` is traj_compare's rule: the first recording against every other, then every
    other against each later one; a pair (a, b) uses ``offset[b] - offset[a]``.
    `compare_run(run)` compares every stream both sides recorded into a `RunComparison` keyed
    ``"<a>→<b>/<stream>"`` (recording names hold no `/` or `→`, so a key splits back); a stream
    only one side has, two streams of different types, and a pair `compare` refuses go into
    `skipped` with the reason, never silently (``errors="raise"`` lets the error through).
    `compare_dataset(dataset)` → `DatasetComparison`: `rows()` is the tidy table across runs (run,
    a, b, stream, metric, every `ErrorStats` field, the run's tags), `worst(metric, stat)` the run,
    pair and stream to drill into. WHY here (Zach, 2026-11-25, "Sim to real agent"): the dataset
    lived in traj_compare and bam_ws as a joint-trajectory-only type; robot_msgs takes the nouns
    and the math, so every tool reads one shape for every stream.

Prior art behind the rules: evo and trajectopy (alignment, pose error, RPE, time shift), ROS 2
`joint_trajectory_controller` and `control_msgs` (tolerances, result codes), ISO 9283
(accuracy / repeatability), CNC contouring (contour and lag), MATLAB `stepinfo` (settling,
overshoot), Eiter–Mannila (Fréchet).

Not yet in: MCAP / Parquet export of a `Comparison` (that lives in traj_compare, not here), and
any viewer beyond the stock `plot()` / Rerun `log()` every type has (robot_msg_viz builds the
polished ones). Each further analysis is a new metric or a new function over trajectories; none
changes the shape above.

## Shapes — solids for any 3-D scene

Zach, 2026-11-10: a robot description's solids *"don't just apply to robot descriptions but can be
used to describe like any 3D scene"*, so they live at the bottom of the stack
(`robot_msgs` ← `robot_description` ← everything), not in the description.

- **`SolidPrimitive` is ROS-literal**: `type` (`BOX=1 … PRISM=5`), `dimensions` in ROS's index
  layout (`BOX_X/Y/Z`, `SPHERE_RADIUS`, `CYLINDER_HEIGHT/RADIUS`, `CONE_HEIGHT/RADIUS`,
  `PRISM_HEIGHT`) and an `(N, 2)` `polygon` for the prism. `box()`, `sphere()`, `cylinder()`,
  `cone()`, `prism()` build one; `.size`, `.radius`, `.height` read it.
- **Origin: ROS centres every primitive's bounding box on 0,0,0.** Cylinder and cone lie along z,
  the cone's base at `z = -h/2` and apex at `+h/2`, so its centroid is at `z = -h/4`. `inertia(m)`
  is about the centroid and reports that offset as `com` (URDF `<inertial><origin>`).
- **Standard functions live with the type**: `volume()`, `centroid()`, `inertia(mass)`,
  `bounding_box()` (min, max `Point`s), `contains(points)` (surface inside), `to_mesh(resolution)`
  (closed, outward, vertices on the surface, so it converges from below). A pose that places the
  solid is the owner's, as with `BoundingBox3D.center`.
- **`Mesh`** is `shape_msgs/Mesh` as two arrays (`vertices (N, 3)`, `triangles (M, 3)` int32), plus
  the divergence-theorem solid maths (`volume()` signed, `centroid()`, `inertia(mass)`) and
  `transformed(pose)`. **`Plane`** is ROS's `coef` (a, b, c, d) with a signed `distance`.
- **`MeshResource(filename, scale)`** is a URDF/visualization_msgs-style file reference. ROS
  `shape_msgs` has no such type, so it has no ROS conversion; load the file into a `Mesh` to send it.
- **`Material` stays in `robot_description`**: ROS has no Material message (URDF's name, colour
  and texture is a description concept). The colour itself is `std.ColorRGBA`.

## Manipulation — grasps in, waypoint requests to the generator

Decided 2026-11-11 (Zach: "The input to the trajectory gen pipeline should be a WaypointTrajectory
or WaypointAction ... The grasping networks can output just a Grasp Pose ... this should be a msg
in robot_msgs"). Prior art: his bam_msgs `WaypointAction` and bam_traj_gen's working copy.

- **Two levels, one boundary each.** A detector that only knows where to grip emits a `GraspPose`
  (tool pose, close `width`, optional `pre_grasp_width` and `score`, frame in `header`). A pick
  recipe (bam_traj_gen `make_pick_and_place`), or a detector that wants its own approach,
  emits a `WaypointAction`. The generator takes only `WaypointAction` and returns a
  `JointTrajectory`.
- **One frame for every waypoint.** `WaypointAction.header.frame_id`, once; a waypoint holds a
  bare `Pose`, so a mixed-frame plan cannot be written (Zach's v3 note: "all points should be for
  the same frame ID no?"). `start_state` is a `JointState`, after MoveIt's MotionPlanRequest:
  you always plan from the current state.
- **Names are free text; labels are a closed set.** `Waypoint.name` is any string, unique in its
  action. `Waypoint.label` is a `WaypointLabel` (`start`, `home`, `approach`, `pre_grasp`,
  `grasp`, `close`, `lift`, `carry`, `pre_place`, `place`, `release`, `retreat`, `ready`), the
  word tools key on: a cycle ends at `ready`, and a moving target is caught at `close` unless
  `sync_target` names another waypoint. A `Literal` of strings, not an `enum.Enum`, so it
  serialises as itself and reads in JSON. Today's pick-and-place names map one to one (table in
  `waypoint.py`).
- **Intent only, never generated data.** A waypoint carries what its author decides: gripper
  width and effort cap, duration, dwell, blend radius, interpolation density, the moving
  target's velocity. IK solutions, blend bookkeeping (`start_blend_point`, offsets) and per-
  waypoint joint angles are the generator's working state and stay in bam_traj_gen; so do
  `call_function` hooks and Cartesian impedance (an executor setting, no producer yet).
- **Limits are the action's.** `cartesian_limits` (`CartesianLimits`, bam_traj_gen's defaults),
  `arm_limits` and `hand_limits` (`trajectory.JointLimits`, checked against `arm_joints` /
  `hand_joints`).

## Named poses — a robot's saved positions, by name

Decided 2026-11-15 (Zach: *"each robot description can basically have a number of saved positions
that could be like in joint space or Cartesian space. And then ready position is just a saved
position in space. And so all the code should not be like referencing an exact position, but should
rather be like accessing that saved position by name"*). The rig's READY had been copied into four
modules at three precisions (65.2 / 65.246 / the IK's full float); moving it meant finding them all.

```python
ready = NamedPose.from_degrees(name="ready", group="arm", degrees={"revolute_3": -12.7, ...},
                               why="where the arm waits between picks")
find_named_pose(description_poses, "ready").position_of("revolute_3")    # rad
ready.digest()                                  # "where it is", 16 hex: records carry it
named_poses_to_srdf(poses, "bam_fb_v2")         # MoveIt's <group_state>, values exact
named_poses_to_json(poses, "bam_fb_v2")         # lossless, each pose with its digest
```

- **The stock part is MoveIt's SRDF `<group_state>`** (`setNamedTarget` / `getNamedTargetValues`):
  name + group + one value per joint, in the URDF's joint frame. `named_poses_to_srdf` writes it
  (values printed with `repr`, so they read back to the same float) and `named_poses_from_srdf`
  reads one. Prior art at home: Zach's 2025 `RobotParam._init_saved_poses` (ZERO / HOME / READY as a
  joint state *and* the tool pose it gives, captured from the arm by `save_curr_pose.py`, which also
  printed the `<group_state>` to paste into moveit_config).
- **Joint or Cartesian, one type.** A joint pose's `positions` are its target. A Cartesian pose
  (`tool_pose` + `tip_link`) is resolved by an IK the caller passes, which returns every solution;
  the tie-break keeps the one whose largest single-joint move from the seed (`positions`) is
  smallest, then the smallest overall move (`nearest_solution`). WHY that order: a branch flip is
  one joint swinging far, which a sum over joints hides. A Cartesian pose refuses to hand out its
  seed as if it were the target (`position_of` raises; `resolved(ik)` first). robot_msgs has no IK
  of its own: kinematics belong to the description.
- **The digest names where the pose is.** sha256 (16 hex) over name, group, kind and every joint
  value sorted by joint name (plus the tool pose and tip link of a Cartesian one). Not `why`
  (rewording moves nothing) and not `tolerance` (how close counts is the reader's judgement). A
  record measured at "ready" stores the digest; a reader compares it with the description's and
  calls a mismatch stale, without comparing numbers. The algorithm is pinned by a test: changing it
  would call every record stale.
- **The description publishes a JSON beside the SRDF** (`named_poses_to_json`): each pose's own
  `to_dict` and its digest, so a reader with only a JSON parser (the rig's stdlib monitor) can check
  a record. `named_poses_from_json` refuses an entry whose digest is not its pose's own: the file
  was edited by hand; regenerate it from the description.
- **The description owns the poses; nothing else holds a number.** A consumer asks by name. A
  calibration fitted at a pose (an aim, a feed-forward basis) records the digest it was fitted at,
  so moving the pose shows up as a failing check, not a silent mismatch.
- **Not a message on any wire:** no ROS mapping (an SRDF entry is configuration, not traffic).

## Extending without forking

1. **Extra per-sample data → `channels`.** Gains, feed-forward torque, controller mode,
   temperature, a waypoint's name: `traj.with_channel("kp", kp)`. No new type needed.
2. **A new kind of trajectory → subclass `Trajectory` in your own file** and declare its
   `sample()` fields. Indexing, iteration, interpolation, resampling, plotting and serialisation
   come for free:

   ```python
   @dataclass(frozen=True, eq=False, repr=False, kw_only=True)
   class GripperTrajectory(Trajectory):
       width: Array = sample()  # metres
       grasping: Array | None = sample(default=None, interp="hold", dtype=bool)
   ```
3. **Business types → your own dataclasses that hold robot_msgs values** (composition). A
   `GraspCandidate(target: Detection3D, approach: Pose, quality: float)` lives in your
   application, not here. The exception is `GraspPose` (*Manipulation*): it crosses the
   grasp-detector → trajectory-generator seam, and every detector (Contact-GraspNet, GraspNet,
   GPD, a camera baseline) already emits the same pose + width + score.
4. **A new standard type** (it exists in a ROS package and crosses a wire between domains) →
   add it here, one file, with a catalog example; the contract tests then cover it.
   A type ROS keeps inside a process but that crosses *our* seams qualifies too when every
   vendor already agrees on its shape: `ServoCommand` is the position/velocity/feed-forward/kp/kd
   command every smart-servo board takes (Zach, 2026-11-11: named `Servo…`, not `Actuator…`, so
   the type says at a glance what it drives; a pneumatic or suction actuator gets its own type).

## Files — `save` / `load`

- `msg.save(path)` / `Type.load(path)` for one message; `robot_msgs.save(objs, path)` /
  `robot_msgs.load(path)` for a list, tuple or dict of messages (nested freely; the container
  comes back as it was). One tree format for both: `to_dict`, plus `{"_bundle": kind, "items"}`.
- Suffix rule: `.json` → indented JSON, arrays inline (readable); `.npz` → `save_npz`'s format
  (JSON tree in `__msg__`, raw arrays, no pickle); anything else → `.npz` appended, as
  `numpy.savez` does. WHY not a folder or a by-content pick: one file per name, one atomic rename,
  numpy's own rule.
- `Type.load` checks the file's `_type` before decoding and names both types when they differ;
  `Msg.load` returns whatever the file holds. Writes go to a temporary file in the same folder,
  are fsynced, then `os.replace`d: a failed save keeps the old file.
- No `RobotMemory` (Zach, 2026-11-26): a robot's memory is its `RobotState`. `RobotState` and
  `RobotConfig` are separate types, bundled into one file on demand — keras separates
  `get_config` from weights but `model.save` zips both; torch keeps `state_dict` apart from the
  code that builds the model.

## Storage is the cross-language contract; the API is the views

Zach, 2026-11-10: *"I don't really care about the internal implementation, do it for the best
performance on any language. With the views it can act however we want it to be."* So:

- **What callers see is the views** — `traj.positions`, `traj[i]`, iteration, `at(t)`, slices.
  They are the stable API; how a type stores its data behind them may change for speed.
- **What crosses a language boundary is the storage**, and it is chosen to be zero-copy in every
  language: one contiguous, little-endian, C-ordered array per field, time on axis 0
  (struct-of-arrays). That is exactly an Arrow record batch, an `Eigen::Map`, an `ndarray` view in
  Rust, or a flatbuffer vector — a C++ or Rust `robot_msgs` maps the same bytes without parsing.
- **Field names, units, frames and conventions are the semantics**, bound early and identical in
  every language; the technology (numpy today, C++/Rust bindings later) binds late.


## Collision scene

Zach, 2026-11-26 (Daily Note, "collision scene agent"): *"our model of the collision scene can continue
to improve, thats the key type though... in the simplest case its basically empty, in v1 it just has some
static objects in v2 we incorporate the camera occupancy map, dynamic objects etc."*

**Decision:** `robot_msgs.manipulation.CollisionScene` = moveit_msgs/PlanningScene's collision part, with
`CollisionElement` = moveit_msgs/CollisionObject (field names verbatim: `id`, `pose`, `primitives` +
`primitive_poses`, `meshes` + `mesh_poses`, `planes` + `plane_poses`) plus the tags MoveIt has no field for.

- **Two axes per element, from Zach's words:** `motion` fixed / moving and `compliance` rigid /
  deformable ("A concrete floor is fixed and non deformable. Any collision is not acceptable."; "objects
  are moving and deformable. actually a success grasp is one where you do get some deformation"), plus
  `source` scene / object / self (SceneDescription, the handled objects, the RobotDescription) and a
  `margin` in metres (coal/FCL `security_margin`, MoveIt link padding).
- **`contact` is a policy enum, v1 uses only `avoid`** ("as v1 we should say like any collision body
  assume its Avoid"). `allow` and `touch` are reserved names so a planner that needs contact later reads
  them without a schema change.
- **Planes are half-spaces** (solid opposite the normal). WHY: a two-sided thin plane lets a discrete
  check step through a floor.
- **The occupancy map is v2 and additive:** an optional field with source `sensor`, a minor version.
- **The checker and the viewer are robot_collision's** (`robot_collision.scene`); the type carries no
  geometry engine. `digest()` hashes what a checker reads, so a planner can say which scene it planned in.

## Fiducial markers — `robot_msgs.fiducial`

Zach, 2026-11-26: *"we want some standard robot_msgs for dealing with fiduciary markers (aruco
markers what I am doing right now)"*. Requirements came from the three tracks consuming them the
same day — the ArUco Kalman tracker (`~/aruco_marker_tracker_agent`), the marker bundle adjustment
(`~/aruco_bundle_adjustment_agent`) and the camera calibration (`~/aruco_camera_calibration_agent`)
— and from the marker types already in use (Aruco Studio's `TagSpec` / `BoardSpec`,
bam_rack_twin's `Fiducial` / `TagPoses`, bam_cad_descriptions' `Marker`, the rig `Board`s).

```python
frame = FiducialArray(header=image.header, detections=dets, poses=poses)   # one image
frame.as_arrays()["T_camera_marker"]     # (k, 2, 4, 4): both IPPE solutions per marker
mounts = FiducialMap(mounts=stickers) .merged(board.mounts())             # where they are
mounts.get(7).corners_in_parent()        # (4, 3) on its link, OpenCV corner order
tracks.predicted()                       # markers the filter carried through an occlusion
```

1. **Three layers, one key.** Measured (`FiducialDetection`, `FiducialPose`, `FiducialArray`),
   described (`FiducialMount`, `FiducialMap`, `FiducialBoard`), estimated (`FiducialTrack`,
   `FiducialTrackArray`). A marker is ``(dictionary, id)`` everywhere, with OpenCV's dictionary
   names (`"DICT_4X4_50"`, `"DICT_APRILTAG_36h11"`). WHY: the bundle adjustment must take
   measurements without depending on the detection pipeline (Zach); one key joins all three.
2. **`Fiducial*` everywhere, never `Marker*`** (the bundle-adjustment track asked for one prefix).
   It names the class of thing, ArUco and AprilTag alike, as fiducial_msgs does.
3. **Corners are OpenCV's**: top-left, top-right, bottom-right, bottom-left *as printed*, (4, 2)
   px. **The marker frame is OpenCV's `SOLVEPNP_IPPE_SQUARE` frame** (`marker_object_points`):
   origin at the centre, x right, y up, z out of the face. Every repo already agreed; tests check
   both against a marker OpenCV rendered, detected and solved.
4. **`marker_length` is the black square, border included**, m — the number PnP scales by
   (OpenCV's `markerLength`, Studio's `marker_length_m`), never the sticker's outer size.
5. **Both planar PnP solutions are kept** (`pose` + `alt_pose`, each with its reprojection error;
   `ambiguity`, `flipped()`). WHY: all three tracks asked; a small single marker has two poses
   that fit its corners (Collins & Bartoli 2014), and a consumer with a prior (robot FK, a filter)
   picks better than the solver can. `pose` is the chosen one, not necessarily the lower error.
6. **A mount's parent is any frame** (`header.frame_id`: a robot link, a board, the world), with an
   optional covariance (a CAD prior, or what a bundle adjustment estimated) and provenance in
   `header.meta`. A board is a special case: `FiducialBoard.mounts()` lowers a grid or ChArUco
   spec to mounts in OpenCV's board frame (top-left origin, x right, y down, z into the board;
   checked against `cv2.aruco.GridBoard` / `CharucoBoard`, legacy pattern included).
7. **Every 6×6 covariance here is local-tangent**: ``[δp, δθ]`` in the pose's own axes,
   ``p_true = p + R δp``, ``R_true = R Exp(δθ)``. WHY not ROS's fixed-axis ``[x y z rx ry rz]``:
   it is what the tracker and the bundle adjustment estimate in, and it survives a change of
   parent unchanged. `local_to_parent_covariance` gives ROS's (first order, ``blockdiag(R, R)``,
   checked by a Monte-Carlo draw).
8. **A track says whether it was seen.** `FiducialTrack.observed` (updated by a detection this
   frame or only predicted), `last_seen`, `nis` (the truth-free consistency check); a tracker
   reports every marker it knows, so occluded ones are in the array as predicted.
9. **The image header is the join.** `FiducialArray.header.stamp` (mid-exposure when known) and
   `frame_id` (the camera's optical frame) are how a solver finds the `JointState` and the
   `CameraInfo` of the same capture; no dedicated observation type. Capture settings are a `sensor.CameraSettings` (SI: exposure in s) stamped on
   the same camera frame; vendor knobs without a field ride in `header.meta`.
10. **Not yet:** ROS conversion (aruco_msgs / fiducial_msgs are not in rosbags' stock typestore),
    ChArUco interpolated-corner detections (the board gives their object points:
    `chessboard_corners()`), a stamped `PoseWithCovariance` for camera extrinsics, plotting.

