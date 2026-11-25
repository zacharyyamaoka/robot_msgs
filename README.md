# robot_msgs

A cohesive domain model for robot data: the standard message types — ROS's `geometry_msgs`,
`sensor_msgs`, `trajectory_msgs`, `vision_msgs` — as frozen Python dataclasses, **with the
functions that move between them**. Offsetting a pose, interpolating a trajectory, projecting a
point, turning a depth image into a cloud: first-class methods, not separate libraries.

- **ROS's shape, not ROS.** Same packages and field names (the ROS docs are our docs), no ROS
  install, no codegen. ROS 2 messages and bags are an export (`robot_msgs.ros`).
- **Immutable values over compact numpy storage.** Every message is frozen; its arrays are
  read-only, so slices, rows and columns are free views.
- **Runtime dependencies: numpy and the stdlib-only `dataclass-values` (`Value`/`Space`).** Plots (`[viz]`), the Rerun viewer (`[rerun]`) and ROS export (`[ros]`) are extras.

## Look at any message: `.show()` and `.plot()`

```python
traj.show()                       # the Rerun viewer, with a dashboard for this type
traj.plot().savefig("joints.png") # a matplotlib figure, for reports

from robot_msgs.viz.rerun import show, save
show(target=planned, real=measured, tool=tool_path)   # one plot per joint, one line per name
save("pick.rrd", target=planned, real=measured)       # open later with `rerun pick.rrd`
```

`.show()` needs `pip install 'robot-msgs[rerun]'`; `.plot()` needs `[viz]`. Every message type
has both; the core never imports either.

### A JointTrajectory as its robot: `robot=`

```python
traj.show(robot="ur5e.urdf")                              # the arm, from the robot's own URDF
traj.show(robot="ur5e.urdf", tool=tcp)                    # tcp: the tool's PoseTrajectory
save("plan.rrd", plan=traj, robot="ur5e.urdf", tool=tcp)  # ghosts=False for a web bundle of many
```

The page is split in half. On the left, always: the URDF arm moving on the time bar (Rerun's own
URDF loader), faint copies at its start (blue) and end (orange) poses, and the tool's path, solid
up to the playhead and faint after it. On the right, tabs: **q**, **qd**, **qdd**, **qddd** (a
plot per joint in degrees with its limit lines; the title gives its travel and distance to the
limit, or its peak and % of limit; qddd is the slope of qdd), **q / qd / qdd** (each joint's three
as % of its own limits under a table of the values at the playhead) and **TCP** (position,
roll / pitch / yaw, speed, angular speed). It opens paused at t = 0, on q; play loops.

- `robot` is a URDF path; robot_msgs gains no dependency (Rerun's `rerun.urdf` reads it). The
  trajectory's joint names must be URDF joints. Revolute joints are shown in degrees, prismatic
  ones in metres; the message itself stays SI.
- Limits: the message's `JointLimits` first, else the URDF's (position, velocity).
- `tool`'s `header.frame_id` names a URDF link (empty = the root link). Without `tool`, the tool
  is the URDF's last moving link, by forward kinematics.
- A tab whose data the message lacks is left out (no velocities: no qd, qdd, qddd).
- `log(traj, name, recording=rec, robot=…, tool=…)` composes into your own recording, and
  `robot_blueprint(traj, name, robot=…)` is its layout. Without `robot=`, nothing changes.
- No URDF? `log_joint_tabs(traj, name, recording=rec)` logs the same right-hand tabs (q, qd, qdd,
  qddd, q / qd / qdd; every joint read as revolute) and `joint_tabs_blueprint(traj, name)` lays
  them out at full width. robot_msg_viz's JointTrajectory view uses them.

## A trajectory is one entity with many views

```python
import numpy as np

from robot_msgs.geometry import Pose
from robot_msgs.trajectory import JointTrajectory

t = np.linspace(0.0, 2.0, 201)
traj = JointTrajectory(
    t=t,
    joint_names=("shoulder", "elbow"),
    positions=np.stack([np.sin(t), np.cos(t)], axis=1),
).with_derivatives()

print(traj)                                 # a summary, never 400 numbers
print(traj.positions.shape)                 # batch view, no copy
print(traj[100])                            # one point
print(traj.at(0.123).positions)             # interpolated at any time
for point in traj[:2]:                      # slices are views too
    print(point.t, point.positions)
print(traj.joint("elbow").joint_names)      # one joint, still a JointTrajectory
print(traj.resample(dt=0.5).t)              # onto a fixed grid

grasp = Pose.from_xyz_rpy(0.5, 0.0, 0.2, np.pi, 0.0, 0.0)  # tool pointing down
pregrasp = grasp.shifted(dz=-0.10, local=True)              # back off along the tool
print(pregrasp.position)
```

```text
JointTrajectory(N=201, t=[0 … 2] s, joint_names=('shoulder', 'elbow'), positions=float64[201×2], velocities=float64[201×2], accelerations=float64[201×2])
(201, 2)
JointTrajectoryPoint(t=1.0, positions=[0.8415, 0.5403], velocities=[ 0.5403, -0.8415], accelerations=[-0.8414, -0.5403])
[0.12268879 0.99243461]
0.0 [0. 1.]
0.01 [0.00999983 0.99995   ]
('elbow',)
[0.  0.5 1.  1.5 2. ]
Point(x=0.5, y=1.2246467991473533e-17, z=0.30000000000000004)
```

`JointTrajectory` and `PoseTrajectory` subclass the domain-free `Trajectory` (samples at strictly
increasing times). Anything else you want to send along rides as `channels`
(`traj.with_channel("kp", kp)`); a new kind of trajectory subclasses `Trajectory` and declares
its `sample()` fields — it gets indexing, interpolation, plotting and serialisation for free.

Context ROS has no field for (an origin, units, provenance) rides `Header.meta`, a frozen
`str → str` mapping that survives every view and file and is dropped on ROS export. A recording
whose stamps repeat or step back is built with
`JointTrajectory.from_samples(..., duplicate_times="drop_later" | "average" | "error")`, which
returns `(trajectory, repair)` and writes the same receipt into `header.meta`; never silently.

## Packages

| Import | Types |
|---|---|
| `robot_msgs.std` | `Header` `ColorRGBA` · a running component's own reports: `LogRecord` `DiagnosticRecord` `Metrics` |
| `robot_msgs.geometry` | `Vector3` `Point` `Quaternion` `Pose` `Pose2D` `Transform` `Twist` `Wrench` `Accel` `Inertia` `PoseStamped` `TransformStamped` · batch maths in `geometry.rotations` (incl. direct `euler_to_matrix` / `matrix_to_euler`) |
| `robot_msgs.sensor` | `JointState` `Image` `CameraInfo` `PointCloud` `Imu` |
| `robot_msgs.control` | `ServoState` `ServoCommand` |
| `robot_msgs.shape` | `SolidPrimitive` (box, sphere, cylinder, cone, prism: `volume` `inertia` `bounding_box` `contains` `to_mesh`) `Mesh` `Plane` `MeshResource` (a URDF-style file reference; no ROS message) |
| `robot_msgs.vision` | `BoundingBox2D` `BoundingBox3D` `ObjectHypothesis` `ObjectHypothesisWithPose` `Detection2D` `Detection2DArray` `Detection3D` `Detection3DArray` |
| `robot_msgs.trajectory` | `Trajectory` `JointTrajectory` `PoseTrajectory` (+ points) `JointLimits` (physical / software `position_kind`) `JointLimitViolation` `joint_limit_violations` `concatenate` `TimeRepair` |
| `robot_msgs.viz` `[viz]` | `plot(msg)` for every type, `animate(*trajectories, path=...)` |
| `robot_msgs.ros` `[ros]` | `to_ros` / `from_ros` (ROS 2 Jazzy via rosbags), `write_bag` / `read_bag` (mcap) |

Every message: `==` and `hash` by value, a short `repr`, `replace(...)`, `to_dict()` /
`from_dict()`, `save_npz` / `load_npz`, pickling.
`robot_msgs.adopt(array)` hands a numpy array over without a copy (a view of a received buffer).

## Conventions

SI units · quaternions `(x, y, z, w)` (MuJoCo/viser `wxyz` only through `xyzw_to_wxyz`) ·
Euler = ROS roll-pitch-yaw (fixed X, Y, Z) · `a @ b` applies `b` in `a`'s frame ·
`Header.stamp` float seconds, trajectory `t` relative to it · `Header.meta` str → str · absent optional arrays are `None`.
The full contract, what we keep from ROS and what we drop, and how to extend: [docs/DESIGN.md](docs/DESIGN.md).

## Develop

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msgs --isolated --extra test python -m pytest -q
```

Examples: `examples/tour.py` (above), `examples/ur5e_pick.py` (waypoints → minimum-jerk joint
trajectory → tool pose trajectory through UR5e FK), `examples/gallery.py OUTDIR` (every plot).
