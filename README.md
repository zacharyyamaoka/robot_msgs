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

**Landing page: [robotmsgs.com](https://robotmsgs.com)**

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
- Every sample is a dot: each joint, jerk and TCP plot draws one per sample over its line (they
  come apart as you zoom in), and the tool path has them as `/world/<name>/tool/samples`. Hide
  the 3D ones with that entity's eye in the blueprint panel; in a plot, select the series and
  remove SeriesPoints under Selection › Visualizers (**+** › SeriesPoints puts them back).
- `log(traj, name, recording=rec, robot=…, tool=…)` composes into your own recording, and
  `robot_blueprint(traj, name, robot=…)` is its layout. Without `robot=`, nothing changes.
- No URDF? `log_joint_tabs(traj, name, recording=rec)` logs the same right-hand tabs (q, qd, qdd,
  qddd, q / qd / qdd; every joint read as revolute) and `joint_tabs_blueprint(traj, name)` lays
  them out at full width. robot_msg_viz's JointTrajectory view uses them.

## Save and load: `state.save("arm")`

```python
state = RobotState(components={"bt": BTState(tree_id="Bringup"), "joints": joints})
state.save("arm")                          # arm.npz: compact, arrays raw, no pickle
state = RobotState.load("arm")             # TypeError if arm.npz holds another type
bt.save("bt.json")                         # .json: human-readable, arrays inline
robot_msgs.save({"config": config, "state": state, "history": states}, "step.npz")
step = robot_msgs.load("step.npz")         # the same dict, list and tuple back
```

The suffix picks the encoding: `.json` readable, `.npz` compact, anything else gets `.npz`
appended (numpy.savez's rule). Every write is atomic (temporary file, then `os.replace`), so a
crash mid-save keeps the old file. A bundle is any list, tuple or dict of messages, nested as deep
as you like. `RobotState` and `RobotConfig` hold named component messages (`state["joints"]`,
`state.with_components(joints=measured)`); `BTState` is a behaviour tree's statuses, blackboard,
tick and clock.

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

## Compare anything a robot records: `compare(reference, measured)`

```python
from robot_msgs.compare import Dataset, Recording, Run, compare, compare_dataset

result = target.compare(real)                  # error = target - real, same type, real's samples
result.metrics["positions/j2"].rms             # ErrorStats: rms mean median max_abs p95_abs final iae …

compare(sim_servos, real_servos)               # ServoStateTrajectory: positions/<j> … fault_mismatch
compare(sim_camera, real_camera)               # ImageTrajectory: pixels, mae, rmse, psnr
compare(sim_scan, real_scan)                   # PointCloud: distance, distance_reverse, chamfer
imu_a.compare(imu_b)                           # two single messages: values only

run = Run(name="pick_0.5", tags={"speed": 0.5}, recordings={
    "target": Recording(streams={"joints": plan}),
    "sim": Recording(streams={"joints": sim_q, "servo_state": sim_servos, "camera": sim_camera}),
    "real": Recording(streams={"joints": real_q, "servo_state": real_servos, "camera": real_camera}),
})
sweep = compare_dataset(Dataset(runs=(run, ...)))  # target→sim, target→real, sim→real, per stream
sweep.rows()                                       # one tidy row per run, pair, stream, metric + tags
sweep.worst("positions/j2")                        # (run, a, b, stream, rms): where to drill in
```

Streams are trajectories: `ServoStateTrajectory`, `ServoCommandTrajectory`, `ImuTrajectory`,
`WrenchTrajectory` and `ImageTrajectory` beside `JointTrajectory` and `PoseTrajectory`, each built
from its message log (`ServoStateTrajectory.from_servo_states(states)`). Rules and every metric
name: [docs/DESIGN.md](docs/DESIGN.md), "Comparing trajectories".

## Joint commands: a recipe the controller checks as it runs

```python
from robot_msgs.control import JointCommandController, JointCommandSequence
from robot_msgs.control.helpers import find_hardstop, hold, move_to, stop_all

recipe = JointCommandSequence(name="home elbow, then park", steps=[
    hold("shoulder", max_effort=12.0),
    find_hardstop("elbow", direction=-1, max_vel=0.3, max_effort=3.0, max_travel=1.2,
                  repeats=3, define=-2.356),          # 3 touches, mean defines; spread > 1 mrad refuses
    move_to("elbow", -1.0, max_vel=1.0, max_effort=8.0),
    stop_all(),
])
jcc = JointCommandController(joint_names)
goal = jcc.submit(recipe, client="homing")
out = jcc.tick(servo_state, dt)      # out.command (ServoCommand with .mode), out.defines, out.log
jcc.result(goal)                      # done | aborted | preempted | stopped | rejected | faulted, + hardstops
```

Helpers are plain functions returning lists of three primitives (`Command`, `Delay`, `Condition`), so
a recipe is data any viewer or language reads. A sequence that ends badly holds where the arm is;
stop and brake are board modes, sent only for a fault, a lost bus or an explicit `stop()`.

Each primitive remembers where it came from in the recipe, as metadata that never changes what runs:
`call` is the helper call as written (`'find_hardstop("elbow", direction=-1, max_vel=0.3, ...)'`,
stamped by every helper), and `helpers.step("Home the elbow", find_hardstop(...), move_by(...))` adds
your own words as `label`. `helpers.outline(recipe)` reads them back as rows of label > call >
primitives, which is what a log or a viewer shows first, with the three primitive kinds one level lower.

`robot_msgs.control.recipe` is the short way to write the same recipe, so every number says what it is.
One vocabulary of keywords (`pos` `by` rad, `vel` rad/s, `max_tau` `ff` `tau` N·m, `tol`, `dir`, `travel`,
`timeout`, `repeats`, `define`; `recipe.signature("move_by")` lists a verb's own), a `Joint` that declares
its defaults once, a `Group` addressed all / a set / one as moteus tview does (`A>`, `11,12>`, `11>`), and
tview-style text lines. Each calls the helper of the same name, so the primitives are identical; only
`call` reads as the step was written.

```python
from robot_msgs.control.recipe import ALL, Group, Joint, deg, script, wait

J1 = Joint("shoulder", max_tau=12, ff=5)            # defaults, declared once
J2 = Joint("elbow", vel=0.5, max_tau=3, id=12)
Arm = Group(J1, J2, name="Arm")

J1.hold()                                            # a keyword left out takes the joint's default
J2.find_hardstop(dir=-1, travel=1.0, define=deg(-135))
J2.move_by(deg(20), max_tau=8)                       # ... and one on the step overrides it
Arm.stop()                                           # ONE Command to every joint; also stop(Arm)
Arm[ALL].limp(); Arm[J1, J2].brake(); Arm[12].hold() # all, a set, or one (by Joint, name or id)
script("""
    Park 20° off the stop:                           # a label for the indented steps under it
        elbow> move_by 20deg max_tau=8
        wait 0.3
    A> stop
""", joints=Arm)
```

## Packages

| Import | Types |
|---|---|
| `robot_msgs.std` | `Header` `ColorRGBA` · a running component's own reports: `LogRecord` `DiagnosticRecord` `Metrics` |
| `robot_msgs.geometry` | `Vector3` `Point` `Quaternion` `Pose` `Pose2D` `Transform` `Twist` `Wrench` `Accel` `Inertia` `PoseStamped` `TransformStamped` · batch maths in `geometry.rotations` (incl. direct `euler_to_matrix` / `matrix_to_euler`) |
| `robot_msgs.sensor` | `JointState` `Image` `CameraInfo` `CameraSettings` `PointCloud` `Imu` |
| `robot_msgs.control` | `ServoState` `ServoCommand` (per-servo `mode`: `SERVO_MODES`) · routines as data: `JointCommand`, the primitives `Command` `Delay` `Condition` (+ `Record` `Reduce` `FailWhen`, predicates `InCollision` `TravelEnd` `FollowEnded` `Reached` `Settled` `Stall` `EffortStall` `Elapsed`), `JointCommandSequence`; helper functions in `control.helpers` (`find_hardstop` `move_to` `hold` `stop_all` `follow` `wait_until` `at_once` ...); the short way to write them in `control.recipe` (`Joint` `Group` `script`, keywords `pos` `by` `vel` `max_tau` `ff` ...); `JointCommandController` (the one writer to the servos) · the Arm's `JointTable` (joint name -> `Servo`: bus, CAN id, sign, offset; `to_servos` / `to_joints` / `with_zero`) · `HardwareInterface(deployment, backend="sim" \| "real")` (a `HardwareDeployment`: the table + a backend factory per name; `read()` / `write()` joint-named; `Arm` is the deprecated name) and `GroundTruth`, the simulator's truth a simulated `ServoState` carries in `truth` (None on real) |
| `robot_msgs.shape` | `SolidPrimitive` (box, sphere, cylinder, cone, prism: `volume` `inertia` `bounding_box` `contains` `to_mesh`) `Mesh` `Plane` `MeshResource` (a URDF-style file reference; no ROS message) |
| `robot_msgs.manipulation` | `GraspPose` `Waypoint` `WaypointAction` `CartesianLimits` `NamedPose` `AllowedCollisionMatrix` · `CollisionScene` of `CollisionElement`s (moveit_msgs PlanningScene / CollisionObject; each element tagged scene / object / self, fixed / moving, rigid / deformable, with a `margin` and a `contact` policy, v1 `avoid`; checked and viewed by robot_collision) |
| `robot_msgs.behavior` | `BTState` (a behaviour tree's node statuses by path, JSON blackboard, tick, clock) |
| `robot_msgs.robot` | `RobotState` `RobotConfig` (the whole robot as named component messages; nest freely) |
| `robot_msgs.vision` | `BoundingBox2D` `BoundingBox3D` `ObjectHypothesis` `ObjectHypothesisWithPose` `Detection2D` `Detection2DArray` `Detection3D` `Detection3DArray` |
| `robot_msgs.fiducial` | ArUco / AprilTag / ChArUco: `FiducialDetection` (id, dictionary, 4 corner px) `FiducialPose` (`T_camera_marker` + the other IPPE solution) `FiducialArray` · `FiducialMount` (a marker on a link, board or world) `FiducialMap` `FiducialBoard` (grid / ChArUco → its mounts, OpenCV geometry) · `FiducialTrack` `FiducialTrackArray` (filtered pose, observed or predicted) |
| `robot_msgs.trajectory` | `Trajectory` `JointTrajectory` `PoseTrajectory` (+ points) `JointLimits` (physical / software `position_kind`) `JointLimitViolation` `joint_limit_violations` `concatenate` `TimeRepair` · recorded streams: `ServoStateTrajectory` `ServoCommandTrajectory` `ImuTrajectory` `WrenchTrajectory` `ImageTrajectory` |
| `robot_msgs.compare` | `compare` → `Comparison` (`ErrorStats` per metric) for every stream, single message and point cloud · `check` `aggregate` `goal` `relative_error` `frechet` `estimate_time_offset` · datasets: `Recording` `Run` `Dataset` → `compare_dataset` → `DatasetComparison` |
| `robot_msgs.viz` `[viz]` | `plot(msg)` for every type, `animate(*trajectories, path=...)` |
| `robot_msgs.ros` `[ros]` | `to_ros` / `from_ros` (ROS 2 Jazzy via rosbags), `write_bag` / `read_bag` (mcap) |

Every message: `==` and `hash` by value, a short `repr`, `replace(...)`, `to_dict()` /
`from_dict()`, `save(path)` / `load(path)` (`.json` or `.npz`), `save_npz` / `load_npz`, pickling.
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
