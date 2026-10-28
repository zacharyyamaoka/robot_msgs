"""The README tour. Every line here is run and its real output pasted into the docs."""

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
