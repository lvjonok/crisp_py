"""Try to follow a "figure eight" target on the yz plane."""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose

left_arm = make_robot("fep")
print(left_arm._current_joint)
left_arm.wait_until_ready()

# %%
print(left_arm.end_effector_pose)
print(left_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
homing_pose = left_arm.end_effector_pose.copy()


# %%
# Paremeters for the circle
radius = 0.2  # [m]
center = np.array([0.4, 0.0, 0.4])
# rot = Rotation.from_matrix([
#     [1.0, -0.0, -0.0],
#     [-0.0, -1.0, 0.0],
#     [-0.0, 0.0, -1.0]
# ])
# pose = Pose(position=center, orientation=rot)
ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# %%
left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    # file_path="config/control/gravity_compensation.yaml"
    # file_path="config/control/default_operational_space_controller.yaml"
    # file_path="config/control/clipped_cartesian_impedance.yaml"
    file_path="config/control/default_cartesian_impedance.yaml"
)

# %%
# The move_to function will publish a pose to /target_pose while interpolation linearly
left_arm.move_to(position=center, speed=0.15)

# %%
# The set_target will directly publish the pose to /target_pose
ee_poses = []
target_poses = []
ts = []

print("Starting to draw a circle...")
t = 0.0
target_pose = left_arm.end_effector_pose.copy()
rate = left_arm.node.create_rate(ctrl_freq)
log_every = max(int(ctrl_freq), 1)
step = 0

while t < max_time:
    x = center[0]
    y = radius * np.sin(2 * np.pi * sin_freq_y * t) + center[1]
    z = radius * np.sin(2 * np.pi * sin_freq_z * t) + center[2]
    target_pose.position = np.array([x, y, z])

    left_arm.set_target(pose=target_pose)

    rate.sleep()

    current_pose = left_arm.end_effector_pose.copy()
    target_pose_sample = left_arm._target_pose.copy()
    z_error = target_pose_sample.position[2] - current_pose.position[2]
    if step % log_every == 0:
        print(f"t={t:.2f}s z_error={z_error:.4f} m")

    ee_poses.append(current_pose)
    target_poses.append(target_pose_sample)
    ts.append(t)

    t += 1.0 / ctrl_freq
    step += 1

while t < max_time + 1.0:
    # Just wait a bit for the end effector to settle

    rate.sleep()

    current_pose = left_arm.end_effector_pose.copy()
    target_pose_sample = left_arm._target_pose.copy()

    ee_poses.append(current_pose)
    target_poses.append(target_pose_sample)
    ts.append(t)

    t += 1.0 / ctrl_freq


print("Done drawing a circle!")


# %%
y_t = [target_pose_sample.position[1] for target_pose_sample in target_poses]
z_t = [target_pose_sample.position[2] for target_pose_sample in target_poses]

# %%
# === Normal params ===
y_ee = [ee_pose.position[1] for ee_pose in ee_poses]
z_ee = [ee_pose.position[2] for ee_pose in ee_poses]

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(y_ee, z_ee, label="current")
ax[0].plot(y_t, z_t, label="target", linestyle="--")
ax[0].set_xlabel("$y$")
ax[0].set_ylabel("$z$")
# ax[0].legend()
ax[1].plot(ts, z_ee, label="current")
ax[1].plot(ts, z_t, label="target", linestyle="--")
ax[1].set_xlabel("$t$")
ax[1].legend()

for a in ax:
    a.grid()

fig.tight_layout()

plt.show()

# %%
rot_error_vecs = []
rot_error_angles = []
for target_pose_sample, ee_pose in zip(target_poses, ee_poses):
    rot_error = target_pose_sample.orientation * ee_pose.orientation.inv()
    rotvec = rot_error.as_rotvec()
    rot_error_vecs.append(rotvec)
    rot_error_angles.append(np.linalg.norm(rotvec))

rot_error_vecs = np.array(rot_error_vecs)
rot_error_angles = np.array(rot_error_angles)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ts, rot_error_vecs[:, 0], label="rx")
ax[0].plot(ts, rot_error_vecs[:, 1], label="ry")
ax[0].plot(ts, rot_error_vecs[:, 2], label="rz")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("rotvec [rad]")
ax[0].legend()
ax[1].plot(ts, rot_error_angles, label="angle")
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("angle [rad]")
ax[1].legend()

for a in ax:
    a.grid()

fig.tight_layout()

plt.show()

# %%

print("Going back home.")
left_arm.home()

# %%
left_arm.shutdown()
