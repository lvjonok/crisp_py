"""Try to follow a "figure eight" target on the yz plane."""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose

left_arm = make_robot("panda")
print(left_arm._current_joint)
left_arm.wait_until_ready()

# %%
print(left_arm.end_effector_pose)
print(left_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()

# %%
left_arm.shutdown()
