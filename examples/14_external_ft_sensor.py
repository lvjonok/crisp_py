"""Example demonstrating the use of the External FT Sensor.

This example shows how to:
1. Create an external FT sensor that subscribes to WrenchStamped messages
2. Use the franka_ft_sensor ROS2 node for gravity bias subtraction
3. Transform wrench measurements to a different frame (e.g., robot tip)

Prerequisites:
- ROS2 robot bringup running with TF tree published
- franka_ft_sensor node running:
  
  # For raw data only:
  ros2 launch franka_ft_sensor franka_ft_sensor.launch.py \\
      ft_ip:=172.16.0.12 \\
      frame_id:=ft_sensor_link

  # With gravity compensation and EE transformation:
  ros2 launch franka_ft_sensor franka_ft_sensor.launch.py \\
      ft_ip:=172.16.0.12 \\
      frame_id:=ft_sensor_link \\
      subtract_bias:=true \\
      transform_to_ee:=true \\
      ee_frame_id:=panda_hand_tcp \\
      joint_names:="['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']"

Published topics by franka_ft_sensor node:
- netft_data: Raw wrench from sensor
- netft_data_unbiased: Gravity-compensated wrench (if subtract_bias:=true)
- netft_data_ee: Wrench at EE frame (if transform_to_ee:=true)
"""

# %%
import time

import numpy as np

from crisp_py.sensors import ExternalFTSensor, ExternalFTSensorConfig, make_external_ft_sensor

# %% Option 1: Subscribe to raw data
sensor_raw = make_external_ft_sensor(
    config_name="external_ft_sensor",
    data_topic="/netft_data",  # Raw data from sensor
)

# %% Option 2: Subscribe to gravity-compensated data
# This uses the Pinocchio-based compensation in the ROS2 node
sensor_unbiased = make_external_ft_sensor(
    config_name="external_ft_sensor",
    data_topic="/netft_data_unbiased",  # Gravity-compensated
)

# %% Option 3: Subscribe to EE-transformed data
sensor_ee = make_external_ft_sensor(
    config_name="external_ft_sensor",
    data_topic="/netft_data_ee",  # Transformed to EE frame
)

# %% Option 4: Create sensor with inline configuration
config = ExternalFTSensorConfig(
    name="inline_ft_sensor",
    data_topic="/netft_data_unbiased",
    shape=(6,),
    # These processing options are for additional Python-side processing
    # (the main compensation is done by the ROS2 node)
    subtract_bias=False,
    transform_to_tip=False,
    # Data parameters
    max_data_delay=0.5,
    buffer_size=100,
)

sensor_inline = ExternalFTSensor(config=config)

# %% Wait for sensor to be ready
print("Waiting for external FT sensor...")
sensor_raw.wait_until_ready(timeout=10.0)
print("Sensor ready!")

# %% Read sensor values
print(f"Raw wrench: {sensor_raw.value}")
print(f"Force: {sensor_raw.force}")
print(f"Torque: {sensor_raw.torque}")

# %% Continuous reading loop
print("\nStarting continuous reading (Ctrl+C to stop)...")
try:
    while True:
        force = sensor_raw.force
        torque = sensor_raw.torque
        force_magnitude = np.linalg.norm(force)

        print(
            f"Force: [{force[0]:7.2f}, {force[1]:7.2f}, {force[2]:7.2f}] N "
            f"(|F|={force_magnitude:.2f} N), "
            f"Torque: [{torque[0]:7.3f}, {torque[1]:7.3f}, {torque[2]:7.3f}] Nm"
        )
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")

# %% Access the buffer for recent measurements
recent_wrenches = sensor_raw.buffer.get()
print(f"Buffer shape: {recent_wrenches.shape}")
print(f"Average force magnitude: {np.mean(np.linalg.norm(recent_wrenches[:, :3], axis=1)):.2f} N")
