"""Example demonstrating action-based gripper control.

This example shows how to use the ActionGripper class which controls
grippers via ROS2 actions (control_msgs.action.GripperCommand) instead
of topics. This provides better feedback and control over the gripper state.
"""

# %%
import time

from crisp_py.gripper.gripper import ActionGripper, make_gripper

# Method 1: Create ActionGripper directly from config
print("Creating ActionGripper from config...")
gripper = ActionGripper.from_yaml("action_gripper_example")
gripper.wait_until_ready(timeout=10.0)
print(f"Gripper ready! Current value: {gripper.value}")

# Method 2: Use make_gripper factory with use_action=True
# gripper = make_gripper("action_gripper_example", use_action=True)
# gripper.wait_until_ready()

# %%
# Check initial gripper state
print("\n--- Initial State ---")
print(f"Position (normalized 0-1): {gripper.value}")
print(f"Position (raw meters): {gripper.raw_value}")
print(f"Torque/Effort: {gripper.torque}")
print(f"Target: {gripper.target}")
print(f"Min value: {gripper.min_value}")
print(f"Max value: {gripper.max_value}")

# %%
# Non-blocking commands (default)
print("\n--- Non-blocking Commands ---")
print("Opening gripper (non-blocking)...")
gripper.open()
time.sleep(3.0)
print(f"Gripper opened to: {gripper.value}")

print("Closing gripper (non-blocking)...")
gripper.close()
time.sleep(3.0)
print(f"Gripper closed to: {gripper.value}")

# %%
# Blocking commands - wait for action to complete
print("\n--- Blocking Commands ---")
print("Opening gripper (blocking, will wait for completion)...")
gripper.set_target(1.0, blocking=True)
print(f"Gripper opened to: {gripper.value}")

print("Closing gripper (blocking)...")
gripper.set_target(0.0, blocking=True)
print(f"Gripper closed to: {gripper.value}")

# %%
# Custom max_effort for different objects
print("\n--- Custom Max Effort ---")
print("Gentle grasp with low effort...")
gripper.set_target(0.3, blocking=True, max_effort=10.0)
print(f"Position: {gripper.value}, Effort: {gripper.torque}")

print("Strong grasp with high effort...")
gripper.set_target(0.3, blocking=True, max_effort=50.0)
print(f"Position: {gripper.value}, Effort: {gripper.torque}")

# %%
# Pick and place cycle
print("\n--- Pick and Place Cycle ---")
for i in range(3):
    print(f"Cycle {i+1}/3")
    
    # Open gripper
    print("  Opening...")
    gripper.set_target(0.8, blocking=True, max_effort=20.0)
    time.sleep(0.5)
    
    # Close to grasp
    print("  Grasping...")
    gripper.set_target(0.2, blocking=True, max_effort=30.0)
    time.sleep(0.5)
    
    # Release
    print("  Releasing...")
    gripper.set_target(0.8, blocking=True, max_effort=10.0)
    time.sleep(0.5)

print("\nDone!")

# %%
# Monitor gripper state continuously
print("\n--- Continuous Monitoring (10 seconds) ---")
freq = 10.0
rate = gripper.node.create_rate(freq)
t = 0.0

# Move gripper while monitoring
gripper.set_target(0.5, max_effort=25.0)

while t < 10.0:
    print(f"Time: {t:.1f}s | Position: {gripper.value:.3f} | "
          f"Raw: {gripper.raw_value:.4f}m | Torque: {gripper.torque}")
    rate.sleep()
    t += 1.0 / freq

print("Monitoring complete!")
