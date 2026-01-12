#!/usr/bin/env python3
"""
Example script demonstrating the ROHand gripper interface with crisp_py.

This example shows:
1. How to connect to the ROHand gripper
2. Basic open/close operations
3. Setting individual joint positions
4. Executing predefined gestures
5. Reading joint positions and efforts

Prerequisites:
- ROHand hardware connected and powered
- rohand_ros2 package launched with appropriate parameters
  
Example launch command:
  ros2 launch rohand rohand_ap001.launch.py \\
    port_name:=/dev/ttyUSB0 \\
    baudrate:=115200 \\
    hand_ids:=[2] \\
    namespace:=rohand_left
"""

import time
from crisp_py.gripper.rohand_gripper import make_rohand_gripper


def main():
    """Main example demonstrating ROHand gripper usage."""
    
    print("=" * 60)
    print("ROHand Gripper Example with crisp_py")
    print("=" * 60)
    
    # Method 1: Create gripper from YAML config using factory function
    print("\n[1] Creating ROHand gripper from config file...")
    gripper = make_rohand_gripper(
        config_name="rohand_left",
        namespace="rohand_left",
    )
    
    print("[✓] Gripper initialized")
    
    # Wait for gripper to be ready
    print("\n[2] Waiting for gripper to be ready...")
    try:
        gripper.wait_until_ready(timeout=10.0)
        print("[✓] Gripper is ready!")
    except TimeoutError as e:
        print(f"[✗] Error: {e}")
        print("    Make sure the rohand node is running:")
        print("    ros2 launch rohand rohand_ap001.launch.py namespace:=rohand_left")
        return
    
    # Display current joint positions
    print("\n[3] Current joint positions:")
    print("-" * 60)
    for joint_name in gripper.JOINT_NAMES:
        pos = gripper.get_joint_position(joint_name)
        eff = gripper.get_joint_effort(joint_name)
        print(f"  {joint_name:20s}: {pos:.3f} rad, effort: {eff:.2f}")
    print("-" * 60)
    
    # Execute predefined gestures
    print("\n[4] Executing gestures...")
    
    gestures = ['open', 'close', 'rock', 'pointing', 'open']
    
    for gesture in gestures:
        print(f"\n  → Executing gesture: {gesture}")
        gripper.execute_gesture(gesture)
        time.sleep(3.0)
        
        # Show some joint positions
        if_prox = gripper.get_joint_position('if_proximal_link')
        mf_prox = gripper.get_joint_position('mf_proximal_link')
        th_root = gripper.get_joint_position('th_root_link')
        print(f"    Index: {if_prox:.3f}, Middle: {mf_prox:.3f}, Thumb rot: {th_root:.3f}")
    
    # Test basic open/close
    print("\n[5] Testing basic open/close operations...")
    
    print("  → Opening gripper...")
    gripper.open()
    time.sleep(2.0)
    print(f"    Is open? {gripper.is_open()}")
    
    print("  → Closing gripper...")
    gripper.close()
    time.sleep(2.0)
    print(f"    Is closed? {gripper.is_closed()}")
    
    print("  → Using grasp() method...")
    gripper.grasp()
    time.sleep(2.0)
    
    # Control individual joints - close and open each finger sequentially
    print("\n[6] Controlling individual fingers...")
    
    print("  → Opening hand completely...")
    gripper.open()
    time.sleep(2.0)
    
    # Define fingers with their proximal joints and limits
    fingers = [
        ('Thumb', 'th_proximal_link', 0.7),
        ('Index', 'if_proximal_link', 1.44),
        ('Middle', 'mf_proximal_link', 1.44),
        ('Ring', 'rf_proximal_link', 1.44),
        ('Little', 'lf_proximal_link', 1.44),
    ]
    
    # Close each finger one by one
    print("\n  Closing each finger individually:")
    for finger_name, joint_name, max_pos in fingers:
        # Closing
        print(f"    → Closing {finger_name} finger...")
        gripper.set_joint_target(joint_name, max_pos)
        time.sleep(1.5)
        pos = gripper.get_joint_position(joint_name)
        print(f"       {finger_name} position: {pos:.3f} rad")
        # Opening 
        print(f"    → Opening {finger_name} finger...")
        gripper.set_joint_target(joint_name, 0.0)
        time.sleep(1.5)
        pos = gripper.get_joint_position(joint_name)
        print(f"       {finger_name} position: {pos:.3f} rad")
    
    time.sleep(1.0)
    
    # Monitor joint efforts during grasp
    print("\n[7] Monitoring joint efforts during grasp...")
    gripper.open()
    time.sleep(2.0)
    
    print("  → Closing to generate contact forces...")
    gripper.close()
    time.sleep(2.0)
    
    print("\n  Current joint efforts:")
    for joint_name in ['th_proximal_link', 'if_proximal_link', 
                       'mf_proximal_link', 'rf_proximal_link', 'lf_proximal_link']:
        effort = gripper.get_joint_effort(joint_name)
        print(f"    {joint_name:20s}: {effort:.2f}")
    
    # Final position
    print("\n[8] Returning to open position...")
    gripper.open()
    time.sleep(2.0)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Cleanup
    gripper.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
