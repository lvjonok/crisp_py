#!/usr/bin/env python3
"""
MetaQuest Hand Pose Teleoperation for Panda Robot.

Uses the hand-pose channel from MetaQuest to control Panda robot position
via delta tracking. Control is only active while a key is held.

Coordinate Systems:
- Quest/Unity: X-right, Y-up, Z-forward (left-handed)
- FLU (Robot): X-forward, Y-left, Z-up (right-handed)

Usage:
    python metaquest_panda_teleop.py

Controls:
    SPACE - Toggle teleoperation on/off
    q     - Quit

Prerequisites:
- MetaQuest streaming hand tracking data
- Panda robot connected and ready
"""

import sys
import time
import select
import termios
import tty
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
from meta_teleop import MetaTeleopClient, HandPoseData, CoordinateSystem
from meta_teleop.transforms import positions_unity_to_flu, quaternions_unity_to_flu


class KeyboardInput:
    """Non-blocking keyboard input handler."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> Optional[str]:
        """Get key if pressed, None otherwise (non-blocking)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def unity_position_to_flu(pos_x: float, pos_y: float, pos_z: float) -> np.ndarray:
    """
    Convert Unity position to FLU coordinate system.

    Unity: X-right, Y-up, Z-forward
    FLU:   X-forward, Y-left, Z-up

    Returns:
        np.ndarray: [flu_x, flu_y, flu_z]
    """
    flu_x = pos_z   # forward = unity forward
    flu_y = -pos_x  # left = negative unity right
    flu_z = pos_y   # up = unity up
    return np.array([flu_x, flu_y, flu_z])


def unity_quaternion_to_flu(rot_x: float, rot_y: float, rot_z: float, rot_w: float) -> Rotation:
    """
    Convert Unity quaternion to FLU coordinate system as scipy Rotation.

    Both Unity and scipy use scalar-last quaternion format (x, y, z, w).

    Two-step transform:
    1. Axis mapping (same as position):
       flu_qx = unity_qz, flu_qy = -unity_qx, flu_qz = unity_qy

    2. Handedness correction (conjugate - negate vector part):
       Unity is LEFT-handed, FLU is RIGHT-handed.
       Conjugating flips rotation direction to account for handedness change.

    Combined result:
       flu_qx = -unity_qz
       flu_qy = +unity_qx  (double negative: -(-unity_qx))
       flu_qz = -unity_qy

    Args:
        rot_x, rot_y, rot_z, rot_w: Quaternion components in Unity coords

    Returns:
        Rotation: scipy Rotation object in FLU coords
    """
    # Step 1: Axis mapping, Step 2: Conjugate (negate vector part)
    flu_qx = -rot_z      # axis: unity_qz, then negate
    flu_qy = rot_x       # axis: -unity_qx, then negate = +unity_qx
    flu_qz = -rot_y      # axis: unity_qy, then negate
    flu_qw = rot_w       # scalar unchanged

    return Rotation.from_quat([flu_qx, flu_qy, flu_qz, flu_qw])


def main():
    """Main teleoperation loop."""
    print("=" * 60)
    print("MetaQuest Hand Pose Teleoperation for Panda")
    print("=" * 60)

    # Configuration
    control_rate_hz = 50.0
    position_scale = 1.0  # Scale factor for hand movement
    hand_to_use = "left"  # "left" or "right"

    # Connect to robot
    print("\n[1] Connecting to robot...")
    robot = make_robot("fep")
    robot.wait_until_ready()
    print(f"    Robot ready. EE position: {robot.end_effector_pose.position}")

    # Switch to cartesian impedance control
    print("\n[2] Switching to cartesian impedance controller...")
    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    # Connect to MetaQuest
    print("\n[3] Connecting to MetaQuest...")
    with MetaTeleopClient() as client:
        print("    Waiting for channels...")
        client.wait_for_channels(timeout=10.0)
        print(f"    Connected to {client.device_name} ({client.device_model})")

        channel = client['hand_pose']
        print(f"    Using hand_pose channel at ~{channel.stats.target_hz:.0f}Hz")

        # State variables
        teleoperation_active = False
        initial_hand_pos: Optional[np.ndarray] = None
        initial_hand_orientation: Optional[Rotation] = None
        initial_robot_pos: Optional[np.ndarray] = None
        initial_robot_orientation: Optional[Rotation] = None

        print("\n" + "=" * 60)
        print("Controls:")
        print("  SPACE - Toggle teleoperation on/off")
        print("  q     - Quit")
        print("=" * 60)
        print(f"\nUsing {hand_to_use} hand for control")
        print("Press SPACE to start teleoperation...\n")

        rate_period = 1.0 / control_rate_hz

        with KeyboardInput() as keyboard:
            running = True
            last_time = time.time()

            while running:
                current_time = time.time()

                # Rate limiting
                if current_time - last_time < rate_period:
                    time.sleep(0.001)
                    continue
                last_time = current_time

                # Handle keyboard input
                key = keyboard.get_key()
                if key == 'q':
                    print("\nQuitting...")
                    running = False
                    continue
                elif key == ' ':
                    # Toggle teleoperation on/off
                    if not teleoperation_active:
                        # Activation - capture initial positions and orientations
                        data = channel.last_packet
                        if data is not None:
                            hand = data.left if hand_to_use == "left" else data.right
                            if hand.is_tracked:
                                initial_hand_pos = unity_position_to_flu(
                                    hand.pos_x, hand.pos_y, hand.pos_z
                                )
                                initial_hand_orientation = unity_quaternion_to_flu(
                                    hand.rot_x, hand.rot_y, hand.rot_z, hand.rot_w
                                )
                                initial_robot_pose = robot.end_effector_pose.copy()
                                initial_robot_pos = initial_robot_pose.position.copy()
                                initial_robot_orientation = initial_robot_pose.orientation
                                teleoperation_active = True
                                print(f"\n[ACTIVE] Teleoperation enabled - references reset")
                                print(f"         Robot pos: [{initial_robot_pos[0]:.3f}, {initial_robot_pos[1]:.3f}, {initial_robot_pos[2]:.3f}]")
                            else:
                                print(f"[WARN] {hand_to_use} hand not tracked")
                    else:
                        # Deactivation
                        teleoperation_active = False
                        initial_hand_pos = None
                        initial_hand_orientation = None
                        initial_robot_pos = None
                        initial_robot_orientation = None
                        print("[INACTIVE] Teleoperation disabled - pose frozen")

                # Update robot pose if teleoperation is active
                if teleoperation_active and initial_hand_pos is not None:
                    data = channel.last_packet
                    if data is not None:
                        hand = data.left if hand_to_use == "left" else data.right
                        if hand.is_tracked:
                            # Get current hand pose in FLU
                            current_hand_pos = unity_position_to_flu(
                                hand.pos_x, hand.pos_y, hand.pos_z
                            )
                            current_hand_orientation = unity_quaternion_to_flu(
                                hand.rot_x, hand.rot_y, hand.rot_z, hand.rot_w
                            )

                            # Compute position delta
                            pos_delta = (current_hand_pos - initial_hand_pos) * position_scale

                            # Compute orientation delta: delta_rot = current * initial^-1
                            # Then apply: target_rot = delta_rot * initial_robot_rot
                            delta_rotation = current_hand_orientation * initial_hand_orientation.inv()
                            target_orientation = delta_rotation * initial_robot_orientation

                            # Apply deltas to robot pose
                            target_pos = initial_robot_pos + pos_delta

                            # Update robot target
                            target_pose = Pose(
                                position=target_pos,
                                orientation=target_orientation
                            )
                            robot.set_target(pose=target_pose)

                            # Print status occasionally
                            if int(current_time * 10) % 10 == 0:
                                euler = delta_rotation.as_euler('xyz', degrees=True)
                                print(f"  Pos: [{pos_delta[0]:+.3f}, {pos_delta[1]:+.3f}, {pos_delta[2]:+.3f}]m  "
                                      f"Rot: [{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}]°", end="\r")
                        else:
                            print(f"[WARN] {hand_to_use} hand lost tracking")

    print("\nShutting down...")
    robot.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
