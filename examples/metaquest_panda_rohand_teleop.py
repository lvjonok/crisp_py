#!/usr/bin/env python3
"""
Combined Panda + ROHand Teleoperation using MetaQuest.

Uses two channels from MetaQuest:
- hand_pose: Controls Panda arm position and orientation via delta tracking
- finger_joints: Controls ROHand gripper via retargeting

Coordinate Systems:
- Quest/Unity: X-right, Y-up, Z-forward (left-handed)
- FLU (Robot): X-forward, Y-left, Z-up (right-handed)

Usage:
    python metaquest_panda_rohand_teleop.py

Controls:
    SPACE - Toggle teleoperation on/off
    q     - Quit

Prerequisites:
- MetaQuest streaming hand tracking data
- Panda robot connected and ready
- ROHand gripper connected:
    ros2 launch rohand rohand_ap001.launch.py \\
        port_name:=/dev/ttyUSB0 \\
        namespace:=rohand_left
"""

import sys
import time
import select
import termios
import tty
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
from crisp_py.gripper.rohand_gripper import make_rohand_gripper, ROHandGripper
from meta_teleop import MetaTeleopClient, CoordinateSystem
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig

# Path to dex-retargeting assets
DEX_RETARGETING_PATH = Path(__file__).absolute().parent.parent / "third_party" / "dex-retargeting"

# Mapping from VR Quest 3 joint indices to MediaPipe 21-joint format
VR_TO_MEDIAPIPE_INDICES = np.array([
    1,   # WRIST <- Wrist
    2,   # THUMB_CMC <- ThumbMetacarpal
    3,   # THUMB_MCP <- ThumbProximal
    4,   # THUMB_IP <- ThumbDistal
    5,   # THUMB_TIP <- ThumbTip
    7,   # INDEX_MCP <- IndexProximal
    8,   # INDEX_PIP <- IndexIntermediate
    9,   # INDEX_DIP <- IndexDistal
    10,  # INDEX_TIP <- IndexTip
    12,  # MIDDLE_MCP <- MiddleProximal
    13,  # MIDDLE_PIP <- MiddleIntermediate
    14,  # MIDDLE_DIP <- MiddleDistal
    15,  # MIDDLE_TIP <- MiddleTip
    17,  # RING_MCP <- RingProximal
    18,  # RING_PIP <- RingIntermediate
    19,  # RING_DIP <- RingDistal
    20,  # RING_TIP <- RingTip
    22,  # PINKY_MCP <- LittleProximal
    23,  # PINKY_PIP <- LittleIntermediate
    24,  # PINKY_DIP <- LittleDistal
    25,  # PINKY_TIP <- LittleTip
])

# Coordinate transform from FLU to MANO convention
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])


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
    """
    flu_x = pos_z   # forward = unity forward
    flu_y = -pos_x  # left = negative unity right
    flu_z = pos_y   # up = unity up
    return np.array([flu_x, flu_y, flu_z])


def unity_quaternion_to_flu(rot_x: float, rot_y: float, rot_z: float, rot_w: float) -> Rotation:
    """
    Convert Unity quaternion to FLU coordinate system as scipy Rotation.

    Two-step transform:
    1. Axis mapping (same as position):
       flu_qx = unity_qz, flu_qy = -unity_qx, flu_qz = unity_qy

    2. Handedness correction (conjugate - negate vector part):
       Unity is LEFT-handed, FLU is RIGHT-handed.
       Conjugating flips rotation direction to account for handedness change.

    Combined result:
       flu_qx = -unity_qz
       flu_qy = +unity_qx  (double negative)
       flu_qz = -unity_qy
    """
    flu_qx = -rot_z      # axis: unity_qz, then negate
    flu_qy = rot_x       # axis: -unity_qx, then negate = +unity_qx
    flu_qz = -rot_y      # axis: unity_qy, then negate
    flu_qw = rot_w       # scalar unchanged

    return Rotation.from_quat([flu_qx, flu_qy, flu_qz, flu_qw])


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Compute the 3D coordinate frame from detected 3D key points."""
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index mcp, middle mcp

    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


def extract_joint_positions(data, hand_type: str) -> Optional[np.ndarray]:
    """Extract and transform joint positions from MetaQuest finger joints data."""
    hand_joints = data.left if hand_type == "Left" else data.right

    if not hand_joints.is_tracked:
        return None

    positions = hand_joints.positions(CoordinateSystem.FLU)  # (26, 3)
    joint_pos = positions[VR_TO_MEDIAPIPE_INDICES]  # (21, 3)
    joint_pos = joint_pos - joint_pos[0:1, :]

    mediapipe_wrist_rot = estimate_frame_from_hand_points(joint_pos)
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
    joint_pos = joint_pos @ mediapipe_wrist_rot @ operator2mano

    return joint_pos


def qpos_to_gripper_target(qpos: np.ndarray, joint_names: list) -> Dict[str, float]:
    """Convert retargeting qpos to gripper target dictionary."""
    controllable_joints = set(ROHandGripper.JOINT_NAMES)
    target = {}
    for i, name in enumerate(joint_names):
        if name in controllable_joints:
            target[name] = float(qpos[i])
    return target


def main():
    """Main teleoperation loop."""
    print("=" * 60)
    print("Combined Panda + ROHand Teleoperation")
    print("=" * 60)

    # Configuration
    control_rate_hz = 50.0
    position_scale = 1.0
    hand_to_use = "left"
    hand_type_str = "Left" if hand_to_use == "left" else "Right"
    hand_type_enum = HandType.left if hand_to_use == "left" else HandType.right
    rohand_namespace = "rohand_left"
    rohand_config = "rohand_left"

    # Setup retargeting for ROHand
    print("\n[1] Setting up finger retargeting...")
    robot_dir = DEX_RETARGETING_PATH / "assets" / "robots" / "hands"
    config_path = get_default_config_path(RobotName.rohand, RetargetingType.vector, hand_type_enum)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    joint_names = retargeting.joint_names
    print(f"    Retargeting loaded: {len(joint_names)} joints")

    # Connect to Panda robot
    print("\n[2] Connecting to Panda robot...")
    robot = make_robot("fep")
    robot.wait_until_ready()
    print(f"    Panda ready. EE position: {robot.end_effector_pose.position}")

    # Switch to cartesian impedance control
    print("\n[3] Switching to cartesian impedance controller...")
    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    # Connect to ROHand gripper
    print(f"\n[4] Connecting to ROHand gripper (namespace: {rohand_namespace})...")
    gripper = make_rohand_gripper(
        config_name=rohand_config,
        namespace=rohand_namespace,
    )
    try:
        gripper.wait_until_ready(timeout=10.0)
        print("    ROHand gripper ready!")
        gripper.open()
    except TimeoutError as e:
        print(f"    [WARN] ROHand not available: {e}")
        print("    Continuing with Panda only...")
        gripper = None

    # Connect to MetaQuest
    print("\n[5] Connecting to MetaQuest...")
    with MetaTeleopClient() as client:
        print("    Waiting for channels...")
        client.wait_for_channels(timeout=10.0)
        print(f"    Connected to {client.device_name} ({client.device_model})")

        hand_pose_channel = client['hand_pose']
        finger_joints_channel = client['finger_joints']
        print(f"    hand_pose channel at ~{hand_pose_channel.stats.target_hz:.0f}Hz")
        print(f"    finger_joints channel at ~{finger_joints_channel.stats.target_hz:.0f}Hz")

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
                        pose_data = hand_pose_channel.last_packet
                        if pose_data is not None:
                            hand = pose_data.left if hand_to_use == "left" else pose_data.right
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

                # Always update finger positions (ROHand) if gripper available
                if gripper is not None:
                    finger_data = finger_joints_channel.last_packet
                    if finger_data is not None:
                        joint_pos = extract_joint_positions(finger_data, hand_type_str)
                        if joint_pos is not None:
                            # Compute retargeting
                            retargeting_type = retargeting.optimizer.retargeting_type
                            indices = retargeting.optimizer.target_link_human_indices

                            if retargeting_type == "POSITION":
                                ref_value = joint_pos[indices, :]
                            else:
                                origin_indices = indices[0, :]
                                task_indices = indices[1, :]
                                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                            qpos = retargeting.retarget(ref_value)
                            target = qpos_to_gripper_target(qpos, joint_names)
                            gripper.set_target(target)

                # Update Panda pose if teleoperation is active
                if teleoperation_active and initial_hand_pos is not None:
                    pose_data = hand_pose_channel.last_packet
                    if pose_data is not None:
                        hand = pose_data.left if hand_to_use == "left" else pose_data.right
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

                            # Compute orientation delta
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
                                      f"Rot: [{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}]deg", end="\r")
                        else:
                            print(f"[WARN] {hand_to_use} hand lost tracking")

    print("\nShutting down...")
    if gripper is not None:
        gripper.open()
        time.sleep(1.0)
        gripper.shutdown()
    robot.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
