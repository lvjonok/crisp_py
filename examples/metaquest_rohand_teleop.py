#!/usr/bin/env python3
"""
Real-time teleoperation of ROHand using MetaQuest hand tracking.

This script:
1. Receives hand tracking data from MetaQuest via UDP multicast
2. Retargets human hand pose to ROHand joint configuration
3. Sends control commands to the real ROHand gripper

Prerequisites:
- MetaQuest streaming hand tracking data (meta-teleop Unity app running)
- ROHand hardware connected and powered
- rohand_ros2 package launched:
    ros2 launch rohand rohand_ap001.launch.py \\
        port_name:=/dev/ttyUSB0 \\
        baudrate:=115200 \\
        hand_ids:=[2] \\
        namespace:=rohand_left

Usage:
    python metaquest_rohand_teleop.py --hand-type left --namespace rohand_left
    python metaquest_rohand_teleop.py --hand-type right --namespace rohand_right
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import tyro
from loguru import logger

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from meta_teleop import MetaTeleopClient, CoordinateSystem
from crisp_py.gripper.rohand_gripper import make_rohand_gripper, ROHandGripper

# Path to dex-retargeting assets
dex_retargeting_path = Path(__file__).absolute().parent.parent / "third_party" / "dex-retargeting"

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

# Coordinate transform from FLU (Forward-Left-Up) to MANO convention
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


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame from detected 3D key points.
    """
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
    """
    Extract and transform joint positions from MetaQuest finger joints data.
    """
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
    """
    Convert retargeting qpos to gripper target dictionary.

    Args:
        qpos: Joint positions from retargeting (11 values for ROHand)
        joint_names: Joint names in same order as qpos

    Returns:
        Dictionary mapping joint names to positions for gripper.set_target()
    """
    # Only send commands for joints the gripper can control
    controllable_joints = set(ROHandGripper.JOINT_NAMES)

    target = {}
    for i, name in enumerate(joint_names):
        if name in controllable_joints:
            target[name] = float(qpos[i])

    return target


def main(
    hand_type: str = "left",
    namespace: str = "rohand_left",
    config_name: str = "rohand_left",
    connection_timeout: float = 10.0,
    control_rate_hz: float = 30.0,
):
    """
    Real-time teleoperation of ROHand using MetaQuest hand tracking.

    Args:
        hand_type: Which hand to track from MetaQuest ("left" or "right")
        namespace: ROS2 namespace for the ROHand gripper
        config_name: Name of the gripper config file
        connection_timeout: Timeout in seconds for connecting to MetaQuest
        control_rate_hz: Target control rate in Hz
    """
    # Determine hand type enum
    hand_type_enum = HandType.left if hand_type.lower() == "left" else HandType.right
    hand_type_str = "Left" if hand_type.lower() == "left" else "Right"

    # Setup retargeting
    robot_dir = dex_retargeting_path / "assets" / "robots" / "hands"
    config_path = get_default_config_path(RobotName.rohand, RetargetingType.vector, hand_type_enum)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Loading retargeting config from {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    # Get joint names from retargeting (order matches qpos)
    joint_names = retargeting.joint_names
    logger.info(f"Retargeting joint names: {joint_names}")

    # Setup ROHand gripper
    logger.info(f"Connecting to ROHand gripper (namespace: {namespace})...")
    gripper = make_rohand_gripper(
        config_name=config_name,
        namespace=namespace,
    )

    try:
        gripper.wait_until_ready(timeout=10.0)
        logger.info("ROHand gripper is ready!")
    except TimeoutError as e:
        logger.error(f"Failed to connect to gripper: {e}")
        logger.error("Make sure the rohand node is running:")
        logger.error(f"  ros2 launch rohand rohand_ap001.launch.py namespace:={namespace}")
        return

    # Open gripper to start position
    logger.info("Opening gripper to start position...")
    gripper.open()

    # Connect to MetaQuest
    logger.info("Connecting to MetaQuest...")
    with MetaTeleopClient() as client:
        logger.info("Waiting for MetaQuest channels...")
        client.wait_for_channels(timeout=connection_timeout)
        logger.info(f"Connected to {client.device_name} ({client.device_model})")

        channel = client['finger_joints']
        logger.info(f"Receiving finger joints at ~{channel.stats.target_hz:.0f} Hz")
        logger.info(f"Tracking {hand_type_str} hand, controlling ROHand at {control_rate_hz} Hz")
        logger.info("Starting teleoperation. Press Ctrl+C to stop.")

        import time
        control_period = 1.0 / control_rate_hz
        last_control_time = 0.0
        frames_processed = 0

        try:
            while True:
                current_time = time.time()

                # Rate limiting
                if current_time - last_control_time < control_period:
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue

                last_control_time = current_time

                # Get latest finger joints data
                data = channel.last_packet

                if data is None:
                    continue

                joint_pos = extract_joint_positions(data, hand_type_str)

                if joint_pos is None:
                    if frames_processed % 100 == 0:
                        logger.warning(f"{hand_type_str} hand is not tracked.")
                    continue

                # Compute retargeting reference value
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices

                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                # Retarget to robot joint positions
                qpos = retargeting.retarget(ref_value)

                # Convert to gripper target and send
                target = qpos_to_gripper_target(qpos, joint_names)
                gripper.set_target(target)

                frames_processed += 1
                if frames_processed % 300 == 0:
                    logger.info(f"Processed {frames_processed} frames")

        except KeyboardInterrupt:
            logger.info("\nStopping teleoperation...")

    # Return to open position
    logger.info("Returning gripper to open position...")
    gripper.open()
    import time
    time.sleep(2.0)

    logger.info("Teleoperation complete.")
    gripper.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
