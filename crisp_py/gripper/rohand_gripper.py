"""RoHand gripper wrapper for crisp_py interface."""

import threading
from typing import Dict, List, Optional

import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import JointState

from crisp_py.config.path import find_config
from crisp_py.gripper.gripper_config import GripperConfig
from crisp_py.utils.callback_monitor import CallbackMonitor


class ROHandGripper:
    """Interface for ROHand dexterous hand gripper.

    This class provides control and monitoring for the ROHand gripper with 11 joints:
    - th_root_link: thumb rotation/abduction
    - th_proximal_link, th_distal_link: thumb joints
    - if_proximal_link, if_distal_link: index finger joints
    - mf_proximal_link, mf_distal_link: middle finger joints
    - rf_proximal_link, rf_distal_link: ring finger joints
    - lf_proximal_link, lf_distal_link: little finger joints

    All joint values are in radians where 0.0 = fully open hand.
    """

    THREADS_REQUIRED = 2

    # Joint names in the order they appear in JointState messages
    JOINT_NAMES = [
        "th_root_link",  # 0: thumb rotation (abduction)
        "th_proximal_link",  # 1: thumb proximal
        "th_distal_link",  # 2: thumb distal
        "if_proximal_link",  # 3: index proximal
        "if_distal_link",  # 4: index distal
        "mf_proximal_link",  # 5: middle proximal
        "mf_distal_link",  # 6: middle distal
        "rf_proximal_link",  # 7: ring proximal
        "rf_distal_link",  # 8: ring distal
        "lf_proximal_link",  # 9: little proximal
        "lf_distal_link",  # 10: little distal
    ]

    # Indices for proximal joints (used for open/close commands)
    PROXIMAL_INDICES = [1, 3, 5, 7, 9]  # thumb, index, middle, ring, little proximals

    # Joint limits in radians (min, max) - where 0.0 = fully open
    JOINT_LIMITS = {
        "th_root_link": (0.0, 1.57),  # Thumb rotation: 0 to 90 degrees
        "th_proximal_link": (0.0, 0.7),  # Thumb proximal: limited range
        "th_distal_link": (0.0, 0.7),  # Thumb distal: follows proximal
        "if_proximal_link": (0.0, 1.44),  # Index proximal
        "if_distal_link": (0.0, 1.44),  # Index distal
        "mf_proximal_link": (0.0, 1.44),  # Middle proximal
        "mf_distal_link": (0.0, 1.44),  # Middle distal
        "rf_proximal_link": (0.0, 1.44),  # Ring proximal
        "rf_distal_link": (0.0, 1.44),  # Ring distal
        "lf_proximal_link": (0.0, 1.44),  # Little proximal
        "lf_distal_link": (0.0, 1.44),  # Little distal
    }

    # Predefined gestures (using joint limits where applicable)
    GESTURES = {
        "open": {
            "th_root_link": JOINT_LIMITS["th_root_link"][0],  # 1.57 - thumb out
            "th_proximal_link": JOINT_LIMITS["th_proximal_link"][0],  # 0.0 - open
            "if_proximal_link": JOINT_LIMITS["if_proximal_link"][0],  # 0.0 - open
            "mf_proximal_link": JOINT_LIMITS["mf_proximal_link"][0],  # 0.0 - open
            "rf_proximal_link": JOINT_LIMITS["rf_proximal_link"][0],  # 0.0 - open
            "lf_proximal_link": JOINT_LIMITS["lf_proximal_link"][0],  # 0.0 - open
        },
        "close": {
            "th_root_link": JOINT_LIMITS["th_root_link"][0],  # 1.57 - thumb out
            "th_proximal_link": 0.4,  # 0.4 - maximum does not close properly
            "if_proximal_link": JOINT_LIMITS["if_proximal_link"][1],  # 1.44 - max
            "mf_proximal_link": JOINT_LIMITS["mf_proximal_link"][1],  # 1.44 - max
            "rf_proximal_link": JOINT_LIMITS["rf_proximal_link"][1],  # 1.44 - max
            "lf_proximal_link": JOINT_LIMITS["lf_proximal_link"][1],  # 1.44 - max
        },
        "rock": {  # Rock gesture: middle and ring closed, thumb in, others open
            "th_root_link": JOINT_LIMITS["th_root_link"][0],  # 0.0 - thumb in
            "th_proximal_link": 0.3,
            "if_proximal_link": JOINT_LIMITS["if_proximal_link"][0],  # 0.0 - open
            "mf_proximal_link": JOINT_LIMITS["mf_proximal_link"][1],  # 1.44 - closed
            "rf_proximal_link": JOINT_LIMITS["rf_proximal_link"][1],  # 1.44 - closed
            "lf_proximal_link": JOINT_LIMITS["lf_proximal_link"][0],  # 0.0 - open
        },
        "pointing": {  # Pointing: index open, others closed
            "th_root_link": JOINT_LIMITS["th_root_link"][0],  # 0.0 - thumb in
            "th_proximal_link": JOINT_LIMITS["th_proximal_link"][1],  # 0.7 - closed
            "if_proximal_link": JOINT_LIMITS["if_proximal_link"][0],  # 0.0 - open
            "mf_proximal_link": JOINT_LIMITS["mf_proximal_link"][1],  # 1.44 - closed
            "rf_proximal_link": JOINT_LIMITS["rf_proximal_link"][1],  # 1.44 - closed
            "lf_proximal_link": JOINT_LIMITS["lf_proximal_link"][1],  # 1.44 - closed
        },
    }

    def __init__(
        self,
        node: Node | None = None,
        namespace: str = "rohand_left",
        gripper_config: GripperConfig | None = None,
        spin_node: bool = True,
        frame_id: str = "rohand_2",
    ):
        """Initialize the ROHand gripper client.

        Args:
            node (Node, optional): ROS2 node to use. If None, creates a new node.
            namespace (str, optional): ROS2 namespace for the gripper. Default "rohand_left".
            gripper_config (GripperConfig, optional): Configuration for the gripper class.
            spin_node (bool, optional): Whether to spin the node in a separate thread.
            frame_id (str, optional): Frame ID used in header of target messages. Default "rohand_2".
        """
        if not rclpy.ok() and node is None:
            rclpy.init()

        self.node = (
            rclpy.create_node(node_name="rohand_gripper_client", namespace="", parameter_overrides=[])
            if not node
            else node
        )

        self.config = (
            gripper_config
            if gripper_config
            else GripperConfig(
                min_value=0.0,
                max_value=1.57,  # ~90 degrees in radians
                command_topic=f"{namespace}/target_joint_states",
                joint_state_topic=f"{namespace}/joint_states",
            )
        )

        self._namespace = namespace
        self._frame_id = frame_id
        self._positions: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._efforts: Optional[np.ndarray] = None
        self._target: Optional[Dict[str, float]] = None

        self._callback_monitor = CallbackMonitor(self.node, stale_threshold=self.config.max_joint_delay)

        # Publisher for target joint states
        self._command_publisher = self.node.create_publisher(
            JointState,
            self.config.command_topic,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )

        # Subscriber for current joint states
        self._joint_subscriber = self.node.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self._callback_monitor.monitor(f"{namespace.capitalize()} ROHand Joint State", self._callback_joint_state),
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )

        # Timer for publishing target commands
        self.node.create_timer(
            1.0 / self.config.publish_frequency,
            self._callback_monitor.monitor(
                f"{namespace.capitalize()} ROHand Target Publisher", self._callback_publish_target
            ),
            ReentrantCallbackGroup(),
        )

        if spin_node:
            threading.Thread(target=self._spin_node, daemon=True).start()

    @classmethod
    def from_yaml(
        cls,
        config_name: str,
        node: Node | None = None,
        namespace: str = "rohand_left",
        spin_node: bool = True,
        **overrides,
    ) -> "ROHandGripper":
        """Create a ROHandGripper instance from a YAML configuration file.

        Args:
            config_name: Name of the config file (with or without .yaml extension)
            node: ROS2 node to use. If None, creates a new node.
            namespace: ROS2 namespace for the gripper.
            spin_node: Whether to spin the node in a separate thread.
            **overrides: Additional parameters to override YAML values

        Returns:
            ROHandGripper: Configured gripper instance
        """

        if not config_name.endswith(".yaml"):
            config_name = f"{config_name}.yaml"

        config_path = find_config(f"grippers/{config_name}")
        if config_path is None:
            raise FileNotFoundError(f"Config file {config_name} not found")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        data.update(overrides)

        namespace = data.pop("namespace", namespace)
        frame_id = data.pop("frame_id", "rohand_2")
        config_data = data.pop("gripper_config", data)

        gripper_config = GripperConfig(**config_data)

        return cls(
            node=node,
            namespace=namespace,
            gripper_config=gripper_config,
            spin_node=spin_node,
            frame_id=frame_id,
        )

    def _spin_node(self):
        """Spin the ROS2 node in a separate thread."""
        if not rclpy.ok():
            rclpy.init()
        executor = MultiThreadedExecutor(num_threads=self.THREADS_REQUIRED)
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _callback_joint_state(self, msg: JointState):
        """Save the latest joint state values.

        Args:
            msg (JointState): The message containing the joint state.
        """
        # Store joint states in the order defined by JOINT_NAMES
        self._positions = np.array(msg.position)
        self._velocities = np.array(msg.velocity) if msg.velocity else np.zeros(len(msg.position))
        self._efforts = np.array(msg.effort) if msg.effort else np.zeros(len(msg.position))

    def _callback_publish_target(self):
        """Publish the target command."""
        if self._target is None:
            return

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        # Populate message with target values
        msg.name = []
        msg.position = []
        msg.velocity = []

        # Map target dict to message arrays
        for joint_name in self.JOINT_NAMES:
            if joint_name in self._target:
                msg.name.append(joint_name)
                msg.position.append(float(self._target[joint_name]))
                msg.velocity.append(100.0)  # Default velocity

        self._command_publisher.publish(msg)

    # Properties for accessing joint data
    @property
    def positions(self) -> Optional[np.ndarray]:
        """Returns current joint positions or None if not initialized."""
        return self._positions

    @property
    def velocities(self) -> Optional[np.ndarray]:
        """Returns current joint velocities or None if not initialized."""
        return self._velocities

    @property
    def efforts(self) -> Optional[np.ndarray]:
        """Returns current joint efforts or None if not initialized."""
        return self._efforts

    def get_joint_position(self, joint_name: str) -> Optional[float]:
        """Get position of a specific joint by name.

        Args:
            joint_name: Name of the joint (e.g., 'if_proximal_link')

        Returns:
            Joint position in radians or None if not available
        """
        if self._positions is None or joint_name not in self.JOINT_NAMES:
            return None
        idx = self.JOINT_NAMES.index(joint_name)
        return float(self._positions[idx])

    def get_joint_effort(self, joint_name: str) -> Optional[float]:
        """Get effort of a specific joint by name.

        Args:
            joint_name: Name of the joint

        Returns:
            Joint effort or None if not available
        """
        if self._efforts is None or joint_name not in self.JOINT_NAMES:
            return None
        idx = self.JOINT_NAMES.index(joint_name)
        return float(self._efforts[idx])

    def get_joint_limits(self, joint_name: str) -> Optional[tuple]:
        """Get position limits for a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Tuple of (min, max) position limits in radians, or None if unknown
        """
        return self.JOINT_LIMITS.get(joint_name)

    def is_ready(self) -> bool:
        """Returns True if the gripper is fully ready to operate."""
        return self._positions is not None

    def wait_until_ready(self, timeout: float = 10.0, check_frequency: float = 10.0):
        """Wait until the gripper is available.

        Args:
            timeout: Maximum time to wait in seconds
            check_frequency: How often to check in Hz
        """
        rate = self.node.create_rate(check_frequency)
        elapsed = 0.0
        while not self.is_ready():
            rate.sleep()
            elapsed += 1.0 / check_frequency
            if elapsed >= timeout:
                raise TimeoutError(f"ROHand gripper not ready after {timeout} seconds")

    # Basic control methods
    def set_target(self, target: Dict[str, float]):
        """Set target positions for specific joints.

        Args:
            target: Dictionary mapping joint names to target positions in radians
                   Example: {'if_proximal_link': 0.5, 'th_root_link': 1.57}
        """
        if self._target is None:
            self._target = {}
        self._target.update(target)

    def set_joint_target(self, joint_name: str, position: float):
        """Set target position for a single joint.

        Args:
            joint_name: Name of the joint to control
            position: Target position in radians (0.0 = fully open)
                     Will be clamped to joint limits automatically
        """
        if joint_name not in self.JOINT_NAMES:
            raise ValueError(f"Unknown joint name: {joint_name}. Must be one of {self.JOINT_NAMES}")

        # Clamp position to joint limits
        if joint_name in self.JOINT_LIMITS:
            min_val, max_val = self.JOINT_LIMITS[joint_name]
            position = np.clip(position, min_val, max_val)

        if self._target is None:
            self._target = {}
        self._target[joint_name] = position

    def open(self):
        """Open the gripper (all proximal joints to 0.0)."""
        self.execute_gesture("open")

    def close(self):
        """Close the gripper (all proximal joints to closed position)."""
        self.execute_gesture("close")

    def grasp(self, width: Optional[float] = None):
        """Grasp with the gripper. For now, behaves like close().

        Args:
            width: Not used for ROHand, kept for interface compatibility
        """
        self.close()

    def execute_gesture(self, gesture_name: str):
        """Execute a predefined gesture.

        Args:
            gesture_name: Name of gesture ('open', 'close', 'rock', 'pointing')
        """
        if gesture_name not in self.GESTURES:
            raise ValueError(f"Unknown gesture: {gesture_name}. Available: {list(self.GESTURES.keys())}")

        self.set_target(self.GESTURES[gesture_name].copy())

    def is_open(self, open_threshold: float = 0.3) -> bool:
        """Returns True if the gripper is open.

        Checks if all proximal joints are below the threshold.

        Args:
            open_threshold: Maximum joint angle (radians) to consider "open"
        """
        if self._positions is None:
            raise RuntimeError("Gripper not ready. Call wait_until_ready() first.")

        # Check all proximal joints
        for idx in self.PROXIMAL_INDICES:
            if self._positions[idx] > open_threshold:
                return False
        return True

    def is_closed(self, close_threshold: float = 0.8) -> bool:
        """Returns True if the gripper is closed.

        Checks if all proximal joints are above the threshold.

        Args:
            close_threshold: Minimum joint angle (radians) to consider "closed"
        """
        if self._positions is None:
            raise RuntimeError("Gripper not ready. Call wait_until_ready() first.")

        # Check all proximal joints
        for idx in self.PROXIMAL_INDICES:
            if self._positions[idx] < close_threshold:
                return False
        return True

    def shutdown(self):
        """Shutdown the node."""
        if rclpy.ok():
            rclpy.shutdown()


def make_rohand_gripper(
    config_name: Optional[str] = None,
    gripper_config: Optional[GripperConfig] = None,
    node: Optional[Node] = None,
    namespace: str = "rohand_left",
    spin_node: bool = True,
    **overrides,
) -> ROHandGripper:
    """Factory function to create a ROHandGripper from a configuration file.

    Args:
        config_name: Name of the gripper config file
        gripper_config: Directly provide a GripperConfig instance instead of loading from file
        node: ROS2 node to use. If None, creates a new node
        namespace: ROS2 namespace for the gripper
        spin_node: Whether to spin the node in a separate thread
        **overrides: Additional parameters to override config values

    Returns:
        ROHandGripper: Configured gripper instance
    """
    if not ((config_name is None and gripper_config) or (config_name and gripper_config is None)):
        raise ValueError("Either config_name or gripper_config must be provided, not both.")

    if config_name is not None:
        return ROHandGripper.from_yaml(
            config_name=config_name.replace(".yaml", ""),
            node=node,
            namespace=namespace,
            spin_node=spin_node,
            **overrides,
        )

    return ROHandGripper(gripper_config=gripper_config, node=node, namespace=namespace, spin_node=spin_node)
