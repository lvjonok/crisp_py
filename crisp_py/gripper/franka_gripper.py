"""Simple Node to allow users of crisp_py (https://github.com/utiasDSL/crisp_py) to use the Franka Hand (which we strongly discourage)."""

import threading
import warnings
from time import time

import numpy as np
import rclpy
import yaml
# from franka_msgs.action import Grasp, Homing
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from crisp_py.config.path import find_config
from crisp_py.gripper.gripper import Gripper
from crisp_py.gripper.gripper_config import GripperConfig
from crisp_py.utils.callback_monitor import CallbackMonitor


class GripperClient:
    def __init__(self, node: Node, gripper_namespace: str = "panda_gripper"):
        """Initialize the gripper client."""

        self._node = node

        # The namespace for the gripper might change
        # check https://github.com/frankaemika/franka_ros2/issues/121
        self._action_client = ActionClient(
            self._node,
            GripperCommand,
            f"{gripper_namespace}/gripper_action",
            callback_group=ReentrantCallbackGroup(),
        )
        # self._grasp_client = ActionClient(
        #     node,
        #     Grasp,
        #     f"{gripper_namespace}/grasp",
        #     callback_group=ReentrantCallbackGroup(),
        # )
        # self._home_client = ActionClient(
        #     node,
        #     Homing,
        #     f"{gripper_namespace}/homing",
        #     callback_group=ReentrantCallbackGroup(),
        # )
        self._gripper_state_subscriber = node.create_subscription(
            JointState,
            f"{gripper_namespace}/joint_states",
            self._gripper_state_callback,
            qos_profile_system_default,
        )
        node.get_logger().warn(self._gripper_state_subscriber.topic_name)
        self._width = None

    @property
    def width(self) -> float | None:
        """Returns the current width of the gripper or None if not initialized."""
        return self._width

    def is_open(self, open_threshold: float = 0.05) -> bool:
        """Returns True if the gripper is open."""
        print(self.width)
        res = self.width > open_threshold

        self._node.get_logger().info(f"Gripper is_open check: {res} (width: {self.width})")

        return res

    def is_ready(self) -> bool:
        """Returns True if the gripper is fully ready to operate."""
        return self.width is not None

    def wait_until_ready(self, timeout_sec: float = 5.0):
        """Waits until the gripper is fully ready to operate."""
        time_start = time()
        while not self.is_ready():
            rclpy.spin_once(self._node, timeout_sec=1.0)
            if time() - time_start > timeout_sec:
                raise TimeoutError("Gripper client is not ready after timeout.")

    def _gripper_state_callback(self, msg: JointState):
        """Updates the gripper width using the current joint state."""
        self._width = msg.position[0] + msg.position[1]

    def home(self):
        """Homes the gripper."""

        # Create and send goal
        goal = GripperCommand.Goal()
        goal.command.position = 0.03
        goal.command.max_effort = 30.0

        # Send goal asynchronously with feedback callback
        self._action_client.send_goal_async(goal)
        
    def grasp(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 50.0,
        epsilon_outer: float = 0.08,
        epsilon_inner: float = 0.01,
        block: bool = False,
    ):
        """Grasp with the gripper and does not block.
        Args:
            width (float): The width of the gripper.
            speed (float, optional): The speed of the gripper. Defaults to 0.1.
            force (float, optional): The force of the gripper. Defaults to 50.0.
            epsilon_outer (float, optional): The outer epsilon of the gripper. Defaults to 0.08.
            epsilon_inner (float, optional): The inner epsilon of the gripper. Defaults to 0.01.
            block (bool, optional): Whether to block. Defaults to False.
        """

        # Create and send goal
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = force

        future = self._action_client.send_goal_async(goal)

        # # Send goal asynchronously with feedback callback
        # self._action_client.send_goal_async(goal)

        # goal = Grasp.Goal()
        # goal.width = width
        # goal.speed = speed
        # goal.force = force
        # goal.epsilon.outer = epsilon_outer
        # goal.epsilon.inner = epsilon_inner
        # future = self._grasp_client.send_goal_async(goal)  # We assume that the server is running.

        if block:
            rate = self._node.create_rate(10)
            while not future.done():
                rate.sleep()
            goal_handle = future.result()
            future = goal_handle.get_result_async()

            while not future.done():
                rate.sleep()

            rate.destroy()

    def close(self, **grasp_kwargs):
        """Close the gripper.

        Args:
            **grasp_kwargs: Keyword arguments to pass to the grasp function. (check the grasp function for details)
        """
        self.grasp(width=0.0, **grasp_kwargs)

    def open(self, **grasp_kwargs):
        """Open the gripper.

        Args:
            **grasp_kwargs: Keyword arguments to pass to the grasp function. (check the grasp function for details)
        """
        # self.grasp(width=0.08, **grasp_kwargs)
        self.home()

    def toggle(self, **grasp_kwargs):
        """Toggle the gripper between open and closed.

        Args:
            **grasp_kwargs: Keyword arguments to pass to the grasp function. (check the grasp function for details)
        """
        if self.is_open():
            self.close(**grasp_kwargs)
        else:
            self.open(**grasp_kwargs)


class CrispPyGripperAdapater(Node):
    def __init__(self):
        super().__init__("crisp_py_gripper_adapter")
        
        warnings.warn(
            "CrispPyGripperAdapater is deprecated. Use FrankaGripper class instead for "
            "better integration with the crisp_py gripper interface.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.command_topic = "gripper/gripper_position_controller/commands"
        self.joint_state_topic = "gripper/joint_states"

        self.joint_state_freq = 50

        self.gripper_client = GripperClient(self, gripper_namespace="panda_gripper")
        self.gripper_client.wait_until_ready()

        self.gripper_client.open()
        self.is_closing = False

        self.create_subscription(
            Float64MultiArray,
            self.command_topic,
            self.callback_command,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )

        self.joint_state_publisher = self.create_publisher(
            JointState,
            self.joint_state_topic,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )

        self.create_timer(1 / self.joint_state_freq, self.callback_publish_joint_state)
        self.get_logger().info("The crisp_py gripper adapter started.")

    def callback_publish_joint_state(self):
        if self.gripper_client.width is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = ["gripper_joint"]
        msg.position = [self.gripper_client.width / 0.08]
        msg.effort = [0.0]

        self.joint_state_publisher.publish(msg)

    def callback_command(self, msg: Float64MultiArray):
        """Callback to the gripper command."""
        # NOTE: this only allows to open and close. The FrankaHand is not super responsive anyway, this is just a
        # temporary node...
        gripper_command = msg.data[0]
        self.get_logger().info(f"Received a command to move the gripper: {msg}")

        if gripper_command <= 0.5 and self.gripper_client.is_open() and not self.is_closing:
            self.get_logger().info("Closing the gripper.")
            self.gripper_client.close()
            self.is_closing = True
        elif gripper_command > 0.5 and not self.gripper_client.is_open() and self.is_closing:
            self.get_logger().info("Opening the gripper.")
            self.gripper_client.open()
            self.is_closing = False


class FrankaGripper(Gripper):
    """Gripper wrapper for Franka Hand using action-based control.
    
    This class provides the same interface as the base Gripper class but uses
    ROS2 action clients to communicate with the Franka gripper instead of topics.
    """

    def __init__(
        self,
        node: Node | None = None,
        namespace: str = "",
        gripper_config: GripperConfig | None = None,
        spin_node: bool = True,
        gripper_namespace: str = "panda_gripper",
        force: float = 50.0,
    ):
        """Initialize the Franka gripper client.

        Args:
            node (Node, optional): ROS2 node to use. If None, creates a new node.
            namespace (str, optional): ROS2 namespace for the gripper.
            gripper_config (GripperConfig, optional): configuration for the gripper class.
            spin_node (bool, optional): Whether to spin the node in a separate thread.
            gripper_namespace (str, optional): Namespace for Franka gripper actions. Defaults to "panda_gripper".
            force (float, optional): Default force for grasp actions. Defaults to 50.0.
        """
        if not rclpy.ok() and node is None:
            rclpy.init()

        self.node = (
            rclpy.create_node(
                node_name="franka_gripper_client", namespace=namespace, parameter_overrides=[]
            )
            if not node
            else node
        )
        self.config = (
            gripper_config if gripper_config else GripperConfig(min_value=0.0, max_value=0.08)
        )

        self._prefix = f"{namespace}_" if namespace else ""
        self._value = None
        self._torque = None
        self._target = None
        self._index = self.config.index
        self._force = force
        self._callback_monitor = CallbackMonitor(
            self.node, stale_threshold=self.config.max_joint_delay
        )

        # Initialize GripperClient for action-based control
        self.gripper_client = GripperClient(self.node, gripper_namespace=gripper_namespace)
        
        # Wait for action server to be available
        self.node.get_logger().info(f"Waiting for action server at {gripper_namespace}/gripper_action...")
        if not self.gripper_client._action_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"Action server {gripper_namespace}/gripper_action is not available after 5 seconds. "
                f"Is the Franka gripper driver running?"
            )
        self.node.get_logger().info("Action server connected.")

        # Note: We don't create command publisher or timer for _callback_publish_target
        # since we use action-based control instead

        # Subscribe to joint states from GripperClient's subscription
        # The GripperClient already subscribes to joint_states, we just use its data
        
        # Create service clients (kept for compatibility)
        self.reboot_client = self.node.create_client(
            Trigger, self.config.reboot_service
        )
        self.enable_torque_client = self.node.create_client(
            SetBool, self.config.enable_torque_service
        )

        if spin_node:
            threading.Thread(target=self._spin_node, daemon=True).start()

    @classmethod
    def from_yaml(
        cls,
        config_name: str,
        node: Node | None = None,
        namespace: str = "",
        spin_node: bool = True,
        **overrides,
    ) -> "FrankaGripper":
        """Create a FrankaGripper instance from a YAML configuration file.

        Args:
            config_name: Name of the config file (with or without .yaml extension)
            node: ROS2 node to use. If None, creates a new node.
            namespace: ROS2 namespace for the gripper.
            spin_node: Whether to spin the node in a separate thread.
            **overrides: Additional parameters to override YAML values

        Returns:
            FrankaGripper: Configured gripper instance

        Raises:
            FileNotFoundError: If the config file is not found
        """
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config_path = find_config(f"grippers/{config_name}")
        if config_path is None:
            config_path = find_config(config_name)

        if config_path is None:
            raise FileNotFoundError(
                f"Gripper config file '{config_name}' not found in any CRISP config paths"
            )

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        data.update(overrides)

        namespace = data.pop("namespace", namespace)
        gripper_namespace = data.pop("gripper_namespace", "panda_gripper")
        force = data.pop("force", 50.0)
        
        # Remove gripper_type from config_data as it's not part of GripperConfig
        data.pop("gripper_type", None)
        
        config_data = data.pop("gripper_config", data)
        gripper_config = GripperConfig(**config_data)

        return cls(
            node=node,
            namespace=namespace,
            gripper_config=gripper_config,
            spin_node=spin_node,
            gripper_namespace=gripper_namespace,
            force=force,
        )

    def _spin_node(self):
        if not rclpy.ok():
            rclpy.init()
        executor = MultiThreadedExecutor(num_threads=self.THREADS_REQUIRED)
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    @property
    def value(self) -> float | None:
        """Returns the current value of the gripper or None if not initialized."""
        if self.gripper_client.width is None:
            return None
        
        # Convert Franka width to normalized value [0-1]
        self._value = self.gripper_client.width
        return np.clip(self._normalize(self._value), 0.0, 1.0)

    @property
    def raw_value(self) -> float | None:
        """Returns the current raw value of the gripper or None if not initialized."""
        return self.gripper_client.width

    @property
    def value_unnormalized(self) -> float | None:
        """Alias for raw_value for compatibility with ZMQ bridge."""
        return self.raw_value

    @property
    def torque(self) -> float | None:
        """Returns the current torque of the gripper.
        
        Note: Franka gripper does not provide torque feedback through the action interface.
        """
        return None

    def is_ready(self) -> bool:
        """Returns True if the gripper is fully ready to operate."""
        return self.gripper_client.is_ready()

    def wait_until_ready(self, timeout: float = 10.0, check_frequency: float = 10.0):
        """Wait until the gripper is available."""
        self.gripper_client.wait_until_ready(timeout_sec=timeout)

    def set_target(self, target: float, *, epsilon: float = 0.1, block: bool = False):
        """Set gripper target position using action client.

        Args:
            target (float): The target value for the gripper between 0 and 1 from closed to open respectively.
            epsilon (float): allowed zone around the target limits that are allowed to be set.
            block (bool): Whether to block until the action completes. Defaults to False.
        """
        assert 0.0 - epsilon <= target <= 1.0 + epsilon, (
            f"The target should be normalized between 0 and 1, but is currently {target}"
        )
        self._target = self._unnormalize(target)
        
        # Send grasp command with unnormalized width
        width = np.clip(self._target, self.config.min_value, self.config.max_value)
        self.gripper_client.grasp(width=width, force=self._force, block=block)

    def close(self, block: bool = False):
        """Close the gripper.
        
        Args:
            block (bool): Whether to block until the action completes. Defaults to False.
        """
        self.set_target(target=0.0, block=block)

    def open(self, block: bool = False):
        """Open the gripper.
        
        Args:
            block (bool): Whether to block until the action completes. Defaults to False.
        """
        self.set_target(target=1.0, block=block)


def main():
    rclpy.init()
    adapter = CrispPyGripperAdapater()
    rclpy.spin(adapter)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
