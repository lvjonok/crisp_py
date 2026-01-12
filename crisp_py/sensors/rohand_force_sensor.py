"""ROHand tactile force sensor module.

This module provides a sensor class for reading and processing
tactile force data from ROHand dexterous hands.
"""

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from crisp_py.config.path import find_config


# Finger names in order
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little', 'palm']

# Grid shapes for each finger's force sensor array (height, width)
FINGER_GRID_SHAPES = {
    'thumb': (8, 5),    # 40 points, 36 valid
    'index': (12, 5),   # 60 points, 60 valid
    'middle': (12, 5),  # 60 points, 60 valid
    'ring': (12, 5),    # 60 points, 60 valid
    'little': (8, 4),   # 32 points, 32 valid
    'palm': (11, 5),    # 55 points, 56 valid
}

# Boolean masks indicating which positions are dummy points (True = dummy/invalid)
DUMMY_MASKS = {
    'thumb': np.array([
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [True, False, False, False, True],
        [True, False, True, False, True],
        [True, True, True, True, True],
    ], dtype=bool),
    'index': np.array([
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [True, False, False, False, True],
        [True, True, False, False, True],
    ], dtype=bool),
    'middle': np.array([
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [True, False, False, False, True],
        [True, True, False, False, True],
    ], dtype=bool),
    'ring': np.array([
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [True, False, False, False, True],
        [True, True, False, False, True],
    ], dtype=bool),
    'little': np.array([
        [True, False, False, True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ], dtype=bool),
    'palm': np.array([
        [False, False, False, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
    ], dtype=bool),
}


@dataclass
class FingerForceData:
    """Force data for a single finger."""
    
    name: str
    """Name of the finger."""
    
    grid: np.ndarray
    """2D grid of force values (NaN for dummy positions)."""
    
    valid_mask: np.ndarray
    """Boolean mask of valid (non-dummy) positions."""
    
    timestamp: float = 0.0
    """Timestamp of the last update."""
    
    @property
    def force_sum(self) -> float:
        """Total force across all valid sensor points."""
        return float(np.nansum(self.grid))
    
    @property
    def force_mean(self) -> float:
        """Mean force across valid sensor points."""
        return float(np.nanmean(self.grid))
    
    @property
    def force_max(self) -> float:
        """Maximum force value."""
        return float(np.nanmax(self.grid))
    
    @property
    def valid_values(self) -> np.ndarray:
        """Array of valid (non-NaN) force values."""
        return self.grid[self.valid_mask]
    
    @property
    def num_valid_points(self) -> int:
        """Number of valid sensor points."""
        return int(np.sum(self.valid_mask))


@dataclass
class ROHandForceData:
    """Complete force data from ROHand tactile sensors."""
    
    fingers: dict[str, FingerForceData] = field(default_factory=dict)
    """Force data for each finger."""
    
    @property
    def thumb(self) -> Optional[FingerForceData]:
        """Get thumb force data."""
        return self.fingers.get('thumb')
    
    @property
    def index(self) -> Optional[FingerForceData]:
        """Get index finger force data."""
        return self.fingers.get('index')
    
    @property
    def middle(self) -> Optional[FingerForceData]:
        """Get middle finger force data."""
        return self.fingers.get('middle')
    
    @property
    def ring(self) -> Optional[FingerForceData]:
        """Get ring finger force data."""
        return self.fingers.get('ring')
    
    @property
    def little(self) -> Optional[FingerForceData]:
        """Get little finger force data."""
        return self.fingers.get('little')
    
    @property
    def palm(self) -> Optional[FingerForceData]:
        """Get palm force data."""
        return self.fingers.get('palm')
    
    @property
    def total_force(self) -> float:
        """Total force across all fingers."""
        return sum(f.force_sum for f in self.fingers.values())
    
    def get_force_sums(self) -> dict[str, float]:
        """Get force sums for all fingers as a dictionary."""
        return {name: data.force_sum for name, data in self.fingers.items()}
    
    def get_force_array(self) -> np.ndarray:
        """Get force sums as array in order: thumb, index, middle, ring, little, palm."""
        return np.array([
            self.fingers.get(name, FingerForceData(name, np.array([]), np.array([]))).force_sum
            for name in FINGER_NAMES
        ])


class ROHandForceSensor:
    """Sensor class for ROHand tactile force sensors.
    
    This sensor subscribes to PointCloud2 messages published by the ROHand
    node and provides access to the tactile force data in a structured format.
    
    Example:
        >>> sensor = ROHandForceSensor(namespace="rohand_left")
        >>> sensor.wait_until_ready()
        >>> data = sensor.force_data
        >>> print(f"Total force: {data.total_force}")
        >>> print(f"Index finger force: {data.index.force_sum}")
    """
    
    THREADS_REQUIRED = 2
    
    def __init__(
        self,
        namespace: str = "rohand_left",
        node: Optional[Node] = None,
        spin_node: bool = True,
    ):
        """Initialize the ROHand force sensor.
        
        Args:
            namespace: ROS2 namespace for the ROHand node.
            node: Existing ROS2 node to use. If None, creates a new node.
            spin_node: Whether to spin the node in a background thread.
        """
        if not rclpy.ok() and node is None:
            rclpy.init()
        
        self._namespace = namespace
        self._spin_node = spin_node
        
        # Create or use existing node
        self.node = (
            rclpy.create_node(
                node_name="rohand_force_sensor",
                namespace=namespace,
            )
            if node is None
            else node
        )
        
        # Initialize force data storage
        self._force_data: dict[str, FingerForceData] = {}
        self._subscribers: dict[str, rclpy.subscription.Subscription] = {}
        self._data_lock = threading.Lock()
        
        # Create subscribers for each finger
        for finger_name in FINGER_NAMES:
            topic = f"force_{finger_name}"
            self._subscribers[finger_name] = self.node.create_subscription(
                PointCloud2,
                topic,
                lambda msg, fn=finger_name: self._force_callback(msg, fn),
                qos_profile_sensor_data,
                callback_group=ReentrantCallbackGroup(),
            )
            
            # Initialize with empty data
            shape = FINGER_GRID_SHAPES[finger_name]
            grid = np.full(shape, np.nan, dtype=np.float32)
            valid_mask = ~DUMMY_MASKS[finger_name]
            self._force_data[finger_name] = FingerForceData(
                name=finger_name,
                grid=grid,
                valid_mask=valid_mask,
            )
        
        # Start spinning if requested
        if spin_node:
            self._spin_thread = threading.Thread(target=self._spin_node_thread, daemon=True)
            self._spin_thread.start()
    
    def _force_callback(self, msg: PointCloud2, finger_name: str):
        """Handle incoming force data for a finger.
        
        Args:
            msg: PointCloud2 message containing force grid.
            finger_name: Name of the finger this data is for.
        """
        print("Callback!")
        try:
            # Parse PointCloud2 message
            height = msg.height
            width = msg.width
            point_step = msg.point_step
            
            # Extract force values from message data
            grid = np.full((height, width), np.nan, dtype=np.float32)
            
            for row in range(height):
                for col in range(width):
                    offset = row * msg.row_step + col * point_step
                    force_value = struct.unpack('f', msg.data[offset:offset + 4])[0]
                    grid[row, col] = force_value
            
            # Update stored data
            with self._data_lock:
                valid_mask = ~np.isnan(grid)
                self._force_data[finger_name] = FingerForceData(
                    name=finger_name,
                    grid=grid,
                    valid_mask=valid_mask,
                    timestamp=time.time(),
                )
        except Exception as e:
            self.node.get_logger().error(f"Error parsing force data for {finger_name}: {e}")
    
    def _spin_node_thread(self):
        """Background thread to spin the node."""
        executor = MultiThreadedExecutor(num_threads=self.THREADS_REQUIRED)
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    
    @property
    def force_data(self) -> ROHandForceData:
        """Get the current force data for all fingers.
        
        Returns:
            ROHandForceData containing force grids and summaries for all fingers.
        """
        with self._data_lock:
            return ROHandForceData(fingers=dict(self._force_data))
    
    def get_finger_force(self, finger_name: str) -> Optional[FingerForceData]:
        """Get force data for a specific finger.
        
        Args:
            finger_name: Name of the finger ('thumb', 'index', 'middle', 'ring', 'little', 'palm').
            
        Returns:
            FingerForceData for the specified finger, or None if not available.
        """
        with self._data_lock:
            return self._force_data.get(finger_name)
    
    def get_force_sums(self) -> np.ndarray:
        """Get force sums for all fingers as a numpy array.
        
        Returns:
            Array of force sums in order: [thumb, index, middle, ring, little, palm]
        """
        return self.force_data.get_force_array()
    
    def is_ready(self) -> bool:
        """Check if sensor data is available for all fingers.
        
        Returns:
            True if data has been received for at least one finger.
        """
        with self._data_lock:
            return any(
                data.timestamp > 0 
                for data in self._force_data.values()
            )
    
    def wait_until_ready(self, timeout: float = 10.0, check_frequency: float = 10.0):
        """Wait until sensor data is available.
        
        Args:
            timeout: Maximum time to wait in seconds.
            check_frequency: How often to check (Hz).
            
        Raises:
            TimeoutError: If sensor doesn't become ready within timeout.
        """
        rate = self.node.create_rate(check_frequency)
        elapsed = 0.0
        dt = 1.0 / check_frequency
        
        while not self.is_ready():
            rate.sleep()
            elapsed += dt
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout waiting for ROHand force sensor. "
                    f"Are the topics being published in namespace '{self._namespace}'?"
                )
    
    def shutdown(self):
        """Shutdown the sensor and clean up resources."""
        for sub in self._subscribers.values():
            self.node.destroy_subscription(sub)
        if self._spin_node:
            self.node.destroy_node()
    
    @classmethod
    def from_yaml(
        cls,
        config_name: str = "rohand_force_left",
        namespace: Optional[str] = None,
        node: Optional[Node] = None,
        spin_node: bool = True,
    ) -> "ROHandForceSensor":
        """Create a ROHandForceSensor from a YAML configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension).
            namespace: Override namespace from config.
            node: Existing ROS2 node to use.
            spin_node: Whether to spin the node in a background thread.
            
        Returns:
            Configured ROHandForceSensor instance.
        """
        import yaml
        
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"
        
        config_path = find_config(f"sensors/{config_name}")
        if config_path is None:
            # Use default namespace if no config found
            ns = namespace or "rohand_left"
            return cls(namespace=ns, node=node, spin_node=spin_node)
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        ns = namespace or data.get("namespace", "rohand_left")
        
        return cls(namespace=ns, node=node, spin_node=spin_node)


def make_rohand_force_sensor(
    namespace: str = "rohand_left",
    node: Optional[Node] = None,
    spin_node: bool = True,
) -> ROHandForceSensor:
    """Factory function to create a ROHandForceSensor.
    
    Args:
        namespace: ROS2 namespace for the ROHand node.
        node: Existing ROS2 node to use.
        spin_node: Whether to spin the node in a background thread.
        
    Returns:
        Configured ROHandForceSensor instance.
    """
    return ROHandForceSensor(namespace=namespace, node=node, spin_node=spin_node)
