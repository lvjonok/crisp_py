#!/usr/bin/env python3
"""
ROHand Force Sensor Visualization Example.

This example demonstrates how to use the ROHandForceSensor to visualize
tactile force data from the ROHand dexterous hand, including:
- Real-time heatmap visualization of force distribution
- Force curve chart showing force trends over time

Prerequisites:
- ROHand hardware connected and powered
- rohand_ros2 package launched:
  ros2 launch rohand rohand_ap001.launch.py \\
    port_name:=/dev/ttyUSB0 \\
    baudrate:=115200 \\
    hand_ids:=[2] \\
    namespace:=rohand_left

Usage:
    python3 examples/rohand_force_visualization.py
"""

import os
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from crisp_py.sensors import (
    ROHandForceSensor,
    FINGER_NAMES,
    FINGER_GRID_SHAPES,
    DUMMY_MASKS,
)


# Visualization configuration
MAX_FORCE_SUM = 5000  # Maximum expected total force per finger
MAX_FORCE = 200       # Maximum expected single point force
COLOR_SCALE = 5       # Scale factor for heatmap colors
POINT_RADIUS = 2      # Radius for finger force points
PALM_POINT_RADIUS = 4 # Radius for palm force points
CURVE_HISTORY = 100   # Number of data points to show in curve


# Force point locations for heatmap visualization (left hand)
# These coordinates map to the hand reference image
LEFT_FORCE_POINTS = {
    'thumb': [
        (23, 231), (30, 231), (37, 231), (44, 231), (51, 231),
        (23, 225), (30, 225), (37, 225), (44, 225), (51, 225),
        (23, 219), (30, 219), (37, 219), (44, 219), (51, 219),
        (23, 213), (30, 213), (37, 213), (44, 213), (51, 213),
        (23, 207), (30, 207), (37, 207), (44, 207), (51, 207),
        (-1, -1), (30, 208), (37, 208), (44, 208), (-1, -1),
        (-1, -1), (33, 202), (-1, -1), (41, 202), (-1, -1),
        (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
    ],
    'index': [
        (-1, -1), (289, 109), (295, 109), (301, 109), (-1, -1),
        (-1, -1), (289, 103), (295, 103), (301, 103), (-1, -1),
        (-1, -1), (289, 97), (295, 97), (301, 97), (-1, -1),
        (-1, -1), (289, 91), (295, 91), (301, 91), (-1, -1),
        (-1, -1), (289, 85), (295, 85), (301, 85), (-1, -1),
        (283, 84), (289, 79), (295, 79), (301, 79), (307, 84),
        (283, 78), (289, 73), (295, 73), (301, 73), (307, 78),
        (283, 72), (289, 67), (295, 67), (301, 67), (307, 72),
        (283, 66), (289, 61), (295, 61), (301, 61), (307, 66),
        (283, 60), (289, 55), (295, 55), (301, 55), (307, 60),
        (-1, -1), (289, 49), (295, 49), (301, 49), (-1, -1),
        (-1, -1), (-1, -1), (292, 43), (298, 43), (-1, -1),
    ],
    'middle': [
        (-1, -1), (343, 89), (349, 89), (355, 89), (-1, -1),
        (-1, -1), (343, 83), (349, 83), (355, 83), (-1, -1),
        (-1, -1), (343, 77), (349, 77), (355, 77), (-1, -1),
        (-1, -1), (343, 71), (349, 71), (355, 71), (-1, -1),
        (-1, -1), (343, 65), (349, 65), (355, 65), (-1, -1),
        (337, 66), (343, 59), (349, 59), (355, 59), (361, 66),
        (337, 60), (343, 53), (349, 53), (355, 53), (361, 60),
        (337, 54), (343, 47), (349, 47), (355, 47), (361, 54),
        (337, 48), (343, 41), (349, 41), (355, 41), (361, 48),
        (337, 42), (343, 35), (349, 35), (355, 35), (361, 42),
        (-1, -1), (343, 29), (349, 29), (355, 29), (-1, -1),
        (-1, -1), (-1, -1), (346, 23), (352, 23), (-1, -1),
    ],
    'ring': [
        (-1, -1), (400, 109), (406, 109), (412, 109), (-1, -1),
        (-1, -1), (400, 103), (406, 103), (412, 103), (-1, -1),
        (-1, -1), (400, 97), (406, 97), (412, 97), (-1, -1),
        (-1, -1), (400, 91), (406, 91), (412, 91), (-1, -1),
        (-1, -1), (400, 85), (406, 85), (412, 85), (-1, -1),
        (394, 84), (400, 79), (406, 79), (412, 79), (418, 84),
        (394, 78), (400, 73), (406, 73), (412, 73), (418, 78),
        (394, 72), (400, 67), (406, 67), (412, 67), (418, 72),
        (394, 66), (400, 61), (406, 61), (412, 61), (418, 66),
        (394, 60), (400, 55), (406, 55), (412, 55), (418, 60),
        (-1, -1), (400, 49), (406, 49), (412, 49), (-1, -1),
        (-1, -1), (-1, -1), (403, 43), (409, 43), (-1, -1),
    ],
    'little': [
        (-1, -1), (460, 90), (454, 90), (-1, -1),
        (466, 95), (460, 95), (454, 95), (448, 95),
        (466, 100), (460, 100), (454, 100), (448, 100),
        (466, 105), (460, 105), (454, 105), (448, 105),
        (466, 110), (460, 110), (454, 110), (448, 110),
        (466, 115), (460, 115), (454, 115), (448, 115),
        (466, 120), (460, 120), (454, 120), (448, 120),
        (466, 125), (460, 125), (454, 125), (448, 125),
    ],
    'palm': [
        (350, 295), (350, 305), (350, 315), (-1, -1), (-1, -1),
        (360, 295), (360, 305), (360, 315), (-1, -1), (-1, -1),
        (370, 295), (370, 305), (370, 315), (370, 325), (-1, -1),
        (380, 295), (380, 305), (380, 315), (380, 325), (380, 335),
        (390, 295), (390, 305), (390, 315), (390, 325), (390, 335),
        (400, 295), (400, 305), (400, 315), (400, 325), (400, 335),
        (410, 295), (410, 305), (410, 315), (410, 325), (410, 335),
        (420, 295), (420, 305), (420, 315), (420, 325), (420, 335),
        (430, 295), (430, 305), (430, 315), (430, 325), (430, 335),
        (440, 295), (440, 305), (440, 315), (440, 325), (440, 335),
        (450, 295), (450, 305), (450, 315), (450, 325), (450, 335),
    ],
}

# Right hand force points (mirrored)
RIGHT_FORCE_POINTS = {
    'thumb': [
        (440, 231), (447, 231), (454, 231), (461, 231), (468, 231),
        (440, 225), (447, 225), (454, 225), (461, 225), (468, 225),
        (440, 219), (447, 219), (454, 219), (461, 219), (468, 219),
        (440, 213), (447, 213), (454, 213), (461, 213), (468, 213),
        (440, 207), (447, 207), (454, 207), (461, 207), (468, 207),
        (-1, -1), (447, 208), (454, 208), (461, 208), (-1, -1),
        (-1, -1), (450, 202), (-1, -1), (458, 202), (-1, -1),
        (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
    ],
    'index': [
        (-1, -1), (190, 109), (196, 109), (202, 109), (-1, -1),
        (-1, -1), (190, 103), (196, 103), (202, 103), (-1, -1),
        (-1, -1), (190, 97), (196, 97), (202, 97), (-1, -1),
        (-1, -1), (190, 91), (196, 91), (202, 91), (-1, -1),
        (-1, -1), (190, 85), (196, 85), (202, 85), (-1, -1),
        (184, 84), (190, 79), (196, 79), (202, 79), (208, 84),
        (184, 78), (190, 73), (196, 73), (202, 73), (208, 78),
        (184, 72), (190, 67), (196, 67), (202, 67), (208, 72),
        (184, 66), (190, 61), (196, 61), (202, 61), (208, 66),
        (184, 60), (190, 55), (196, 55), (202, 55), (208, 60),
        (-1, -1), (190, 49), (196, 49), (202, 49), (-1, -1),
        (-1, -1), (-1, -1), (193, 43), (199, 43), (-1, -1),
    ],
    'middle': [
        (-1, -1), (136, 89), (142, 89), (148, 89), (-1, -1),
        (-1, -1), (136, 83), (142, 83), (148, 83), (-1, -1),
        (-1, -1), (136, 77), (142, 77), (148, 77), (-1, -1),
        (-1, -1), (136, 71), (142, 71), (148, 71), (-1, -1),
        (-1, -1), (136, 65), (142, 65), (148, 65), (-1, -1),
        (130, 66), (136, 59), (142, 59), (148, 59), (154, 66),
        (130, 60), (136, 53), (142, 53), (148, 53), (154, 60),
        (130, 54), (136, 47), (142, 47), (148, 47), (154, 54),
        (130, 48), (136, 41), (142, 41), (148, 41), (154, 48),
        (130, 42), (136, 35), (142, 35), (148, 35), (154, 42),
        (-1, -1), (136, 29), (142, 29), (148, 29), (-1, -1),
        (-1, -1), (-1, -1), (139, 23), (145, 23), (-1, -1),
    ],
    'ring': [
        (-1, -1), (78, 107), (84, 107), (90, 107), (-1, -1),
        (-1, -1), (78, 101), (84, 101), (90, 101), (-1, -1),
        (-1, -1), (78, 95), (84, 95), (90, 95), (-1, -1),
        (-1, -1), (78, 89), (84, 89), (90, 89), (-1, -1),
        (-1, -1), (78, 83), (84, 83), (90, 83), (-1, -1),
        (72, 82), (78, 77), (84, 77), (90, 77), (96, 82),
        (72, 76), (78, 71), (84, 71), (90, 71), (96, 76),
        (72, 70), (78, 65), (84, 65), (90, 65), (96, 70),
        (72, 64), (78, 59), (84, 59), (90, 59), (96, 64),
        (72, 58), (78, 53), (84, 53), (90, 53), (96, 58),
        (-1, -1), (78, 47), (84, 47), (90, 47), (-1, -1),
        (-1, -1), (-1, -1), (81, 41), (87, 41), (-1, -1),
    ],
    'little': [
        (-1, -1), (37, 90), (31, 90), (-1, -1),
        (43, 95), (37, 95), (31, 95), (25, 95),
        (43, 100), (37, 100), (31, 100), (25, 100),
        (43, 105), (37, 105), (31, 105), (25, 105),
        (43, 110), (37, 110), (31, 110), (25, 110),
        (43, 115), (37, 115), (31, 115), (25, 115),
        (43, 120), (37, 120), (31, 120), (25, 120),
        (43, 125), (37, 125), (31, 125), (25, 125),
    ],
    'palm': [
        (-1, -1), (-1, -1), (141, 315), (141, 305), (141, 295),
        (-1, -1), (-1, -1), (131, 315), (131, 305), (131, 295),
        (-1, -1), (121, 325), (121, 315), (121, 305), (121, 295),
        (111, 335), (111, 325), (111, 315), (111, 305), (111, 295),
        (101, 335), (101, 325), (101, 315), (101, 305), (101, 295),
        (91, 335), (91, 325), (91, 315), (91, 305), (91, 295),
        (81, 335), (81, 325), (81, 315), (81, 305), (81, 295),
        (71, 335), (71, 325), (71, 315), (71, 305), (71, 295),
        (61, 335), (61, 325), (61, 315), (61, 305), (61, 295),
        (51, 335), (51, 325), (51, 315), (51, 305), (51, 295),
        (41, 335), (41, 325), (41, 315), (41, 305), (41, 295),
    ],
}


def interpolate(n: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    """Linear interpolation between ranges."""
    return (n - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


def clamp(n: float, smallest: float, largest: float) -> float:
    """Clamp value to range."""
    return max(smallest, min(n, largest))


class ROHandForceVisualizer:
    """Visualization class for ROHand tactile force sensors."""
    
    def __init__(
        self,
        namespace: str = "rohand_left",
        hand_type: str = "left",
        show_heatmap: bool = True,
        show_curve: bool = True,
    ):
        """Initialize the visualizer.
        
        Args:
            namespace: ROS2 namespace for the ROHand node.
            hand_type: 'left' or 'right' hand.
            show_heatmap: Whether to show the heatmap visualization.
            show_curve: Whether to show the force curve chart.
        """
        self.namespace = namespace
        self.hand_type = hand_type
        self.show_heatmap = show_heatmap
        self.show_curve = show_curve
        
        # Select force point locations based on hand type
        self.force_points = LEFT_FORCE_POINTS if hand_type == "left" else RIGHT_FORCE_POINTS
        
        # Initialize sensor
        print(f"Connecting to ROHand force sensor in namespace '{namespace}'...")
        self.sensor = ROHandForceSensor(namespace=namespace)
        
        # Data history for curve visualization
        self.force_history: dict[str, deque] = {
            name: deque(maxlen=CURVE_HISTORY) for name in FINGER_NAMES
        }
        self.time_history: deque = deque(maxlen=CURVE_HISTORY)
        self.start_time = time.time()
        
        # Load reference image
        self.hand_image = self._load_hand_image()
        self.img_height, self.img_width = self.hand_image.shape[:2]
        
        # Initialize matplotlib for curve
        if self.show_curve:
            self._init_curve_plot()
        
        # Initialize OpenCV window for heatmap
        if self.show_heatmap:
            cv2.namedWindow("ROHand Force Heatmap", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROHand Force Heatmap", 600, 400)
    
    def _load_hand_image(self) -> np.ndarray:
        """Load the hand reference image."""
        # Try to find the image in common locations
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / "media" / f"force_{self.hand_type}.png",
            script_dir / f"force_{self.hand_type}.png",
            Path.home() / "github.com" / "franka_hardware" / "crisp_py" / "media" / f"force_{self.hand_type}.png",
            Path.home() / "github.com" / "franka_hardware" / "roh_gen2_demos" / "force_on_rohand" / "pic" / f"force_{self.hand_type}.png",
        ]
        
        for path in possible_paths:
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    print(f"Loaded hand image from: {path}")
                    return img
        
        # Create a blank placeholder image if no image found
        print("Warning: Hand reference image not found, using blank background")
        return np.zeros((400, 500, 3), dtype=np.uint8) + 50
    
    def _init_curve_plot(self):
        """Initialize the matplotlib figure for force curves."""
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.fig.subplots_adjust(bottom=0.15)
        
        self.lines = {}
        self.annotations = {}
        colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
        finger_labels = ["Thumb", "Index", "Middle", "Ring", "Little", "Palm"]
        
        for i, (name, label) in enumerate(zip(FINGER_NAMES, finger_labels)):
            line, = self.ax.plot([], [], colors[i], label=label, linewidth=2)
            self.lines[name] = line
            
            annotation = self.ax.text(
                0, 0, "",
                ha="left", va="center",
                bbox=dict(facecolor="white", pad=0.7),
                fontsize=10
            )
            self.annotations[name] = annotation
        
        self.ax.set_ylabel("Force Value")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylim(0, MAX_FORCE_SUM)
        self.ax.yaxis.set_major_locator(MultipleLocator(MAX_FORCE_SUM / 5))
        self.ax.grid(True)
        self.ax.legend(loc="upper right")
        self.ax.set_title("Real-time ROHand Force Data")
        self.fig.show()
    
    def update_heatmap(self, force_data):
        """Update the heatmap visualization.
        
        Args:
            force_data: ROHandForceData from the sensor.
        """
        # Create blank heatmap
        heatmap = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        for finger_name in FINGER_NAMES:
            finger_data = force_data.fingers.get(finger_name)
            if finger_data is None:
                continue
            
            force_points = self.force_points[finger_name]
            force_grid = finger_data.grid.flatten()
            
            for idx, (x, y) in enumerate(force_points):
                # Skip dummy points
                if x < 0 or y < 0:
                    continue
                if x >= self.img_width or y >= self.img_height:
                    continue
                
                # Get force value
                if idx < len(force_grid):
                    value = force_grid[idx]
                    if np.isnan(value):
                        continue
                    value = value * COLOR_SCALE
                else:
                    continue
                
                # Map force to color (HSV color space)
                color = interpolate(value, 0, MAX_FORCE, 120, 1)
                color = int(clamp(color, 1, 120))
                
                # Draw circle at force point
                radius = PALM_POINT_RADIUS if finger_name == 'palm' else POINT_RADIUS
                cv2.circle(heatmap, (x, y), radius, color, -1)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
        heatmap_colored = cv2.resize(heatmap_colored, (self.img_width, self.img_height))
        
        # Create mask for blending
        mask = heatmap > 0
        
        # Blend with hand image
        result = self.hand_image.copy()
        if np.any(mask):
            result[mask] = cv2.addWeighted(
                self.hand_image[mask], 0.2,
                heatmap_colored[mask], 0.8, 0
            )
        
        cv2.imshow("ROHand Force Heatmap", result)
    
    def update_curve(self, force_data):
        """Update the force curve chart.
        
        Args:
            force_data: ROHandForceData from the sensor.
        """
        current_time = time.time() - self.start_time
        self.time_history.append(current_time)
        
        # Update data history
        finger_labels = ["Thumb", "Index", "Middle", "Ring", "Little", "Palm"]
        for name in FINGER_NAMES:
            finger_data = force_data.fingers.get(name)
            force_sum = finger_data.force_sum if finger_data else 0.0
            self.force_history[name].append(force_sum)
        
        # Update plot lines
        for i, (name, label) in enumerate(zip(FINGER_NAMES, finger_labels)):
            if self.force_history[name]:
                x_data = list(self.time_history)
                y_data = list(self.force_history[name])
                self.lines[name].set_data(x_data, y_data)
                
                # Update annotation at curve end
                if x_data and y_data:
                    self.annotations[name].set_position((x_data[-1], y_data[-1]))
                    self.annotations[name].set_text(f"{label}: {y_data[-1]:.1f}")
        
        # Adjust view
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=False)
        self.ax.legend(loc="upper right")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Run the visualization loop."""
        print("\nWaiting for force sensor data...")
        try:
            self.sensor.wait_until_ready(timeout=10.0)
            print("✓ Force sensor ready!")
        except TimeoutError as e:
            print(f"✗ Error: {e}")
            return
        
        print("\nStarting visualization...")
        print("Press 'q' in the heatmap window or Ctrl+C to exit\n")
        
        try:
            while True:
                # Get latest force data
                force_data = self.sensor.force_data
                
                # Update visualizations
                if self.show_heatmap:
                    self.update_heatmap(force_data)
                
                if self.show_curve:
                    self.update_curve(force_data)
                
                # Print force sums
                force_sums = force_data.get_force_sums()
                print(
                    f"\rForce: Th={force_sums.get('thumb', 0):6.1f} "
                    f"If={force_sums.get('index', 0):6.1f} "
                    f"Mf={force_sums.get('middle', 0):6.1f} "
                    f"Rf={force_sums.get('ring', 0):6.1f} "
                    f"Lf={force_sums.get('little', 0):6.1f} "
                    f"Pm={force_sums.get('palm', 0):6.1f}",
                    end="", flush=True
                )
                
                # Check for exit
                if self.show_heatmap:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    time.sleep(0.033)  # ~30 Hz
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        if self.show_heatmap:
            cv2.destroyAllWindows()
        if self.show_curve:
            plt.close(self.fig)
        self.sensor.shutdown()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ROHand Force Sensor Visualization")
    parser.add_argument(
        "--namespace", "-n",
        type=str,
        default="rohand_left",
        help="ROS2 namespace for the ROHand node"
    )
    parser.add_argument(
        "--hand", "-H",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Hand type (left or right)"
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable heatmap visualization"
    )
    parser.add_argument(
        "--no-curve",
        action="store_true",
        help="Disable force curve chart"
    )
    
    args = parser.parse_args()
    
    visualizer = ROHandForceVisualizer(
        namespace=args.namespace,
        hand_type=args.hand,
        show_heatmap=not args.no_heatmap,
        show_curve=not args.no_curve,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
