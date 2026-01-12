#!/usr/bin/env python3
"""
Interactive GUI for controlling ROHand gripper joints independently.

This example provides a graphical interface with sliders for each joint,
allowing real-time control and monitoring of the ROHand gripper.

Prerequisites:
- ROHand hardware connected and powered
- rohand_ros2 package launched:
  ros2 launch rohand rohand_ap001.launch.py \\
    port_name:=/dev/ttyUSB0 \\
    baudrate:=115200 \\
    hand_ids:=[2] \\
    namespace:=rohand_left
"""

import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from crisp_py.gripper.rohand_gripper import ROHandGripper
import numpy as np


class ROHandGUI:
    """GUI for controlling ROHand gripper with individual joint sliders."""
    
    def __init__(self, namespace: str = "rohand_left"):
        """Initialize the GUI and gripper connection.
        
        Args:
            namespace: ROS2 namespace for the ROHand gripper
        """
        # Initialize ROS2 and gripper
        rclpy.init()
        self.gripper = ROHandGripper.from_yaml(
            config_name="rohand_left",
            namespace=namespace,
            spin_node=True
        )
        
        # Wait for gripper to be ready
        print("Waiting for ROHand gripper...")
        try:
            self.gripper.wait_until_ready(timeout=10.0)
            print("✓ Gripper ready!")
        except TimeoutError:
            print("✗ Error: Gripper not ready. Make sure rohand node is running.")
            raise
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("ROHand Gripper Control")
        self.root.geometry("800x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Joint sliders dictionary
        self.sliders = {}
        self.position_labels = {}
        self.effort_labels = {}
        
        # Create GUI elements
        self._create_widgets()
        
        # Start position update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_positions, daemon=True)
        self.update_thread.start()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="ROHand Gripper Joint Control",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Status bar (create early so callbacks can access it)
        self.status_label = ttk.Label(
            main_frame,
            text="Status: Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        # Gesture buttons
        gesture_frame = ttk.LabelFrame(main_frame, text="Quick Gestures", padding="10")
        gesture_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        gestures = [
            ("Open", "open"),
            ("Close", "close"),
            ("Rock", "rock"),
            ("Pointing", "pointing")
        ]
        
        for idx, (label, gesture) in enumerate(gestures):
            btn = ttk.Button(
                gesture_frame,
                text=label,
                command=lambda g=gesture: self._execute_gesture(g)
            )
            btn.grid(row=0, column=idx, padx=5)
        
        # Joint controls
        controls_frame = ttk.LabelFrame(main_frame, text="Joint Controls", padding="10")
        controls_frame.grid(row=2, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create sliders for each joint
        row = 0
        
        # Thumb joints
        self._add_joint_section(controls_frame, row, "THUMB JOINTS")
        row += 1
        self._add_joint_slider(controls_frame, row, 'th_root_link', "Thumb Rotation")
        row += 1
        self._add_joint_slider(controls_frame, row, 'th_proximal_link', "Thumb Proximal")
        row += 1
        
        # Index finger
        self._add_joint_section(controls_frame, row, "INDEX FINGER")
        row += 1
        self._add_joint_slider(controls_frame, row, 'if_proximal_link', "Index Proximal")
        row += 1
        
        # Middle finger
        self._add_joint_section(controls_frame, row, "MIDDLE FINGER")
        row += 1
        self._add_joint_slider(controls_frame, row, 'mf_proximal_link', "Middle Proximal")
        row += 1
        
        # Ring finger
        self._add_joint_section(controls_frame, row, "RING FINGER")
        row += 1
        self._add_joint_slider(controls_frame, row, 'rf_proximal_link', "Ring Proximal")
        row += 1
        
        # Little finger
        self._add_joint_section(controls_frame, row, "LITTLE FINGER")
        row += 1
        self._add_joint_slider(controls_frame, row, 'lf_proximal_link', "Little Proximal")
        row += 1
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def _add_joint_section(self, parent, row, title):
        """Add a section header for joint group."""
        label = ttk.Label(
            parent,
            text=title,
            font=('Arial', 10, 'bold')
        )
        label.grid(row=row, column=0, columnspan=4, pady=(10, 5), sticky=tk.W)
    
    def _add_joint_slider(self, parent, row, joint_name, display_name):
        """Add a slider and labels for a joint.
        
        Args:
            parent: Parent widget
            row: Grid row number
            joint_name: Internal joint name
            display_name: Human-readable name
        """
        # Get joint limits
        limits = self.gripper.get_joint_limits(joint_name)
        if limits is None:
            limits = (0.0, 1.57)
        min_val, max_val = limits
        
        # Joint name label
        name_label = ttk.Label(parent, text=display_name, width=20)
        name_label.grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Slider
        slider = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            command=lambda val, jn=joint_name: self._on_slider_change(jn, float(val))
        )
        slider.set(0.0)  # Start at open position
        slider.grid(row=row, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.sliders[joint_name] = slider
        
        # Current position label
        pos_label = ttk.Label(parent, text="0.000 rad", width=10)
        pos_label.grid(row=row, column=2, padx=5, pady=2)
        self.position_labels[joint_name] = pos_label
        
        # Effort label
        effort_label = ttk.Label(parent, text="0.00 N", width=10)
        effort_label.grid(row=row, column=3, padx=5, pady=2)
        self.effort_labels[joint_name] = effort_label
        
        # Configure column weights
        parent.columnconfigure(1, weight=1)
    
    def _on_slider_change(self, joint_name, value):
        """Handle slider value change.
        
        Args:
            joint_name: Name of the joint
            value: New slider value
        """
        self.gripper.set_joint_target(joint_name, value)
        self.status_label.config(text=f"Status: Setting {joint_name} to {value:.3f} rad")
    
    def _execute_gesture(self, gesture_name):
        """Execute a predefined gesture.
        
        Args:
            gesture_name: Name of the gesture to execute
        """
        self.gripper.execute_gesture(gesture_name)
        self.status_label.config(text=f"Status: Executing gesture '{gesture_name}'")
        
        # Update sliders to match gesture
        gesture = self.gripper.GESTURES[gesture_name]
        for joint_name, position in gesture.items():
            if joint_name in self.sliders:
                self.sliders[joint_name].set(position)
    
    def _update_positions(self):
        """Background thread to update position and effort displays."""
        while self.running:
            try:
                # Update position and effort labels
                for joint_name in self.sliders.keys():
                    pos = self.gripper.get_joint_position(joint_name)
                    effort = self.gripper.get_joint_effort(joint_name)
                    
                    if pos is not None:
                        self.position_labels[joint_name].config(text=f"{pos:.3f} rad")
                    
                    if effort is not None:
                        self.effort_labels[joint_name].config(text=f"{effort:.2f} N")
                
                # Small delay to avoid excessive updates
                threading.Event().wait(0.1)
            except Exception as e:
                print(f"Error updating positions: {e}")
                break
    
    def on_closing(self):
        """Handle window closing."""
        print("\nShutting down...")
        self.running = False
        
        # Open gripper before closing
        print("Opening gripper...")
        self.gripper.open()
        threading.Event().wait(1.0)
        
        self.gripper.shutdown()
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        print("\nROHand GUI Control started!")
        print("Use sliders to control individual joints")
        print("Close window to exit\n")
        self.root.mainloop()


def main():
    """Main entry point."""
    try:
        gui = ROHandGUI(namespace="rohand_left")
        gui.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
