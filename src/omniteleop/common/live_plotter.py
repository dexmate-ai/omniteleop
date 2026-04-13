#!/usr/bin/env python3
"""Live plotting utility for visualizing robot commands in real-time."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time
from typing import Dict, Optional, Any, List
from queue import Queue
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for better performance

class LiveCommandPlotter:
    """Real-time plotting of robot commands with minimal overhead."""

    def __init__(
        self,
        window_size: int = 500,
        update_rate: int = 20,
        components: Optional[List[str]] = None,
        plot_velocity: bool = False,
        plot_style: str = "combined",  # "combined" or "separate"
    ):
        """Initialize live plotter.

        Args:
            window_size: Number of samples to display
            update_rate: Plot update frequency in Hz
            components: List of components to plot (default: arms only)
            plot_velocity: Whether to plot velocity in addition to position
            plot_style: "combined" (all joints on one plot) or "separate" (individual subplots)
        """
        self.window_size = window_size
        self.update_interval = 1000 // update_rate  # milliseconds
        self.plot_velocity = plot_velocity
        self.plot_style = plot_style

        # Default to arm components
        if components is None:
            self.components = ["left_arm", "right_arm"]
        else:
            self.components = components

        # Data storage
        self.data_queues = {}
        self.time_queues = {}
        for comp in self.components:
            self.data_queues[comp] = {
                "pos": deque(maxlen=window_size),
                "vel": deque(maxlen=window_size) if plot_velocity else None,
            }
            self.time_queues[comp] = deque(maxlen=window_size)

        # Thread-safe command queue
        self.command_queue = Queue(maxsize=100)

        # Plotting thread
        self.plot_thread = None
        self.running = False

        # Figure and axes
        self.fig = None
        self.axes = {}
        self.lines = {}

        # Start time for relative timestamps
        self.start_time = None

    def start(self):
        """Start the live plotting in a separate thread."""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()

        # Start plotting in separate thread
        self.plot_thread = threading.Thread(target=self._run_plot, daemon=True)
        self.plot_thread.start()

    def stop(self):
        """Stop the live plotting."""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)
        plt.close("all")

    def update_command(self, components: Dict[str, Dict[str, Any]]):
        """Update with new command data (non-blocking).

        Args:
            components: Dictionary of component commands
        """
        if not self.running:
            return

        # Add to queue if not full (drop if full to avoid blocking)
        if not self.command_queue.full():
            self.command_queue.put((time.time(), components))

    def _run_plot(self):
        """Run the plotting loop in a separate thread."""

        if self.plot_style == "separate":
            # Create individual subplot for each joint
            max_joints = 7  # Assume max 7 DOF for arms
            num_components = len(self.components)
            rows = max_joints
            cols = num_components * (2 if self.plot_velocity else 1)

            self.fig = plt.figure(figsize=(4 * cols, 2 * rows))
            self.fig.suptitle("Live Robot Command Visualization (Per-Joint)")

            # Create grid spec for better layout control
            gs = self.fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)

            # Initialize axes for each component and joint
            for i, comp in enumerate(self.components):
                col_offset = i * (2 if self.plot_velocity else 1)

                # Create position axes for each joint
                for j in range(max_joints):
                    ax_pos = self.fig.add_subplot(gs[j, col_offset])
                    if j == 0:
                        ax_pos.set_title(f"{comp} - Position")
                    ax_pos.set_ylabel(f"J{j + 1} (rad)", fontsize=8)
                    if j == rows - 1:
                        ax_pos.set_xlabel("Time (s)", fontsize=8)
                    ax_pos.grid(True, alpha=0.3)
                    ax_pos.tick_params(labelsize=8)
                    self.axes[f"{comp}_pos_j{j}"] = ax_pos

                # Create velocity axes if enabled
                if self.plot_velocity:
                    for j in range(max_joints):
                        ax_vel = self.fig.add_subplot(gs[j, col_offset + 1])
                        if j == 0:
                            ax_vel.set_title(f"{comp} - Velocity")
                        ax_vel.set_ylabel(f"J{j + 1} (rad/s)", fontsize=8)
                        if j == rows - 1:
                            ax_vel.set_xlabel("Time (s)", fontsize=8)
                        ax_vel.grid(True, alpha=0.3)
                        ax_vel.tick_params(labelsize=8)
                        self.axes[f"{comp}_vel_j{j}"] = ax_vel

        else:  # combined style
            # Create figure with combined plots
            num_components = len(self.components)
            rows = 2 if self.plot_velocity else 1

            self.fig, axes_array = plt.subplots(
                rows,
                num_components,
                figsize=(5 * num_components, 4 * rows),
                squeeze=False,
            )
            self.fig.suptitle("Live Robot Command Visualization (Combined)")

            # Initialize axes for each component
            for i, comp in enumerate(self.components):
                # Position plot
                ax_pos = axes_array[0, i]
                ax_pos.set_title(f"{comp} - Position")
                ax_pos.set_xlabel("Time (s)")
                ax_pos.set_ylabel("Joint Position (rad)")
                ax_pos.grid(True, alpha=0.3)
                self.axes[f"{comp}_pos"] = ax_pos

                # Velocity plot (if enabled)
                if self.plot_velocity:
                    ax_vel = axes_array[1, i]
                    ax_vel.set_title(f"{comp} - Velocity")
                    ax_vel.set_xlabel("Time (s)")
                    ax_vel.set_ylabel("Joint Velocity (rad/s)")
                    ax_vel.grid(True, alpha=0.3)
                    self.axes[f"{comp}_vel"] = ax_vel

        # Initialize empty lines (will be created on first data)
        self.lines = {}

        # Set up animation
        self.ani = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show(block=True)

    def _update_plot(self, frame):
        """Update plot with latest data."""
        # Process all pending commands
        while not self.command_queue.empty():
            try:
                timestamp, components = self.command_queue.get_nowait()
                relative_time = timestamp - self.start_time

                # Store data for each component
                for comp in self.components:
                    if comp in components:
                        comp_data = components[comp]

                        # Update time
                        self.time_queues[comp].append(relative_time)

                        # Update position data
                        if "pos" in comp_data:
                            pos_data = np.array(comp_data["pos"])
                            self.data_queues[comp]["pos"].append(pos_data)

                        # Update velocity data if enabled
                        if self.plot_velocity and "vel" in comp_data:
                            vel_data = np.array(comp_data["vel"])
                            self.data_queues[comp]["vel"].append(vel_data)
            except Exception:
                break

        # Update plots for each component
        for comp in self.components:
            if len(self.time_queues[comp]) == 0:
                continue

            times = np.array(self.time_queues[comp])

            # Update position plot
            self._update_component_plot(comp, "pos", times)

            # Update velocity plot if enabled
            if self.plot_velocity:
                self._update_component_plot(comp, "vel", times)

        # Adjust x-axis limits
        if any(len(self.time_queues[comp]) > 0 for comp in self.components):
            current_time = time.time() - self.start_time
            x_min = max(0, current_time - 10)  # Show last 10 seconds
            x_max = current_time + 0.5

            for ax in self.axes.values():
                ax.set_xlim(x_min, x_max)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

    def _update_component_plot(self, component: str, field: str, times: np.ndarray):
        """Update plot for a specific component and field."""
        data_queue = self.data_queues[component][field]

        if data_queue is None or len(data_queue) == 0:
            return

        # Convert to numpy array for easier manipulation
        data_array = np.array(list(data_queue))

        # Determine number of joints
        if data_array.ndim == 1:
            num_joints = 1
            data_array = data_array.reshape(-1, 1)
        else:
            num_joints = data_array.shape[1]

        if self.plot_style == "separate":
            # Update each joint's individual subplot
            for j in range(num_joints):
                ax_key = f"{component}_{field}_j{j}"
                if ax_key not in self.axes:
                    continue

                ax = self.axes[ax_key]
                line_key = ax_key

                # Create or update line for this joint
                if line_key not in self.lines:
                    # Create single line for this joint
                    color = plt.cm.tab10(j / 7.0)  # Color based on joint index
                    (line,) = ax.plot([], [], color=color, linewidth=1.5)
                    self.lines[line_key] = line

                # Update line data
                self.lines[line_key].set_data(times, data_array[:, j])

        else:  # combined style
            ax_key = f"{component}_{field}"
            if ax_key not in self.axes:
                return

            ax = self.axes[ax_key]
            line_key = ax_key

            # Create or update lines for each joint
            if line_key not in self.lines:
                # Create lines for each joint
                self.lines[line_key] = []
                colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
                for j in range(num_joints):
                    (line,) = ax.plot(
                        [], [], label=f"Joint {j + 1}", color=colors[j], linewidth=1.5
                    )
                    self.lines[line_key].append(line)
                ax.legend(loc="upper left", fontsize=8)

            # Update line data
            for j, line in enumerate(self.lines[line_key]):
                if j < num_joints:
                    line.set_data(times, data_array[:, j])

class SimpleLivePlotter:
    """Simplified live plotter for quick visualization."""

    def __init__(self, components: List[str] = None, plot_style: str = "combined"):
        """Initialize simple plotter.

        Args:
            components: Components to plot (default: left_arm, right_arm)
            plot_style: "combined" or "separate" for joint visualization
        """
        self.components = components or ["left_arm", "right_arm"]
        self.plotter = LiveCommandPlotter(
            window_size=500,
            update_rate=20,
            components=self.components,
            plot_velocity=False,
            plot_style=plot_style,
        )
        self.started = False

    def update(self, command_dict: Dict):
        """Update plotter with command data.

        Args:
            command_dict: Full command dictionary with 'components' key
        """
        if not self.started:
            self.plotter.start()
            self.started = True

        if "components" in command_dict:
            self.plotter.update_command(command_dict["components"])

    def stop(self):
        """Stop the plotter."""
        if self.started:
            self.plotter.stop()
            self.started = False
