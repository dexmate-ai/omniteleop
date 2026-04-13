#!/usr/bin/env python3
"""Telemetry data simulator for debugging without real robot.

This script simulates the telemetry data that would be published by
the robot_controller, allowing you to test the telemetry_viewer without
needing actual robot hardware.
"""

import time
import numpy as np
from typing import Dict, Any
import tyro
from loguru import logger

from dexcomm import Node
from omniteleop.common import get_config
from dexcomm.codecs import DictDataCodec
from omniteleop.common.logging import setup_logging

class TelemetrySimulator:
    """Simulates robot telemetry data for debugging purposes."""

    def __init__(
        self,
        namespace: str = "",
        telemetry_topic: str = "robot/telemetry",
        publish_rate: int = 50,
        simulation_mode: str = "sine",  # sine, random, or static
        noise_level: float = 0.01,
    ):
        """Initialize telemetry simulator.

        Args:
            namespace: Optional namespace prefix.
            telemetry_topic: Topic to publish telemetry data.
            publish_rate: Publishing frequency in Hz.
            simulation_mode: Simulation pattern (sine, random, static).
            noise_level: Amount of noise to add (0.0 to 1.0).
        """
        self.node = Node(name="telemetry_simulator", namespace=namespace)

        self.telemetry_topic = telemetry_topic
        self.publish_rate = publish_rate
        self.simulation_mode = simulation_mode
        self.noise_level = noise_level

        # Load configuration for default positions
        self.config = get_config()

        # Component definitions with joint counts
        self.component_joints = {
            "left_arm": 7,
            "right_arm": 7,
            "torso": 3,
            "head": 2,
            "left_hand": 6,
            "right_hand": 6,
        }

        # Initialize base positions (home positions)
        self.base_positions = self._get_base_positions()

        # Publisher for telemetry
        self.telemetry_publisher = None

        # Simulation state
        self.start_time = time.time()
        self.frame_count = 0

        # Per-component state for richer dynamics (e.g., head lag)
        self._prev_filtered: Dict[str, np.ndarray] = {}
        self._prev_robot: Dict[str, np.ndarray] = {}

        logger.info(
            f"Telemetry simulator initialized: {simulation_mode} mode at {publish_rate}Hz"
        )

    def _get_base_positions(self) -> Dict[str, np.ndarray]:
        """Get base positions for each component from config."""
        base_pos = {}

        # Default positions if not in config
        defaults = {
            "left_arm": [0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0],
            "right_arm": [0.0, 0.5, 0.0, -1.5, 0.0, 0.5, 0.0],
            "torso": [0.0, 0.0, 0.0],
            "head": [0.0, 0.0],
            "left_hand": [0.0] * 6,
            "right_hand": [0.0] * 6,
        }

        for comp, default in defaults.items():
            # Try to get from config first
            init_pos = self.config.get_init_pos(comp)
            if init_pos is not None:
                base_pos[comp] = np.array(init_pos)
            else:
                base_pos[comp] = np.array(default)

        return base_pos

    def initialize(self):
        """Initialize publisher."""
        # Create telemetry publisher
        self.telemetry_publisher = self.node.create_publisher(
            self.telemetry_topic,
            encoder=DictDataCodec.encode,
        )

    def _generate_positions(self, component: str, t: float) -> np.ndarray:
        """Generate simulated positions for a component.

        Args:
            component: Component name.
            t: Current time in seconds.

        Returns:
            Array of joint positions.
        """
        base_pos = self.base_positions[component]
        n_joints = len(base_pos)

        if self.simulation_mode == "sine":
            # Sine wave oscillation around base position
            # Different frequency for each joint to create interesting patterns
            positions = base_pos.copy()
            for i in range(n_joints):
                freq = 0.1 * (i + 1)  # Different frequency for each joint
                amplitude = 0.2  # Radians
                positions[i] += amplitude * np.sin(2 * np.pi * freq * t)

        elif self.simulation_mode == "random":
            # Random walk around base position
            positions = base_pos + np.random.randn(n_joints) * 0.1

        else:  # static
            # Static at base position
            positions = base_pos.copy()

        # Add noise if specified
        if self.noise_level > 0:
            positions += np.random.randn(n_joints) * self.noise_level

        return positions

    def _generate_telemetry_message(self) -> Dict[str, Any]:
        """Generate a complete telemetry message.

        Returns:
            Telemetry message dictionary.
        """
        current_time = time.time()
        t = current_time - self.start_time

        # Generate raw command (before filtering)
        raw_command = {}
        for comp in self.component_joints:
            positions = self._generate_positions(comp, t)
            raw_command[comp] = {
                "pos": positions.tolist(),
                "vel": (np.random.randn(len(positions)) * 0.5).tolist(),
            }

        # Generate filtered command (simulate filtering effect)
        filtered_command = {}
        for comp, data in raw_command.items():
            # Apply simulated filtering (smoothing)
            filtered_pos = np.array(data["pos"])
            # Default damping
            if self.frame_count > 0:
                filtered_pos = filtered_pos * 0.9

            # Make head more different: stronger low-pass with memory to add lag
            if comp == "head":
                prev = self._prev_filtered.get(comp, filtered_pos)
                # Heavier smoothing -> more difference from raw
                filtered_pos = 0.6 * prev + 0.4 * filtered_pos
                self._prev_filtered[comp] = filtered_pos.copy()

            filtered_command[comp] = {
                "pos": filtered_pos.tolist(),
                "vel": (np.array(data["vel"]) * 0.8).tolist(),  # Reduced velocity
            }

        # Generate robot state (actual positions)
        # Simulate some tracking error
        robot_state = {}
        for comp in self.component_joints:
            if comp in filtered_command:
                # Robot lags behind command slightly
                command_pos = np.array(filtered_command[comp]["pos"])
                if comp == "head":
                    # Larger, structured difference: add temporal lag + bias + more noise
                    prev_robot = self._prev_robot.get(comp, command_pos)
                    # IIR to induce lag relative to command
                    lagged = 0.7 * prev_robot + 0.3 * command_pos
                    # Small periodic bias per joint for visibility
                    n = len(command_pos)
                    phase = np.linspace(0.0, np.pi / 2.0, n, endpoint=True)
                    bias = 0.05 * np.sin(2 * np.pi * 0.5 * t + phase)
                    noise = np.random.randn(n) * 0.03
                    robot_pos = lagged + bias + noise
                    self._prev_robot[comp] = robot_pos.copy()
                else:
                    tracking_error = np.random.randn(len(command_pos)) * 0.02
                    robot_pos = command_pos + tracking_error
            else:
                # For components not in command (like head)
                robot_pos = self._generate_positions(comp, t)

            robot_state[comp] = robot_pos.tolist()

        # Build complete telemetry message with proper timestamp
        telemetry_msg = {
            "timestamp_ns": time.time_ns(),  # Nanosecond timestamp for precision
            "timestamp": current_time,  # Seconds timestamp for compatibility
            "components": filtered_command,  # Filtered commands
            "robot_state": robot_state,  # Actual robot positions
            "raw_command": raw_command,  # Raw commands before filtering
        }

        return telemetry_msg

    def run(self):
        """Main simulation loop."""
        self.initialize()

        logger.info(f"Starting telemetry simulation at {self.publish_rate}Hz")
        logger.info(f"Mode: {self.simulation_mode}, Noise: {self.noise_level}")

        # Rate control
        period = 1.0 / self.publish_rate
        next_publish_time = time.time()

        try:
            while True:
                current_time = time.time()

                if current_time >= next_publish_time:
                    # Generate and publish telemetry
                    telemetry_msg = self._generate_telemetry_message()
                    self.telemetry_publisher.publish(telemetry_msg)

                    self.frame_count += 1

                    if self.frame_count % self.publish_rate == 0:
                        logger.debug(
                            f"Published {self.frame_count} frames, "
                            f"components: {len(telemetry_msg['components'])}"
                        )

                    # Schedule next publish
                    next_publish_time += period

                    # Handle drift - if we're far behind, reset
                    if current_time > next_publish_time + period:
                        next_publish_time = current_time + period

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Telemetry simulator stopped by user")
        finally:
            self.node.shutdown()

def main(
    namespace: str = "",
    telemetry_topic: str = "robot/telemetry",
    publish_rate: int = 50,
    simulation_mode: str = "sine",
    noise_level: float = 0.01,
    debug: bool = False,
):
    """Run telemetry simulator.

    Args:
        namespace: Optional namespace prefix.
        telemetry_topic: Topic to publish telemetry data.
        publish_rate: Publishing frequency in Hz.
        simulation_mode: Simulation pattern - 'sine', 'random', or 'static'.
        noise_level: Amount of noise to add (0.0 to 1.0).
        debug: Enable debug logging.
    """
    # Setup logging
    logger = setup_logging(debug)

    logger.info(
        f"Starting Telemetry Simulator"
        f"{f' (namespace: {namespace})' if namespace else ''}"
    )

    # Validate simulation mode
    valid_modes = ["sine", "random", "static"]
    if simulation_mode not in valid_modes:
        logger.error(f"Invalid simulation mode. Choose from: {valid_modes}")
        return 1

    # Create and run simulator
    simulator = TelemetrySimulator(
        namespace=namespace,
        telemetry_topic=telemetry_topic,
        publish_rate=publish_rate,
        simulation_mode=simulation_mode,
        noise_level=noise_level,
    )

    try:
        simulator.run()
    except Exception as e:
        logger.error(f"Error running telemetry simulator: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys

    sys.exit(tyro.cli(main))
