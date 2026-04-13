"""Input handler for exoskeleton arms with JoyCon controllers."""

import threading
import time
from typing import Dict, Any, Optional

from loguru import logger
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from .base_handler import BaseInputHandler, RobotCommand, CommandMode, ArmCommandType
from omniteleop.common import get_config
from omniteleop.follower.input_handlers.control.joycon.controller import (
    JoyConController,
)

class ExoJoyconHandler(BaseInputHandler):
    """Handler for exoskeleton arm inputs combined with JoyCon controllers.

    This handler processes:
    - Joint positions from leader arms (exoskeleton)
    - JoyCon controller inputs for additional controls
    - Combines both into unified robot commands
    """

    def __init__(self, config: Dict[str, Any], namespace: str = ""):
        """Initialize exoskeleton + JoyCon handler.

        Args:
            config: Configuration dictionary
            namespace: Optional namespace for topics
        """
        # Initialize base handler
        BaseInputHandler.__init__(self, config, namespace)

        # Initialize Node for communication
        self.node = Node(name="exo_joycon_handler", namespace=namespace)

        # JoyCon controller for processing joycon inputs
        # Build config for JoyConController from component configs
        joycon_controller_config = {
            "base": config.get("base", {}),
            "torso": config.get("torso", {}),
            "hands": config.get("hands", {}),
            "gripper": config.get("gripper", {}),
            # Add any joycon-specific settings (like deadzones)
            **config.get("joycon", {}),
        }
        self.joycon_controller = JoyConController(joycon_controller_config)
        self.config = config

        # Latest data (protected by locks)
        self.latest_leader_arm_joints: Optional[Dict[str, Any]] = None
        self.latest_joycon: Optional[Dict[str, Any]] = None
        self._leader_arm_joints_lock = threading.RLock()
        self._joycon_lock = threading.RLock()

        # Subscribers (will be initialized in setup_subscribers)
        self.joints_sub = None
        self.joycon_sub = None

        # Publisher for recorder control
        self.recorder_pub = None

        # Control state
        self._motion_control_started = False
        self.exit_after_publish = False

        logger.info("ExoJoyconHandler initialized")

    def initialize(self) -> bool:
        """Initialize communication and connections.

        Returns:
            True if successful, False otherwise
        """
        if self.initialized:
            return True

        # Set up subscribers
        self.setup_subscribers()

        self.initialized = True
        self.running = True
        logger.info("ExoJoyconHandler initialization complete")
        return True

    def setup_subscribers(self) -> None:
        """Set up Zenoh subscribers for exo joints and JoyCon data."""
        # Subscribe to exoskeleton joints
        joints_topic = self.config.get("topics", {}).get("exo_joints", "exo/joints")
        self.joints_sub = self.node.create_subscriber(
            joints_topic,
            self._on_leader_arm_joints_received,
            decoder=DictDataCodec.decode,
        )

        # Subscribe to JoyCon inputs
        joycon_topic = self.config.get("topics", {}).get("exo_joycon", "exo/joycon")
        self.joycon_sub = self.node.create_subscriber(
            joycon_topic, self._on_joycon_received, decoder=DictDataCodec.decode
        )

        # Create publisher for recorder control
        recorder_topic = get_config().get_topic("recorder_control")
        self.recorder_pub = self.node.create_publisher(
            recorder_topic, encoder=DictDataCodec.encode
        )

        resolved_joints = self.node.resolve_topic(joints_topic)
        resolved_joycon = self.node.resolve_topic(joycon_topic)
        resolved_recorder = self.node.resolve_topic(recorder_topic)

        logger.info(f"Subscribed to {resolved_joints} and {resolved_joycon}")
        logger.info(f"Publishing recorder commands to {resolved_recorder}")

    def _on_leader_arm_joints_received(self, joints_data):
        """Handle incoming leader arm joint positions.

        Args:
            joints_data: Deserialized joint data from exoskeleton
        """
        with self._leader_arm_joints_lock:
            self.latest_leader_arm_joints = joints_data

    def _on_joycon_received(self, joycon_data):
        """Handle incoming JoyCon data.

        Args:
            joycon_data: Deserialized JoyCon data
        """
        with self._joycon_lock:
            self.latest_joycon = joycon_data

    def process_inputs(self) -> Optional[RobotCommand]:
        """Process exoskeleton and JoyCon inputs into robot commands.

        Returns:
            RobotCommand if inputs available, None otherwise
        """
        # Get latest data safely with locks
        with self._leader_arm_joints_lock:
            joints = (
                self.latest_leader_arm_joints.copy()
                if self.latest_leader_arm_joints
                else None
            )

        with self._joycon_lock:
            joycon = self.latest_joycon.copy() if self.latest_joycon else None

        if not joints and not joycon:
            return None

        # Initialize new command
        command = RobotCommand(timestamp_ns=time.time_ns())

        # Process joint positions from exoskeleton
        if joints:
            self._process_arm_joints(joints, command)

        # Process JoyCon inputs for additional controls
        # This will set the emergency stop state based on JoyCon controller
        if joycon:
            self._process_joycon_commands(joycon, command)

        # If no JoyCon data, default to emergency stop for safety
        elif not self._motion_control_started:
            command.safety_flags.emergency_stop = True

        # Update latest command
        self.update_command(command)

        return command

    def _process_arm_joints(self, joints: Dict[str, Any], command: RobotCommand):
        """Process arm joint positions and velocities.

        Args:
            joints: Joint position data (ExoJointData as dict)
            command: Command to update
        """
        # Handle array-based format with velocities
        left_arm_pos = joints.get("left_arm_pos", [])
        left_arm_vel = joints.get("left_arm_vel", [])
        right_arm_pos = joints.get("right_arm_pos", [])
        right_arm_vel = joints.get("right_arm_vel", [])

        # Process left arm positions and velocities using input_components
        # These will be processed by command processor
        if left_arm_pos:
            command.input_components["left_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": left_arm_pos,
            }
            if left_arm_vel:
                command.input_components["left_arm"]["vel"] = left_arm_vel

        # Process right arm positions and velocities
        if right_arm_pos:
            command.input_components["right_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": right_arm_pos,
            }
            if right_arm_vel:
                command.input_components["right_arm"]["vel"] = right_arm_vel

    def _process_joycon_commands(self, joycon: Dict[str, Any], command: RobotCommand):
        """Process JoyCon inputs into robot commands.

        Args:
            joycon: JoyCon input data
            command: Command to update
        """
        # Process through centralized controller
        robot_commands = self.joycon_controller.process(joycon)
        # Always update emergency stop state from JoyCon controller
        command.safety_flags.emergency_stop = robot_commands.estop

        # Check for exit request
        if robot_commands.exit_requested:
            command.safety_flags.exit_requested = True
            logger.critical("Exit requested via JoyCon")
            self.exit_after_publish = True

        # Check if motion control should start (when estop is released)
        if not robot_commands.estop and not self._motion_control_started:
            self._motion_control_started = True
            logger.info("Motion control started - estop released")

        # Check for recording commands and publish them
        recording_command = self.joycon_controller.get_recording_command()
        if recording_command and self.recorder_pub:
            recording_msg = {
                "command": recording_command,
                "metadata": {
                    "source": "joycon",
                    "timestamp": time.time_ns(),
                },
            }
            try:
                self.recorder_pub.publish(recording_msg)
                logger.info(
                    f"Published recording command: {recording_command} to recorder/control"
                )
                logger.debug(f"Recording message content: {recording_msg}")
            except Exception as e:
                logger.error(f"Failed to publish recording command: {e}")
        elif recording_command:
            logger.warning(
                f"Recording command '{recording_command}' generated but no publisher available"
            )

        # Handle based on priority
        if robot_commands.priority == "base":
            self._handle_base_command(robot_commands.base, command)
        elif robot_commands.priority == "head":
            # Head command using unified structure with relative mode
            if robot_commands.head.active:
                command.input_components["head"] = {
                    "mode": CommandMode.RELATIVE,
                    "pos": [
                        robot_commands.head.delta_j1,
                        robot_commands.head.delta_j2,
                        robot_commands.head.delta_j3,
                    ],
                }
        elif robot_commands.priority == "torso":
            # Torso command using unified structure with relative mode
            if robot_commands.torso.active:
                command.input_components["torso"] = {
                    "mode": CommandMode.RELATIVE,
                    "pos": [
                        robot_commands.torso.delta_x,
                        0.0,
                        robot_commands.torso.delta_z,
                    ],
                }
        elif robot_commands.priority == "hands":
            # Hand commands using unified structure
            if robot_commands.hands.active:
                # Process left hand
                if robot_commands.hands.left_positions:
                    command.input_components["left_hand"] = {
                        "mode": CommandMode.ABSOLUTE
                        if robot_commands.hands.left_mode == "absolute"
                        else CommandMode.RELATIVE,
                        "pos": robot_commands.hands.left_positions,
                    }

                # Process right hand
                if robot_commands.hands.right_positions:
                    command.input_components["right_hand"] = {
                        "mode": CommandMode.ABSOLUTE
                        if robot_commands.hands.right_mode == "absolute"
                        else CommandMode.RELATIVE,
                        "pos": robot_commands.hands.right_positions,
                    }

    def _handle_base_command(self, base_cmd, command: RobotCommand):
        """Handle mobile base velocity commands.

        Args:
            base_cmd: BaseCommand with velocity commands
            command: Command to update
        """
        if not base_cmd.active:
            return

        # Base uses velocity control directly as input component
        # Command processor will pass it through to output
        command.input_components["chassis"] = {
            "vx": base_cmd.vx,  # Forward/backward velocity
            "vy": base_cmd.vy,  # Left/right velocity
            "wz": base_cmd.wz,  # Angular velocity
        }

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        self.running = False

        # Node handles communication cleanup
        self.node.shutdown()

        logger.info("ExoJoyconHandler cleaned up")

    def needs_motion_manager(self) -> bool:
        """Check if this handler needs motion manager for processing.

        Returns:
            True since torso IK and hand processing need motion manager
        """
        return True

    def get_exit_requested(self) -> bool:
        """Check if exit has been requested.

        Returns:
            True if exit requested, False otherwise
        """
        return self.exit_after_publish
