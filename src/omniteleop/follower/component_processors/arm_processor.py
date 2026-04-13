"""Arm component processor."""

from typing import Dict, Any, List
import numpy as np

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import (
    RobotCommand,
    CommandMode,
    ArmCommandType,
)

class ArmProcessor(BaseComponentProcessor):
    """Processes arm commands for left or right arm.

    Supports two command types:
    - Joint control: Direct joint position/velocity commands
    - EE pose control: End-effector pose commands (IK handled externally)

    Features:
    - Torso pitch compensation (optional)
    - Safety limiting of joint angle steps
    - Absolute and relative positioning
    """

    def __init__(self, side: str, config, motion_manager, robot_info, teleop_mode):
        """Initialize arm processor.

        Args:
            side: Arm side ('left' or 'right')
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        assert side in ["left", "right"], "Arm side must be 'left' or 'right'"
        super().__init__(side, config, motion_manager, robot_info, teleop_mode)

        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints(f"{side}_arm")

        # Get arm-specific motion manager component
        self.arm = (
            motion_manager.left_arm if side == "left" else motion_manager.right_arm
        )

        # Torso pitch compensation configuration
        handler_config = self.config.get("input_handlers", {}).get(teleop_mode, {})
        leader_arms_config = handler_config.get("leader_arms", {})
        self.compensate_torso_pitch = (
            self.robot_info.has_torso
            and leader_arms_config.get("compensate_torso_pitch", False)
        )

        # Safety limits
        self.max_joint_angle_step = np.deg2rad(10)  # Maximum 10 degrees per step

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "arm"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process arm command.

        Routes to joint or EE pose processing based on command type.
        Note: EE pose commands require both arms and are handled externally
        in the orchestrator's dual-arm processing.

        Args:
            input_data: Arm input data
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        command_type = input_data.get("command_type", ArmCommandType.JOINT)

        if command_type == ArmCommandType.JOINT:
            return self._process_joint_command(input_data, command)
        elif command_type == ArmCommandType.EE_POSE:
            # EE pose is handled externally by dual-arm coordinator
            # Store the data for external processing
            return False

        return False

    def _process_joint_command(
        self, input_data: Dict[str, Any], command: RobotCommand
    ) -> bool:
        """Process joint position command.

        Handles both absolute and relative joint position commands.
        Applies torso pitch compensation if enabled.

        Args:
            input_data: Arm input data with 'pos', 'vel', and 'mode' fields
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        mode = input_data.get("mode", CommandMode.ABSOLUTE)
        positions = input_data.get("pos", [])
        velocities = input_data.get("vel", [])

        if not positions:
            return False

        # Handle relative mode
        if mode == CommandMode.RELATIVE:
            current_positions = self.motion_manager.get_joint_pos_dict()
            current_arm_pos = [
                current_positions.get(name, 0) for name in self.joint_names
            ]
            positions = [
                curr + delta for curr, delta in zip(current_arm_pos, positions)
            ]

        # Apply torso pitch compensation if enabled
        if self.compensate_torso_pitch:
            positions = self._compensate_for_torso_pitch(positions)

        # Build output
        output = {
            "pos": positions,
            "mode": CommandMode.ABSOLUTE.value,
        }
        if velocities:
            output["vel"] = velocities

        command.output_components[self.component_name] = output
        return True

    def _compensate_for_torso_pitch(self, positions: List[float]) -> List[float]:
        """Compensate arm positions for torso pitch.

        Adjusts the first arm joint (shoulder pitch) to counteract
        the torso pitch, preventing the robot from falling over.

        Args:
            positions: Original arm joint positions

        Returns:
            Compensated arm joint positions
        """
        # Only called when self.compensate_torso_pitch is True,
        # which already verifies robot_info.has_torso

        # Get torso pitch
        torso_pos = self.motion_manager.torso.get_joint_pos()
        torso_pitch = torso_pos[0] + torso_pos[2] - torso_pos[1]

        # Apply compensation to shoulder pitch (first joint)
        compensated = positions.copy()
        pitch_adjustment = torso_pitch if self.side == "left" else -torso_pitch
        compensated[0] += pitch_adjustment

        return compensated

    def limit_joint_step(self, target_positions: List[float]) -> np.ndarray:
        """Limit joint position changes for safety.

        Clips the change in joint positions to max_joint_angle_step
        to prevent sudden movements.

        Args:
            target_positions: Target joint positions

        Returns:
            Safety-limited joint positions
        """
        current = self.arm.get_joint_pos()
        target = np.array(target_positions)
        position_diff = target - current

        # Calculate the maximum allowed step for each joint
        step_sizes = np.abs(position_diff)
        alphas = np.minimum(1.0, self.max_joint_angle_step / (step_sizes + 1e-10))
        clipped_position = current + alphas * position_diff

        return clipped_position

    def apply_positions(self, positions: List[float]) -> None:
        """Apply joint positions to motion manager.

        Args:
            positions: Joint positions to apply
        """
        self.arm.set_joint_pos(positions)

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync arm to robot state.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        arm_pos = robot_joints.get(self.component_name, [])
        if arm_pos:
            self.arm.set_joint_pos(arm_pos)
