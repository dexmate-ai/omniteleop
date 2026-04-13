"""Hand component processor."""

import os
from typing import Dict, Any

from dexmotion.utils import robot_utils

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand, CommandMode

class HandProcessor(BaseComponentProcessor):
    """Processes hand commands for left or right hand."""

    def __init__(self, side: str, config, motion_manager, robot_info, teleop_mode):
        """Initialize hand processor.

        Args:
            side: Hand side ('left' or 'right')
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        assert side in ["left", "right"], "Hand side must be 'left' or 'right'"
        super().__init__(side, config, motion_manager, robot_info, teleop_mode)
        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints(f"{side}_hand")
        # Determine hand type from ROBOT_CONFIG env var
        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            self.hand_type = "gripper"
        elif "f5d6" in robot_config:
            self.hand_type = "f5d6"
        else:
            self.hand_type = None  # No hands
        self.joint_pos_limits = self.robot_info.get_joint_pos_limits(self.joint_names)

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "hand"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process hand command based on hand type.

        Routes to hand processing based on configuration.

        Args:
            input_data: Hand input data with 'pos' and 'mode' fields
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        return self._process_hand_command(input_data, command)

    def _process_hand_command(
        self, input_data: Dict[str, Any], command: RobotCommand
    ) -> bool:
        """Process hand_f5d6 joint position commands.

        Updates motion manager state and applies joint limits.

        Args:
            input_data: Hand input data
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        mode = input_data.get("mode", CommandMode.ABSOLUTE)
        positions = input_data.get("pos", [])

        if not positions:
            return False

        # Get current positions if needed for relative mode
        if mode == CommandMode.RELATIVE:
            current_positions = self.motion_manager.get_joint_pos_dict()
            current_hand_pos = [
                current_positions.get(name, 0) for name in self.joint_names
            ]
            positions = [
                curr + delta for curr, delta in zip(current_hand_pos, positions)
            ]

        # Update motion manager state with clipping
        updated_positions = self.motion_manager.get_joint_pos_dict()
        for i, pos in enumerate(positions):
            if i < len(self.joint_names):
                updated_positions[self.joint_names[i]] = pos

        # Clip to joint limits
        updated_positions = robot_utils.clip_joint_positions_to_limits(
            self.motion_manager.pin_robot, updated_positions
        )

        if updated_positions:
            self.motion_manager.set_joint_pos(updated_positions)
            command.output_components[self.component_name] = {
                "pos": positions,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

        return False

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync hand to robot state.

        Syncs motion manager.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        hand_pos = robot_joints.get(self.component_name, [])
        if not hand_pos:
            return
        motion_manager_joints = {}
        for i, pos in enumerate(hand_pos):
            if i < len(self.joint_names):
                motion_manager_joints[self.joint_names[i]] = pos
        if motion_manager_joints:
            self.motion_manager.set_joint_pos(motion_manager_joints)
