"""Head component processor."""

from typing import Dict, Any, List
import numpy as np

from loguru import logger
from dexmotion.utils import robot_utils

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand, CommandMode

class HeadProcessor(BaseComponentProcessor):
    """Processes head position commands.

    Supports two modes:
    - 'manual': Uses manual control from input (e.g., JoyCon sticks)
    - 'fixed': Always sends fixed position from configuration

    Head positions are always clipped to joint limits for safety.
    """

    def __init__(self, side, config, motion_manager, robot_info, teleop_mode):
        """Initialize head processor.

        Args:
            side: Not used (always None for head)
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        super().__init__(None, config, motion_manager, robot_info, teleop_mode)

        # Get head configuration from input handler config
        handler_config = self.config.get("input_handlers", {}).get(teleop_mode, {})
        head_config = handler_config.get("head", {})

        self.head_mode = head_config.get("mode", "manual")  # 'manual' or 'fixed'
        self.head_fixed_position = head_config.get("fixed_position", [0.5, 0.0, 0.0])
        self.head_sensitivity = head_config.get("sensitivity", 0.1)

        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints("head")

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "head"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process head command based on configuration mode.

        In 'fixed' mode, always sends the fixed position from config.
        In 'manual' mode, processes input deltas if provided.

        Args:
            input_data: Head input data (can be None in fixed mode)
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        try:
            head_positions = None

            if self.head_mode == "fixed":
                # Fixed mode: Always send fixed position regardless of input
                head_positions = np.array(self.head_fixed_position)

            elif self.head_mode == "manual":
                # Manual mode: Only process if head input is present
                if input_data is not None:
                    deltas = np.array(input_data.get("pos", [0.0, 0.0, 0.0]))

                    # Get current head positions
                    current_head_pos = self.motion_manager.head.get_joint_pos()

                    # Apply deltas to get target positions
                    head_positions = current_head_pos + deltas

            # Clip head positions to joint limits and apply
            if head_positions is not None:
                clipped_head_pos = self._clip_to_limits(head_positions)

                # Apply to motion manager
                self.motion_manager.head.set_joint_pos(clipped_head_pos)

                # Set head command output
                command.output_components["head"] = {
                    "pos": clipped_head_pos.tolist()
                    if isinstance(clipped_head_pos, np.ndarray)
                    else clipped_head_pos,
                    "mode": CommandMode.ABSOLUTE.value,
                }
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to compute head command: {e}")
            return False

    def _clip_to_limits(self, head_positions: np.ndarray) -> List[float]:
        """Clip head positions to joint limits.

        Args:
            head_positions: Target head positions

        Returns:
            Clipped head positions
        """
        current_positions = self.motion_manager.get_joint_pos_dict()

        # Update with target head positions
        for i, pos in enumerate(head_positions):
            current_positions[f"head_j{i + 1}"] = pos

        # Clip to joint limits
        clipped_positions = robot_utils.clip_joint_positions_to_limits(
            self.motion_manager.pin_robot, current_positions
        )

        # Extract clipped head positions
        return [clipped_positions[f"head_j{i + 1}"] for i in range(3)]

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync head to robot state.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        head_pos = robot_joints.get("head", [])
        if head_pos:
            self.motion_manager.head.set_joint_pos(head_pos)
