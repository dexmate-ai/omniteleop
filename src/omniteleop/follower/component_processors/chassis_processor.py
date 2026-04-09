"""Chassis/base component processor."""

from typing import Dict, Any

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand

class ChassisProcessor(BaseComponentProcessor):
    """Processes chassis velocity commands.

    The chassis processor is a simple pass-through that forwards velocity
    commands (vx, vy, wz) directly to the output without modification.
    Only active for robots with mobile base support.
    """

    def __init__(self, side, config, motion_manager, robot_info, teleop_mode):
        """Initialize chassis processor.

        Args:
            side: Not used (always None for chassis)
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        super().__init__(None, config, motion_manager, robot_info, teleop_mode)

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "chassis"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process chassis velocity command.

        Chassis commands are pass-through - velocity commands (vx, vy, wz)
        are forwarded directly to output without modification.

        Args:
            input_data: Chassis velocity data (vx, vy, wz)
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        if not input_data:
            return False

        # Pass through velocity commands directly
        command.output_components["chassis"] = input_data
        return True

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync chassis to robot state.

        Chassis has no persistent state to sync.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        pass  # Chassis has no state to sync
