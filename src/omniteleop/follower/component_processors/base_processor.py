"""Base class for component processors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from dexmotion.motion_manager import MotionManager
from omniteleop.follower.input_handlers.base_handler import RobotCommand
from dexbot_utils import RobotInfo

class BaseComponentProcessor(ABC):
    """Abstract base class for component processors.

    Each component processor handles processing for a specific robot component
    (e.g., arm, hand, torso). Processors can be instantiated per-side for
    bilateral components (left_arm, right_arm) or as singletons for unique
    components (torso, head).

    Attributes:
        side: Component side ('left', 'right', or None for single components)
        config: Full configuration dictionary
        motion_manager: Shared motion manager instance
        teleop_mode: Teleoperation mode (e.g., 'exo_joycon', 'vr')
        component_name: Full component name (e.g., 'left_arm', 'torso')
    """

    def __init__(
        self,
        side: Optional[str],
        config: Dict[str, Any],
        motion_manager: MotionManager,
        robot_info: RobotInfo,
        teleop_mode: str,
    ):
        """Initialize component processor.

        Args:
            side: Component side ('left', 'right', or None for single components)
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_type: Robot type identifier
            teleop_mode: Teleoperation mode identifier
        """
        self.side = side
        self.config = config
        self.motion_manager = motion_manager
        self.robot_info = robot_info
        self.teleop_mode = teleop_mode

        # Construct full component name
        if side:
            self.component_name = f"{side}_{self.component_type}"
        else:
            self.component_name = self.component_type

    @property
    @abstractmethod
    def component_type(self) -> str:
        """Component type identifier.

        Returns:
            Component type string: 'arm', 'hand', 'torso', 'head', 'chassis'
        """
        pass

    @abstractmethod
    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process component input and update command output.

        Takes input data from command.input_components, processes it according
        to component-specific logic, and writes results to command.output_components.

        Args:
            input_data: Component-specific input data from input handler
            command: RobotCommand to update (writes to output_components)

        Returns:
            True if processing succeeded, False otherwise
        """
        pass

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Synchronize motion manager to actual robot state.

        Called during startup to sync motion manager's internal state
        with the real robot's current configuration.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this processor is enabled.

        Returns:
            True if processor is enabled (default), False otherwise
        """
        return True
