"""Abstract base controller for all input devices."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from omniteleop.follower.input_handlers.control.commands import RobotCommands

class AbstractController(ABC):
    """Abstract base class for all input device controllers.

    Defines the common interface that all controllers (JoyCon, VR, etc.)
    must implement for consistent teleoperation control.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize controller.

        Args:
            config: Configuration dictionary with device-specific settings
        """
        self.config = config or {}
        self.estop_active = True
        self.exit_requested = False
        self.stats = {
            "total_updates": 0,
            "estop_toggles": 0,
        }

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> RobotCommands:
        """Process input data and return robot commands.

        Args:
            input_data: Raw input data from the device

        Returns:
            RobotCommands with appropriate component commands
        """
        pass

    @abstractmethod
    def get_control_mode(self, input_data: Dict[str, Any]) -> str:
        """Get current control mode without processing.

        Args:
            input_data: Raw input data from the device

        Returns:
            String indicating active control mode
        """
        pass

    def is_estop_active(self) -> bool:
        """Check if emergency stop is active.

        Returns:
            True if emergency stop is active
        """
        return self.estop_active

    def is_exit_requested(self) -> bool:
        """Check if exit has been requested.

        Returns:
            True if exit was requested
        """
        return self.exit_requested

    def reset_estop(self) -> None:
        """Reset emergency stop state."""
        self.estop_active = False
        self.exit_requested = False

    def get_stats(self) -> Dict[str, int]:
        """Get controller statistics.

        Returns:
            Dictionary with activation counts
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset controller statistics."""
        for key in self.stats:
            self.stats[key] = 0
