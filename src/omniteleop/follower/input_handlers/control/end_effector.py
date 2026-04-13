"""Abstract end-effector controller for hands/grippers."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from loguru import logger

class AbstractEndEffectorController(ABC):
    """Abstract base class for end-effector controllers.

    Defines the interface for controlling different types of end-effectors
    (hands, grippers, etc.) from various input devices.
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize end-effector controller.

        Args:
            side: "left" or "right"
            config: End-effector configuration from robot config
        """
        self.side = side
        self.config = config
        self.joint_positions: Dict[str, float] = {}

    @abstractmethod
    def process_input(self, input_data: Any) -> Dict[str, float]:
        """Process input and return joint positions.

        Args:
            input_data: Parsed input data (device-specific format)

        Returns:
            Dictionary of joint positions or tuple of (positions, mode)
        """
        pass

    def get_positions(self) -> Dict[str, float]:
        """Get current joint positions.

        Returns:
            Dictionary of joint positions
        """
        return self.joint_positions.copy()

    def reset(self):
        """Reset end-effector to default position."""
        for joint in self.joint_positions:
            self.joint_positions[joint] = 0.0
        logger.info(f"{self.side} end-effector reset to default position")
