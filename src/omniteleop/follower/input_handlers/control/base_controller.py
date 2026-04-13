"""Abstract base controller for mobile robot base."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from omniteleop.follower.input_handlers.control.commands import BaseCommand

class AbstractBaseController(ABC):
    """Abstract base class for mobile base controllers.

    Defines the interface for controlling robot mobile base movement
    from different input devices (JoyCon, VR, etc.).
    """

    def __init__(
        self,
        linear_velocity: float = 0.5,
        angular_velocity: float = 1.0,
        deadzone: float = 0.1,
    ):
        """Initialize base controller.

        Args:
            linear_velocity: Maximum linear velocity in m/s
            angular_velocity: Maximum angular velocity in rad/s
            deadzone: Input deadzone threshold
        """
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.deadzone = deadzone

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> BaseCommand:
        """Process input data for base control.

        Args:
            input_data: Raw input data from the device

        Returns:
            BaseCommand with velocity commands
        """
        pass

    @abstractmethod
    def is_active(self, input_data: Dict[str, Any]) -> bool:
        """Check if base control is active.

        Args:
            input_data: Raw input data from the device

        Returns:
            True if base control inputs are active
        """
        pass
