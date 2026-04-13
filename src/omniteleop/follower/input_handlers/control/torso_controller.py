"""Abstract torso controller for robot waist/lift mechanism."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from omniteleop.follower.input_handlers.control.commands import TorsoCommand

class AbstractTorsoController(ABC):
    """Abstract base class for torso controllers.

    Defines the interface for controlling robot torso movement
    from different input devices (JoyCon, VR, etc.).
    """

    def __init__(self, sensitivity: float = 0.015, deadzone: float = 0.1):
        """Initialize torso controller.

        Args:
            sensitivity: Movement sensitivity in meters per frame
            deadzone: Input deadzone threshold
        """
        self.sensitivity = sensitivity
        self.deadzone = deadzone

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> TorsoCommand:
        """Process input data for torso control.

        Args:
            input_data: Raw input data from the device

        Returns:
            TorsoCommand with movement deltas
        """
        pass

    @abstractmethod
    def is_active(self, input_data: Dict[str, Any]) -> bool:
        """Check if torso control is active.

        Args:
            input_data: Raw input data from the device

        Returns:
            True if torso control inputs are active
        """
        pass
