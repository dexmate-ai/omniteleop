"""Abstract hand controller for end-effectors."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from omniteleop.follower.input_handlers.control.commands import HandCommand

class AbstractHandController(ABC):
    """Abstract base class for hand/end-effector controllers.

    Defines the interface for controlling robot hands/grippers
    from different input devices (JoyCon, VR, etc.).
    """

    def __init__(
        self,
        left_config: Dict[str, Any] = None,
        right_config: Dict[str, Any] = None,
        stick_deadzone: float = 0.1,
    ):
        """Initialize hand controller.

        Args:
            left_config: Configuration for left end-effector
            right_config: Configuration for right end-effector
        """
        self.left_config = left_config or {"type": "none"}
        self.right_config = right_config or {"type": "none"}

        # Add stick_deadzone to configs if not already present
        if "stick_deadzone" not in self.left_config:
            self.left_config["stick_deadzone"] = stick_deadzone
        if "stick_deadzone" not in self.right_config:
            self.right_config["stick_deadzone"] = stick_deadzone

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> HandCommand:
        """Process input data for hand control.

        Args:
            input_data: Raw input data from the device

        Returns:
            HandCommand with joint positions or deltas
        """
        pass

    @abstractmethod
    def is_active(self, input_data: Dict[str, Any]) -> bool:
        """Check if hand control is active.

        Args:
            input_data: Raw input data from the device

        Returns:
            True if hand control should be active
        """
        pass

    @abstractmethod
    def _has_modifiers(self, input_data: Dict[str, Any]) -> bool:
        """Check if modifier inputs are pressed (device-specific).

        Args:
            input_data: Raw input data from the device

        Returns:
            True if modifiers are pressed (which typically disable hand control)
        """
        pass
