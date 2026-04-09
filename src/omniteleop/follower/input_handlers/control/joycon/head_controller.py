"""Head controller for manual head positioning."""

from typing import Dict, Any
from loguru import logger
from omniteleop.follower.input_handlers.control.commands import HeadCommand

class HeadController:
    """Interprets JoyCon inputs for manual head control.

    Control scheme:
    - Right stick Y axis: Control head_j1 (pitch)
    - Right stick X axis: Control head_j2 (yaw)
    - Left stick Y axis: Control head_j3 (roll)

    Uses relative control with configurable sensitivity.
    """

    def __init__(self, sensitivity: float = 0.1, deadzone: float = 0.1):
        """Initialize head controller.

        Args:
            sensitivity: Sensitivity for head control (higher = faster movement)
            deadzone: Deadzone for stick inputs
        """
        self.sensitivity = sensitivity
        self.deadzone = deadzone
        logger.info(
            f"Head controller initialized with sensitivity={sensitivity}, deadzone={deadzone}"
        )

    def process(self, joycon_data: Dict[str, Any]) -> HeadCommand:
        """Process JoyCon data for head control.

        Args:
            joycon_data: Raw JoyCon data with 'left' and 'right' controller info

        Returns:
            HeadCommand with movement deltas
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        left_stick = left_data.get("stick", {})
        right_stick = right_data.get("stick", {})

        # Initialize command
        command = HeadCommand()

        # Right stick Y controls head_j1 (pitch) - reversed polarity
        right_stick_y = right_stick.get("y", 0.0)
        if abs(right_stick_y) > self.deadzone:
            command.delta_j1 = -right_stick_y * self.sensitivity

        # Right stick X controls head_j2 (yaw) - reversed polarity
        right_stick_x = right_stick.get("x", 0.0)
        if abs(right_stick_x) > self.deadzone:
            command.delta_j2 = -right_stick_x * self.sensitivity

        # Left stick Y controls head_j3 (roll)
        left_stick_y = left_stick.get("y", 0.0)
        if abs(left_stick_y) > self.deadzone:
            command.delta_j3 = left_stick_y * self.sensitivity

        command.active = True

        return command

    def is_active(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if any head control inputs are active.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if any stick inputs are above deadzone
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        left_stick = left_data.get("stick", {})
        right_stick = right_data.get("stick", {})

        left_stick_y = abs(left_stick.get("y", 0.0))
        right_stick_x = abs(right_stick.get("x", 0.0))
        right_stick_y = abs(right_stick.get("y", 0.0))

        return (
            left_stick_y > self.deadzone
            or right_stick_x > self.deadzone
            or right_stick_y > self.deadzone
        )
