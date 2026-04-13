"""Base controller for mobile robot platform."""

from typing import Dict, Any
from omniteleop.follower.input_handlers.control.commands import BaseCommand
from omniteleop.follower.input_handlers.control.base_controller import (
    AbstractBaseController,
)

class BaseController(AbstractBaseController):
    """Interprets JoyCon inputs for mobile base control.

    Control schemes (when activated via +/- buttons):
    - Left stick: Translational velocity (vx, vy)
    - Right buttons: Discrete velocity control
        - X: Forward (vx > 0)
        - B: Backward (vx < 0)
        - Y: Turn left (wz > 0)
        - A: Turn right (wz < 0)
    """

    def __init__(
        self,
        linear_velocity: float = 0.5,
        angular_velocity: float = 1.0,
        deadzone: float = 0.1,
    ):
        """Initialize base controller.

        Args:
            linear_velocity: Max linear velocity in m/s
            angular_velocity: Max angular velocity in rad/s
        """
        super().__init__(linear_velocity, angular_velocity, deadzone)

    def process(self, joycon_data: Dict[str, Any]) -> BaseCommand:
        """Process JoyCon data for base control.

        Args:
            joycon_data: Raw JoyCon data with 'left' and 'right' controller info

        Returns:
            BaseCommand with velocity commands
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        # Extract buttons
        right_buttons = right_data.get("buttons", {})

        left_stick = left_data.get("stick", {})

        # Initialize command
        command = BaseCommand()

        # Left stick: Analog translational control
        stick_x = left_stick.get("x", 0.0)
        stick_y = left_stick.get("y", 0.0)

        # Apply deadzone and remap to [0, 1]
        if abs(stick_y) > self.deadzone:
            # Remap: (deadzone, 1.0) -> (0.0, 1.0)
            sign = 1.0 if stick_y > 0 else -1.0
            normalized = (abs(stick_y) - self.deadzone) / (1.0 - self.deadzone)
            command.vx = sign * normalized * self.linear_velocity  # Forward/backward

        if abs(stick_x) > self.deadzone:
            # Remap: (deadzone, 1.0) -> (0.0, 1.0)
            sign = 1.0 if stick_x > 0 else -1.0
            normalized = (abs(stick_x) - self.deadzone) / (1.0 - self.deadzone)
            command.vy = (
                -sign * normalized * self.linear_velocity
            )  # Left/right (negated for intuitive control)

        # Right buttons: Discrete control
        # Forward/backward
        if right_buttons.get("x", False):  # X button
            command.vx = self.linear_velocity  # Forward
        elif right_buttons.get("b", False):  # B button
            command.vx = -self.linear_velocity  # Backward

        # Rotation
        if right_buttons.get("y", False):  # Y button
            command.wz = self.angular_velocity  # Turn left (positive)
        elif right_buttons.get("a", False):  # A button
            command.wz = -self.angular_velocity  # Turn right (negative)

        command.active = True

        return command

    def is_active(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if base control is active without processing.

        Note: With the new toggle-based system, this method is called
        only when base control is already activated in the main controller.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if base control inputs are active (always True when called)
        """
        # In the new toggle-based system, this method is called only when
        # base control is already activated, so we always return True
        return True
