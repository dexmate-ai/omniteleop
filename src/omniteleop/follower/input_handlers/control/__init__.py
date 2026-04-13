"""Unified control system for teleoperation input devices.

Provides abstract base classes and device-specific implementations
for JoyCon, VR, and other input devices.
"""

from .commands import BaseCommand, TorsoCommand, HandCommand, RobotCommands
from .controller import AbstractController
from .base_controller import AbstractBaseController
from .torso_controller import AbstractTorsoController
from .hand_controller import AbstractHandController

def create_controller(device_type: str, config: dict = None) -> AbstractController:
    """Factory function to create appropriate controller.

    Args:
        device_type: Type of input device ("joycon", "vr", etc.)
        config: Configuration dictionary for the device

    Returns:
        Appropriate controller implementation

    Raises:
        ValueError: If device_type is not supported
    """
    config = config or {}

    if device_type == "joycon":
        from .joycon.controller import JoyConController

        return JoyConController(config)

        return VRController(config)
    else:
        raise ValueError(f"Unknown device type: {device_type}")

__all__ = [
    # Commands
    "BaseCommand",
    "TorsoCommand",
    "HandCommand",
    "RobotCommands",
    # Abstract classes
    "AbstractController",
    "AbstractBaseController",
    "AbstractTorsoController",
    "AbstractHandController",
    # Factory
    "create_controller",
]
