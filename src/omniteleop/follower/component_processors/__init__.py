"""Component processors for modular command processing.

Each component processor handles a specific robot component (arms, hands, etc.)
with a unified interface for processing commands and syncing state.
"""

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.component_processors.chassis_processor import ChassisProcessor
from omniteleop.follower.component_processors.head_processor import HeadProcessor
from omniteleop.follower.component_processors.torso_processor import TorsoProcessor
from omniteleop.follower.component_processors.hand_processor import HandProcessor
from omniteleop.follower.component_processors.arm_processor import ArmProcessor
from omniteleop.follower.component_processors.safety_validator import SafetyValidator

__all__ = [
    "BaseComponentProcessor",
    "ChassisProcessor",
    "HeadProcessor",
    "TorsoProcessor",
    "HandProcessor",
    "ArmProcessor",
    "SafetyValidator",
]
