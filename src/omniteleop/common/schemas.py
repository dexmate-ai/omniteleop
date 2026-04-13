"""Simple data schemas for communication between components.

These dataclasses define the structure of messages passed via Zenoh.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ExoJointData:
    """Joint positions and velocities from the exoskeleton."""

    timestamp_ns: int
    left_arm_pos: List[float] = field(
        default_factory=list
    )  # Left arm joint positions in radians
    left_arm_vel: List[float] = field(
        default_factory=list
    )  # Left arm joint velocities in rad/s
    right_arm_pos: List[float] = field(
        default_factory=list
    )  # Right arm joint positions in radians
    right_arm_vel: List[float] = field(
        default_factory=list
    )  # Right arm joint velocities in rad/s

@dataclass
class JoyConData:
    """JoyCon controller inputs."""

    timestamp_ns: int
    left: Dict[str, Any]  # Left controller data
    right: Dict[str, Any]  # Right controller data

@dataclass
class SafeJointCommand:
    """Safety-validated commands for the robot."""

    timestamp_ns: int
    components: Dict[str, Dict[str, List[float]]]  # Component commands
    safety_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class PoseData:
    """Pose data from the VR headset."""

    timestamp_ns: int
    left: Dict[str, Any]
    right: Dict[str, Any]
