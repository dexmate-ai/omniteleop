"""Shared command dataclasses for all input control systems."""

from dataclasses import dataclass
from typing import Dict

@dataclass
class BaseCommand:
    """Mobile base velocity command."""

    vx: float = 0.0  # Forward/backward velocity in m/s
    vy: float = 0.0  # Left/right velocity in m/s
    wz: float = 0.0  # Angular velocity in rad/s
    active: bool = False

@dataclass
class TorsoCommand:
    """Torso movement command."""

    delta_x: float = 0.0  # Forward/backward delta in meters
    delta_z: float = 0.0  # Up/down delta in meters (vertical)
    active: bool = False

@dataclass
class HandCommand:
    """Hand/gripper command."""

    left_positions: Dict[str, float] = None  # Joint positions/deltas for left hand
    right_positions: Dict[str, float] = None  # Joint positions/deltas for right hand
    left_mode: str = "absolute"  # "absolute" or "relative"
    right_mode: str = "absolute"  # "absolute" or "relative"
    active: bool = False

    def __post_init__(self):
        if self.left_positions is None:
            self.left_positions = {}
        if self.right_positions is None:
            self.right_positions = {}

@dataclass
class HeadCommand:
    """Head movement command."""

    delta_j1: float = 0.0  # Delta for head_j1 (pitch)
    delta_j2: float = 0.0  # Delta for head_j2 (yaw)
    delta_j3: float = 0.0  # Delta for head_j3 (roll)
    active: bool = False

@dataclass
class RobotCommands:
    """Combined commands for all robot components."""

    base: BaseCommand = None
    torso: TorsoCommand = None
    hands: HandCommand = None
    head: HeadCommand = None
    priority: str = "none"  # Which control mode has priority
    estop: bool = False
    exit_requested: bool = False  # Signal to exit all programs

    def __post_init__(self):
        if self.base is None:
            self.base = BaseCommand()
        if self.torso is None:
            self.torso = TorsoCommand()
        if self.hands is None:
            self.hands = HandCommand()
        if self.head is None:
            self.head = HeadCommand()
