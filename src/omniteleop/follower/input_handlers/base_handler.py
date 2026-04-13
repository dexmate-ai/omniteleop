"""Base abstract class for all input handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import threading
import numpy as np

class CommandMode(Enum):
    """Mode for command execution."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"

class ArmCommandType(Enum):
    """Type of arm command."""

    JOINT = "joint"  # Direct joint positions
    EE_POSE = "ee_pose"  # End-effector pose (requires IK)

@dataclass
class SafetyFlags:
    """Safety status flags."""

    collision_detected: bool = False
    limits_enforced: bool = False
    emergency_stop: bool = False  # True means estop needs to be activated
    exit_requested: bool = False

@dataclass
class RobotCommand:
    """Validated robot command with safety flags.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        input_components: Input-specific components (e.g., torso_delta, hand_command, arm_targets)
                         These are processed and removed by command processor
        output_components: Standard output components (e.g., left_arm, right_arm, torso, chassis)
                          These follow a consistent format with pos/vel and mode
        safety_flags: Safety status flags
        valid: Whether the command passed safety checks
    """

    timestamp_ns: int
    input_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    safety_flags: SafetyFlags = field(default_factory=SafetyFlags)
    valid: bool = True

class BaseInputHandler(ABC):
    """Abstract base class for teleoperation input handlers.

    All input handlers must inherit from this class and implement
    the required methods for processing different input devices.
    """

    def __init__(self, config: Dict[str, Any], namespace: str = ""):
        """Initialize base input handler.

        Args:
            config: Configuration dictionary for the handler
            namespace: Optional namespace for topics
        """
        self.config = config
        self.namespace = namespace

        # Threading locks for thread-safe data access
        self._data_lock = threading.RLock()

        # Latest processed command
        self._latest_command: Optional[RobotCommand] = None

        # Handler state
        self.initialized = False
        self.running = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the input handler and its communication.

        This should set up subscribers, publishers, and any hardware
        connections needed for the specific input device.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def setup_subscribers(self) -> None:
        """Set up communication subscribers for input data.

        Each handler should subscribe to appropriate topics for
        its input device (e.g., exo joints, VR poses, etc.).
        """
        pass

    @abstractmethod
    def process_inputs(self) -> Optional[RobotCommand]:
        """Process the latest inputs and generate robot commands.

        This method should:
        1. Read the latest input data
        2. Transform it to robot commands
        3. Return a RobotCommand object

        Returns:
            RobotCommand if inputs available, None otherwise
        """
        pass

    def get_latest_command(self) -> Optional[RobotCommand]:
        """Get the most recent processed command.

        Thread-safe accessor for the latest command.

        Returns:
            Latest RobotCommand or None
        """
        with self._data_lock:
            return self._latest_command

    def update_command(self, command: RobotCommand) -> None:
        """Update the latest command in a thread-safe manner.

        Args:
            command: New command to store
        """
        with self._data_lock:
            self._latest_command = command

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and connections.

        Should properly close all connections, subscribers,
        and release any hardware resources.
        """
        pass

    def is_initialized(self) -> bool:
        """Check if handler is properly initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self.initialized

    def is_running(self) -> bool:
        """Check if handler is currently running.

        Returns:
            True if running, False otherwise
        """
        return self.running

    def convert_to_publishable(self, command: RobotCommand) -> Dict[str, Any]:
        """Convert RobotCommand to publishable dictionary format.

        Handles numpy array conversion and formatting for serialization.

        Args:
            command: RobotCommand to convert

        Returns:
            Dictionary ready for publishing
        """
        command_dict = {
            "timestamp_ns": command.timestamp_ns,
            "components": {},
            "safety_flags": {
                "collision_detected": command.safety_flags.collision_detected,
                "limits_enforced": command.safety_flags.limits_enforced,
                "emergency_stop": command.safety_flags.emergency_stop,
                "exit_requested": command.safety_flags.exit_requested,
            },
        }

        # Convert numpy arrays to lists
        for name, data in command.components.items():
            component_data = {}

            # Handle position data
            if "pos" in data:
                component_data["pos"] = (
                    data["pos"].tolist()
                    if isinstance(data["pos"], np.ndarray)
                    else data["pos"]
                )

            # Handle velocity data
            if "vel" in data:
                component_data["vel"] = (
                    data["vel"].tolist()
                    if isinstance(data["vel"], np.ndarray)
                    else data["vel"]
                )

            # Add mode if present (for hand commands)
            if "mode" in data:
                component_data["mode"] = data["mode"]

            # Add other fields (for base commands - vx, vy, wz)
            for key in data:
                if key not in ["pos", "vel", "mode"]:
                    component_data[key] = data[key]

            command_dict["components"][name] = component_data

        return command_dict
