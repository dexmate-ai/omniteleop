#!/usr/bin/env python3
"""Leader arm reader for teleoperation system.

Reads joint positions from the leader arms (exoskeleton) and publishes
them via Zenoh for the teleoperation system.
"""

import sys
import time
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple, Any
import tyro
import math
import numpy as np
from numpy.typing import NDArray

# Add dynamixel SDK to path
from dynamixel_sdk import PortHandler, PacketHandler, GroupBulkRead

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common import ExoJointData, get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from loguru import logger

class LeaderArmReader:
    """Reads leader arm (exoskeleton) positions and publishes via Zenoh."""

    # Motor configuration constants
    ADDR_PRESENT_VELOCITY = 128
    LEN_PRESENT_VELOCITY = 4
    ADDR_PRESENT_POSITION = 132
    LEN_PRESENT_POSITION = 4
    PROTOCOL_VERSION = 2.0

    # Conversion constants
    POSITION_SCALE = 4095.0  # Maximum position value
    VELOCITY_SCALE = 0.229  # Velocity unit in rpm
    RPM_TO_RAD_S = 2 * math.pi / 60.0  # Convert rpm to rad/s

    # Default configuration values
    DEFAULT_BAUD_RATE = 3000000
    DEFAULT_PORT = "/dev/ttyUSB0"
    DEFAULT_PUBLISH_RATE = 40
    DEFAULT_DEBUG_REFRESH_RATE = 10

    # Publishing condition threshold (radians)
    PUBLISH_THRESHOLD = np.array([0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5])

    def __init__(
        self,
        namespace: str = "",
        com_port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        publish_rate: Optional[int] = None,
        topic: Optional[str] = None,
        debug: bool = False,
        read_velocity: bool = False,
    ) -> None:
        """Initialize motor reader.

        Args:
            namespace: Optional namespace prefix (e.g., "robot1")
            com_port: Override serial port (uses config if None)
            baud_rate: Override baud rate (uses config if None)
            publish_rate: Override publish rate (uses config if None)
            topic: Override topic (uses config if None)
            debug: Enable debug output
            read_velocity: Enable reading and publishing velocity data
        """
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="leader_arm_reader", namespace=namespace)

        # Load configuration
        config = get_config()
        self.config = config

        # Use config values with overrides
        self.publish_rate = publish_rate or config.get_rate(
            "input_rate", self.DEFAULT_PUBLISH_RATE
        )
        self.topic = topic or config.get_topic("exo_joints")

        # Get leader arm configurations (with backward compatibility)
        self.arm_configs = config.get_leader_arms()
        if not self.arm_configs:
            raise ValueError("No leader arm configuration found in config")

        # Get port and baud rate from config or use overrides
        if com_port:
            self.com_port = com_port
            self.baud_rate = baud_rate or self.DEFAULT_BAUD_RATE
        else:
            # Get from config - using the single port defined at top level
            self.com_port = self.arm_configs.get("port", self.DEFAULT_PORT)
            self.baud_rate = self.arm_configs.get("baud_rate", self.DEFAULT_BAUD_RATE)

        # Hardware interfaces (single port for all motors)
        self.port_handler: Optional[PortHandler] = None
        self.packet_handler: Optional[PacketHandler] = None
        self.bulk_reader: Optional[GroupBulkRead] = None

        # Zenoh publisher (will be created through Node)
        self.publisher: Optional[Any] = None

        # Motor management
        self.motors: Dict[int, Dict[str, Any]] = {}
        self.motor_positions: Dict[int, float] = {}

        # Initial positions for publishing condition
        self.init_pos: Dict[str, NDArray[np.float64]] = {}

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.publish_rate)

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "LeaderArm",
                self.publish_rate,
                refresh_rate=self.DEFAULT_DEBUG_REFRESH_RATE,
            )

        # Velocity reading option
        self.read_velocity = read_velocity

        logger.info(
            f"Leader arm reader configured "
            f"(velocity={'enabled' if read_velocity else 'disabled'})"
        )

    def initialize(self) -> bool:
        """Initialize hardware and Zenoh communication.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize Zenoh publisher through Node (namespace handled automatically)
            self.publisher = self.node.create_publisher(
                self.topic, encoder=DictDataCodec.encode
            )

            # Initialize Dynamixel hardware (single port for all motors)
            if not self._initialize_hardware():
                return False

            # Connect to all motors from both arms
            self._connect_all_motors()

            # Setup bulk reading for all motors
            self._setup_bulk_reader()

            total_motors = len(self.motors)
            if total_motors == 0:
                logger.error("No motors connected")
                return False

            logger.success(f"Initialized with {total_motors} motors total")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _initialize_hardware(self) -> bool:
        """Initialize Dynamixel hardware interfaces.

        Returns:
            True if hardware initialization successful, False otherwise
        """
        try:
            self.port_handler = PortHandler(self.com_port)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if not self.port_handler.openPort():
                logger.error(f"Failed to open port {self.com_port}")
                return False

            if not self.port_handler.setBaudRate(self.baud_rate):
                logger.error(f"Failed to set baud rate {self.baud_rate}")
                return False

            logger.info(f"Opened port {self.com_port} @ {self.baud_rate} baud")
            return True

        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False

    def _connect_all_motors(self) -> None:
        """Connect to all motors from all configured arms."""
        # Iterate through arm configs to get all motor IDs
        self.init_pos = {}
        for arm_name, arm_config in self.arm_configs.items():
            # Skip non-arm entries like 'port' and 'baud_rate'
            if not isinstance(arm_config, dict) or "motor_ids" not in arm_config:
                continue
            self.init_pos[arm_name] = self.config.get_init_pos(arm_name)
            motor_ids = arm_config.get("motor_ids", [])
            self._connect_arm_motors(arm_name, motor_ids)

    def _connect_arm_motors(self, arm_name: str, motor_ids: List[int]) -> None:
        """Connect to motors for a specific arm.

        Args:
            arm_name: Name of the arm
            motor_ids: List of motor IDs for this arm
        """
        for motor_id in motor_ids:
            try:
                # Ping motor
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.port_handler, motor_id
                )

                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[motor_id] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": arm_name,
                    }
                    logger.info(
                        f"{arm_name} Motor {motor_id}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.warning(f"{arm_name} Motor {motor_id}: Not found")

            except Exception as e:
                logger.warning(f"{arm_name} Motor {motor_id}: Connection error - {e}")

    def _setup_bulk_reader(self) -> None:
        """Setup bulk reader for all connected motors."""
        if not self.motors:
            logger.warning("No motors connected, skipping bulk reader setup")
            return

        try:
            self.bulk_reader = GroupBulkRead(self.port_handler, self.packet_handler)

            for motor_id, info in self.motors.items():
                if info["connected"]:
                    self._add_motor_to_bulk_reader(motor_id)

            logger.info(f"Bulk reader configured with {len(self.motors)} motors")

        except Exception as e:
            logger.error(f"Bulk reader setup failed: {e}")
            self.bulk_reader = None

    def _add_motor_to_bulk_reader(self, motor_id: int) -> None:
        """Add a motor to the bulk reader.

        Args:
            motor_id: ID of the motor to add
        """
        # Always add position register
        success = self.bulk_reader.addParam(
            motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        )

        if not success:
            logger.warning(
                f"Failed to add position for motor {motor_id} to bulk reader"
            )
        else:
            logger.debug(f"Added position for motor {motor_id} to bulk reader")

        # Add velocity register if enabled
        if self.read_velocity:
            success = self.bulk_reader.addParam(
                motor_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
            )

            if not success:
                logger.warning(
                    f"Failed to add velocity for motor {motor_id} to bulk reader"
                )
            else:
                logger.debug(f"Added velocity for motor {motor_id} to bulk reader")

    def _read_positions_velocities(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Read all motor positions and velocities and return arm arrays.

        Returns:
            Tuple of (left_arm_positions, left_arm_velocities, right_arm_positions, right_arm_velocities)
        """
        left_arm_positions: List[float] = []
        left_arm_velocities: List[float] = []
        right_arm_positions: List[float] = []
        right_arm_velocities: List[float] = []

        if not self.bulk_reader:
            return (
                left_arm_positions,
                left_arm_velocities,
                right_arm_positions,
                right_arm_velocities,
            )

        comm_result = self.bulk_reader.txRxPacket()

        if comm_result == 0:  # COMM_SUCCESS
            # Process left arm if configured
            if "left_arm" in self.arm_configs:
                left_arm_positions, left_arm_velocities = self._read_arm_pos_vel(
                    self.arm_configs["left_arm"]
                )

            # Process right arm if configured
            if "right_arm" in self.arm_configs:
                right_arm_positions, right_arm_velocities = self._read_arm_pos_vel(
                    self.arm_configs["right_arm"]
                )
        else:
            logger.warning(f"Bulk read failed: {comm_result}")

        return (
            left_arm_positions,
            left_arm_velocities,
            right_arm_positions,
            right_arm_velocities,
        )

    def _read_arm_pos_vel(
        self, arm_config: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Read positions and velocities for a single arm.

        Args:
            arm_config: Configuration for the arm

        Returns:
            Tuple of (joint positions in radians, joint velocities in rad/s)
        """
        positions: List[float] = []
        velocities: List[float] = []
        motor_ids = arm_config.get("motor_ids", [])
        motor_polarities = arm_config.get("motor_polarities", [1] * len(motor_ids))

        for i, motor_id in enumerate(motor_ids):
            position_rad, velocity_rad_s = self._read_motor_data(
                motor_id, i, motor_polarities
            )
            positions.append(position_rad)
            velocities.append(velocity_rad_s)

        return positions, velocities

    def _read_motor_data(
        self, motor_id: int, motor_index: int, motor_polarities: List[int]
    ) -> Tuple[float, float]:
        """Read position and velocity data for a single motor.

        Args:
            motor_id: ID of the motor to read
            motor_index: Index of the motor in the arm
            motor_polarities: List of polarity corrections

        Returns:
            Tuple of (position in radians, velocity in rad/s)
        """
        position_rad = 0.0
        velocity_rad_s = 0.0

        if motor_id not in self.motors or not self.motors[motor_id]["connected"]:
            return position_rad, velocity_rad_s

        try:
            # Read position data
            position_rad = self._read_motor_position(motor_id)

            # Read velocity data if enabled
            if self.read_velocity:
                velocity_rad_s = self._read_motor_velocity(motor_id)

            # Apply polarity correction from config
            if motor_index < len(motor_polarities):
                position_rad *= motor_polarities[motor_index]
                velocity_rad_s *= motor_polarities[motor_index]

        except Exception as e:
            logger.warning(f"Error reading motor {motor_id}: {e}")
            # Keep default values (0.0)

        return position_rad, velocity_rad_s

    def _read_motor_position(self, motor_id: int) -> float:
        """Read position data for a specific motor.

        Args:
            motor_id: ID of the motor to read

        Returns:
            Position in radians
        """
        if not self.bulk_reader.isAvailable(
            motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        ):
            return 0.0

        # Get raw position
        raw_pos = self.bulk_reader.getData(
            motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        )

        # Convert position to radians
        position_rad = (raw_pos / self.POSITION_SCALE) * 2 * math.pi - math.pi
        return position_rad

    def _read_motor_velocity(self, motor_id: int) -> float:
        """Read velocity data for a specific motor.

        Args:
            motor_id: ID of the motor to read

        Returns:
            Velocity in rad/s
        """
        if not self.bulk_reader.isAvailable(
            motor_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
        ):
            return 0.0

        raw_vel = self.bulk_reader.getData(
            motor_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
        )

        # Convert velocity to rad/s
        # Velocity unit is 0.229 rpm, convert to rad/s
        velocity_rad_s = raw_vel * self.VELOCITY_SCALE * self.RPM_TO_RAD_S
        return velocity_rad_s

    def _create_exo_joint_data(
        self,
        left_arm_pos: List[float],
        left_arm_vel: List[float],
        right_arm_pos: List[float],
        right_arm_vel: List[float],
    ) -> ExoJointData:
        """Create ExoJointData object from arm data.

        Args:
            left_arm_pos: Left arm joint positions
            left_arm_vel: Left arm joint velocities
            right_arm_pos: Right arm joint positions
            right_arm_vel: Right arm joint velocities

        Returns:
            ExoJointData object
        """
        return ExoJointData(
            timestamp_ns=time.time_ns(),
            left_arm_pos=left_arm_pos,
            left_arm_vel=left_arm_vel if self.read_velocity else [],
            right_arm_pos=right_arm_pos,
            right_arm_vel=right_arm_vel if self.read_velocity else [],
        )

    def _update_debug_display(
        self, left_arm_pos: List[float], right_arm_pos: List[float]
    ) -> None:
        """Update debug display with current arm positions.

        Args:
            left_arm_pos: Left arm joint positions
            right_arm_pos: Right arm joint positions
        """
        if not self._debug_display:
            return

        # Convert to dict for debug display compatibility
        debug_joints: Dict[str, float] = {}

        # Add left arm joints
        if "left_arm" in self.arm_configs and left_arm_pos:
            joint_names = self.arm_configs["left_arm"].get("follower_joint_names", [])
            for i, name in enumerate(joint_names[: len(left_arm_pos)]):
                debug_joints[name] = left_arm_pos[i]

        # Add right arm joints
        if "right_arm" in self.arm_configs and right_arm_pos:
            joint_names = self.arm_configs["right_arm"].get("follower_joint_names", [])
            for i, name in enumerate(joint_names[: len(right_arm_pos)]):
                debug_joints[name] = right_arm_pos[i]

        self._debug_display.print_leader_arm(debug_joints)

    def run(self) -> None:
        """Main reading loop."""
        resolved_topic = self.node.resolve_topic(self.topic)
        logger.info(f"Started publishing to {resolved_topic} at {self.publish_rate}Hz")

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        start_publishing = False
        try:
            while True:
                # Read motor positions and velocities
                left_arm_pos, left_arm_vel, right_arm_pos, right_arm_vel = (
                    self._read_positions_velocities()
                )

                # Check publishing condition
                if not start_publishing:
                    start_publishing = True
                    logger.info("Publishing condition satisfied")

                if not start_publishing:
                    self.rate_limiter.sleep()
                    continue

                # Create and publish data if we have at least one arm
                if left_arm_pos or right_arm_pos:
                    data = self._create_exo_joint_data(
                        left_arm_pos, left_arm_vel, right_arm_pos, right_arm_vel
                    )

                    # Publish via Zenoh (convert dataclass to dict for DictDataCodec)
                    self.publisher.publish(asdict(data))

                    # Update debug display
                    self._update_debug_display(left_arm_pos, right_arm_pos)

                # Use RateLimiter for precise timing
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        if self.port_handler:
            self.port_handler.closePort()
            logger.info(f"Closed port {self.com_port}")

        # Node handles cleanup
        self.node.shutdown()

        logger.info("Leader arm reader stopped")

def main(
    namespace: str = "",
    com_port: Optional[str] = None,
    baud_rate: Optional[int] = None,
    publish_rate: Optional[int] = None,
    topic: Optional[str] = None,
    debug: bool = False,
    read_velocity: bool = False,
):
    """Main entry point for leader arm reader.

    Configuration is loaded from ROBOT_CONFIG env var (e.g., "vega_1_f5d6" -> vega_1_f5d6.yaml).

    Args:
        namespace: Optional namespace prefix (e.g., "robot1")
        com_port: Override serial port (uses config if None)
        baud_rate: Override baud rate (uses config if None)
        publish_rate: Override publish rate (uses config if None)
        topic: Override topic (uses config if None)
        debug: Enable debug output
        read_velocity: Enable reading and publishing velocity data
    """
    # Setup logging
    logger = setup_logging(debug)
    logger.info(
        f"Starting Leader Arm Reader{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = LeaderArmReader(
        namespace=namespace,
        com_port=com_port,
        baud_rate=baud_rate,
        publish_rate=publish_rate,
        topic=topic,
        debug=debug,
        read_velocity=read_velocity,
    )

    if not reader.initialize():
        logger.error("Failed to initialize leader arm reader")

    try:
        reader.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        reader.cleanup()

if __name__ == "__main__":
    sys.exit(tyro.cli(main))
