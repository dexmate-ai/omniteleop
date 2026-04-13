#!/usr/bin/env python3
"""Simple replay script for recorded MDP data.

Reads recorded pkl files and publishes commands at the same rate as recording,
following the same initialization pattern as command_processor.py.
"""

import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import tyro
from loguru import logger

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging

class ReplayRecorder:
    """Simple replay system for recorded MDP data.

    This class handles replaying previously recorded robot trajectories by
    publishing commands at the original recording rate. It follows the same
    initialization pattern as the command processor to ensure proper robot
    controller state management.
    """

    # Constants for timing and initialization
    DEFAULT_COMMAND_RATE_HZ: int = 20
    ESTOP_ACTIVATION_DELAY_SEC: float = 1.0
    INITIAL_POSITION_DELAY_SEC: float = 6.0
    DEBUG_LOG_INTERVAL: int = 20

    # Default safety flags
    DEFAULT_SAFETY_FLAGS: Dict[str, bool] = {
        "emergency_stop": False,
        "exit_requested": False,
    }

    def __init__(
        self,
        pkl_file: str,
        namespace: str = "",
        debug: bool = False,
    ) -> None:
        """Initialize replay recorder.

        Args:
            pkl_file: Path to the recorded transitions.pkl file.
            namespace: Namespace for Zenoh topics.
            debug: Enable debug output.

        Raises:
            FileNotFoundError: If the specified pkl file does not exist.
        """
        self.node = Node(name="replay_recorder", namespace=namespace)

        self.debug = debug
        self.config = get_config()

        # Load recorded data
        self.recorded_data: List[Dict[str, Any]] = self._load_recorded_data(pkl_file)

        # Publishing rate configuration
        self.publish_rate = self.config.get_rate(
            "command_rate", self.DEFAULT_COMMAND_RATE_HZ
        )
        self.rate_limiter = RateLimiter(self.publish_rate)

        # State tracking
        self.running = False

        # Setup communication channels
        self._setup_communication()

        logger.info(
            f"🎬 ReplayRecorder initialized, will publish at {self.publish_rate}Hz"
        )

    def _load_recorded_data(self, pkl_file: str) -> List[Dict[str, Any]]:
        """Load recorded data from pickle file.

        Args:
            pkl_file: Path to the pickle file containing recorded transitions.

        Returns:
            List of transition dictionaries containing recorded data.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        pkl_path = Path(pkl_file)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_file}")

        logger.info(f"📂 Loading recorded data from: {pkl_path}")

        with open(pkl_path, "rb") as f:
            recorded_data = pickle.load(f)

        logger.info(f"✅ Loaded {len(recorded_data)} transitions")
        return recorded_data

    def _setup_communication(self) -> None:
        """Setup Zenoh publishers.

        Creates publisher for robot commands using the same topic structure
        as command_processor to ensure compatibility.
        """
        # Publisher for commands
        commands_topic = self.config.get_topic("robot_commands")
        self.command_pub = self.node.create_publisher(
            commands_topic, encoder=DictDataCodec.encode
        )

        resolved_commands = self.node.resolve_topic(commands_topic)
        logger.info(f"📡 Publishing commands to: {resolved_commands}")

    def _wait_for_user_start(self) -> None:
        """Wait for user to press Enter to start replay.

        Provides an interactive prompt allowing the user to start the
        replay sequence when ready.
        """
        logger.info("⏸️ Press Enter to start replay...")
        input()
        logger.info("▶️ Starting replay!")

    def _get_first_transition_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get action and safety flags from first transition.

        Returns:
            Tuple of (first_action, first_safety_flags). Returns empty dicts
            if no data is available.
        """
        if not self.recorded_data:
            return {}, {}

        first_transition = self.recorded_data[0]
        first_action = first_transition.get("action", {})
        first_safety_flags = first_transition.get("metadata", {}).get(
            "safety_flags", self.DEFAULT_SAFETY_FLAGS.copy()
        )

        return first_action, first_safety_flags

    def _initialize_emergency_stop_state(self) -> None:
        """Initialize robot controller emergency stop state.

        This method ensures the robot controller is in the proper state
        by first activating emergency stop, then sending the first command
        to disable it and move to the initial position. This two-step
        process ensures reliable state initialization.
        """
        first_action, first_safety_flags = self._get_first_transition_data()

        if not first_action:
            logger.warning("⚠️ No first action available, skipping estop initialization")
            return

        # Step 1: Activate emergency stop to initialize robot controller state
        logger.info(
            "🛑 Step 1: Activating emergency stop to initialize robot controller state..."
        )
        estop_activate_command = {
            "timestamp_ns": time.time_ns(),
            "components": first_action,
            "safety_flags": {
                "emergency_stop": True,
                "exit_requested": False,
            },
        }
        self.command_pub.publish(estop_activate_command)
        time.sleep(self.ESTOP_ACTIVATION_DELAY_SEC)

        # Step 2: Send first command to disable emergency stop
        logger.info(
            "✅ Step 2: Sending first command to disable emergency stop and "
            "move to initial position..."
        )
        first_command = {
            "timestamp_ns": time.time_ns(),
            "components": first_action,
            "safety_flags": first_safety_flags,
        }
        self.command_pub.publish(first_command)

        logger.info(
            "⏳ First command sent, waiting for robot to reach initial position..."
        )
        if self.debug:
            logger.debug(f"🔍 First command safety flags: {first_safety_flags}")

        # Wait for robot to reach the initial position
        time.sleep(self.INITIAL_POSITION_DELAY_SEC)
        logger.info("🚀 Proceeding with remaining commands...")

    def _create_command_from_transition(
        self, transition: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create command dictionary from transition data.

        Args:
            transition: Transition dictionary from recorded data.

        Returns:
            Command dictionary ready for publishing, or None if action
            data is missing.
        """
        action_data = transition.get("action")
        if not action_data:
            return None

        safety_flags = transition.get("metadata", {}).get(
            "safety_flags", self.DEFAULT_SAFETY_FLAGS.copy()
        )

        return {
            "timestamp_ns": time.time_ns(),
            "components": action_data,
            "safety_flags": safety_flags,
        }

    def _replay_transitions(self, start_index: int = 1) -> int:
        """Replay recorded transitions starting from specified index.

        Args:
            start_index: Index to start replay from (default: 1, skipping first).

        Returns:
            Number of transitions successfully published.
        """
        transition_idx = start_index
        total_transitions = len(self.recorded_data)

        logger.info(
            f"🔄 Replaying remaining {total_transitions - start_index} transitions..."
        )

        while self.running and transition_idx < total_transitions:
            try:
                transition = self.recorded_data[transition_idx]
                command_dict = self._create_command_from_transition(transition)

                if command_dict:
                    self.command_pub.publish(command_dict)

                    if self.debug and transition_idx % self.DEBUG_LOG_INTERVAL == 0:
                        self._log_transition_debug(
                            transition_idx, total_transitions, command_dict
                        )

                transition_idx += 1

            except Exception as e:
                logger.error(
                    f"❌ Error in replay loop at transition {transition_idx}: {e}"
                )
                break

            self.rate_limiter.sleep()

        return transition_idx

    def _log_transition_debug(
        self,
        transition_idx: int,
        total_transitions: int,
        command_dict: Dict[str, Any],
    ) -> None:
        """Log debug information for a transition.

        Args:
            transition_idx: Current transition index.
            total_transitions: Total number of transitions.
            command_dict: Command dictionary being published.
        """
        logger.info(f"📤 Published transition {transition_idx}/{total_transitions}")

        action_data = command_dict.get("components", {})
        safety_flags = command_dict.get("safety_flags", {})

        logger.debug(f"🔧 Command components: {list(action_data.keys())}")
        logger.debug(f"🔒 Safety flags: {safety_flags}")

    def run(self) -> None:
        """Main replay loop.

        Orchestrates the entire replay sequence:
        1. Wait for user input
        2. Initialize emergency stop state
        3. Replay recorded transitions
        """
        self.running = True

        logger.info(f"🚀 Starting replay system at {self.publish_rate}Hz")

        # Wait for user input to start replay
        self._wait_for_user_start()

        # Initialize robot controller emergency stop state
        self._initialize_emergency_stop_state()

        # Replay transitions (start from second transition)
        transitions_published = self._replay_transitions(start_index=1)

        logger.info(
            f"✅ Replay completed! Published {transitions_published} transitions"
        )
        self.running = False

    def stop(self) -> None:
        """Stop replay and cleanup resources.

        Shuts down all communication channels.
        """
        self.running = False
        self.node.shutdown()  # Dexcomm Node cleanup
        logger.info("🛑 ReplayRecorder stopped")

def main(
    pkl_file: str,
    namespace: str = "",
    debug: bool = False,
) -> int:
    """Main entry point for replay recorder.

    Args:
        pkl_file: Path to the recorded transitions.pkl file.
        namespace: Zenoh namespace for topic isolation.
        debug: Enable debug output.

    Returns:
        Exit code (0 for success, 1 for error).

    Example:
        python replay_record.py recordings/episode_0001_20250929_143052/transitions.pkl --debug
    """
    setup_logging(debug)

    try:
        recorder = ReplayRecorder(
            pkl_file=pkl_file,
            namespace=namespace,
            debug=debug,
        )
        recorder.run()
    except KeyboardInterrupt:
        logger.info("⚡ Shutting down...")
        recorder.stop()
    except Exception as e:
        logger.error(f"❌ Replay error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(tyro.cli(main))
