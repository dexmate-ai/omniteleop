#!/usr/bin/env python3
"""Standalone MDP Recorder for Policy Learning.

This program subscribes to robot commands and independently collects observations
from the robot (joint angles, camera images) to create MDP datasets for policy learning.
It runs as a separate process and doesn't interfere with the main control loop.
"""

import sys
import time
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import json
from loguru import logger
import tyro
import cv2  # type: ignore
import os
import shutil
from dexcomm import Node
from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config
from dexcomm.utils import RateLimiter
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from dexcomm.codecs import DictDataCodec

# Import pynput for pedal mode (optional dependency)
try:
    from pynput import keyboard as pynput_keyboard

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning(
        "⚠️ pynput not available. Pedal mode will not work. Install with: pip install pynput"
    )

# Import rerun for visual indicator (optional dependency)
try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    logger.debug(
        "⚠️ rerun not available. Visual indicator will not work. Install with: pip install rerun-sdk"
    )

def _save_single_image(args: Tuple[str, np.ndarray, bool, int]) -> None:
    """Helper function to save a single image (for multiprocessing).

    Args:
        args: Tuple of (image_path, image_data, compress_images, jpeg_quality)
    """
    img_path, img, compress_images, jpeg_quality = args
    if compress_images:
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])  # type: ignore
    else:
        cv2.imwrite(str(img_path), img)  # type: ignore

@dataclass
class MDPTransition:
    """Single MDP transition data."""

    timestamp_ns: int
    observation: Dict[str, Any]
    action: Dict[str, Any]
    metadata: Dict[str, Any] = None

class MDPRecorder:
    """Standalone recorder for MDP data collection.

    Subscribes to robot commands and independently collects observations
    from the robot hardware for policy learning datasets.
    """

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        record_mode: str = "joycon",
        show_rerun: bool = False,
    ):
        """Initialize MDP Recorder.

        Args:
            namespace: Namespace for Zenoh topics
            debug: Enable debug output
            record_mode: Recording mode - "keyboard" for keyboard controls, "pedal" for pedal controls (a/b/c keys), "joycon" for joycon/subscriber controls (default)
            show_rerun: If True, show visual indicator using rerun
        """
        self._node = Node(name="mdp_recorder", namespace=namespace)
        self.record_mode = record_mode
        self.show_rerun = show_rerun

        # Pedal mode variables
        self.pedal_key_press_times = {}
        self.pedal_key_lock = threading.Lock()
        self.pedal_hold_duration = 1.0  # seconds
        self.pedal_listener = None
        self.pedal_running = False

        # Initialize rerun if requested
        if self.show_rerun:
            if not RERUN_AVAILABLE:
                logger.warning(
                    "⚠️ rerun not available. Visual indicator disabled. Install with: pip install rerun-sdk"
                )
                self.show_rerun = False
            else:
                try:
                    rr.init("MDP Recorder Status", spawn=True)
                    logger.info("📊 Rerun visual indicator enabled")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to initialize rerun: {e}. Visual indicator disabled."
                    )
                    self.show_rerun = False

        # Load configuration from robot_config.yaml
        self.config = get_config()
        recorder_config = self.config.get("recorder", {})

        # Get configuration values with defaults
        # Convert to absolute path to ensure it works regardless of CWD
        save_dir_str = recorder_config.get("save_dir", "recordings")
        save_path = Path(save_dir_str)
        # If relative path, make it absolute relative to home directory
        # (safer than using cwd which might not exist)
        if not save_path.is_absolute():
            save_path = Path.home() / save_path
        self.save_dir = save_path
        self.episode_prefix = recorder_config.get("episode_prefix", "episode")
        self.record_rate = recorder_config.get("record_rate", 20.0)

        # Parse image resolution
        resolution = recorder_config.get("image_resolution", [640, 480])
        self.image_resolution = (
            tuple(resolution) if isinstance(resolution, list) else (640, 480)
        )

        self.compress_images = recorder_config.get("compress_images", True)
        self.jpeg_quality = recorder_config.get("jpeg_quality", 90)
        self.auto_stop_on_estop = recorder_config.get("auto_stop_on_estop", True)
        self.num_workers = recorder_config.get("num_workers", 4)

        # Determine hand type from ROBOT_CONFIG environment variable
        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            self.hand_type = "gripper"
        else:
            self.hand_type = "hand_f5d6"

        logger.info(
            f"🤏 Hand type detected: {self.hand_type} (from ROBOT_CONFIG={robot_config})"
        )

        # Component recording configuration
        components_config = recorder_config.get("components", {})
        self.record_components = {
            # Joint states
            "left_arm": components_config.get("left_arm", True),
            "right_arm": components_config.get("right_arm", True),
            "torso": components_config.get("torso", True),
            "head": components_config.get("head", True),
            "left_hand": components_config.get("left_hand", True),
            "right_hand": components_config.get("right_hand", True),
            # Camera images
            "head_left_rgb": components_config.get("head_left_rgb", True),
            "head_right_rgb": components_config.get("head_right_rgb", True),
            "left_wrist_rgb": components_config.get("left_wrist_rgb", False),
            "right_wrist_rgb": components_config.get("right_wrist_rgb", False),
        }

        # Episode management
        self.episode_num = self._get_next_episode_num()
        self.episode_dir = None
        self.is_recording = False
        self.episode_start_time = None
        self.episode_data = []

        # Current state
        self.latest_command = None
        self.command_lock = threading.Lock()

        # Action tracking - maintain complete action state at all times
        self.current_action = {}  # Holds the most recent action for each component
        self.last_action = {}  # Holds the last action for each component (for fallback)
        self.action_lock = threading.Lock()

        # Robot interface
        self.robot = None

        # Recording thread
        self.record_thread = None
        self.record_running = False
        self.rate_limiter = RateLimiter(self.record_rate)

        # Statistics
        self.total_transitions = 0
        self.total_episodes = 0
        self.transitions_in_episode = 0

        # Debug mode
        self.debug = debug

        # Check if any images should be saved
        self.save_images = any(
            [
                self.record_components["head_left_rgb"],
                self.record_components["head_right_rgb"],
                self.record_components["left_wrist_rgb"],
                self.record_components["right_wrist_rgb"],
            ]
        )

        logger.info(
            f"🎬 MDP Recorder initialized: save_dir={self.save_dir}, "
            f"record_rate={self.record_rate}Hz, images={self.save_images}, "
            f"workers={self.num_workers}, hand_type={self.hand_type}"
        )

    def _get_next_episode_num(self) -> int:
        """Find the next available episode number."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        existing_episodes = list(self.save_dir.glob(f"{self.episode_prefix}_*"))

        if not existing_episodes:
            return 0

        numbers = []
        for ep_dir in existing_episodes:
            try:
                num = int(ep_dir.name.split("_")[-1])
                numbers.append(num)
            except (ValueError, IndexError):
                continue

        return max(numbers, default=-1) + 1

    def _set_sensor_enabled(
        self, sensors_cfg: Any, sensor_name: str, enabled: bool, warn: bool = True
    ) -> None:
        sensor_cfg = getattr(sensors_cfg, sensor_name, None)
        if sensor_cfg is None or not hasattr(sensor_cfg, "enable"):
            if enabled and warn:
                print(f"Sensor config '{sensor_name}' not found; unable to enable it")
            return
        sensor_cfg.enable = enabled

    def initialize(self):
        """Initialize communication and robot interface."""
        # Subscribe to robot commands
        commands_topic = self.config.get_topic("robot_commands")
        self.subscriber = self._node.create_subscriber(
            commands_topic,
            callback=self._on_command_received,
            decoder=DictDataCodec.decode,
        )
        # logger.info(f"📡 Subscribed to commands: {self._resolve_topic(commands_topic)}")

        # Initialize robot interface for observations
        logger.info("🤖 Initializing robot interface for observations...")
        try:
            # Configure robot with head camera enabled
            robot_configs = get_robot_config()
            robot_configs.sensors["head_camera"].enabled = True
            # robot_configs.sensors.head_camera.use_rtc = False

            if (
                self.record_components["left_wrist_rgb"]
                or self.record_components["right_wrist_rgb"]
            ):
                print("Enabling left wrist camera")
                print(robot_configs.sensors)
                robot_configs.sensors["left_wrist_camera"].enabled = True
                # self._set_sensor_enabled(robot_configs.sensors, "left_wrist_camera", True, warn=True)
                print("Enabling right wrist camera")
                print(robot_configs.sensors)
                robot_configs.sensors["right_wrist_camera"].enabled = True
                # self._set_sensor_enabled(robot_configs.sensors, "right_wrist_camera", True, warn=True)

            self.robot = Robot(configs=robot_configs)

            # Wait for camera to become active if images are needed
            if self.save_images:
                logger.info("📷 Waiting for camera streams to become active...")
                if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
                    logger.success("✅ Camera streams active!")
                else:
                    logger.warning("⚠️ Some camera streams may not be active")

            logger.success("✅ Robot interface initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize robot: {e}")
            logger.warning("⚠️ Recorder will run without robot observations")
            self.robot = None

        # Subscribe to control signals (start/stop recording) unless in keyboard or pedal mode
        if self.record_mode == "keyboard":
            logger.info("⌨️ Keyboard mode: Using keyboard controls for start/stop")
        elif self.record_mode == "pedal":
            if not PYNPUT_AVAILABLE:
                logger.error(
                    "❌ Pedal mode requires pynput. Install with: pip install pynput"
                )
                raise ImportError("pynput is required for pedal mode")
            logger.info(
                "🎹 Pedal mode: Using pedal controls (hold 'a'=start, 'b'=end, 'c'=discard for 1s)"
            )
        else:
            control_topic = self.config.get_topic("recorder_control")
            self.control_subscriber = self._node.create_subscriber(
                control_topic,
                callback=self._on_control_received,
                decoder=DictDataCodec.decode,
            )
            # logger.info(f"📡 Subscribed to control: {self._resolve_topic(control_topic)}")
            logger.info("👂 MDP Recorder is now listening for start/stop commands")

    def _on_command_received(self, data: Dict[str, Any]):
        """Handle incoming robot command and update action state."""
        with self.command_lock:
            self.latest_command = data

        # Update current action state with new components
        # Hold previous values for components not in this command
        with self.action_lock:
            components = data.get("components", {})
            for component_name, component_data in components.items():
                # Update this component's action
                self.current_action[component_name] = component_data

        # Check for recording triggers in safety flags
        safety_flags = data.get("safety_flags", {})

        # Auto-stop recording on emergency stop if configured
        if (
            self.auto_stop_on_estop
            and self.is_recording
            and safety_flags.get("emergency_stop", False)
        ):
            logger.info("🛑 Emergency stop detected, ending episode")
            self.end_episode()

        # Check for exit signal
        if safety_flags.get("exit_requested", False):
            logger.info("🚪 Exit signal received")
            if self.is_recording:
                self.end_episode()

    def _on_control_received(self, data: Dict[str, Any]):
        """Handle recording control commands."""
        logger.info(f"📨 MDP Recorder received control message: {data}")
        command = data.get("command", "")
        logger.info(
            f"🎯 Extracted command: '{command}', current recording state: {self.is_recording}"
        )

        if command == "start" and not self.is_recording:
            metadata = data.get("metadata", {})
            logger.info(f"▶️ Starting episode with metadata: {metadata}")
            self.start_episode(metadata)
        elif command == "stop" and self.is_recording:
            logger.info("⏹️ Stopping current episode")
            self.end_episode()

    def start_episode(self, metadata: Optional[Dict[str, Any]] = None):
        """Start recording a new episode."""
        if self.is_recording:
            logger.warning("⚠️ Already recording, ending current episode first")
            self.end_episode()

        # Create episode directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = (
            self.save_dir / f"{self.episode_prefix}_{self.episode_num:04d}_{timestamp}"
        )
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir)
        self.episode_dir.mkdir(parents=True)

        # Create image subdirectories if needed
        if self.record_components["head_left_rgb"]:
            (self.episode_dir / "head_left_rgb").mkdir(exist_ok=True)
        if self.record_components["head_right_rgb"]:
            (self.episode_dir / "head_right_rgb").mkdir(exist_ok=True)
        if self.record_components["left_wrist_rgb"]:
            (self.episode_dir / "left_wrist_rgb").mkdir(exist_ok=True)
        if self.record_components["right_wrist_rgb"]:
            (self.episode_dir / "right_wrist_rgb").mkdir(exist_ok=True)

        # Initialize episode
        self.is_recording = True
        self.episode_data = []
        self.transitions_in_episode = 0
        self.episode_start_time = time.time()
        self.episode_start_timestamp_ns = time.time_ns()

        # Clear action state at start of new episode to ensure clean slate
        with self.action_lock:
            self.current_action.clear()
            self.last_action.clear()

        # Store metadata
        episode_metadata = {
            "episode_num": self.episode_num,
            "start_time": self.episode_start_time,
            "start_timestamp_ns": self.episode_start_timestamp_ns,
            "datetime": datetime.now().isoformat(),
            "record_rate": self.record_rate,
            "save_images": self.save_images,
            "hand_type": self.hand_type,
        }
        if metadata:
            episode_metadata.update(metadata)

        # Save metadata in human-readable JSON format
        with open(self.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(episode_metadata, f, indent=2, ensure_ascii=False)

        # Start recording thread
        self.record_running = True
        self.record_thread = threading.Thread(
            target=self._record_loop, daemon=True, name="MDPRecordLoop"
        )
        self.record_thread.start()

        # Print formatted banner
        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg green>🎬 RECORDING STARTED - Episode {self.episode_num}</bg green></bold></white>\n"
            f"  📁 Directory: {self.episode_dir.name}\n"
            f"  📊 Record Rate: {self.record_rate} Hz\n"
            f"  📸 Save Images: {self.save_images}\n"
            f"  📝 Metadata: {metadata if metadata else 'None'}\n" + "=" * 80
        )

        # Show rerun visual indicator
        self._show_rerun_indicator(
            "start",
            {
                "episode_num": self.episode_num,
                "directory": self.episode_dir.name,
                "record_rate": self.record_rate,
                "save_images": self.save_images,
                "metadata": metadata if metadata else "None",
            },
        )

    def _record_loop(self):
        """Main recording loop that runs in separate thread."""
        logger.info(f"🔄 Recording loop started at {self.record_rate}Hz")

        while self.record_running and self.is_recording:
            # Get safety flags from latest command
            with self.command_lock:
                safety_flags = (
                    self.latest_command.get("safety_flags", {})
                    if self.latest_command
                    else {}
                )

            # Get observations from robot
            state = self._get_robot_state()
            joint_pos = state.get("joint_pos", {})

            # Build resolved action components with fallbacks
            # For each enabled component:
            # - If action is in current_action, use it
            # - Else if last_action exists, use last_action
            # - Else use observation as action (first occurrence in episode)
            resolved_action = {}
            joint_components = [
                "left_arm",
                "right_arm",
                "torso",
                "head",
                "left_hand",
                "right_hand",
            ]

            with self.action_lock:
                for comp in joint_components:
                    # Only process components that are enabled for recording
                    if not self.record_components.get(comp, False):
                        continue

                    # Priority 1: current action from command
                    if comp in self.current_action:
                        resolved_action[comp] = self.current_action[comp]
                    # Priority 2: last action (from previous command)
                    elif comp in self.last_action:
                        resolved_action[comp] = self.last_action[comp]
                    # Priority 3: use observation as action (first occurrence in episode)
                    elif comp in joint_pos:
                        # Convert observation format to action format: joint_pos -> {"pos": [...]}
                        resolved_action[comp] = {"pos": joint_pos[comp]}

                    # Update last_action with the resolved value for next iteration
                    if comp in resolved_action:
                        self.last_action[comp] = resolved_action[comp]

                # Handle chassis separately (not in joint_pos)
                if "chassis" in self.current_action:
                    resolved_action["chassis"] = self.current_action["chassis"]
                    self.last_action["chassis"] = self.current_action["chassis"]
                elif "chassis" in self.last_action:
                    resolved_action["chassis"] = self.last_action["chassis"]

            # Log first transition to help debug
            if self.transitions_in_episode == 0:
                logger.info(
                    f"📊 First action - from command: {list(self.current_action.keys())}, resolved: {list(resolved_action.keys())}"
                )

            # Get camera images if enabled
            images = {}
            if self.save_images and self.robot:
                images = self._get_camera_images()

            # Create transition with resolved action state
            transition = MDPTransition(
                timestamp_ns=time.time_ns(),
                observation=dict(state=state, **images),
                action=resolved_action,  # Contains all enabled components with fallbacks
                metadata={
                    "safety_flags": safety_flags,
                    "transition_num": self.transitions_in_episode,
                },
            )

            # Save transition
            self._save_transition(transition)
            self.transitions_in_episode += 1
            self.total_transitions += 1

            if self.debug and self.transitions_in_episode % 20 == 0:
                logger.debug(f"📝 Recorded {self.transitions_in_episode} transitions")

            # Maintain recording rate
            self.rate_limiter.sleep()

        logger.info("⏹️ Recording loop stopped")

    def _get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state observation based on configuration."""
        observation = {
            "joint_pos": {},
            "joint_vel": {},
        }

        if self.robot is None:
            return observation

        # Get joint positions only for enabled components
        if self.record_components["left_arm"]:
            observation["joint_pos"]["left_arm"] = (
                self.robot.left_arm.get_joint_pos().tolist()
            )
        if self.record_components["right_arm"]:
            observation["joint_pos"]["right_arm"] = (
                self.robot.right_arm.get_joint_pos().tolist()
            )

        # Check if torso exists before accessing (not all robots have torso)
        if self.record_components["torso"] and hasattr(self.robot, "torso"):
            observation["joint_pos"]["torso"] = (
                self.robot.torso.get_joint_pos().tolist()
            )
        elif self.record_components["torso"]:
            logger.debug("⚠️ torso not available on robot")

        # Check if head exists before accessing (not all robots have head)
        if self.record_components["head"] and hasattr(self.robot, "head"):
            observation["joint_pos"]["head"] = self.robot.head.get_joint_pos().tolist()
        elif self.record_components["head"]:
            logger.debug("⚠️ head not available on robot")

        # Record hand or gripper state based on hand type
        if self.record_components["left_hand"]:
            observation["joint_pos"]["left_hand"] = (
                self.robot.left_hand.get_joint_pos().tolist()
            )

        if self.record_components["right_hand"]:
            observation["joint_pos"]["right_hand"] = (
                self.robot.right_hand.get_joint_pos().tolist()
            )

        # Get joint velocities for enabled arm components
        if self.record_components["left_arm"] and hasattr(
            self.robot.left_arm, "get_joint_vel"
        ):
            observation["joint_vel"]["left_arm"] = (
                self.robot.left_arm.get_joint_vel().tolist()
            )
        if self.record_components["right_arm"] and hasattr(
            self.robot.right_arm, "get_joint_vel"
        ):
            observation["joint_vel"]["right_arm"] = (
                self.robot.right_arm.get_joint_vel().tolist()
            )

        return observation

    def _get_camera_images(self) -> Dict[str, Optional[np.ndarray]]:
        """Get camera images from robot based on configuration."""
        processed_images = {}

        # Determine which camera images to fetch
        head_obs_keys = []
        if self.record_components["head_left_rgb"]:
            head_obs_keys.append("left_rgb")
        if self.record_components["head_right_rgb"]:
            head_obs_keys.append("right_rgb")

        if not head_obs_keys:
            return processed_images

        # Fetch images from head camera
        image_dict = self.robot.sensors.head_camera.get_obs(obs_keys=head_obs_keys)
        image_dict = {f"head_{key}": value for key, value in image_dict.items()}
        if self.record_components["left_wrist_rgb"]:
            image_dict["left_wrist_rgb"] = (
                self.robot.sensors.left_wrist_camera.get_obs()
            )
        if self.record_components["right_wrist_rgb"]:
            image_dict["right_wrist_rgb"] = (
                self.robot.sensors.right_wrist_camera.get_obs()
            )

        # Process images - handle both direct arrays and dict format with timestamp
        for key, value in image_dict.items():
            if value is not None:
                # Extract image data if it's wrapped in a dict with timestamp
                if isinstance(value, dict):
                    img = value.get("data")
                else:
                    img = value

                if img is not None and self.image_resolution:
                    img = cv2.resize(img, self.image_resolution)  # type: ignore

                # Convert RGB to BGR for OpenCV saving
                if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # type: ignore

                processed_images[key] = img

        return processed_images

    def _save_transition(self, transition: MDPTransition):
        """Buffer transition data in memory (images will be saved at episode end)."""
        # Store the full transition including images in memory
        self.episode_data.append(transition)

    def end_episode(self):
        """End current episode and save data."""
        if not self.is_recording:
            logger.warning("⚠️ Not currently recording")
            return

        # Stop recording thread
        self.record_running = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        self.is_recording = False
        episode_duration = time.time() - self.episode_start_time

        # Calculate statistics
        avg_rate = (
            self.transitions_in_episode / episode_duration
            if episode_duration > 0
            else 0
        )

        # Print formatted banner
        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg blue>💾 EPISODE SAVED - Episode {self.episode_num}</bg blue></bold></white>\n"
            f"  📁 Directory: {self.episode_dir.name}\n"
            f"  📊 Transitions: {self.transitions_in_episode}\n"
            f"  ⏱️  Duration: {episode_duration:.1f}s\n"
            f"  📈 Avg Rate: {avg_rate:.1f} Hz\n"
            f"  🎯 Total Episodes: {self.total_episodes + 1}\n"
            f"  📦 Total Transitions: {self.total_transitions}\n" + "=" * 80
        )

        # Show rerun visual indicator
        self._show_rerun_indicator(
            "end",
            {
                "episode_num": self.episode_num,
                "directory": self.episode_dir.name,
                "transitions": self.transitions_in_episode,
                "duration": f"{episode_duration:.1f}",
                "avg_rate": f"{avg_rate:.1f}",
                "total_episodes": self.total_episodes + 1,
                "total_transitions": self.total_transitions,
            },
        )

        # Save episode data
        if self.episode_data:
            logger.info(
                f"💾 Saving episode {self.episode_num} with {len(self.episode_data)} transitions..."
            )

            # Prepare data for saving
            transitions_to_save = []

            # Prepare image saving tasks and transition data
            image_save_tasks = []
            image_types = [
                "head_left_rgb",
                "head_right_rgb",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ]

            for i, transition in enumerate(self.episode_data):
                # Collect images to save
                if self.save_images:
                    for img_type in image_types:
                        img = transition.observation.get(img_type)
                        if img is not None:
                            img_path = (
                                self.episode_dir / img_type / f"frame_{i:06d}.jpg"
                            )
                            image_save_tasks.append(
                                (img_path, img, self.compress_images, self.jpeg_quality)
                            )

                # Create transition data without images for pickle
                transition_data = {
                    "timestamp_ns": transition.timestamp_ns,
                    "state": transition.observation.get("state", {}),
                    "action": transition.action,
                    "metadata": transition.metadata,
                }
                transitions_to_save.append(transition_data)

            # Save images in parallel using multiprocessing
            if image_save_tasks:
                if self.num_workers > 0:
                    logger.info(
                        f"📸 Saving {len(image_save_tasks)} images using {self.num_workers} workers..."
                    )
                    with Pool(processes=self.num_workers) as pool:
                        pool.map(_save_single_image, image_save_tasks)
                else:
                    logger.info(
                        f"📸 Saving {len(image_save_tasks)} images sequentially..."
                    )
                    for task in image_save_tasks:
                        _save_single_image(task)

            # Save transitions pickle
            data_path = self.episode_dir / "transitions.pkl"
            with open(data_path, "wb") as f:
                pickle.dump(transitions_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            logger.warning("⚠️ No data to save for episode")

        # Update counters
        self.episode_num += 1
        self.total_episodes += 1
        self.episode_data = []
        self.transitions_in_episode = 0

    def discard_episode(self):
        """Discard current episode without saving data."""
        if not self.is_recording:
            logger.warning("⚠️ Not currently recording")
            return

        # Stop recording thread
        self.record_running = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        self.is_recording = False
        episode_duration = time.time() - self.episode_start_time

        # Calculate statistics before discarding
        transitions_count = len(self.episode_data)
        avg_rate = transitions_count / episode_duration if episode_duration > 0 else 0

        # Print formatted banner with red background
        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg red>🗑️ EPISODE DISCARDED - Episode {self.episode_num}</bg red></bold></white>\n"
            f"  📁 Directory: {self.episode_dir.name if self.episode_dir else 'N/A'}\n"
            f"  📊 Transitions: {transitions_count}\n"
            f"  ⏱️  Duration: {episode_duration:.1f}s\n"
            f"  📈 Avg Rate: {avg_rate:.1f} Hz\n"
            f"  ⚠️  Data was NOT saved\n" + "=" * 80
        )

        # Show rerun visual indicator
        self._show_rerun_indicator(
            "discard",
            {
                "episode_num": self.episode_num,
                "directory": self.episode_dir.name if self.episode_dir else "N/A",
                "transitions": transitions_count,
                "duration": f"{episode_duration:.1f}",
                "avg_rate": f"{avg_rate:.1f}",
            },
        )

        # Clean up episode directory if it exists
        if self.episode_dir and self.episode_dir.exists():
            try:
                shutil.rmtree(self.episode_dir)
                logger.info(f"🗑️ Deleted episode directory: {self.episode_dir.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete episode directory: {e}")

        # Update counters (but don't increment total_episodes since we discarded)
        self.episode_num += 1
        self.episode_data = []
        self.transitions_in_episode = 0

    def _show_rerun_indicator(self, status: str, info_dict: Dict[str, Any]):
        """Show visual indicator using rerun.

        Args:
            status: Status type - "start", "end", or "discard"
            info_dict: Dictionary containing information to display
        """
        if not self.show_rerun or not RERUN_AVAILABLE:
            return

        try:
            # Color mapping (RGB values 0-255)
            color_map = {
                "start": [0, 255, 0],  # Green
                "end": [0, 128, 255],  # Blue
                "discard": [255, 0, 0],  # Red
            }
            color = color_map.get(status, [128, 128, 128])

            # Create a large rectangle covering the viewport
            # Use 2D space with coordinates from -1000 to 1000 for a large indicator
            rect_size = 5000.0  # Very large rectangle

            # Create text information
            lines = []
            if status == "start":
                lines.append(
                    f"🎬 RECORDING STARTED - Episode {info_dict.get('episode_num', 'N/A')}"
                )
                lines.append(f"📁 Directory: {info_dict.get('directory', 'N/A')}")
                lines.append(
                    f"📊 Record Rate: {info_dict.get('record_rate', 'N/A')} Hz"
                )
                lines.append(f"📸 Save Images: {info_dict.get('save_images', 'N/A')}")
                lines.append(f"📝 Metadata: {info_dict.get('metadata', 'None')}")
                label = f"🎬 RECORDING STARTED - Episode {info_dict.get('episode_num', 'N/A')}"
            elif status == "end":
                lines.append(
                    f"💾 EPISODE SAVED - Episode {info_dict.get('episode_num', 'N/A')}"
                )
                lines.append(f"📁 Directory: {info_dict.get('directory', 'N/A')}")
                lines.append(f"📊 Transitions: {info_dict.get('transitions', 'N/A')}")
                lines.append(f"⏱️  Duration: {info_dict.get('duration', 'N/A')}s")
                lines.append(f"📈 Avg Rate: {info_dict.get('avg_rate', 'N/A')} Hz")
                lines.append(
                    f"🎯 Total Episodes: {info_dict.get('total_episodes', 'N/A')}"
                )
                lines.append(
                    f"📦 Total Transitions: {info_dict.get('total_transitions', 'N/A')}"
                )
                label = (
                    f"💾 EPISODE SAVED - Episode {info_dict.get('episode_num', 'N/A')}"
                )
            elif status == "discard":
                lines.append(
                    f"🗑️ EPISODE DISCARDED - Episode {info_dict.get('episode_num', 'N/A')}"
                )
                lines.append(f"📁 Directory: {info_dict.get('directory', 'N/A')}")
                lines.append(f"📊 Transitions: {info_dict.get('transitions', 'N/A')}")
                lines.append(f"⏱️  Duration: {info_dict.get('duration', 'N/A')}s")
                lines.append(f"📈 Avg Rate: {info_dict.get('avg_rate', 'N/A')} Hz")
                lines.append("⚠️  Data was NOT saved")
                label = f"🗑️ EPISODE DISCARDED - Episode {info_dict.get('episode_num', 'N/A')}"

            # Log rectangle as a box in 2D space
            rr.log(
                "status_indicator/background",
                rr.Boxes3D(
                    half_sizes=[[rect_size, rect_size, rect_size]],
                    centers=[[0, 0, 0]],
                    colors=[color],
                    quaternions=[0, 0, 0, 1],
                    fill_mode="solid",
                    labels=[label],
                    show_labels=True,
                ),
            )

            # Log text information to rerun's text log
            full_text = "\n".join(lines)
            rr.log(
                "status_indicator/text",
                rr.TextLog(full_text, level=rr.TextLogLevel.INFO),
            )

        except Exception as e:
            logger.debug(f"Failed to show rerun indicator: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get recorder statistics."""
        return {
            "total_transitions": self.total_transitions,
            "total_episodes": self.total_episodes,
            "current_episode_num": self.episode_num,
            "is_recording": self.is_recording,
            "transitions_in_episode": self.transitions_in_episode,
        }

    def run(self):
        """Main run loop for status monitoring."""
        if self.record_mode == "keyboard":
            self._run_keyboard_mode()
        elif self.record_mode == "pedal":
            self._run_pedal_mode()
        else:
            self._run_subscriber_mode()

    def _run_keyboard_mode(self):
        """Run recorder with keyboard controls (keyboard mode)."""
        logger.info("⌨️ MDP Recorder running in keyboard mode")
        logger.info(
            "🎮 Commands: 's' = start recording, 'e' = end episode, 'd' = delete/discard, 'q' = quit"
        )

        try:
            while True:
                # Print current status
                if self.is_recording:
                    stats = self.get_statistics()
                    logger.info(
                        f"📹 Recording episode {stats['current_episode_num']}: "
                        f"{stats['transitions_in_episode']} transitions"
                    )
                else:
                    logger.info("⏸️  Idle - Press 's' to start recording")

                # Wait for user input
                user_input = input("Command (s/e/d/q): ").strip().lower()

                if user_input == "s" and not self.is_recording:
                    logger.info("▶️ Starting new episode...")
                    self.start_episode()
                elif user_input == "e" and self.is_recording:
                    logger.info("⏹️ Ending current episode...")
                    self.end_episode()
                elif user_input == "d" and self.is_recording:
                    logger.info("🗑️ Discarding current episode...")
                    self.discard_episode()
                elif user_input == "q":
                    logger.info("👋 Quitting recorder...")
                    break
                elif user_input == "s" and self.is_recording:
                    logger.warning(
                        "⚠️ Already recording! Press 'e' to end or 'd' to discard current episode first."
                    )
                elif user_input == "e" and not self.is_recording:
                    logger.warning("⚠️ Not recording! Press 's' to start a new episode.")
                elif user_input == "d" and not self.is_recording:
                    logger.warning("⚠️ Not recording! Press 's' to start a new episode.")
                else:
                    logger.warning(f"❓ Unknown command: '{user_input}'")

        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")

    def _on_pedal_key_press(self, key):
        """Handle pedal key press event."""
        print("Hello")
        if not PYNPUT_AVAILABLE:
            return

        try:
            # Get character from key
            if hasattr(key, "char") and key.char:
                key_char = key.char.lower()
                print("key_char", key_char)
                if key_char in {"a", "b", "c"}:
                    with self.pedal_key_lock:
                        if key_char not in self.pedal_key_press_times:
                            self.pedal_key_press_times[key_char] = time.time()
                            logger.debug(
                                f"🎹 Pedal key '{key_char}' pressed, waiting for {self.pedal_hold_duration}s hold..."
                            )
        except AttributeError:
            # Special keys don't have char attribute
            pass

    def _on_pedal_key_release(self, key):
        """Handle pedal key release event."""
        if not PYNPUT_AVAILABLE:
            return False

        try:
            # Get character from key
            if hasattr(key, "char") and key.char:
                key_char = key.char.lower()
                if key_char in {"a", "b", "c"}:
                    with self.pedal_key_lock:
                        if key_char in self.pedal_key_press_times:
                            press_time = self.pedal_key_press_times[key_char]
                            hold_time = time.time() - press_time
                            del self.pedal_key_press_times[key_char]

                            if hold_time >= self.pedal_hold_duration:
                                # Execute the action based on key
                                if key_char == "a" and not self.is_recording:
                                    logger.info(
                                        "▶️ Pedal 'a' detected: Starting new episode..."
                                    )
                                    self.start_episode()
                                elif key_char == "b" and self.is_recording:
                                    logger.info(
                                        "⏹️ Pedal 'b' detected: Ending current episode..."
                                    )
                                    self.end_episode()
                                elif key_char == "c" and self.is_recording:
                                    logger.info(
                                        "🗑️ Pedal 'c' detected: Discarding current episode..."
                                    )
                                    self.discard_episode()
                                elif key_char == "a" and self.is_recording:
                                    logger.warning(
                                        "⚠️ Already recording! Press 'b' to end or 'c' to discard first."
                                    )
                                elif key_char in {"b", "c"} and not self.is_recording:
                                    logger.warning(
                                        "⚠️ Not recording! Press 'a' to start a new episode."
                                    )
                            else:
                                logger.debug(
                                    f"❌ Key '{key_char}' released too early ({hold_time:.2f}s < {self.pedal_hold_duration}s)"
                                )
        except AttributeError:
            # Special keys don't have char attribute
            pass

        # Stop listener on escape key
        if PYNPUT_AVAILABLE and key == pynput_keyboard.Key.esc:
            self.pedal_running = False
            return False

        return True

    def _check_pedal_hold_duration(self):
        """Periodically check if pedal keys have been held for the required duration."""
        while self.pedal_running:
            current_time = time.time()
            with self.pedal_key_lock:
                keys_to_remove = []
                for key_char, press_time in self.pedal_key_press_times.items():
                    hold_time = current_time - press_time
                    if hold_time >= self.pedal_hold_duration:
                        # Execute the action based on key
                        if key_char == "a" and not self.is_recording:
                            logger.info(
                                f"✅ Pedal 'a' held for {self.pedal_hold_duration}s: Starting new episode..."
                            )
                            self.start_episode()
                            keys_to_remove.append(key_char)
                        elif key_char == "b" and self.is_recording:
                            logger.info(
                                f"✅ Pedal 'b' held for {self.pedal_hold_duration}s: Ending current episode..."
                            )
                            self.end_episode()
                            keys_to_remove.append(key_char)
                        elif key_char == "c" and self.is_recording:
                            logger.info(
                                f"✅ Pedal 'c' held for {self.pedal_hold_duration}s: Discarding current episode..."
                            )
                            self.discard_episode()
                            keys_to_remove.append(key_char)
                        elif key_char == "a" and self.is_recording:
                            logger.warning(
                                "⚠️ Already recording! Press 'b' to end or 'c' to discard first."
                            )
                            keys_to_remove.append(key_char)
                        elif key_char in {"b", "c"} and not self.is_recording:
                            logger.warning(
                                "⚠️ Not recording! Press 'a' to start a new episode."
                            )
                            keys_to_remove.append(key_char)

                for key_char in keys_to_remove:
                    if key_char in self.pedal_key_press_times:
                        del self.pedal_key_press_times[key_char]

            time.sleep(0.1)  # Check every 100ms

    def _run_pedal_mode(self):
        """Run recorder with pedal controls (pedal mode)."""
        if not PYNPUT_AVAILABLE:
            logger.error(
                "❌ Pedal mode requires pynput. Install with: pip install pynput"
            )
            return

        logger.info("🎹 MDP Recorder running in pedal mode")
        logger.info(
            "🎮 Pedal controls: Hold 'a' for 1s = start, 'b' for 1s = end, 'c' for 1s = discard"
        )
        logger.info("Press 'esc' to quit\n")

        self.pedal_running = True

        # Start checking hold duration in a separate thread
        check_thread = threading.Thread(
            target=self._check_pedal_hold_duration, daemon=True
        )
        check_thread.start()

        # Create and start keyboard listener
        try:
            print("trying to listen to pedal")
            with pynput_keyboard.Listener(
                on_press=self._on_pedal_key_press, on_release=self._on_pedal_key_release
            ) as listener:
                while self.pedal_running:
                    # Print current status periodically
                    if self.is_recording:
                        stats = self.get_statistics()
                        logger.info(
                            f"📹 Recording episode {stats['current_episode_num']}: "
                            f"{stats['transitions_in_episode']} transitions"
                        )
                    else:
                        logger.info("⏸️  Idle - Hold 'a' for 1s to start recording")

                    time.sleep(5.0)  # Status update interval

                    if not listener.running:
                        break
        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")
        finally:
            self.pedal_running = False
            logger.info("👋 Exiting pedal mode...")

    def _run_subscriber_mode(self):
        """Run recorder with topic subscriber controls (normal mode)."""
        logger.info("🎧 MDP Recorder running (waiting for commands)")

        try:
            while True:
                # Print statistics periodically
                if self.is_recording:
                    stats = self.get_statistics()
                    logger.info(
                        f"📹 Recording episode {stats['current_episode_num']}: "
                        f"{stats['transitions_in_episode']} transitions"
                    )

                time.sleep(5.0)  # Status update interval

        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")

    def cleanup(self):
        """Clean up resources."""
        # Stop pedal mode if running
        if self.record_mode == "pedal":
            self.pedal_running = False
            if self.pedal_listener:
                try:
                    self.pedal_listener.stop()
                except Exception:
                    pass

        # End any ongoing episode
        if self.is_recording:
            self.end_episode()

        # Shutdown robot interface
        if self.robot:
            self.robot.shutdown()

        # Node cleanup
        self.shutdown()

        logger.info(
            f"🛑 MDP Recorder shutdown: recorded {self.total_episodes} episodes, "
            f"{self.total_transitions} total transitions"
        )

def main(
    namespace: str = "",
    debug: bool = False,
    record_mode: str = "joycon",
    show_rerun: bool = False,
):
    """Main entry point for MDP Recorder.

    Configuration is loaded from robot_config.yaml.

    Args:
        namespace: Optional namespace prefix
        debug: Enable debug output
        record_mode: Recording mode - "keyboard" for keyboard controls, "pedal" for pedal controls (hold a/b/c for 1s), "joycon" for joycon/subscriber controls (default)
        show_rerun: If True, show visual indicator using rerun
    """
    # Setup logging
    logger = setup_logging(debug)
    mode_str = f" in {record_mode} mode" if record_mode != "joycon" else ""
    logger.info(
        f"🚀 Starting MDP Recorder{f' (namespace: {namespace})' if namespace else ''}{mode_str}"
    )

    # Load config to check if recorder is enabled
    config = get_config()
    recorder_config = config.get("recorder", {})

    if not recorder_config.get("enabled", False):
        logger.warning(
            "⚠️ MDP Recorder is disabled in config. "
            "Set 'recorder.enabled: true' in robot_config.yaml to enable."
        )
        return 0

    logger.info(
        f"⚙️ Recorder config: save_dir={recorder_config.get('save_dir')}, "
        f"rate={recorder_config.get('record_rate')}Hz, "
        f"images={recorder_config.get('save_images')}"
    )

    # Create and run recorder
    recorder = MDPRecorder(
        namespace=namespace,
        debug=debug,
        record_mode=record_mode,
        show_rerun=show_rerun,
    )

    try:
        recorder.initialize()
        recorder.run()
    except Exception as e:
        logger.error(f"❌ Recorder error: {e}")
    finally:
        recorder.cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(tyro.cli(main))
