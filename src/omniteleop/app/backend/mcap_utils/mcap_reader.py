"""
MCAP reader for teleop data with Protobuf deserialization.
Supports two reading modes:
- load_episode(): Full load for inspection
- iter_frames(): Memory-efficient streaming for large episodes
"""

import copy
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
from bisect import bisect_left

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

# Import generated protobuf classes
from .proto.robot_pb2 import (
    RobotState as RobotStateProto,
    RobotAction as RobotActionProto,
)
from .proto.policy_pb2 import PolicyStepAnnotation as PolicyStepAnnotationProto
from .proto.camera_pb2 import CameraImage

# Import dataclasses from episode.py (single source of truth)
from .episode import (
    Episode,
    RobotState,
    RobotAction,
    CameraStream,
    PolicyStepAnnotation,
)
from .metadata import EpisodeMetadata

logger = logging.getLogger(__name__)

class TeleopMcapReader:
    """
    MCAP reader for teleoperation data.
    Handles multiple MCAP files (state/action + separate camera files).

    File discovery is done at initialization - no repeated existence checks.
    """

    def __init__(self, base_path: Path):
        """
        Initialize MCAP reader and discover all episode files.

        Args:
            base_path: Episode directory path containing:
                      - state_action.mcap
                      - camera_{name}.mcap for each camera
                      - metadata.json
        """
        self.base_path = Path(base_path)
        self._decoder_factory = DecoderFactory()

        # Discover files at init (once)
        self._file_paths = self._discover_files()
        self._metadata = self._load_metadata()

    def _discover_files(self) -> Dict[str, Any]:
        """Discover all MCAP files in episode directory."""
        state_action_path = self.base_path / "state_action.mcap"

        cameras = {}
        for camera_file in self.base_path.glob("camera_*.mcap"):
            camera_name = camera_file.stem[7:]  # Remove "camera_" prefix
            cameras[camera_name] = camera_file

        return {
            "state_action": state_action_path if state_action_path.exists() else None,
            "cameras": cameras,
        }

    def _load_metadata(self) -> EpisodeMetadata:
        """Load metadata from JSON file."""
        metadata_path = self.base_path / "metadata.json"
        if metadata_path.exists():
            return EpisodeMetadata.load(metadata_path)
        return EpisodeMetadata()

    @property
    def metadata(self) -> EpisodeMetadata:
        return self._metadata

    @property
    def file_paths(self) -> Dict[str, Any]:
        return self._file_paths

    @property
    def available_cameras(self) -> List[str]:
        return list(self._file_paths["cameras"].keys())

    # ========== Protobuf Parsing ==========

    def _parse_camera_image(self, msg: CameraImage) -> np.ndarray:
        """Parse CameraImage protobuf to numpy array."""
        if msg.encoding == "jpeg":
            # Decode JPEG
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode JPEG image")

            # User Rule: "we write in bgr. We convert to rgb in the read."
            # imdecode returns BGR. Convert to RGB.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume raw bytes (bgr8) because writer wrote BGR bytes
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )

            # User Rule: "we write in bgr. We convert to rgb in the read."
            # Convert BGR -> RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ========== Timestamp Alignment ==========

    def _find_nearest_idx(self, timestamps: np.ndarray, target: int) -> int:
        """Find index of nearest timestamp using binary search."""
        idx = bisect_left(timestamps, target)
        if idx == 0:
            return 0
        if idx == len(timestamps):
            return len(timestamps) - 1
        if abs(timestamps[idx - 1] - target) <= abs(timestamps[idx] - target):
            return idx - 1
        return idx

    def _align_to_timestamps(
        self,
        target_timestamps: np.ndarray,
        source_data: List[Tuple[int, Any]],
        max_tolerance_ns: Optional[int] = 50_000_000,  # 50ms default
    ) -> List[Any]:
        """Align source data to target timestamps using nearest-neighbor.

        Args:
            target_timestamps: Reference timestamps to align to
            source_data: List of (timestamp, data) tuples
            max_tolerance_ns: Max allowed gap in nanoseconds. Logs warning if exceeded.
        """
        if not source_data:
            return []
        source_ts = np.array([ts for ts, _ in source_data])

        aligned = []
        for t in target_timestamps:
            idx = self._find_nearest_idx(source_ts, t)
            gap = abs(source_ts[idx] - t)
            if max_tolerance_ns and gap > max_tolerance_ns:
                logger.warning(
                    f"Timestamp alignment gap {gap / 1e6:.1f}ms exceeds tolerance {max_tolerance_ns / 1e6:.0f}ms"
                )
            aligned.append(source_data[idx][1])
        return aligned

    @staticmethod
    def _forward_fill_actions(actions: List[RobotAction]) -> List[RobotAction]:
        """Forward-fill None components in actions with the last valid value.

        During intervention deployments, only the actively controlled arm publishes
        actions. This fills gaps so that inactive components hold their last
        commanded position rather than appearing as missing data.
        """
        _COMPONENT_FIELDS = (
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "head",
            "torso",
        )
        last_valid = {}
        for action in actions:
            for name in _COMPONENT_FIELDS:
                val = getattr(action, name)
                if val is not None:
                    last_valid[name] = val
                elif name in last_valid:
                    setattr(action, name, copy.deepcopy(last_valid[name]))
        return actions

    # ========== Batch Loading ==========

    def load_episode(self, decode_images: bool = True) -> Episode:
        """
        Load complete episode from all MCAP files.

        Args:
            decode_images: Whether to decode camera images

        Returns:
            Episode dataclass with nested state/action/images
        """
        if self._file_paths["state_action"] is None:
            logger.warning(f"No state_action.mcap found in {self.base_path}")
            return None

        # Collect data with timestamps
        states: List[Tuple[int, RobotState]] = []
        actions: List[Tuple[int, RobotAction]] = []
        policy_steps: List[Tuple[int, PolicyStepAnnotation]] = []

        with open(self._file_paths["state_action"], "rb") as f:
            # We use make_reader without decoder factories first to get raw bytes
            # This allows us to manually control decoding with our known proto classes
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                ts = message.log_time
                try:
                    if channel.topic == "/robot/state":
                        proto_msg = RobotStateProto()
                        proto_msg.ParseFromString(message.data)
                        state = RobotState.from_proto(proto_msg)
                        state.timestamp_ns = ts
                        states.append((ts, state))
                    elif channel.topic == "/robot/action":
                        proto_msg = RobotActionProto()
                        proto_msg.ParseFromString(message.data)
                        action = RobotAction.from_proto(proto_msg)
                        action.timestamp_ns = ts
                        actions.append((ts, action))
                    elif channel.topic == "/policy/step":
                        proto_msg = PolicyStepAnnotationProto()
                        proto_msg.ParseFromString(message.data)
                        annotation = PolicyStepAnnotation.from_proto(proto_msg)
                        annotation.timestamp_ns = ts
                        policy_steps.append((ts, annotation))
                except Exception as e:
                    logger.error(
                        f"Failed to decode message on {channel.topic} at {ts}: {e}"
                    )
                    raise e

        logger.info(
            f"Loaded {len(states)} states, {len(actions)} actions, {len(policy_steps)} policy steps"
        )

        if not states:
            return None

        # Reference timestamps from states
        timestamps = np.array([ts for ts, _ in states], dtype=np.int64)

        # Concat states using pytree-style method
        state_list = [s for _, s in states]
        batched_state = RobotState.concat(state_list)

        # Align, forward-fill, and concat actions
        batched_action = None
        if actions:
            aligned_actions = self._align_to_timestamps(timestamps, actions)
            aligned_actions = self._forward_fill_actions(aligned_actions)
            batched_action = RobotAction.concat(aligned_actions)

        # Load and align camera images
        batched_images = None
        if decode_images and self._file_paths["cameras"]:
            camera_frames: Dict[str, List[Tuple[int, np.ndarray]]] = {}

            for camera_name, camera_path in self._file_paths["cameras"].items():
                camera_frames[camera_name] = []
                with open(camera_path, "rb") as f:
                    reader = make_reader(f)
                    for schema, channel, message in reader.iter_messages():
                        if channel.topic.startswith("/camera/"):
                            try:
                                msg = CameraImage()
                                msg.ParseFromString(message.data)
                                img = self._parse_camera_image(msg)
                                camera_frames[camera_name].append(
                                    (message.log_time, img)
                                )
                            except Exception as e:
                                logger.warning(f"Failed to decode image: {e}")
                logger.info(
                    f"Loaded {len(camera_frames[camera_name])} images from {camera_name}"
                )

            # Align each camera to state timestamps and build CameraStream
            if camera_frames:
                aligned_images = {}
                for camera_name, frames in camera_frames.items():
                    if frames:
                        aligned = self._align_to_timestamps(timestamps, frames)
                        aligned_images[camera_name] = np.stack(aligned)

                batched_images = CameraStream(
                    images=aligned_images,
                    timestamp_ns=timestamps,
                    num_frames=len(timestamps),
                )

        # Align policy steps if present (only in policy deployment recordings)
        batched_policy_steps = None
        if policy_steps:
            aligned_policy_steps = self._align_to_timestamps(timestamps, policy_steps)
            batched_policy_steps = PolicyStepAnnotation.concat(aligned_policy_steps)

        return Episode(
            state=batched_state,
            action=batched_action,
            images=batched_images,
            timestamps_ns=timestamps,
            metadata=self._metadata,
            policy_steps=batched_policy_steps,
        )

    # ========== Streaming ==========

    def iter_frames(
        self, cameras: Optional[List[str]] = None
    ) -> Iterator[
        Tuple[
            RobotState,
            Optional[RobotAction],
            CameraStream,
            Optional[PolicyStepAnnotation],
        ]
    ]:  # TODO: make it true streaming, and add optional speicifc image retrieval
        """
        Iterate over synchronized frames (memory-efficient).

        Uses state timestamps as reference, aligns actions, cameras, and policy steps.

        Yields:
            Tuple of (RobotState, RobotAction or None, CameraStream, PolicyStepAnnotation or None)
        """
        if self._file_paths["state_action"] is None:
            return

        # Load state/action/policy data
        states: List[Tuple[int, RobotState]] = []
        actions: List[Tuple[int, RobotAction]] = []
        policy_steps: List[Tuple[int, PolicyStepAnnotation]] = []

        with open(self._file_paths["state_action"], "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                ts = message.log_time
                if channel.topic == "/robot/state":
                    proto_msg = RobotStateProto()
                    proto_msg.ParseFromString(message.data)
                    states.append((ts, RobotState.from_proto(proto_msg)))
                elif channel.topic == "/robot/action":
                    proto_msg = RobotActionProto()
                    proto_msg.ParseFromString(message.data)
                    actions.append((ts, RobotAction.from_proto(proto_msg)))
                elif channel.topic == "/policy/step":
                    proto_msg = PolicyStepAnnotationProto()
                    proto_msg.ParseFromString(message.data)
                    policy_steps.append(
                        (ts, PolicyStepAnnotation.from_proto(proto_msg))
                    )

        if not states:
            return

        # Load camera data
        camera_data: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        for camera_name, camera_path in self._file_paths["cameras"].items():
            camera_data[camera_name] = []
            with open(camera_path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages():
                    if channel.topic.startswith("/camera/"):
                        try:
                            msg = CameraImage()
                            msg.ParseFromString(message.data)
                            img = self._parse_camera_image(msg)
                            camera_data[camera_name].append((message.log_time, img))
                        except:
                            pass

        # Prepare lookups
        action_ts = np.array([ts for ts, _ in actions]) if actions else np.array([])
        policy_ts = (
            np.array([ts for ts, _ in policy_steps]) if policy_steps else np.array([])
        )
        camera_ts = {
            name: np.array([ts for ts, _ in data]) for name, data in camera_data.items()
        }

        # Yield synchronized frames
        for state_ts, state in states:
            # Find nearest action
            action = None
            if len(actions) > 0:
                action = actions[self._find_nearest_idx(action_ts, state_ts)][1]

            # Find nearest policy step (if any)
            policy_step = None
            if len(policy_steps) > 0:
                policy_step = policy_steps[self._find_nearest_idx(policy_ts, state_ts)][
                    1
                ]

            # Build camera frame using CameraStream
            camera_frame = CameraStream(timestamp_ns=state_ts)
            for cam_name, data in camera_data.items():
                if data:
                    idx = self._find_nearest_idx(camera_ts[cam_name], state_ts)
                    camera_frame[cam_name] = data[idx][1]

            yield state, action, camera_frame, policy_step

    # ========== Raw Loading ==========

    def load_raw(self, decode_images: bool = True) -> Dict[str, List[dict]]:
        """Load raw MCAP messages without alignment or interpolation.

        Returns every message as-is with both MCAP envelope timestamps.
        No alignment, no forward-fill, no stacking — just ordered messages
        per channel.

        Args:
            decode_images: Whether to decode camera images to numpy arrays.

        Returns:
            Dict mapping channel name to dict with:
                - log_time_ns (np.ndarray int64): Recorder wall-clock times
                - publish_time_ns (np.ndarray int64): Sensor capture times
                - data (list): Parsed messages (RobotState, RobotAction, np.ndarray, etc.)

            Channel names: "state", "action", "policy_step",
                          "camera/<name>" (e.g. "camera/head_left.rgb")
        """
        result: Dict[str, dict] = {}

        # Accumulators for state/action/policy
        state_log, state_pub, state_data = [], [], []
        action_log, action_pub, action_data = [], [], []
        policy_log, policy_pub, policy_data = [], [], []

        # State/action/policy from state_action.mcap
        if self._file_paths["state_action"] is not None:
            with open(self._file_paths["state_action"], "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages():
                    if channel.topic == "/robot/state":
                        proto_msg = RobotStateProto()
                        proto_msg.ParseFromString(message.data)
                        state = RobotState.from_proto(proto_msg)
                        state.timestamp_ns = message.log_time
                        state_log.append(message.log_time)
                        state_pub.append(message.publish_time)
                        state_data.append(state)
                    elif channel.topic == "/robot/action":
                        proto_msg = RobotActionProto()
                        proto_msg.ParseFromString(message.data)
                        action = RobotAction.from_proto(proto_msg)
                        action.timestamp_ns = message.log_time
                        action_log.append(message.log_time)
                        action_pub.append(message.publish_time)
                        action_data.append(action)
                    elif channel.topic == "/policy/step":
                        proto_msg = PolicyStepAnnotationProto()
                        proto_msg.ParseFromString(message.data)
                        annotation = PolicyStepAnnotation.from_proto(proto_msg)
                        annotation.timestamp_ns = message.log_time
                        policy_log.append(message.log_time)
                        policy_pub.append(message.publish_time)
                        policy_data.append(annotation)

        if state_data:
            result["state"] = {
                "log_time_ns": np.array(state_log, dtype=np.int64),
                "publish_time_ns": np.array(state_pub, dtype=np.int64),
                "data": state_data,
            }
        if action_data:
            result["action"] = {
                "log_time_ns": np.array(action_log, dtype=np.int64),
                "publish_time_ns": np.array(action_pub, dtype=np.int64),
                "data": action_data,
            }
        if policy_data:
            result["policy_step"] = {
                "log_time_ns": np.array(policy_log, dtype=np.int64),
                "publish_time_ns": np.array(policy_pub, dtype=np.int64),
                "data": policy_data,
            }

        # Camera files
        for camera_name, camera_path in self._file_paths["cameras"].items():
            key = f"camera/{camera_name}"
            cam_log, cam_pub, cam_data = [], [], []
            with open(camera_path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages():
                    if channel.topic.startswith("/camera/"):
                        msg = CameraImage()
                        msg.ParseFromString(message.data)
                        cam_log.append(message.log_time)
                        cam_pub.append(message.publish_time)
                        cam_data.append(
                            self._parse_camera_image(msg) if decode_images else msg
                        )
            if cam_data:
                result[key] = {
                    "log_time_ns": np.array(cam_log, dtype=np.int64),
                    "publish_time_ns": np.array(cam_pub, dtype=np.int64),
                    "data": cam_data,
                }
            logger.info(f"Loaded {len(cam_data)} raw messages from {camera_name}")

        return result

    # ========== Utilities ==========

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the episode without full parsing."""
        summary = {
            "base_path": str(self.base_path),
            "metadata": self._metadata.to_dict(),
            "available_cameras": self.available_cameras,
            "files": {
                "state_action": str(self._file_paths["state_action"])
                if self._file_paths["state_action"]
                else None,
                "cameras": {
                    name: str(path)
                    for name, path in self._file_paths["cameras"].items()
                },
            },
        }

        # Add file sizes
        if self._file_paths["state_action"]:
            summary["state_action_size_kb"] = (
                self._file_paths["state_action"].stat().st_size / 1024
            )
        for name, path in self._file_paths["cameras"].items():
            summary[f"camera_{name}_size_kb"] = path.stat().st_size / 1024

        return summary
