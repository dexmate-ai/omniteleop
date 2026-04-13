import cv2

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from mcap_protobuf.writer import Writer as ProtobufWriter
    from mcap.writer import CompressionType
except ImportError:
    raise ImportError(
        "MCAP libraries not installed. Install with: pip install mcap mcap-protobuf-support"
    )

# Import generated protobuf classes (after compiling .proto files)
try:
    from .proto.camera_pb2 import CameraImage
except ImportError:
    raise ImportError(
        "Protobuf classes not found. Compile .proto files first: "
        "cd internal/we_record/mcap_utils/proto && ./compile_proto.sh"
    )

from .episode import RobotState, RobotAction, PolicyStepAnnotation
from .metadata import EpisodeMetadata

class TeleopMcapWriter:
    """
    MCAP writer for teleoperation data.
    Writes state+action to one file and each camera to its own separate file.
    Uses EpisodeMetadata for progressive metadata building.
    """

    def __init__(
        self,
        output_path: Path,
        task_name: str,
        task_language_instruction: str,
        operator: str = "unknown",
        record_rate_hz: float = 20.0,
        init_pos: Optional[Dict] = None,
        record_components: Optional[Dict[str, bool]] = None,
        compression: CompressionType = CompressionType.ZSTD,
        image_encoding: str = "jpeg",  # 'rgb8' or 'jpeg'
        jpeg_quality: int = 90,
        source_mode: Optional[str] = None,
        policy_tag: Optional[str] = None,
        action_chunk_threshold: Optional[float] = None,
    ):
        """
        Initialize MCAP writer.

        Args:
            output_path: Episode folder path.
            task_name: Name of the task being recorded.
            task_language_instruction: Natural language instruction for the task.
            operator: Name of the operator.
            record_rate_hz: Recording rate in Hz.
            init_pos: Initial joint positions from config (may include inactive joints).
            record_components: Dict of component names to enabled status.
            image_encoding: Image encoding format ('rgb8' or 'jpeg')
            jpeg_quality: JPEG quality (1-100) if encoding is 'jpeg'
        """
        self.episode_dir = Path(output_path)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.image_encoding = image_encoding
        self.jpeg_quality = jpeg_quality

        # Initialize metadata with user-provided fields
        self.metadata = EpisodeMetadata(
            task_name=task_name,
            task_language_instruction=task_language_instruction,
            operator=operator,
            record_rate_hz=record_rate_hz,
            init_pos=init_pos,
            record_components=record_components,
            source_mode=source_mode,
            policy_tag=policy_tag,
            action_chunk_threshold=action_chunk_threshold,
        )
        self.metadata.start()  # Sets start_time, initializes counts

        # Accumulate per-step inference durations for avg_inference_time_ms
        self._policy_step_times_ns: list[int] = []

        # Generate state/action file path inside episode folder
        self.state_action_path = self.episode_dir / "state_action.mcap"

        # Create state/action writer
        self._state_action_file = open(self.state_action_path, "wb")
        self._state_action_writer = ProtobufWriter(
            self._state_action_file, compression=compression
        )

        # Camera writers (created on first write per camera)
        self._camera_writers: Dict[str, ProtobufWriter] = {}
        self._camera_files: Dict[str, Any] = {}
        self._camera_paths: Dict[str, Path] = {}

    def write_state(
        self,
        state: RobotState,
        timestamp_ns: int,
        publish_time_ns: Optional[int] = None,
    ):
        """
        Write robot state to MCAP.

        Args:
            state: RobotState dataclass instance
            timestamp_ns: Timestamp in nanoseconds (used as log_time)
            publish_time_ns: Sensor timestamp in nanoseconds (used as publish_time).
                            Falls back to timestamp_ns if None.
        """
        self._state_action_writer.write_message(
            topic="/robot/state",
            message=state.to_proto(),
            log_time=timestamp_ns,
            publish_time=publish_time_ns
            if publish_time_ns is not None
            else timestamp_ns,
        )
        self.metadata.state_count += 1

    def write_action(
        self,
        action: RobotAction,
        timestamp_ns: int,
        publish_time_ns: Optional[int] = None,
    ):
        """
        Write robot action to MCAP.

        Args:
            action: RobotAction dataclass instance
            timestamp_ns: Timestamp in nanoseconds (used as log_time)
            publish_time_ns: Sensor timestamp in nanoseconds (used as publish_time).
                            Falls back to timestamp_ns if None.
        """
        self._state_action_writer.write_message(
            topic="/robot/action",
            message=action.to_proto(),
            log_time=timestamp_ns,
            publish_time=publish_time_ns
            if publish_time_ns is not None
            else timestamp_ns,
        )
        self.metadata.action_count += 1

    def write_policy_step(self, annotation: PolicyStepAnnotation, timestamp_ns: int):
        """Write policy step annotation to MCAP.

        Args:
            annotation: PolicyStepAnnotation dataclass instance.
            timestamp_ns: Timestamp in nanoseconds.
        """
        self._state_action_writer.write_message(
            topic="/policy/step",
            message=annotation.to_proto(),
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
        )
        # Accumulate inference duration for episode-level avg
        if (
            annotation.inference_begin_ns is not None
            and annotation.inference_end_ns is not None
        ):
            self._policy_step_times_ns.append(
                int(annotation.inference_end_ns) - int(annotation.inference_begin_ns)
            )

    def write_camera(
        self,
        image: np.ndarray,
        camera_name: str,
        timestamp_ns: int,
        publish_time_ns: Optional[int] = None,
    ):
        """
        Write camera image to MCAP (separate file per camera).

        Args:
            image: Image array of shape (H, W, 3) with dtype uint8
            camera_name: Name of camera (e.g., "head_left", "head_right")
            timestamp_ns: Timestamp in nanoseconds (used as log_time)
            publish_time_ns: Sensor capture timestamp in nanoseconds (used as publish_time).
                            Falls back to timestamp_ns if None.
        """
        # Create camera writer if it doesn't exist
        if camera_name not in self._camera_writers:
            camera_path = self.episode_dir / f"camera_{camera_name}.mcap"

            camera_file = open(camera_path, "wb")
            camera_writer = ProtobufWriter(camera_file, compression=self.compression)

            self._camera_writers[camera_name] = camera_writer
            self._camera_files[camera_name] = camera_file
            self._camera_paths[camera_name] = camera_path

            # Track available cameras in metadata
            self.metadata.available_cameras.append(camera_name)

        height, width = image.shape[:2]

        # Ensure image is uint8 and contiguous
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8")
        image = np.ascontiguousarray(image)

        image_msg = CameraImage()
        image_msg.height = height
        image_msg.width = width

        if self.image_encoding == "jpeg":  # NOTE: IMAGE INPUT IS BGR
            image_msg.encoding = "jpeg"
            # Encode to jpeg
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encoded_img = cv2.imencode(".jpg", image, encode_param)
            if not success:
                raise RuntimeError("Failed to encode image to JPEG")
            image_msg.data = encoded_img.tobytes()
        else:
            image_msg.encoding = "rgb8"

            image_msg.data = image.tobytes()

        # Write message to camera-specific file
        self._camera_writers[camera_name].write_message(
            topic=f"/camera/{camera_name.replace('.', '/')}",
            message=image_msg,
            log_time=timestamp_ns,
            publish_time=publish_time_ns
            if publish_time_ns is not None
            else timestamp_ns,
        )

    def write_frame(
        self,
        state: RobotState,
        action: RobotAction,
        images: Dict[str, np.ndarray],
        timestamp_ns: Optional[int] = None,
    ):
        """
        Write a complete frame of data (state, action, all cameras) at once.

        Args:
            state: RobotState dataclass instance
            action: RobotAction dataclass instance
            images: Dict mapping camera names to image arrays
            timestamp_ns: Optional timestamp. If None, uses current time.
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)

        # Write state
        if state is not None:
            self.write_state(state, timestamp_ns)

        # Write action
        if action is not None:
            self.write_action(action, timestamp_ns)

        # Write cameras
        for camera_name, image in images.items():
            self.write_camera(image, camera_name, timestamp_ns)

        self.metadata.frame_count += 1

    def close(self, success: bool = True, extra_metadata: Optional[Dict] = None):
        """
        Close all MCAP writers, finalize files, and save metadata.

        Args:
            success: Whether the episode was successful
            extra_metadata: Optional dict of extra runtime metadata (e.g., episode_num, avg_rate)
        """
        # Compute avg inference time from accumulated per-step data
        if self._policy_step_times_ns:
            self.metadata.avg_inference_time_ms = (
                float(np.mean(self._policy_step_times_ns)) / 1e6
            )

        # Finalize metadata
        self.metadata.finalize(success)
        if extra_metadata:
            self.metadata.extra = extra_metadata
        # self.metadata.save(self.episode_dir / "metadata.json")
        self.metadata.save(self.episode_dir / "info.json")

        # Close state/action writer
        self._state_action_writer.finish()
        self._state_action_file.close()

        # Close all camera writers
        for camera_name, writer in self._camera_writers.items():
            writer.finish()
            self._camera_files[camera_name].close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If exception occurred, mark as unsuccessful
        success = exc_type is None
        self.close(success=success)
        return False
