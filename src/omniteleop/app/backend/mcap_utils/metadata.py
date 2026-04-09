"""
Episode metadata for MCAP teleop recording.
Serves three purposes:
1. Writing aid - progressively built during recording
2. Reading composability - easily loaded and accessed
3. Visible structure - all fields explicit for reference
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import hashlib
import time
import logging
from omniteleop.app.backend.mcap_utils.utils import get_robot_and_hand_types

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "0.2.0"

@dataclass
class EpisodeMetadata:
    """
    Complete episode metadata structure.
    All fields Optional for easy composition from reader or writer.
    """

    # === Task info (user provides) ===
    task_name: Optional[str] = None
    task_language_instruction: Optional[str] = None

    # === Source info ===
    operator: Optional[str] = None
    source_mode: Optional[str] = None  # "teleop" | "policy" | "intervention"

    # === Policy deployment info (meaningful when source_mode is "policy" or "intervention") ===
    policy_tag: Optional[str] = None  # artifact ID or pretrained_path
    action_chunk_threshold: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None  # computed at finalize()

    # === Recording config ===
    record_rate_hz: Optional[float] = None

    # === Initial positions (from robot config) ===
    # May include joints not in active channels (e.g., chassis)
    init_pos: Optional[Dict[str, List[float]]] = None

    # === Active recording components ===
    # Shows which joints/cameras were enabled for recording
    record_components: Optional[Dict[str, bool]] = None

    # === Timestamps (auto-populated) ===
    start_time: Optional[datetime] = None
    start_timestamp_ns: Optional[int] = None
    end_time: Optional[datetime] = None
    end_timestamp_ns: Optional[int] = None

    # === Counts (auto-populated during recording) ===
    frame_count: Optional[int] = None
    state_count: Optional[int] = None
    action_count: Optional[int] = None

    # === Camera info ===
    available_cameras: Optional[List[str]] = None

    # === Episode result ===
    success: Optional[bool] = None

    # === Extra metadata (runtime info) ===
    extra: Optional[Dict[str, Any]] = None

    # === Schema ===
    schema_version: Optional[str] = None

    # ------ Writer helpers ------

    def start(self):
        """Called when recording starts."""
        self.start_time = datetime.now()
        self.start_timestamp_ns = int(time.time() * 1e9)
        self.frame_count = 0
        self.state_count = 0
        self.action_count = 0
        self.available_cameras = []
        self.schema_version = SCHEMA_VERSION

    def finalize(self, success: bool = True):
        """Called at close() to complete the metadata."""
        self.end_time = datetime.now()
        self.end_timestamp_ns = int(time.time() * 1e9)
        self.success = success

    # ------ Computed properties ------

    @property
    def episode_id(self) -> Optional[str]:
        """Generate unique ID from start time."""
        if self.start_time:
            return hashlib.sha256(self.start_time.isoformat().encode()).hexdigest()[:16]
        return None

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.end_timestamp_ns and self.start_timestamp_ns:
            return (self.end_timestamp_ns - self.start_timestamp_ns) / 1e9
        return None

    @property
    def camera_count(self) -> int:
        """Number of cameras."""
        return len(self.available_cameras) if self.available_cameras else 0

    # ------ Serialization ------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict with dot-notation keys for JSON."""
        robot_type, hand_type = get_robot_and_hand_types()
        return {
            "schema_version": self.schema_version,
            "datetime": self.start_time.isoformat() if self.start_time else None,
            # Episode info
            "episode.id": self.episode_id,
            "episode.start_time": self.start_time.isoformat()
            if self.start_time
            else None,
            "episode.end_time": self.end_time.isoformat() if self.end_time else None,
            "episode.start_timestamp_ns": self.start_timestamp_ns,
            "episode.end_timestamp_ns": self.end_timestamp_ns,
            "episode.duration": self.duration,
            "episode.length": self.state_count,  # Use state_count as canonical length
            "episode.state_count": self.state_count,
            "episode.action_count": self.action_count,
            "episode.success": self.success,
            # Robot (hardcoded for now)
            "robot.model": robot_type,
            "robot.hand_type": hand_type,
            "robot.record_rate_hz": self.record_rate_hz,
            "robot.init_pos": self.init_pos,
            # Recording config
            "recording.components": self.record_components,
            # Task
            "task.name": self.task_name,
            "task.language_instruction": self.task_language_instruction,
            # Source
            "source.operator": self.operator,
            "source.mode": self.source_mode,
            # Policy
            "policy.tag": self.policy_tag,
            "policy.action_chunk_threshold": self.action_chunk_threshold,
            "policy.avg_inference_time_ms": self.avg_inference_time_ms,
            # Data
            "data.available_cameras": self.available_cameras,
            "data.camera_count": self.camera_count,
            # Extra runtime metadata
            "extra": self.extra,
        }

    def save(self, path: Path):
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        """Load from flat dict - all fields populated."""
        meta = cls(
            task_name=data["task.name"],
            task_language_instruction=data["task.language_instruction"],
            operator=data["source.operator"],
            source_mode=data.get("source.mode"),
            policy_tag=data.get("policy.tag"),
            action_chunk_threshold=data.get("policy.action_chunk_threshold"),
            avg_inference_time_ms=data.get("policy.avg_inference_time_ms"),
            record_rate_hz=data["robot.record_rate_hz"],
            init_pos=data.get("robot.init_pos"),  # Optional for backward compat
            record_components=data.get(
                "recording.components"
            ),  # Optional for backward compat
            frame_count=data["episode.length"],
            state_count=data["episode.state_count"],
            action_count=data["episode.action_count"],
            available_cameras=data["data.available_cameras"],
            start_timestamp_ns=data["episode.start_timestamp_ns"],
            end_timestamp_ns=data["episode.end_timestamp_ns"],
            success=data["episode.success"],
            extra=data.get("extra"),  # Optional for backward compat
            schema_version=data["schema_version"],
        )
        if SCHEMA_VERSION != meta.schema_version:
            logger.warning(
                f"Schema version mismatch: {SCHEMA_VERSION} != {meta.schema_version}"
            )
        # Parse datetime strings
        if data["episode.start_time"]:
            meta.start_time = datetime.fromisoformat(data["episode.start_time"])
        if data["episode.end_time"]:
            meta.end_time = datetime.fromisoformat(data["episode.end_time"])
        return meta

    @classmethod
    def load(cls, path: Path) -> "EpisodeMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
