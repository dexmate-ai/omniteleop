"""Format-agnostic episode loading for replay.

Two on-disk episode layouts exist:

* **MDP** (legacy, :class:`~omniteleop.record.mdp_recorder.MDPEpisode`):
  ``transitions.pkl`` (list of ``{timestamp_ns, state, action, metadata}``
  dicts), ``metadata.json`` (flat dict with ``record_rate``), and one
  ``<camera>/frame_{i:06d}.jpg`` folder per recorded camera, where ``i``
  is the transition index.

* **MCAP** (:class:`~omniteleop.record.mcap_recorder.MCAPRecorder`):
  ``episode.mcap`` + dexdata ``metadata.json`` sidecar. Actions live on
  ``/robot/action/<component>/qpos`` (float32) and
  ``/chassis/action/velocity`` (``[vx, vy, wz]``); camera frames on
  ``/camera/<name>/rgb/video`` (AV1, decoded to RGB by dexdata).

Both loaders present the same surface: ``rate_hz`` / ``num_frames`` /
``cameras`` properties plus a ``frames()`` iterator yielding
:class:`ReplayFrame` — the per-tick ``components`` dict shaped exactly
like the ``robot_commands`` payload, safety flags, and per-camera JPEG
bytes. Consumers: ``replay_record.py`` (full replay) and the app backend
(:func:`inspect_episode` for cheap validation — no image decode).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import cv2
import numpy as np
from loguru import logger

TRANSITIONS_PKL = "transitions.pkl"
EPISODE_MCAP = "episode.mcap"

DEFAULT_RATE_HZ = 20.0
DEFAULT_SAFETY_FLAGS: Dict[str, bool] = {
    "emergency_stop": False,
    "exit_requested": False,
}

# MCAP camera topic → canonical camera name. Mirrors the naming in
# omniteleop.record.component_map (COMPONENT_TOPICS) and the MDP image
# folder names in MDPEpisode._IMAGE_TYPES.
MCAP_CAMERA_TOPICS: Dict[str, str] = {
    "/camera/head_left/rgb/video": "head_left_rgb",
    "/camera/head_right/rgb/video": "head_right_rgb",
    "/camera/left_wrist/rgb/video": "left_wrist_rgb",
    "/camera/right_wrist/rgb/video": "right_wrist_rgb",
}
MDP_CAMERA_DIRS = tuple(MCAP_CAMERA_TOPICS.values())

# Joint components whose /robot/action/<comp>/qpos topics map straight to
# components[comp] = {"pos": [...]} in the robot_commands payload.
ACTION_COMPONENTS = (
    "left_arm",
    "right_arm",
    "head",
    "torso",
    "left_hand",
    "right_hand",
)
CHASSIS_ACTION_TOPIC = "/chassis/action/velocity"


@dataclass(frozen=True)
class ReplayFrame:
    """One replay tick: command components + recorded camera JPEGs."""

    index: int
    components: Dict[str, Any]  # robot_commands "components" payload
    safety_flags: Dict[str, Any]
    images: Dict[str, bytes]  # canonical camera name -> JPEG bytes


def resolve_episode_dir(path: Union[str, Path]) -> Path:
    """Normalize a user-supplied path to an episode directory.

    Accepts either the episode dir itself or (backward compat with the
    old replay_record CLI) a direct path to ``transitions.pkl``.
    """
    p = Path(path).expanduser()
    if p.is_file() and p.name == TRANSITIONS_PKL:
        return p.parent
    return p


def detect_format(path: Union[str, Path]) -> Optional[str]:
    """Return "mdp", "mcap", or None if the path isn't a valid episode."""
    d = resolve_episode_dir(path)
    if not d.is_dir():
        return None
    if (d / TRANSITIONS_PKL).exists():
        return "mdp"
    if (d / EPISODE_MCAP).exists():
        return "mcap"
    return None


def load_episode(path: Union[str, Path]) -> "EpisodeLoader":
    """Construct the right loader for the episode at ``path``."""
    d = resolve_episode_dir(path)
    fmt = detect_format(d)
    if fmt == "mdp":
        return MDPEpisodeLoader(d)
    if fmt == "mcap":
        return MCAPEpisodeLoader(d)
    raise FileNotFoundError(
        f"Not a recorded episode (no {TRANSITIONS_PKL} or {EPISODE_MCAP}): {d}"
    )


def inspect_episode(path: Union[str, Path]) -> Dict[str, Any]:
    """Cheap episode metadata for UI validation — never decodes images."""
    loader = load_episode(path)
    return {
        "name": loader.episode_dir.name,
        "path": str(loader.episode_dir),
        "format": loader.format,
        "num_frames": loader.num_frames,
        "rate_hz": loader.rate_hz,
        "cameras": loader.cameras,
    }


class MDPEpisodeLoader:
    """transitions.pkl + JPEG-folder episode loader."""

    format = "mdp"

    def __init__(self, episode_dir: Union[str, Path]):
        self.episode_dir = Path(episode_dir)
        pkl_path = self.episode_dir / TRANSITIONS_PKL
        with open(pkl_path, "rb") as f:
            self._transitions: List[Dict[str, Any]] = pickle.load(f)
        self.num_frames = len(self._transitions)

        self.rate_hz = DEFAULT_RATE_HZ
        meta_path = self.episode_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                self.rate_hz = float(meta.get("record_rate") or DEFAULT_RATE_HZ)
            except Exception as e:  # noqa: BLE001 — metadata is best-effort
                logger.warning(f"Could not read {meta_path}: {e}")

        self.cameras = [
            cam for cam in MDP_CAMERA_DIRS if (self.episode_dir / cam).is_dir()
        ]

    def frames(self) -> Iterator[ReplayFrame]:
        for i, transition in enumerate(self._transitions):
            action = transition.get("action") or {}
            safety_flags = (transition.get("metadata") or {}).get(
                "safety_flags"
            ) or dict(DEFAULT_SAFETY_FLAGS)
            images: Dict[str, bytes] = {}
            for cam in self.cameras:
                img_path = self.episode_dir / cam / f"frame_{i:06d}.jpg"
                if img_path.exists():
                    images[cam] = img_path.read_bytes()
            yield ReplayFrame(i, action, safety_flags, images)


class MCAPEpisodeLoader:
    """episode.mcap loader built on dexdata's StreamingReader.

    Actions (small float arrays) are materialized up front; camera frames
    stream one-at-a-time in reference order so memory stays O(1) in
    episode length. The reference timeline is the first present
    ``/robot/action/*/qpos`` topic's receive timestamps — every action
    topic is written on the same recorder tick, so this is the recording
    cadence.
    """

    format = "mcap"

    def __init__(self, episode_dir: Union[str, Path], jpeg_quality: int = 85):
        # Deferred import: dexdata is only needed for MCAP episodes.
        from dexdata.mcap_utils.streaming_reader import (
            StreamingReader,
            nearest_ref_to_src,
        )

        self.episode_dir = Path(episode_dir)
        self._jpeg_quality = jpeg_quality
        self._reader = StreamingReader(self.episode_dir)
        self._nearest_ref_to_src = nearest_ref_to_src

        spec_topics = {s.topic for s in self._reader.spec.signals}
        self._action_topics: Dict[str, str] = {
            comp: f"/robot/action/{comp}/qpos"
            for comp in ACTION_COMPONENTS
            if f"/robot/action/{comp}/qpos" in spec_topics
        }
        self._has_chassis = CHASSIS_ACTION_TOPIC in spec_topics
        if not self._action_topics:
            raise ValueError(
                f"{self.episode_dir}: no /robot/action/*/qpos topics in spec — "
                "episode has no replayable actions"
            )

        self._ref_topic = next(iter(self._action_topics.values()))
        self._ref_recv_ns = self._reader.topic_timestamps(self._ref_topic).recv_ns
        self.num_frames = int(self._ref_recv_ns.size)

        self._camera_topics: Dict[str, str] = {
            topic: name
            for topic, name in MCAP_CAMERA_TOPICS.items()
            if topic in spec_topics
        }
        self.cameras = list(self._camera_topics.values())

        self.rate_hz = DEFAULT_RATE_HZ
        md = self._reader.metadata
        record_hz = getattr(getattr(md, "collection", None), "record_hz", 0)
        if record_hz:
            self.rate_hz = float(record_hz)

    def frames(self) -> Iterator[ReplayFrame]:
        episode = self._reader.read_non_image_episode()
        ref_ts = self._ref_recv_ns

        # Align every action topic onto the reference timeline (they share
        # the recorder tick, so this is essentially an index passthrough).
        aligned_actions: Dict[str, np.ndarray] = {}
        for comp, topic in self._action_topics.items():
            vals = episode.signals.get(topic)
            src_ts = episode.recv_timestamps.get(topic)
            if vals is None or src_ts is None or not len(vals):
                continue
            idx = self._nearest_ref_to_src(ref_ts, src_ts)
            aligned_actions[comp] = vals[idx]

        chassis_vals: Optional[np.ndarray] = None
        if self._has_chassis:
            vals = episode.signals.get(CHASSIS_ACTION_TOPIC)
            src_ts = episode.recv_timestamps.get(CHASSIS_ACTION_TOPIC)
            if vals is not None and src_ts is not None and len(vals):
                idx = self._nearest_ref_to_src(ref_ts, src_ts)
                chassis_vals = vals[idx]

        # One lazy decoded-frame iterator per camera, in reference order.
        cam_iters: Dict[str, Iterator[np.ndarray]] = {}
        for topic, name in self._camera_topics.items():
            src_ts = self._reader.topic_timestamps(topic).recv_ns
            if src_ts.size == 0:
                logger.warning(f"{name}: no frames in {topic}, skipping camera")
                continue
            ref_to_src = self._nearest_ref_to_src(ref_ts, src_ts)
            cam_iters[name] = self._reader.iter_frames_in_ref_order(
                topic, ref_to_src
            )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        for i in range(self.num_frames):
            components: Dict[str, Any] = {
                comp: {"pos": vals[i].tolist()}
                for comp, vals in aligned_actions.items()
            }
            if chassis_vals is not None:
                vx, vy, wz = (float(x) for x in chassis_vals[i][:3])
                components["chassis"] = {"vx": vx, "vy": vy, "wz": wz}
            images: Dict[str, bytes] = {}
            for name, it in cam_iters.items():
                frame = next(it, None)
                if frame is not None:
                    # dexdata decodes to RGB; imencode expects BGR.
                    ok, buf = cv2.imencode(
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_params
                    )
                    if ok:
                        images[name] = buf.tobytes()
            yield ReplayFrame(i, components, dict(DEFAULT_SAFETY_FLAGS), images)


# Either loader satisfies this shape; kept as a Union alias (a Protocol
# would be overkill for two classes in one file).
EpisodeLoader = Union[MDPEpisodeLoader, MCAPEpisodeLoader]
