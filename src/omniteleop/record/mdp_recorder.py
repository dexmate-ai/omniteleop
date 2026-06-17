#!/usr/bin/env python3
"""Standalone MDP Recorder for Policy Learning.

This program subscribes to robot commands and independently collects observations
from the robot (joint angles, camera images) to create MDP datasets for policy learning.
It runs as a separate process and doesn't interfere with the main control loop.

The class inherits from :class:`BaseRecorder` and plugs in pickle + JPEG storage via
four hooks: ``_setup_storage``, ``_collect_observation``, ``_write_frame``,
``_finalize_storage``.
"""

import os
import sys
import time
import pickle
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import cv2  # type: ignore
import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.record.base_recorder import BaseRecorder


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


class MDPEpisode:
    """Pickle + JPEG episode writer for the MDP format.

    Reusable outside MDPRecorder: construct once with format config, then call
    ``start`` / ``append`` / ``finalize`` per episode. Holds per-episode state
    (``episode_dir``, transition buffer); config stays constant across episodes.

    The ``observation`` dict passed to ``append`` uses the same shape MDPRecorder
    produces: joint state at top level under ``joint_pos``/``joint_vel``, camera
    frames at top level keyed by component name (``head_left_rgb`` etc.).
    """

    _IMAGE_TYPES = ("head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb")

    def __init__(
        self,
        save_dir: Path,
        episode_prefix: str,
        record_components: Dict[str, bool],
        *,
        record_rate: float,
        hand_type: str,
        compress_images: bool = True,
        num_workers: int = 4,
        jpeg_quality: int = 90,
    ):
        self.save_dir = save_dir
        self.episode_prefix = episode_prefix
        self.record_components = record_components
        self.record_rate = record_rate
        self.hand_type = hand_type
        self.compress_images = compress_images
        self.num_workers = num_workers
        self.jpeg_quality = jpeg_quality
        self.save_images = any(
            record_components.get(k, False) for k in self._IMAGE_TYPES
        )

        self.episode_dir: Path | None = None
        self._transitions: list = []

    def start(
        self,
        episode_num: int,
        start_time: float | None,
        start_timestamp_ns: int | None,
        metadata: Dict[str, Any] | None = None,
    ) -> Path:
        """Create the episode directory, write metadata.json, return the path."""
        # Group episodes by calendar day: <save_dir>/MM-DD-YYYY/<prefix>_<num>_<ts>.
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        day_dir = self.save_dir / now.strftime("%m-%d-%Y")
        self.episode_dir = day_dir / f"{self.episode_prefix}_{episode_num:04d}_{timestamp}"
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir)
        self.episode_dir.mkdir(parents=True)
        for img_type in self._IMAGE_TYPES:
            if self.record_components.get(img_type, False):
                (self.episode_dir / img_type).mkdir(exist_ok=True)
        self._transitions = []
        ep_meta = {
            "episode_num": episode_num,
            "start_time": start_time,
            "start_timestamp_ns": start_timestamp_ns,
            "datetime": datetime.now().isoformat(),
            "record_rate": self.record_rate,
            "save_images": self.save_images,
            "hand_type": self.hand_type,
        }
        ep_meta.update(metadata or {})
        with open(self.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(ep_meta, f, indent=2, ensure_ascii=False)
        return self.episode_dir

    def append(
        self,
        timestamp_ns: int,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        frame_metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Buffer one transition. Images flushed to disk on ``finalize``."""
        state = {
            "joint_pos": observation.get("joint_pos", {}),
            "joint_vel": observation.get("joint_vel", {}),
        }
        images = {k: v for k, v in observation.items() if k not in ("joint_pos", "joint_vel")}
        self._transitions.append(MDPTransition(
            timestamp_ns=timestamp_ns,
            observation=dict(state=state, **images),
            action=action,
            metadata=frame_metadata or {},
        ))

    def finalize(self, success: bool) -> None:
        """On success: flush images + transitions.pkl. On failure: delete dir."""
        if not success:
            if self.episode_dir and self.episode_dir.exists():
                shutil.rmtree(self.episode_dir)
            self._transitions = []
            return
        if not self._transitions:
            logger.warning("No data to save")
            return
        transitions_to_save = []
        image_save_tasks = []
        for i, t in enumerate(self._transitions):
            if self.save_images:
                for img_type in self._IMAGE_TYPES:
                    img = t.observation.get(img_type)
                    if img is not None:
                        p = self.episode_dir / img_type / f"frame_{i:06d}.jpg"
                        image_save_tasks.append((p, img, self.compress_images, self.jpeg_quality))
            transitions_to_save.append({
                "timestamp_ns": t.timestamp_ns,
                "state": t.observation.get("state", {}),
                "action": t.action,
                "metadata": t.metadata,
            })
        if image_save_tasks:
            if self.num_workers > 0:
                with Pool(processes=self.num_workers) as pool:
                    pool.map(_save_single_image, image_save_tasks)
            else:
                for task in image_save_tasks:
                    _save_single_image(task)
        with open(self.episode_dir / "transitions.pkl", "wb") as f:
            pickle.dump(transitions_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._transitions = []


class MDPRecorder(BaseRecorder):
    """Standalone pickle+JPEG recorder. Storage format unchanged (obsolete, can't delete)."""

    def __init__(self, namespace: str = "", debug: bool = False,
                 record_mode: str = "joycon", show_rerun: bool = False):
        super().__init__(namespace=namespace, debug=debug,
                          record_mode=record_mode, show_rerun=show_rerun)

        recorder_config = self.config.get("recorder", {})
        self.compress_images = recorder_config.get("compress_images", True)
        self.num_workers = recorder_config.get("num_workers", 4)

        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        self.hand_type = "gripper" if "gripper" in robot_config else "hand_f5d6"

        self.save_images = any(self.record_components[k] for k in (
            "head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"
        ))

        self.robot = None
        self.episode_num = self._get_next_episode_num()

        self._episode = MDPEpisode(
            save_dir=self.save_dir,
            episode_prefix=self.episode_prefix,
            record_components=self.record_components,
            record_rate=self.record_rate,
            hand_type=self.hand_type,
            compress_images=self.compress_images,
            num_workers=self.num_workers,
            jpeg_quality=self.jpeg_quality,
        )

    def _node_name(self) -> str:
        return "mdp_recorder"

    def _get_next_episode_num(self) -> int:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # rglob so episodes nested under MM-DD-YYYY day folders (and any legacy
        # flat ones) still count toward the resume sequence number.
        existing = list(self.save_dir.rglob(f"{self.episode_prefix}_*"))
        prefix_len = len(self.episode_prefix.split("_"))
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.split("_")[prefix_len]))
            except (ValueError, IndexError):
                continue
        return max(nums, default=-1) + 1

    def initialize(self) -> None:
        super().initialize()
        robot_configs = get_robot_config()
        robot_configs.sensors["head_camera"].enabled = True
        if self.record_components["left_wrist_rgb"] or self.record_components["right_wrist_rgb"]:
            robot_configs.sensors["left_wrist_camera"].enabled = True
            robot_configs.sensors["right_wrist_camera"].enabled = True
        try:
            self.robot = Robot(configs=robot_configs)
            if self.save_images:
                self.robot.sensors.head_camera.wait_for_active(timeout=5.0)
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            self.robot = None

    # ─── Hooks ─────────────────────────────────────────────────────────────

    def _setup_storage(self, metadata: Dict[str, Any]) -> None:
        self.episode_dir = self._episode.start(
            episode_num=self.episode_num,
            start_time=self.episode_start_time,
            start_timestamp_ns=self.episode_start_timestamp_ns,
            metadata=metadata,
        )

    def _collect_observation(self) -> Dict[str, Any]:
        observation: Dict[str, Any] = {"joint_pos": {}, "joint_vel": {}}
        if self.robot is None:
            return observation
        # Joint positions — unchanged from pre-refactor MDPRecorder.
        if self.record_components["left_arm"]:
            observation["joint_pos"]["left_arm"] = self.robot.left_arm.get_joint_pos().tolist()
        if self.record_components["right_arm"]:
            observation["joint_pos"]["right_arm"] = self.robot.right_arm.get_joint_pos().tolist()
        if self.record_components["torso"] and hasattr(self.robot, "torso"):
            observation["joint_pos"]["torso"] = self.robot.torso.get_joint_pos().tolist()
        if self.record_components["head"] and hasattr(self.robot, "head"):
            observation["joint_pos"]["head"] = self.robot.head.get_joint_pos().tolist()
        if self.record_components["left_hand"]:
            observation["joint_pos"]["left_hand"] = self.robot.left_hand.get_joint_pos().tolist()
        if self.record_components["right_hand"]:
            observation["joint_pos"]["right_hand"] = self.robot.right_hand.get_joint_pos().tolist()
        if self.record_components["left_arm"] and hasattr(self.robot.left_arm, "get_joint_vel"):
            observation["joint_vel"]["left_arm"] = self.robot.left_arm.get_joint_vel().tolist()
        if self.record_components["right_arm"] and hasattr(self.robot.right_arm, "get_joint_vel"):
            observation["joint_vel"]["right_arm"] = self.robot.right_arm.get_joint_vel().tolist()
        # Camera images merged into observation under per-camera keys.
        if self.save_images and self.robot:
            head_keys = []
            if self.record_components["head_left_rgb"]:
                head_keys.append("left_rgb")
            if self.record_components["head_right_rgb"]:
                head_keys.append("right_rgb")
            if head_keys:
                img_dict = self.robot.sensors.head_camera.get_obs(obs_keys=head_keys)
                img_dict = {f"head_{k}": v for k, v in img_dict.items()}
                if self.record_components["left_wrist_rgb"]:
                    img_dict["left_wrist_rgb"] = self.robot.sensors.left_wrist_camera.get_obs()
                if self.record_components["right_wrist_rgb"]:
                    img_dict["right_wrist_rgb"] = self.robot.sensors.right_wrist_camera.get_obs()
                for k, v in img_dict.items():
                    if v is None:
                        continue
                    img = v.get("data") if isinstance(v, dict) else v
                    if img is None:
                        continue
                    if self.image_resolution:
                        img = cv2.resize(img, self.image_resolution)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    observation[k] = img
        return observation

    def _write_frame(self, timestamp_ns: int, resolved_action: Dict[str, Any],
                     observation: Dict[str, Any], safety_flags: Dict[str, Any]) -> None:
        self._episode.append(
            timestamp_ns=timestamp_ns,
            observation=observation,
            action=resolved_action,
            frame_metadata={"safety_flags": safety_flags, "transition_num": self.transitions_in_episode},
        )

    def _finalize_storage(self, success: bool) -> None:
        self._episode.finalize(success)

    def cleanup(self) -> None:
        super().cleanup()
        if self.robot:
            self.robot.shutdown()


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
