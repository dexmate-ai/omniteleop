"""Rerun visualization utilities for dataset and episode visualization."""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tqdm
from tensordict import TensorDict

logger = logging.getLogger(__name__)

def read_video_frames(video_path: str | Path) -> np.ndarray:
    """Read all frames from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        np.ndarray: Array of shape (N, H, W, C) with dtype uint8.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If the video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return np.stack(frames, axis=0)

VideoData = Union[str, Path, torch.Tensor, np.ndarray]

class RerunViz:
    """Visualization helper for episode data using Rerun.

    Supports visualization of:
    - Video data (from paths or tensors)
    - State data (scalar or vector time series)
    - Image data (individual frames)

    Example:
        >>> viz = RerunViz(
        ...     video_data={"camera.rgb": "/path/to/video.mp4"},
        ...     state_data={"observation.state.joint_pos": joint_positions},
        ... )
        >>> viz.visualize()
    """

    def __init__(
        self,
        video_data: dict[str, VideoData] | TensorDict | None = None,
        state_data: dict[str, torch.Tensor | np.ndarray] | TensorDict | None = None,
        image_data: dict[str, torch.Tensor | np.ndarray] | TensorDict | None = None,
        app_name: str = "dexpolicy_viz",
        fps: float = 30.0,
    ):
        """Initialize RerunViz.

        Args:
            video_data: Dict mapping video keys to video paths or frame tensors.
                        Shape for tensors: (N, H, W, C) or (N, C, H, W).
            state_data: Dict mapping state keys to state tensors.
                        Shape: (N,) for scalars or (N, D) for vectors.
            image_data: Dict mapping image keys to image tensors.
                        Shape: (N, H, W, C) or (N, C, H, W).
            app_name: Name for the Rerun application.
            fps: Frames per second for timestamp calculation.
        """
        self.video_data = self._to_dict(video_data) or {}
        self.state_data = self._to_dict(state_data) or {}
        self.image_data = self._to_dict(image_data) or {}
        self.app_name = app_name
        self.fps = fps

        # Loaded video frames cache
        self._video_frames: dict[str, np.ndarray] = {}
        self._episode_len: int | None = None

    @staticmethod
    def _to_dict(data: dict | TensorDict | None) -> dict | None:
        """Convert TensorDict to regular dict if needed."""
        if data is None:
            return None
        if isinstance(data, TensorDict):
            return {k: data[k] for k in data.keys()}
        return data

    def _load_video_frames(self, video_source: VideoData) -> np.ndarray:
        """Load video frames from path or convert tensor.

        Args:
            video_source: Path to video file or tensor of frames.

        Returns:
            np.ndarray: Array of shape (N, H, W, C) with dtype uint8.
        """
        if isinstance(video_source, (str, Path)):
            return read_video_frames(video_source)

        # Handle tensor input
        if isinstance(video_source, torch.Tensor):
            frames = video_source.cpu().numpy()
        else:
            frames = np.asarray(video_source)

        # Handle (N, C, H, W) -> (N, H, W, C)
        if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
            frames = np.transpose(frames, (0, 2, 3, 1))

        # Convert to uint8 if needed
        if frames.dtype in (np.float32, np.float64):
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)

        return frames

    def _validate_lengths(self) -> int:
        """Validate that all sequences have the same length.

        Returns:
            The common sequence length.

        Raises:
            ValueError: If lengths are not aligned or no data provided.
        """
        lengths = {}

        # Check video frames
        for key, frames in self._video_frames.items():
            lengths[f"video:{key}"] = len(frames)

        # Check state data
        for key, values in self.state_data.items():
            if isinstance(values, torch.Tensor):
                lengths[f"state:{key}"] = values.shape[0]
            else:
                lengths[f"state:{key}"] = len(values)

        # Check image data
        for key, values in self.image_data.items():
            if isinstance(values, torch.Tensor):
                lengths[f"image:{key}"] = values.shape[0]
            else:
                lengths[f"image:{key}"] = len(values)

        if not lengths:
            raise ValueError("No data provided for visualization")

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            logger.warning(
                f"Data lengths are not aligned: {lengths}. "
                f"Using minimum length: {min(unique_lengths)}"
            )

        return min(unique_lengths)

    @staticmethod
    def _key_to_entity_path(key: str) -> str:
        """Convert a dotted key to a hierarchical Rerun entity path.

        Examples:
            "observation.state.left_hand" -> "state/observation.state/left_hand"
            "action.left_hand" -> "state/action/left_hand"
            "observation.state.robot.joint_pos" -> "state/observation.state/robot/joint_pos"
            "index" -> "state/other/index"

        Args:
            key: Dotted key string.

        Returns:
            Hierarchical entity path for Rerun.
        """
        # Identify the prefix group (observation.state or action)
        if key.startswith("observation.state."):
            prefix = "observation.state"
            remainder = key[len("observation.state.") :]
        elif key.startswith("action."):
            prefix = "action"
            remainder = key[len("action.") :]
        else:
            # Fallback: put under "other" group
            return f"state/other/{key.replace('.', '/')}"

        # Convert remaining dots to path separators
        subpath = remainder.replace(".", "/")
        return f"state/{prefix}/{subpath}"

    def _get_plot_groups(self) -> dict[str, str]:
        """Get plot groups - one plot per unique second-level path.

        Groups keys by their full path up to the second level:
        - "observation.state.left_hand" -> "observation.state/left_hand"
        - "observation.state.right_hand" -> "observation.state/right_hand"
        - "action.left_hand" -> "action/left_hand"
        - "index" -> "other/index"

        Returns:
            Dict mapping entity path origin to display name for the plot.
        """
        plot_groups: dict[str, str] = {}

        for key in self.state_data.keys():
            if key.startswith("observation.state."):
                remainder = key[len("observation.state.") :]
                # Get the first part after the prefix (e.g., "left_hand" from "left_hand.something")
                second_level = remainder.split(".")[0]
                origin = f"state/observation.state/{second_level}"
                display_name = f"obs.state.{second_level}"
            elif key.startswith("action."):
                remainder = key[len("action.") :]
                second_level = remainder.split(".")[0]
                origin = f"state/action/{second_level}"
                display_name = f"action.{second_level}"
            else:
                # Other keys - group by the key itself (or first part if dotted)
                first_part = key.split(".")[0]
                origin = f"state/other/{first_part}"
                display_name = first_part

            # Only add if not already present
            if origin not in plot_groups:
                plot_groups[origin] = display_name

        return plot_groups

    def _init_rerun(
        self,
        mode: str = "local",
        web_port: int = 9090,
        ws_port: int = 9087,
    ) -> None:
        """Initialize Rerun recording.

        Args:
            mode: 'local' spawns viewer, 'distant' creates server.
            web_port: Web port for distant mode.
            ws_port: WebSocket port for distant mode.
        """
        # Don't spawn viewer during init - we'll spawn after logging all data
        # This avoids IPC timeout issues when logging lots of data
        rr.init(self.app_name, spawn=False)
        gc.collect()

        if mode == "distant":
            rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    def _create_blueprint(self) -> rrb.Blueprint:
        """Create Rerun blueprint for layout.

        Returns:
            Blueprint with cameras in grid and states grouped by second-level path.
        """
        # Video views
        video_views = [
            rrb.Spatial2DView(
                name=key.split(".")[-1] if "." in key else key, origin=f"video/{key}"
            )
            for key in self._video_frames.keys()
        ]

        # Image views
        image_views = [
            rrb.Spatial2DView(
                name=key.split(".")[-1] if "." in key else key, origin=f"image/{key}"
            )
            for key in self.image_data.keys()
        ]

        all_visual_views = video_views + image_views

        # State views - one plot per second-level grouping
        # e.g., separate plots for obs.state.left_hand, obs.state.right_hand, action.left_hand
        plot_groups = self._get_plot_groups()
        state_group_views = []

        for origin, display_name in plot_groups.items():
            group_view = rrb.TimeSeriesView(
                name=display_name,
                origin=origin,
            )
            state_group_views.append(group_view)

        # Build layout
        if all_visual_views and state_group_views:
            # Cameras on left, states on right
            n_cols = min(2, len(all_visual_views))
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Grid(*all_visual_views, grid_columns=n_cols),
                    rrb.Vertical(*state_group_views),
                    column_shares=[2, 1],
                ),
                collapse_panels=False,
            )
        elif all_visual_views:
            n_cols = min(2, len(all_visual_views))
            blueprint = rrb.Blueprint(
                rrb.Grid(*all_visual_views, grid_columns=n_cols),
                collapse_panels=False,
            )
        elif state_group_views:
            blueprint = rrb.Blueprint(
                rrb.Vertical(*state_group_views),
                collapse_panels=False,
            )
        else:
            blueprint = rrb.Blueprint(collapse_panels=False)

        return blueprint

    def _log_frame(self, frame_idx: int) -> None:
        """Log data for a single frame.

        Args:
            frame_idx: Index of the frame to log.
        """
        rr.set_time("frame_index", sequence=frame_idx)
        rr.set_time("timestamp", duration=frame_idx / self.fps)

        # Log video frames
        for key, frames in self._video_frames.items():
            if frame_idx < len(frames):
                rr.log(f"video/{key}", rr.Image(frames[frame_idx]))

        # Log image data
        for key, images in self.image_data.items():
            if isinstance(images, torch.Tensor):
                img = images[frame_idx].cpu().numpy()
            else:
                img = np.asarray(images[frame_idx])

            # Handle (C, H, W) -> (H, W, C)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))

            # Convert to uint8
            if img.dtype in (np.float32, np.float64):
                img = (img * 255).clip(0, 255).astype(np.uint8)

            rr.log(f"image/{key}", rr.Image(img))

        # Log state data with hierarchical grouping
        # Keys like "observation.state.left_hand" -> "state/observation.state/left_hand/0"
        # Keys like "action.left_hand" -> "state/action/left_hand/0"
        for key, values in self.state_data.items():
            if isinstance(values, torch.Tensor):
                value = values[frame_idx].cpu().numpy()
            else:
                value = np.asarray(values[frame_idx])

            # Build hierarchical path from dotted key
            entity_path = self._key_to_entity_path(key)

            # Handle scalar vs vector
            if np.isscalar(value) or (
                isinstance(value, np.ndarray) and value.ndim == 0
            ):
                rr.log(entity_path, rr.Scalars(float(value)))
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                for dim_idx, val in enumerate(value):
                    rr.log(f"{entity_path}/{dim_idx}", rr.Scalars(float(val)))
            else:
                # Flatten complex structures
                flat_value = np.asarray(value).flatten()
                for dim_idx, val in enumerate(flat_value):
                    rr.log(f"{entity_path}/{dim_idx}", rr.Scalars(float(val)))

    def visualize(
        self,
        mode: str = "local",
        web_port: int = 9090,
        ws_port: int = 9087,
        save: bool = False,
        output_path: Path | str | None = None,
        states_dict: dict | None = None,
        videos_dict: dict | None = None,
    ) -> Path | None:
        """Run visualization.

        Args:
            mode: 'local' spawns viewer, 'distant' creates server.
            web_port: Web port for distant mode.
            ws_port: WebSocket port for distant mode.
            save: Whether to save as .rrd file.
            output_path: Path for .rrd file when save=True.
            states_dict: Override state data (for compatibility).
            videos_dict: Override video data (for compatibility).

        Returns:
            Path to saved .rrd file if save=True, else None.
        """
        # Allow overriding data via method args (backward compat)
        if states_dict is not None:
            self.state_data = self._to_dict(states_dict) or {}
        if videos_dict is not None:
            self.video_data = self._to_dict(videos_dict) or {}

        if save and output_path is None:
            raise ValueError("output_path required when save=True")

        # Load all video frames
        logger.info("Loading video frames...")
        for key, source in self.video_data.items():
            self._video_frames[key] = self._load_video_frames(source)
            logger.info(f"  {key}: {self._video_frames[key].shape}")

        # Validate lengths
        self._episode_len = self._validate_lengths()
        logger.info(f"Episode length: {self._episode_len} frames")

        # Initialize Rerun
        self._init_rerun(mode=mode, web_port=web_port, ws_port=ws_port)

        # Send blueprint
        blueprint = self._create_blueprint()
        rr.send_blueprint(blueprint)

        # Log all frames
        logger.info("Logging frames to Rerun...")
        for frame_idx in tqdm.tqdm(range(self._episode_len), desc="Logging"):
            self._log_frame(frame_idx)

        # Handle output
        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(output_path))
            logger.info(f"Saved to {output_path}")
            return output_path

        if mode == "local":
            # Save to temp file and spawn viewer with it
            # This avoids IPC timeout issues when logging lots of data
            import tempfile
            import subprocess

            temp_file = tempfile.NamedTemporaryFile(suffix=".rrd", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            rr.save(temp_path)
            logger.info("Opening visualization in Rerun viewer...")

            # Spawn rerun viewer with the temp file
            subprocess.Popen(
                ["rerun", temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return None

        if mode == "distant":
            logger.info(f"Serving at ws://localhost:{ws_port}")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted, exiting.")

        return None

    @staticmethod
    def load(
        rrd_path: str | Path,
        spawn: bool = True,
    ) -> None:
        """Load and display a saved .rrd visualization file.

        Args:
            rrd_path: Path to the .rrd file.
            spawn: If True, spawn the Rerun viewer. If False, just load the recording.

        Raises:
            FileNotFoundError: If the .rrd file doesn't exist.

        Example:
            >>> RerunViz.load("/path/to/visualization.rrd")
        """
        rrd_path = Path(rrd_path)
        if not rrd_path.exists():
            raise FileNotFoundError(f"RRD file not found: {rrd_path}")

        if not rrd_path.suffix == ".rrd":
            logger.warning(f"File does not have .rrd extension: {rrd_path}")

        logger.info(f"Loading visualization from {rrd_path}")

        # Initialize rerun and spawn viewer if requested
        rr.init(rrd_path.stem, spawn=spawn)
        gc.collect()

        # Load the saved recording
        rr.log_file_from_path(str(rrd_path))
        logger.info("Visualization loaded successfully")

def visualize_episode(
    episode_path: str | Path,
    mode: str = "local",
    save: bool = False,
    output_path: str | Path | None = None,
    fps: float = 30.0,
) -> Path | None:
    """Visualize an MCAP episode directory.

    Args:
        episode_path: Path to episode directory containing state_action.mcap and camera_*.mcap
        mode: 'local' spawns viewer, 'distant' creates server
        save: Whether to save as .rrd file
        output_path: Path for .rrd file when save=True
        fps: Frames per second for timestamp calculation

    Returns:
        Path to saved .rrd file if save=True, else None

    Example:
        >>> visualize_episode("/path/to/episode_0001_20260128_120000/")
    """
    # Handle import for both module and direct script execution
    try:
        from .mcap_utils import TeleopMcapReader
    except ImportError:
        from mcap_utils import TeleopMcapReader

    episode_path = Path(episode_path)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode path not found: {episode_path}")

    logger.info(f"Loading episode from {episode_path}")

    # Load episode using MCAP reader
    reader = TeleopMcapReader(episode_path)
    episode = reader.load_episode(decode_images=True)

    if episode is None:
        raise ValueError(f"Failed to load episode from {episode_path}")

    logger.info(f"Loaded episode with {len(episode)} frames")

    # Debug: print raw episode data before converting
    logger.info("Episode state components:")
    logger.info(f"  left_arm: {episode.state.left_arm is not None}")
    logger.info(f"  right_arm: {episode.state.right_arm is not None}")
    logger.info(f"  left_hand: {episode.state.left_hand is not None}")
    logger.info(f"  right_hand: {episode.state.right_hand is not None}")
    logger.info(f"  head: {episode.state.head is not None}")
    logger.info(f"  torso: {episode.state.torso is not None}")
    if episode.action:
        logger.info("Episode action components:")
        logger.info(f"  left_arm: {episode.action.left_arm is not None}")
        logger.info(f"  right_arm: {episode.action.right_arm is not None}")
        logger.info(f"  left_hand: {episode.action.left_hand is not None}")
        logger.info(f"  right_hand: {episode.action.right_hand is not None}")
        logger.info(f"  head: {episode.action.head is not None}")
        logger.info(f"  torso: {episode.action.torso is not None}")

    # Convert episode to visualization format using to_numpy_dict
    data_dict = episode.to_numpy_dict()
    # Separate state and video data
    state_data = {}
    video_data = {}

    for key, value in data_dict.items():
        if key.startswith("observation.images."):
            # Extract camera name from key like "observation.images.head_left_rgb"
            camera_name = key.replace("observation.images.", "").replace("_rgb", "")
            video_data[camera_name] = value
        else:
            state_data[key] = value

    # Extract policy step annotations if present
    if episode.policy_steps is not None:
        ps = episode.policy_steps
        if ps.inference_begin_ns is not None and ps.inference_end_ns is not None:
            inference_time_ms = (ps.inference_end_ns - ps.inference_begin_ns) / 1e6
            state_data["policy.inference_time_ms"] = inference_time_ms.astype(
                np.float32
            )
        if ps.action_queue_size is not None:
            state_data["policy.action_queue_size"] = np.asarray(
                ps.action_queue_size, dtype=np.float32
            )
        if ps.policy_control_flag is not None:
            state_data["policy.control_flag"] = np.asarray(
                ps.policy_control_flag, dtype=np.float32
            )
        logger.info(
            f"Policy annotation keys added: {[k for k in state_data if k.startswith('policy.')]}"
        )
    else:
        logger.info("No policy step annotations found in episode")

    logger.info(f"State keys: {list(state_data.keys())}")
    logger.info(f"Video keys: {list(video_data.keys())}")

    # Create visualizer and run
    viz = RerunViz(
        video_data=video_data,
        state_data=state_data,
        app_name=episode_path.name,
        fps=fps,
    )

    return viz.visualize(
        mode=mode,
        save=save,
        output_path=output_path,
    )

def visualize_camera_lag(
    episode_path: str | Path,
    mode: str = "local",
    save: bool = False,
    output_path: str | Path | None = None,
    align: str = "log",
) -> Path | None:
    """Visualize camera pipeline lag from raw MCAP timestamps.

    Three per-camera lag metrics:
    - log_lag: inter-frame log_time delta (ms)
    - publish_lag: inter-frame publish_time delta (ms)
    - log_publish_delta: log_time - publish_time per frame (ms)

    Also shows camera streams and head joint state.
    Timeline uses real timestamps (log or publish) from the MCAP data.

    Args:
        episode_path: Path to episode directory.
        mode: 'local' spawns viewer, 'distant' creates server.
        save: Whether to save as .rrd file.
        output_path: Path for .rrd file when save=True.
        align: Which timestamp to use as the timeline: 'log' or 'publish'.

    Returns:
        Path to saved .rrd file if save=True, else None.
    """
    try:
        from .mcap_utils import TeleopMcapReader
    except ImportError:
        from mcap_utils import TeleopMcapReader

    episode_path = Path(episode_path)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode path not found: {episode_path}")

    if save and output_path is None:
        raise ValueError("output_path required when save=True")

    reader = TeleopMcapReader(episode_path)
    raw = reader.load_raw(decode_images=True)

    # Collect camera channels
    camera_channels = {k: v for k, v in raw.items() if k.startswith("camera/")}
    if not camera_channels:
        raise ValueError(f"No camera data found in {episode_path}")

    # Get state data for head joint
    state_channel = raw.get("state")

    # Common t0 for relative timestamps
    ts_key = "log_time_ns" if align == "log" else "publish_time_ns"
    all_t0 = [v[ts_key][0] for v in camera_channels.values()]
    if state_channel is not None:
        all_t0.append(state_channel[ts_key][0])
    t0 = min(all_t0)

    # Compute per-camera lag metrics and keep image data
    camera_data: dict[str, dict] = {}
    for channel_key, channel_data in camera_channels.items():
        camera_name = channel_key.removeprefix("camera/")
        log_ns = channel_data["log_time_ns"]
        pub_ns = channel_data["publish_time_ns"]

        # Timestamps for this camera (from index 1 to match diff arrays)
        timestamps_ns = channel_data[ts_key][1:]

        # log_lag (ms): time between consecutive log_time entries
        log_lag_ms = np.diff(log_ns) / 1e6

        # publish_lag (ms): time between consecutive publish_time entries
        pub_lag_ms = np.diff(pub_ns) / 1e6

        # log_publish_delta (ms): log_time - publish_time per frame
        lp_delta_ms = (log_ns[1:] - pub_ns[1:]) / 1e6

        camera_data[camera_name] = {
            "log_lag_ms": log_lag_ms,
            "publish_lag_ms": pub_lag_ms,
            "lp_delta_ms": lp_delta_ms,
            "timestamps_ns": timestamps_ns,
            "images": channel_data["data"][1:],
        }
        logger.info(
            f"  {camera_name}: {len(log_lag_ms)} frames, "
            f"lp_delta mean={np.mean(lp_delta_ms):.1f}ms, "
            f"log_lag mean={np.mean(log_lag_ms):.1f}ms, "
            f"pub_lag mean={np.mean(pub_lag_ms):.1f}ms"
        )

    # Initialize Rerun
    rr.init(f"{episode_path.name}_camera_lag", spawn=False)
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False)

    # Prepare state lag metrics
    state_lag_data = None
    state_timestamps_ns = None
    head_qpos = None
    if state_channel is not None:
        state_log_ns = state_channel["log_time_ns"]
        state_pub_ns = state_channel["publish_time_ns"]
        state_timestamps_ns = state_channel[ts_key]

        state_lag_data = {
            "log_lag_ms": np.diff(state_log_ns) / 1e6,
            "publish_lag_ms": np.diff(state_pub_ns) / 1e6,
            "lp_delta_ms": (state_log_ns[1:] - state_pub_ns[1:]) / 1e6,
            "timestamps_ns": state_channel[ts_key][1:],
        }
        logger.info(
            f"  state: {len(state_lag_data['log_lag_ms'])} frames, "
            f"lp_delta mean={np.mean(state_lag_data['lp_delta_ms']):.1f}ms, "
            f"log_lag mean={np.mean(state_lag_data['log_lag_ms']):.1f}ms, "
            f"pub_lag mean={np.mean(state_lag_data['publish_lag_ms']):.1f}ms"
        )

        states = state_channel["data"]
        head_arrays = [s.head.qpos for s in states if s.head is not None]
        if head_arrays:
            head_qpos = np.stack(head_arrays)
            logger.info(f"  head.qpos: {head_qpos.shape}")

    # Build blueprint: cameras left, camera lag + state lag + state plots right
    camera_views = [
        rrb.Spatial2DView(
            name=cam_name.replace(".rgb", ""),
            origin=f"camera/{cam_name}",
        )
        for cam_name in camera_data.keys()
    ]

    # Camera lag views
    cam_lag_views = [
        rrb.TimeSeriesView(name="Camera log_lag (ms)", origin="camera_lag/log_lag"),
        rrb.TimeSeriesView(
            name="Camera publish_lag (ms)", origin="camera_lag/publish_lag"
        ),
        rrb.TimeSeriesView(
            name="Camera log-publish delta (ms)", origin="camera_lag/lp_delta"
        ),
    ]

    # State lag views
    state_lag_views = []
    if state_lag_data is not None:
        state_lag_views = [
            rrb.TimeSeriesView(name="State log_lag (ms)", origin="state_lag/log_lag"),
            rrb.TimeSeriesView(
                name="State publish_lag (ms)", origin="state_lag/publish_lag"
            ),
            rrb.TimeSeriesView(
                name="State log-publish delta (ms)", origin="state_lag/lp_delta"
            ),
        ]

    state_views = []
    if head_qpos is not None:
        state_views.append(
            rrb.TimeSeriesView(name="head.qpos", origin="state/head_qpos")
        )

    n_cols = min(2, len(camera_views))
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Grid(*camera_views, grid_columns=n_cols),
            rrb.Vertical(*cam_lag_views, *state_lag_views, *state_views),
            column_shares=[2, 1],
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    # Log camera data at real timestamps
    for camera_name, cd in camera_data.items():
        timestamps_ns = cd["timestamps_ns"]
        for i in range(len(cd["log_lag_ms"])):
            elapsed_s = float(timestamps_ns[i] - t0) / 1e9
            rr.set_time("timestamp", duration=elapsed_s)
            rr.log(f"camera/{camera_name}", rr.Image(cd["images"][i]))
            rr.log(f"camera_lag/log_lag/{camera_name}", rr.Scalars(cd["log_lag_ms"][i]))
            rr.log(
                f"camera_lag/publish_lag/{camera_name}",
                rr.Scalars(cd["publish_lag_ms"][i]),
            )
            rr.log(
                f"camera_lag/lp_delta/{camera_name}", rr.Scalars(cd["lp_delta_ms"][i])
            )

    # Log state lag metrics
    if state_lag_data is not None:
        for i in range(len(state_lag_data["log_lag_ms"])):
            elapsed_s = float(state_lag_data["timestamps_ns"][i] - t0) / 1e9
            rr.set_time("timestamp", duration=elapsed_s)
            rr.log(
                "state_lag/log_lag/state", rr.Scalars(state_lag_data["log_lag_ms"][i])
            )
            rr.log(
                "state_lag/publish_lag/state",
                rr.Scalars(state_lag_data["publish_lag_ms"][i]),
            )
            rr.log(
                "state_lag/lp_delta/state", rr.Scalars(state_lag_data["lp_delta_ms"][i])
            )

    # Log head state at its own real timestamps
    if head_qpos is not None and state_timestamps_ns is not None:
        for i in range(len(head_qpos)):
            elapsed_s = float(state_timestamps_ns[i] - t0) / 1e9
            rr.set_time("timestamp", duration=elapsed_s)
            for dim_idx in range(head_qpos.shape[1]):
                rr.log(
                    f"state/head_qpos/{dim_idx}",
                    rr.Scalars(float(head_qpos[i, dim_idx])),
                )

    # Handle output
    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(output_path))
        logger.info(f"Saved to {output_path}")
        return output_path

    if mode == "local":
        import tempfile
        import subprocess

        temp_file = tempfile.NamedTemporaryFile(suffix=".rrd", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        rr.save(temp_path)
        logger.info("Opening camera lag visualization in Rerun viewer...")
        subprocess.Popen(
            ["rerun", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return None

    if mode == "distant":
        logger.info("Serving camera lag visualization...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted, exiting.")

    return None

def main():
    """CLI entry point for MCAP episode visualization."""
    import argparse
    import sys

    # Add project root to path for direct script execution
    script_dir = Path(__file__).resolve().parent
    project_root = (
        script_dir.parent.parent
    )  # Go up from internal/we_record/ to project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Visualize MCAP teleop episodes using Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize an episode (opens Rerun viewer)
  python mcap_viz.py /path/to/episode_0001_20260128_120000/
  
  # Save visualization to .rrd file
  python mcap_viz.py /path/to/episode/ --save --output viz.rrd
  
  # Start as distant server (for remote viewing)
  python mcap_viz.py /path/to/episode/ --mode distant
  
  # Load a previously saved .rrd file
  python mcap_viz.py /path/to/saved.rrd --load

  # Visualize camera pipeline lag
  python mcap_viz.py /path/to/episode/ --lag
        """,
    )

    parser.add_argument(
        "path", type=str, help="Path to episode directory or .rrd file (with --load)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="Visualization mode: 'local' spawns viewer, 'distant' creates server",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save visualization to .rrd file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .rrd file (required with --save)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames per second for timestamp calculation",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load a .rrd file instead of an episode directory",
    )
    parser.add_argument(
        "--lag",
        action="store_true",
        help="Visualize camera pipeline lag instead of episode data",
    )
    parser.add_argument(
        "--align",
        type=str,
        default="log",
        choices=["log", "publish"],
        help="Timestamp source for --lag timeline: 'log' (recorder wall-clock) or 'publish' (sensor capture)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    path = Path(args.path)

    if args.load:
        # Load existing .rrd file
        RerunViz.load(path)
    elif args.lag:
        # Visualize camera lag
        visualize_camera_lag(
            episode_path=path,
            mode=args.mode,
            save=args.save,
            output_path=args.output,
            align=args.align,
        )
    else:
        # Visualize episode
        if args.save and args.output is None:
            # Default output path
            args.output = path / "visualization.rrd"

        visualize_episode(
            episode_path=path,
            mode=args.mode,
            save=args.save,
            output_path=args.output,
            fps=args.fps,
        )

if __name__ == "__main__":
    main()
