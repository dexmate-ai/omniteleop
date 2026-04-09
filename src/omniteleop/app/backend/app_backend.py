#!/usr/bin/env python3
"""OmniTeleop App Backend.

Orchestrates the full teleoperation stack:
  - Launches leader readers (arm, joycon) and follower processes
    (command_processor, robot_controller) as managed subprocesses.
  - Streams camera images and robot state to the frontend via WebSocket.
  - Optionally runs the MDP recorder when --record-mode is set.

Usage:
    python -m omniteleop.app.backend.app_backend                 # teleop only
    python -m omniteleop.app.backend.app_backend --record-mode   # teleop + recording
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2  # type: ignore
import numpy as np
import tyro
from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter
from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger
import uvicorn

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging

from omniteleop.app.backend.state_checker import StateChecker
from omniteleop.app.backend.backend_utils.state_manager import state_manager
from omniteleop.app.backend.backend_utils.state_publisher import state_publisher
from omniteleop.app.backend.backend_utils.video_publisher import video_publisher

# ---------------------------------------------------------------------------
# Optional recorder import — only needed when --record-mode is active
# ---------------------------------------------------------------------------
try:
    from omniteleop.app.backend.mcap_utils import TeleopMcapWriter
    from omniteleop.app.backend.mcap_utils.episode import (
        RobotState,
        RobotAction,
        JointState,
        WrenchState,
    )

    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

class TeleopApp:
    """Manages the full teleop stack and exposes a FastAPI server to the frontend."""

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        rpi_mode: bool = False,
    ):
        self.namespace = namespace
        self.debug = debug
        self.rpi_mode = rpi_mode
        self.record_mode = False  # set per-launch from /teleop/start body

        self.config = get_config()
        recorder_cfg = self.config.get("recorder", {})

        # ---- paths & rates ------------------------------------------------
        save_dir_str = recorder_cfg.get("save_dir", "recordings")
        save_path = Path(save_dir_str)
        if not save_path.is_absolute():
            save_path = Path.home() / save_path
        self.save_dir = save_path
        self.episode_prefix = recorder_cfg.get("episode_prefix", "episode")
        self.record_rate = recorder_cfg.get("record_rate", 20.0)
        resolution = recorder_cfg.get("image_resolution", [640, 480])
        self.image_resolution = (
            tuple(resolution) if isinstance(resolution, list) else (640, 480)
        )
        self.auto_stop_on_estop = recorder_cfg.get("auto_stop_on_estop", True)

        # ---- component flags -----------------------------------------------
        # Load all component flags from config; any key present in the yaml is
        # respected. Keys not in the yaml default to True for the core joints,
        # False for optional sensors (wrist cameras/wrenches).
        components_cfg = recorder_cfg.get("components", {})
        _defaults: Dict[str, bool] = {
            "left_arm": True,
            "right_arm": True,
            "torso": True,
            "head": True,
            "left_hand": True,
            "right_hand": True,
            "head_left_rgb": True,
            "head_right_rgb": True,
            "left_wrist_rgb": False,
            "right_wrist_rgb": False,
            "left_wrist_wrench": False,
            "right_wrist_wrench": False,
        }
        self.record_components: Dict[str, bool] = {
            **_defaults,
            **{k: bool(v) for k, v in components_cfg.items()},
        }
        self.save_images = any(
            self.record_components.get(k)
            for k in [
                "head_left_rgb",
                "head_right_rgb",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ]
        )

        # ---- subprocess handles --------------------------------------------
        self._procs: List[subprocess.Popen] = []
        self._proc_names: List[str] = []  # parallel name list for each proc
        self._teleop_running = False  # True once start_teleop() succeeds
        self._teleop_env: Dict[str, str] = {}  # env passed from frontend at start

        # ---- per-process log ring-buffers & subscribers -------------------
        # _proc_logs[name] = deque of {"source": name, "line": text}
        self._proc_logs: Dict[str, List[Dict[str, str]]] = {}
        self._log_lock = threading.Lock()
        self._log_subscribers: List[asyncio.Queue] = []
        self._log_loop: Optional[asyncio.AbstractEventLoop] = None

        # ---- robot / state -------------------------------------------------
        self.robot: Optional[Robot] = None
        self._node = Node(name="app_backend", namespace=namespace)

        self.state_checker = None  # initialized in _start_teleop_procs

        self.robot_state = "BOOT"
        self._robot_state_lock = threading.Lock()
        self._state_checker_thread: Optional[threading.Thread] = None
        self._state_checker_running = False

        self.frontend_ready = False

        # ---- recording state (only meaningful when record_mode=True) -------
        self.is_recording = False
        self.episode_dir: Optional[Path] = None
        self.mcap_writer = None
        self.episode_start_time: float = 0.0
        self.transitions_in_episode: int = 0
        self.total_episodes: int = 0
        self.total_transitions: int = 0
        self._record_thread: Optional[threading.Thread] = None
        self._record_running = False
        self._action_lock = threading.Lock()
        self.current_action: Dict[str, Any] = {}
        self.last_action: Dict[str, Any] = {}
        self._component_last_seen: Dict[
            str, float
        ] = {}  # epoch time of last command per component
        self._active_component_window = 0.5  # seconds
        self._latest_robot_joints: Dict[str, Any] = {}  # from robot/joints topic
        self._rate_limiter = RateLimiter(self.record_rate)

        # ---- FastAPI -------------------------------------------------------
        self.app = FastAPI(title="OmniTeleop App Backend")
        self._setup_fastapi()

    # =========================================================================
    # FastAPI setup
    # =========================================================================

    def _setup_fastapi(self):
        app = self.app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # ------------------------------------------------------------------
        # State callback fed into state_publisher
        # ------------------------------------------------------------------
        async def build_state_dict() -> Dict[str, Any]:
            if not self.frontend_ready:
                return {
                    "timestamp_posix": time.time(),
                    "robots": {
                        "robot": [
                            {
                                "topic": "control_state",
                                "data_type": "string",
                                "data": "BOOT",
                            },
                            {"topic": "episode_id", "data_type": "string", "data": ""},
                            {
                                "topic": "error_state",
                                "data_type": "json",
                                "data": {
                                    "state": "BOOT",
                                    "message": "Waiting for frontend…",
                                },
                            },
                        ]
                    },
                }

            is_recording = self.is_recording
            episode_id = (
                str(self.episode_dir.name)
                if (is_recording and self.episode_dir)
                else None
            )

            is_estop = await state_manager.is_estop()

            with self._robot_state_lock:
                robot_state = self.robot_state

            # Compute unified control_state
            if is_estop:
                control_state = "PAUSE"
            elif is_recording:
                control_state = "RECORD"
            elif robot_state == "BOOT":
                control_state = "BOOT"
            elif robot_state == "DIAGNOSIS":
                control_state = "DIAGNOSIS"
            elif robot_state == "ALIGN":
                control_state = "ALIGN"
            elif robot_state == "ACTIVE":
                control_state = "ACTIVE"
            else:
                control_state = "BOOT"

            # Robot observations
            robot_states = [
                {
                    "topic": "control_state",
                    "data_type": "string",
                    "data": control_state,
                },
                {
                    "topic": "episode_id",
                    "data_type": "string",
                    "data": episode_id or "",
                },
                {
                    "topic": "record_mode",
                    "data_type": "boolean",
                    "data": self.record_mode,
                },
                {
                    "topic": "teleop_running",
                    "data_type": "boolean",
                    "data": self._teleop_running,
                },
            ]

            if self.robot:
                try:
                    obs = self._get_robot_state()
                    for comp, positions in obs.get("joint_pos", {}).items():
                        robot_states.append(
                            {
                                "topic": f"observation/state/{comp}",
                                "data_type": "array",
                                "data": positions,
                            }
                        )
                except Exception:
                    pass

            if self.state_checker:
                try:
                    exo = self.state_checker.get_latest_exo_joints()
                    if exo:
                        for side in ("left", "right"):
                            joints = exo.get(side, [])
                            if joints:
                                robot_states.append(
                                    {
                                        "topic": f"observation/exo/{side}_arm",
                                        "data_type": "array",
                                        "data": joints,
                                    }
                                )
                except Exception:
                    pass

            # Diagnostics / error_state
            if self.state_checker:
                if control_state == "BOOT":
                    topic_info = self.state_checker.get_missing_topics()
                    error_data = {
                        "state": "BOOT",
                        "message": "Required topics not available",
                        "missing_topics": topic_info["missing"],
                        "found_topics": topic_info["found"],
                    }
                elif control_state == "DIAGNOSIS":
                    diag = self.state_checker.get_diagnosis_details()
                    error_data = {
                        "state": "DIAGNOSIS",
                        "message": "Robot not ready — diagnostic checks failed",
                        "diagnostic_checks": diag,
                    }
                    if diag.get("no_component_errors") is False:
                        errors, _ = self.state_checker.get_component_errors()
                        if errors:
                            error_data["component_errors"] = errors
                elif control_state == "ALIGN":
                    error_data = self.state_checker.get_out_of_limit_joints()
                    error_data["state"] = "ALIGN"
                else:
                    error_data = {
                        "state": control_state,
                        "message": "Robot is running"
                        if control_state == "ACTIVE"
                        else control_state,
                    }

                robot_states.append(
                    {"topic": "error_state", "data_type": "json", "data": error_data}
                )
            else:
                robot_states.append(
                    {
                        "topic": "error_state",
                        "data_type": "json",
                        "data": {"state": control_state, "message": "No state checker"},
                    }
                )

            robot_states.append(
                {
                    "topic": "estop_state",
                    "data_type": "json",
                    "data": {"software_estop": is_estop},
                }
            )

            with self._action_lock:
                cutoff = time.time() - self._active_component_window
                active_components = [
                    c for c, t in self._component_last_seen.items() if t >= cutoff
                ]
            robot_states.append(
                {
                    "topic": "active_components",
                    "data_type": "json",
                    "data": active_components,
                }
            )

            return {
                "timestamp_posix": time.time(),
                "robots": {"robot": robot_states},
            }

        @app.on_event("startup")
        async def _capture_loop():
            self._log_loop = asyncio.get_running_loop()

            # Sink backend's own loguru output into the log stream
            def _backend_log_sink(msg):
                entry = {"source": "app_backend", "line": msg.rstrip("\n")}
                with self._log_lock:
                    buf = self._proc_logs.setdefault("app_backend", [])
                    buf.append(entry)
                    if len(buf) > 500:
                        buf.pop(0)
                    subs = list(self._log_subscribers)
                loop = self._log_loop
                if loop and loop.is_running():
                    for q in subs:
                        try:
                            loop.call_soon_threadsafe(q.put_nowait, entry)
                        except Exception:
                            pass

            logger.add(_backend_log_sink, format="{level}: {message}")

        state_publisher.set_state_callback(build_state_dict)
        state_publisher.set_connection_callback(self.on_frontend_connected)
        state_publisher.set_disconnection_callback(self.on_frontend_disconnected)

        # ------------------------------------------------------------------
        # REST endpoints
        # ------------------------------------------------------------------

        @app.get("/health")
        async def health():
            return {"status": "ok", "timestamp": time.time()}

        @app.get("/state")
        async def get_state():
            return JSONResponse(content=await build_state_dict())

        @app.get("/sensors")
        async def get_sensors():
            sensors = [
                {"id": "camera_1", "data_type": "video/x-motion-jpeg"},
                {"id": "camera_2", "data_type": "video/x-motion-jpeg"},
            ]
            return JSONResponse(content=sensors)

        # -- Teleop controls ------------------------------------------------

        @app.post("/teleop/start")
        async def teleop_start(request: Request):
            """Launch the teleop subprocess stack.

            Body (all fields optional):
            {
                "record_mode": false,
                "env": {
                    "ROBOT_NAME": "...",
                    "ROBOT_CONFIG": "vega_1_f5d6",
                    "ZENOH_CONFIG": "/path/to/zenoh.json"
                }
            }
            """
            if self._teleop_running:
                raise HTTPException(status_code=409, detail="Teleop already running")
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Cannot start: E-Stop is active"
                )
            try:
                body = await request.json()
            except Exception:
                body = {}
            self.record_mode = bool(body.get("record_mode", False))
            user_env: Dict[str, str] = {
                k: str(v) for k, v in (body.get("env") or {}).items() if v
            }
            self._start_teleop_procs(user_env)
            return JSONResponse(content=await build_state_dict())

        @app.post("/teleop/stop")
        async def teleop_stop():
            """Stop the teleop subprocess stack."""
            if self.is_recording:
                self._end_episode(is_success=False)
            self._stop_teleop_procs()
            return JSONResponse(content=await build_state_dict())

        # -- Recording controls (only active when record_mode was set at /teleop/start) --

        @app.post("/record/start")
        async def record_start(request: Request):
            if not self.record_mode:
                raise HTTPException(
                    status_code=403,
                    detail="Record mode not enabled — set record_mode=true in /teleop/start",
                )
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Robot is in emergency stop"
                )
            if self.is_recording:
                raise HTTPException(status_code=409, detail="Already recording")
            try:
                body = await request.json()
            except Exception:
                body = {}
            self._start_episode(metadata=body.get("metadata"))
            return JSONResponse(content=await build_state_dict())

        @app.post("/record/stop")
        async def record_stop(request: Request):
            if not self.record_mode:
                raise HTTPException(status_code=403, detail="Record mode not enabled")
            if not self.is_recording:
                raise HTTPException(status_code=409, detail="Not currently recording")
            try:
                body = await request.json()
                is_success = body.get("is_success", True)
            except Exception:
                is_success = True
            try:
                episode_id = self._end_episode(is_success=is_success)
                return JSONResponse(
                    content={
                        "timestamp_posix": time.time(),
                        "is_success": is_success,
                        "episode_id": episode_id or "",
                    },
                    status_code=201,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        # -- E-Stop ----------------------------------------------------------

        @app.post("/robots/estop")
        async def robots_estop():
            await state_manager.estop()
            await video_publisher.stop_all_streams()
            if self.is_recording:
                self._end_episode(is_success=False)
            if self.robot and hasattr(self.robot, "estop"):
                try:
                    self.robot.estop.activate()
                except Exception as exc:
                    logger.error(f"Hardware estop failed: {exc}")
            return JSONResponse(content=await build_state_dict())

        @app.delete("/robots/estop")
        async def robots_estop_clear():
            if not await state_manager.is_estop():
                raise HTTPException(status_code=409, detail="Not in emergency stop")
            await state_manager.clear_estop()
            if self.robot and hasattr(self.robot, "estop"):
                try:
                    self.robot.estop.deactivate()
                except Exception as exc:
                    logger.error(f"Hardware estop clear failed: {exc}")
            return JSONResponse(content=await build_state_dict())

        # -- WebSocket endpoints --------------------------------------------

        @app.websocket("/ws/state")
        async def ws_state(websocket: WebSocket):
            await state_publisher.start_stream(websocket)

        @app.websocket("/ws/sensors/stream/{camera_id}")
        async def ws_camera(websocket: WebSocket, camera_id: str):
            if await state_manager.is_estop():
                await websocket.close(code=1008, reason="E-Stop active")
                return
            await video_publisher.start_stream(websocket, camera_id)

        @app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket):
            """Stream per-process log lines as JSON: {"source": "omni-arm", "line": "..."}"""
            await websocket.accept()
            import json as _json

            q: asyncio.Queue = asyncio.Queue(maxsize=500)
            with self._log_lock:
                # Send buffered history first
                history = []
                for name, lines in self._proc_logs.items():
                    history.extend(lines[-200:])
                self._log_subscribers.append(q)
            try:
                for entry in history:
                    await websocket.send_text(_json.dumps(entry))
                while True:
                    entry = await q.get()
                    await websocket.send_text(_json.dumps(entry))
            except Exception:
                pass
            finally:
                with self._log_lock:
                    try:
                        self._log_subscribers.remove(q)
                    except ValueError:
                        pass

        # -- Serve built frontend if present --------------------------------
        frontend_path = Path("/var/www/html")
        if frontend_path.exists() and (frontend_path / "index.html").exists():

            @app.get("/{full_path:path}")
            async def serve_frontend(full_path: str):
                skip = ["state", "sensors", "health"]
                skip_prefixes = ("robots/", "record/", "teleop/", "ws/")
                if full_path in skip or full_path.startswith(skip_prefixes):
                    raise HTTPException(status_code=404)
                file_path = frontend_path / full_path
                if file_path.exists() and file_path.is_file():
                    return FileResponse(str(file_path))
                return FileResponse(str(frontend_path / "index.html"))

    # =========================================================================
    # Frontend connection lifecycle
    # =========================================================================

    async def on_frontend_connected(self):
        await asyncio.sleep(1)
        self.frontend_ready = True
        logger.info("Frontend ready")

    def on_frontend_disconnected(self):
        self.frontend_ready = False
        logger.info("Frontend disconnected")

    # =========================================================================
    # Teleop subprocess management
    # =========================================================================

    def _start_teleop_procs(self, user_env: Dict[str, str]):
        """Spawn all teleop subprocesses with the given env overrides."""
        if self._teleop_running:
            return

        env = os.environ.copy()

        # --- Resolve ROBOT_NAME ---
        if not user_env.get("ROBOT_NAME"):
            # Fall back to whatever is already in the process environment
            user_env.pop("ROBOT_NAME", None)  # let os.environ value pass through

        # --- Resolve ROBOT_CONFIG ---
        if not user_env.get("ROBOT_CONFIG"):
            user_env["ROBOT_CONFIG"] = "vega_1p_gripper"

        # --- Resolve ZENOH_CONFIG ---
        zenoh_val = user_env.get("ZENOH_CONFIG", "").strip()
        zenoh_base = Path.home() / ".dexmate" / "comm" / "zenoh"
        if zenoh_val and not zenoh_val.startswith("/"):
            # Treat as certificate name — expand to full path
            user_env["ZENOH_CONFIG"] = str(
                zenoh_base / zenoh_val / "zenoh_peer_config.json5"
            )
        elif not zenoh_val:
            # Auto-detect: pick the first directory in the zenoh certs folder
            try:
                certs = sorted(p for p in zenoh_base.iterdir() if p.is_dir())
                if certs:
                    user_env["ZENOH_CONFIG"] = str(certs[0] / "zenoh_peer_config.json5")
                    logger.info(
                        f"Auto-detected ZENOH_CONFIG: {user_env['ZENOH_CONFIG']}"
                    )
                else:
                    logger.warning(
                        f"No zenoh cert directories found under {zenoh_base}"
                    )
            except FileNotFoundError:
                logger.warning(f"Zenoh certs directory not found: {zenoh_base}")

        env.update(user_env)
        self._teleop_env = user_env  # store so we can report it in state

        # Re-init StateChecker with ROBOT_NAME from env if provided
        robot_name = env.get("ROBOT_NAME", "")
        try:
            if self.state_checker:
                self._state_checker_running = False
                self.state_checker.cleanup()
            self.state_checker = StateChecker(robot_name, rpi_mode=self.rpi_mode)
            self._state_checker_running = True
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()
            logger.info(f"StateChecker initialised (robot={robot_name!r})")
        except Exception as exc:
            logger.warning(f"StateChecker init failed: {exc}")

        _src = Path(__file__).parents[2]
        ns_args = ["--namespace", self.namespace] if self.namespace else []
        debug_args = ["--debug"] if self.debug else []
        commands = [
            [
                sys.executable,
                str(_src / "follower" / "command_processor.py"),
                *ns_args,
                *debug_args,
            ],
            [
                sys.executable,
                str(_src / "follower" / "robot_controller.py"),
                *ns_args,
                *debug_args,
                "--interpolation-method",
                "linear",
            ],
        ]
        if not self.rpi_mode:
            commands = [
                [
                    sys.executable,
                    str(_src / "leader" / "arm_reader.py"),
                    *ns_args,
                    *debug_args,
                ],
                [
                    sys.executable,
                    str(_src / "leader" / "joycon_reader.py"),
                    *ns_args,
                    *debug_args,
                ],
            ] + commands
        else:
            logger.info("RPi mode: skipping leader scripts (arm_reader, joycon_reader)")

        for cmd in commands:
            name = Path(cmd[1]).stem  # e.g. "arm_reader" from "/path/to/arm_reader.py"
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self._procs.append(proc)
                self._proc_names.append(name)
                with self._log_lock:
                    self._proc_logs[name] = []
                threading.Thread(
                    target=self._tail_proc,
                    args=(proc, name),
                    daemon=True,
                    name=f"log-{name}",
                ).start()
                logger.info(f"Started: {' '.join(cmd)}  (pid={proc.pid})")
            except FileNotFoundError:
                logger.error(f"Command not found: {name} — is omniteleop installed?")

        self._teleop_running = True

        # Subscribe to robot joint feedback published by robot_controller
        joints_topic = self.config.get_topic("robot_joints")
        self._node.create_subscriber(
            joints_topic,
            callback=self._on_joints_received,
            decoder=DictDataCodec.decode,
        )

        # Subscribe to robot commands — always (estop sync) + record mode (action tracking)
        commands_topic = self.config.get_topic("robot_commands")
        self._node.create_subscriber(
            commands_topic,
            callback=self._on_command_received,
            decoder=DictDataCodec.decode,
        )

        mode_str = " [record mode]" if self.record_mode else ""
        logger.success(f"Teleop stack started{mode_str}")

    def _stop_teleop_procs(self):
        """Terminate all teleop subprocesses and their entire process groups."""
        import signal

        for proc in self._procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass  # already dead
        # Wait then force-kill any stragglers
        for proc in self._procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        self._procs.clear()
        self._proc_names.clear()
        self._state_checker_running = False
        self.robot_state = "BOOT"
        self._teleop_running = False
        self.record_mode = False
        self._teleop_env = {}
        logger.info("Teleop stack stopped")

    def _tail_proc(self, proc: subprocess.Popen, name: str):
        """Read stdout/stderr of a subprocess and fan-out to log subscribers."""
        MAX_LINES = 500
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.rstrip("\n")
                entry = {"source": name, "line": line}
                with self._log_lock:
                    buf = self._proc_logs.get(name)
                    if buf is not None:
                        buf.append(entry)
                        if len(buf) > MAX_LINES:
                            buf.pop(0)
                    subs = list(self._log_subscribers)
                loop = self._log_loop
                if loop and loop.is_running():
                    for q in subs:
                        try:
                            loop.call_soon_threadsafe(q.put_nowait, entry)
                        except Exception:
                            pass
        except Exception:
            pass

    # =========================================================================
    # Robot / StateChecker background loop
    # =========================================================================

    def initialize(self):
        """Initialize robot interface, state-checker loop, and video publisher."""
        logger.info("Initializing robot interface…")
        try:
            robot_configs = get_robot_config()
            robot_configs.sensors["head_camera"].enabled = True
            if (
                self.record_components.get("left_wrist_rgb")
                and "left_wrist_camera" in robot_configs.sensors
            ):
                robot_configs.sensors["left_wrist_camera"].enabled = True
            if (
                self.record_components.get("right_wrist_rgb")
                and "right_wrist_camera" in robot_configs.sensors
            ):
                robot_configs.sensors["right_wrist_camera"].enabled = True
            self.robot = Robot(configs=robot_configs)

            if self.save_images:
                if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
                    logger.success("Camera streams active")
                else:
                    logger.warning("Camera streams may not be active")

            logger.success("Robot interface ready")
        except Exception as exc:
            logger.error(f"Robot init failed: {exc}")
            self.robot = None

        if self.state_checker:
            self._state_checker_running = True
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()

        if self.robot:

            def get_frame(camera_id: str) -> bytes:
                try:
                    camera_map = {
                        "camera_1": ("head_camera", "left_rgb"),
                        "camera_2": ("head_camera", "right_rgb"),
                    }
                    if camera_id not in camera_map:
                        return b""
                    sensor_name, obs_key = camera_map[camera_id]
                    sensor = getattr(self.robot.sensors, sensor_name, None)
                    if not sensor:
                        return b""
                    obs = (
                        sensor.get_obs(obs_keys=[obs_key])
                        if obs_key
                        else sensor.get_obs()
                    )
                    img = obs.get(obs_key) if isinstance(obs, dict) and obs_key else obs
                    if isinstance(img, dict):
                        img = img.get("data", img)
                    if not isinstance(img, np.ndarray):
                        return b""
                    if (
                        self.image_resolution
                        and img.shape[:2] != self.image_resolution[::-1]
                    ):
                        img = cv2.resize(img, self.image_resolution)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    return buf.tobytes()
                except Exception:
                    return b""

            video_publisher.set_frame_callback(get_frame)

    def _state_checker_loop(self):
        while self._state_checker_running and self.state_checker:
            try:
                if self.robot:
                    try:
                        left = self.robot.left_arm.get_joint_pos().tolist()
                        right = self.robot.right_arm.get_joint_pos().tolist()
                        self.state_checker.update_robot_joints(left, right)
                    except Exception:
                        pass
                robot_state = self.state_checker.get_state()
                with self._robot_state_lock:
                    self.robot_state = robot_state
            except Exception as exc:
                logger.error(f"StateChecker loop error: {exc}")
                with self._robot_state_lock:
                    self.robot_state = "ERROR"

    # =========================================================================
    # Robot state observation helpers
    # =========================================================================

    def _get_robot_state(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"joint_pos": {}, "joint_vel": {}, "wrench": {}}
        rc = self.record_components

        # Use joint feedback published by robot_controller (robot/joints topic).
        # Falls back to direct hardware read only if no feedback has arrived yet.
        joints_data = self._latest_robot_joints
        joint_positions = joints_data.get("joints", {})
        joint_velocities = joints_data.get("velocities", {})

        if joint_positions:
            for comp in [
                "left_arm",
                "right_arm",
                "torso",
                "head",
                "left_hand",
                "right_hand",
            ]:
                if rc.get(comp) and comp in joint_positions:
                    obs["joint_pos"][comp] = joint_positions[comp]
            for comp in ["left_arm", "right_arm"]:
                if rc.get(comp) and comp in joint_velocities:
                    obs["joint_vel"][comp] = joint_velocities[comp]
        elif self.robot:
            # Fallback: direct hardware read (before robot_controller starts publishing)
            for comp, getter in [
                ("left_arm", lambda: self.robot.left_arm.get_joint_pos().tolist()),
                ("right_arm", lambda: self.robot.right_arm.get_joint_pos().tolist()),
            ]:
                if rc.get(comp):
                    try:
                        obs["joint_pos"][comp] = getter()
                    except Exception:
                        pass
        if self.robot:
            if rc.get("left_wrist_wrench") and self.robot.left_arm.wrench_sensor:
                w = self.robot.left_arm.wrench_sensor.get_wrench_state()
                obs["wrench"]["left"] = {
                    "force": w[:3].tolist(),
                    "torque": w[3:].tolist(),
                }
            if rc.get("right_wrist_wrench") and self.robot.right_arm.wrench_sensor:
                w = self.robot.right_arm.wrench_sensor.get_wrench_state()
                obs["wrench"]["right"] = {
                    "force": w[:3].tolist(),
                    "torque": w[3:].tolist(),
                }
        return obs

    # =========================================================================
    # Recording (only active when record_mode=True)
    # =========================================================================

    def _on_joints_received(self, data: Dict[str, Any]):
        self._latest_robot_joints = data

    def _on_command_received(self, data: Dict[str, Any]):
        now = time.time()
        with self._action_lock:
            for comp, comp_data in data.get("components", {}).items():
                self.current_action[comp] = comp_data
                self._component_last_seen[comp] = now

        estop_active = data.get("safety_flags", {}).get("emergency_stop", False)

        # Sync estop state from command_processor into state_manager so the GUI reflects it
        loop = self._log_loop
        if loop and loop.is_running():
            if estop_active:
                asyncio.run_coroutine_threadsafe(state_manager.estop(), loop)
            else:
                asyncio.run_coroutine_threadsafe(state_manager.clear_estop(), loop)

        if self.auto_stop_on_estop and self.is_recording:
            if estop_active:
                self._end_episode(is_success=False)

    def _get_next_episode_num(self) -> int:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.save_dir.glob(f"{self.episode_prefix}_*.json"))
        numbers = []
        for ep in existing:
            parts = ep.stem.split("_")
            try:
                numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                pass
        return max(numbers, default=-1) + 1

    def _start_episode(self, metadata: Optional[Dict[str, Any]] = None):
        import random

        if not RECORDER_AVAILABLE:
            raise RuntimeError("mcap_utils not available — cannot record")

        save_dir = Path(os.environ.get("RECORDER_SAVE_DIR", "") or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = random.randint(100000, 999999)
        episode_name = f"episode_{ts}_{suffix}"
        self.episode_dir = save_dir / episode_name

        meta = metadata or {}
        self.mcap_writer = TeleopMcapWriter(
            output_path=self.episode_dir,
            task_name=meta.get("task_name", "teleop_recording"),
            task_language_instruction=meta.get(
                "task_instruction", "Teleoperation recording"
            ),
            operator=meta.get("operator_name", "unknown"),
            record_rate_hz=self.record_rate,
            init_pos=self.config.get("init_pos") or None,
            record_components=self.record_components,
        )
        self.is_recording = True
        self.transitions_in_episode = 0
        self.episode_start_time = time.time()
        with self._action_lock:
            self.current_action.clear()
            self.last_action.clear()

        self._record_running = True
        self._record_thread = threading.Thread(
            target=self._record_loop, daemon=True, name="RecordLoop"
        )
        self._record_thread.start()
        logger.success(f"Recording started → {self.episode_dir.name}")

    def _record_loop(self):
        try:
            while self._record_running and self.is_recording:
                ts_ns = time.time_ns()
                self._write_state(ts_ns)
                self._write_action(ts_ns)
                self._write_cameras(ts_ns)
                self.transitions_in_episode += 1
                self.total_transitions += 1
                self._rate_limiter.sleep()
        except Exception as exc:
            logger.error(f"Record loop crashed: {exc}", exc_info=True)
            self.is_recording = False

    def _write_state(self, ts_ns: int):
        if not self.robot or not self.mcap_writer:
            return
        obs = self._get_robot_state()
        jp = obs.get("joint_pos", {})
        jv = obs.get("joint_vel", {})
        rc = self.record_components

        def js(comp, vel=False):
            if comp not in jp:
                return None
            qpos = np.array(jp[comp], dtype=np.float32)
            qvel = np.array(jv[comp], dtype=np.float32) if vel and comp in jv else None
            return JointState(qpos=qpos, qvel=qvel)

        state_obj = RobotState(
            left_arm=js("left_arm", vel=True) if rc.get("left_arm") else None,
            right_arm=js("right_arm", vel=True) if rc.get("right_arm") else None,
            left_hand=js("left_hand") if rc.get("left_hand") else None,
            right_hand=js("right_hand") if rc.get("right_hand") else None,
            head=js("head") if rc.get("head") else None,
            torso=js("torso") if rc.get("torso") else None,
            left_wrist_wrench=WrenchState(
                force=np.array(obs["wrench"]["left"]["force"], dtype=np.float32),
                torque=np.array(obs["wrench"]["left"]["torque"], dtype=np.float32),
            )
            if "left" in obs.get("wrench", {}) and rc.get("left_wrist_wrench")
            else None,
            right_wrist_wrench=WrenchState(
                force=np.array(obs["wrench"]["right"]["force"], dtype=np.float32),
                torque=np.array(obs["wrench"]["right"]["torque"], dtype=np.float32),
            )
            if "right" in obs.get("wrench", {}) and rc.get("right_wrist_wrench")
            else None,
            timestamp_ns=ts_ns,
        )
        self.mcap_writer.write_state(state_obj, ts_ns)

    def _write_action(self, ts_ns: int):
        if not self.mcap_writer:
            return
        obs = self._get_robot_state() if self.robot else {}
        jp = obs.get("joint_pos", {})
        resolved: Dict[str, Any] = {}
        joint_comps = [
            "left_arm",
            "right_arm",
            "torso",
            "head",
            "left_hand",
            "right_hand",
        ]
        rc = self.record_components

        with self._action_lock:
            for comp in joint_comps:
                if not rc.get(comp):
                    continue
                if comp in self.current_action:
                    resolved[comp] = self.current_action[comp]
                elif comp in self.last_action:
                    resolved[comp] = self.last_action[comp]
                elif comp in jp:
                    resolved[comp] = {"pos": jp[comp]}
                if comp in resolved:
                    self.last_action[comp] = resolved[comp]
            if "chassis" in self.current_action:
                resolved["chassis"] = self.current_action["chassis"]
                self.last_action["chassis"] = self.current_action["chassis"]
            elif "chassis" in self.last_action:
                resolved["chassis"] = self.last_action["chassis"]

        if not resolved:
            return

        def qpos(comp):
            d = resolved.get(comp)
            return np.array(d["pos"], dtype=np.float32) if d and "pos" in d else None

        chassis = resolved.get("chassis", {})
        action_obj = RobotAction(
            left_arm=JointState(qpos=qpos("left_arm"))
            if qpos("left_arm") is not None and rc.get("left_arm")
            else None,
            right_arm=JointState(qpos=qpos("right_arm"))
            if qpos("right_arm") is not None and rc.get("right_arm")
            else None,
            left_hand=JointState(qpos=qpos("left_hand"))
            if qpos("left_hand") is not None and rc.get("left_hand")
            else None,
            right_hand=JointState(qpos=qpos("right_hand"))
            if qpos("right_hand") is not None and rc.get("right_hand")
            else None,
            head=JointState(qpos=qpos("head"))
            if qpos("head") is not None and rc.get("head")
            else None,
            torso=JointState(qpos=qpos("torso"))
            if qpos("torso") is not None and rc.get("torso")
            else None,
            chassis_vx=np.array([chassis["vx"]], dtype=np.float32)
            if "vx" in chassis
            else None,
            chassis_vy=np.array([chassis["vy"]], dtype=np.float32)
            if "vy" in chassis
            else None,
            chassis_wz=np.array([chassis["wz"]], dtype=np.float32)
            if "wz" in chassis
            else None,
            timestamp_ns=ts_ns,
        )
        self.mcap_writer.write_action(action_obj, ts_ns)

    def _write_cameras(self, ts_ns: int):
        if not self.save_images or not self.robot or not self.mcap_writer:
            return
        camera_name_map = {
            "head_left_rgb": "head_left.rgb",
            "head_right_rgb": "head_right.rgb",
            "left_wrist_rgb": "left_wrist.rgb",
            "right_wrist_rgb": "right_wrist.rgb",
        }
        imgs: dict = {}
        head_keys = []
        if self.record_components["head_left_rgb"]:
            head_keys.append("left_rgb")
        if self.record_components["head_right_rgb"]:
            head_keys.append("right_rgb")
        if head_keys:
            head_imgs = self.robot.sensors.head_camera.get_obs(obs_keys=head_keys)
            imgs.update({f"head_{k}": v for k, v in head_imgs.items()})
        if self.record_components["left_wrist_rgb"]:
            if self.robot.has_sensor("left_wrist_camera"):
                imgs["left_wrist_rgb"] = self.robot.sensors.left_wrist_camera.get_obs()
            else:
                logger.warning(
                    "left_wrist_rgb enabled in config but left_wrist_camera sensor not available"
                )
        if self.record_components["right_wrist_rgb"]:
            if self.robot.has_sensor("right_wrist_camera"):
                imgs["right_wrist_rgb"] = (
                    self.robot.sensors.right_wrist_camera.get_obs()
                )
            else:
                logger.warning(
                    "right_wrist_rgb enabled in config but right_wrist_camera sensor not available"
                )
        if not imgs:
            return
        for key, val in imgs.items():
            if val is None:
                continue
            img = val.get("data") if isinstance(val, dict) else val
            if img is None:
                continue
            if self.image_resolution:
                img = cv2.resize(img, self.image_resolution)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.mcap_writer.write_camera(img, camera_name_map.get(key, key), ts_ns)

    def _end_episode(self, is_success: bool = True) -> Optional[str]:
        import json

        if not self.is_recording:
            return None
        self._record_running = False
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        self.is_recording = False
        duration = time.time() - self.episode_start_time
        avg_rate = self.transitions_in_episode / duration if duration > 0 else 0

        if self.mcap_writer:
            self.mcap_writer.close(
                success=is_success,
                extra_metadata={"avg_rate": avg_rate, "is_success": is_success},
            )
            self.mcap_writer = None

        if self.episode_dir:
            meta_path = self.episode_dir / "collection-metadata.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "is_success": is_success,
                        "episode_id": self.episode_dir.name,
                        "duration_seconds": duration,
                        "transitions": self.transitions_in_episode,
                        "avg_rate_hz": avg_rate,
                    },
                    f,
                    indent=2,
                )
            state_path = self.episode_dir / ".state.json"
            with open(state_path, "w") as f:
                json.dump({"status": "complete", "is_success": is_success}, f)
            self._fix_ownership(self.episode_dir)

        episode_id = self.episode_dir.name if self.episode_dir else None
        status = "saved" if is_success else "discarded"
        logger.info(
            f"Episode {status}: {episode_id}  ({self.transitions_in_episode} steps, {duration:.1f}s)"
        )
        if is_success:
            self.total_episodes += 1
        self.transitions_in_episode = 0
        return episode_id

    def _fix_ownership(self, path: Path):
        import subprocess as _sp

        if os.getuid() != 0:
            return
        uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
        if uid and gid:
            try:
                _sp.run(
                    ["chown", "-R", f"{uid}:{gid}", str(path)],
                    check=True,
                    capture_output=True,
                )
            except _sp.CalledProcessError:
                pass

    # =========================================================================
    # Run
    # =========================================================================

    def cleanup(self):
        self._state_checker_running = False
        if self._state_checker_thread:
            self._state_checker_thread.join(timeout=2.0)
        if self.state_checker:
            try:
                self.state_checker.cleanup()
            except Exception:
                pass
        if self.is_recording:
            self._end_episode(is_success=False)
        self._stop_teleop_procs()
        if self.robot:
            self.robot.shutdown()
        if hasattr(self._node, "shutdown"):
            self._node.shutdown()

    def run(self):
        import os

        certs_dir = os.path.join(os.path.dirname(__file__), "..", "certs")
        cert_file = os.path.join(certs_dir, "cert.pem")
        key_file = os.path.join(certs_dir, "key.pem")
        ssl_kwargs = {}
        if os.path.isfile(cert_file) and os.path.isfile(key_file):
            ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        uvicorn.run(
            self.app, host="0.0.0.0", port=5006, log_level="warning", **ssl_kwargs
        )

# =============================================================================
# Entry point
# =============================================================================

def main(
    namespace: str = "",
    debug: bool = False,
    rpi_mode: bool = False,
):
    """OmniTeleop App Backend.

    Launches the FastAPI server. Teleoperation processes are started via
    POST /teleop/start from the frontend, which also passes env vars
    (ROBOT_NAME, ROBOT_CONFIG, ZENOH_CONFIG) and the record_mode flag.

    Args:
        namespace: Optional Zenoh namespace prefix.
        debug: Enable verbose logging.
        rpi_mode: RPi mode — leader scripts (arm_reader, joycon_reader) are
            assumed to already be running on a remote machine and will NOT be
            started locally. Only follower scripts are launched.
    """
    setup_logging(debug)
    logger.info("OmniTeleop App Backend starting" + (" [rpi-mode]" if rpi_mode else ""))

    app = TeleopApp(namespace=namespace, debug=debug, rpi_mode=rpi_mode)
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    tyro.cli(main)
