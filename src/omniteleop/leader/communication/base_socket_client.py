#!/usr/bin/env python3
"""Base socket client for VR teleoperation.

Provides abstract base class for connecting to VR servers via SocketIO
and processing VR controller/hand tracking data.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np
import socketio
from loguru import logger
from pytransform3d import transformations as pt

from omniteleop.leader.communication.proto import quest_teleop_pb2

# Transformation matrices for hand convention conversion
LEFT2RIGHT_HAND_CONVENTION = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)

def process_pose(
    pose: quest_teleop_pb2.Pose, is_right_handed: bool = True
) -> np.ndarray:
    """Convert protobuf pose to 4x4 transformation matrix.

    Args:
        pose: Protobuf pose with position and quaternion
        is_right_handed: Whether to use right-handed coordinate system

    Returns:
        4x4 transformation matrix
    """
    pos = [pose.p.x, pose.p.y, pose.p.z]
    quat = [pose.q.w, pose.q.x, pose.q.y, pose.q.z]
    pose_matrix = pt.transform_from_pq(pos + quat)
    if not is_right_handed:
        pose_matrix = (
            LEFT2RIGHT_HAND_CONVENTION @ pose_matrix @ LEFT2RIGHT_HAND_CONVENTION.T
        )
    return pose_matrix

def process_pos(pos: quest_teleop_pb2.Pos, is_right_handed: bool = True) -> np.ndarray:
    """Convert protobuf position to numpy array.

    Args:
        pos: Protobuf position with x, y, z coordinates
        is_right_handed: Whether to use right-handed coordinate system

    Returns:
        3D position array
    """
    position = [pos.x, pos.y, pos.z * (1 if is_right_handed else -1)]
    return np.array(position)

def process_hand_skeleton(
    skeleton: quest_teleop_pb2.Skeleton, is_right_handed: bool = True
) -> np.ndarray:
    """Convert protobuf hand skeleton to joint position array.

    Args:
        skeleton: Protobuf skeleton with joint positions
        is_right_handed: Whether to use right-handed coordinate system

    Returns:
        Array of joint positions (N, 3)
    """
    return np.stack(
        [process_pos(pos, is_right_handed) for pos in skeleton.joint_pos], axis=0
    )

def remove_metacarpal(finger_mat: np.ndarray) -> np.ndarray:
    """Remove metacarpal joints from finger joint array.

    Args:
        finger_mat: Array of finger joint positions (25, 3)

    Returns:
        Array with metacarpal joints removed (21, 3)
    """
    finger_index = np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    )
    return finger_mat[finger_index]

class BaseSocketClient(ABC):
    """Abstract base class for VR socket clients.

    Handles SocketIO communication with VR servers and provides
    framework for processing VR data streams.
    """

    def __init__(
        self,
        server_url: str,
        vr_type: str = "quest",
        rate_limiter_freq: float = 40,
    ):
        """Initialize socket client.

        Args:
            server_url: URL of the VR server to connect to
            vr_type: Type of VR system ("avp", "quest", etc.)
            rate_limiter_freq: Frequency limit for rate limiting
        """
        # Connection settings
        self.server_url = server_url
        self.vr_type = vr_type
        self.log_color = "bold deep_sky_blue3"

        # SocketIO client
        self.sio = socketio.Client()
        self.connected = threading.Event()
        self.condition = threading.Condition()
        self.lock = threading.Lock()

        # Data processing
        self.latest_transformation = None
        self.latest_key = None
        self.event_queue = deque(maxlen=5)
        self.processing_thread = threading.Thread(
            target=self._process_teleop, daemon=True
        )

        # Control flags
        self.should_exit = threading.Event()
        self.reset = False

        # Initialize connection
        self._setup_socket_events()
        self.sio.connect(self.server_url)
        self.processing_thread.start()
        self.connected.wait()
        self.sio.emit("add user", "Teleop Server")
        self.bind()

    def _setup_socket_events(self):
        @self.sio.event
        def connect():
            self.connected.set()

        @self.sio.event
        def disconnect():
            self.connected.clear()
            logger.warning(f"{self.log_header}: Disconnected")

        @self.sio.event
        def connect_error(data):
            logger.error(f"{self.log_header}: Error: {data}")

    def bind(self):
        @self.sio.on("login")
        def on_login(data):
            with self.condition:
                logger.info(
                    f"{self.log_header}: Welcome - there are {data['numUsers']} participants"
                )
                self.condition.notify_all()
            self.sio.handlers["/"]["login"] = lambda data: None

        @self.sio.on(f"{self.vr_type}_teleop_controller")
        def on_teleop_controller(data):
            self.event_queue.append((data, "controller", time.perf_counter()))

        @self.sio.on(f"{self.vr_type}_teleop_hand")
        def on_teleop_hand(data):
            self.event_queue.append((data, "hand", time.perf_counter()))

        @self.sio.on("reset")
        def on_reset():
            logger.info(f"{self.log_header}: Reset signal received")
            self.reset_on()

        @self.sio.on("ping_pong")
        def on_ping_pong(data):
            self.sio.emit(
                "ping_pong",
                {
                    "deviceId": "teleop_server",
                },
            )

    def _process_teleop(self):
        while not self.should_exit.is_set():
            if not self.event_queue:
                time.sleep(0.001)
                continue

            data, key, _ = self.event_queue.popleft()
            if not data:
                logger.error(f"{self.log_header}: Data is empty, aborting operation")
                continue

            # Let child class handle proto parsing
            parsed_data = self._parse_proto_data(data, key)

            transformations = self._process_transformations(parsed_data, key)

            with self.lock:
                self.latest_transformation = transformations
                self.latest_key = key

    @abstractmethod
    def _parse_proto_data(self, data: bytes, key: str) -> Any:
        """Each child class handles its own proto parsing"""
        pass

    @abstractmethod
    def _process_transformations(
        self, parsed_data: Any, key: str
    ) -> dict[str, np.ndarray]:
        """Process transformations from parsed proto data. Override this for custom processing."""
        pass

    @property
    @abstractmethod
    def log_header(self) -> str:
        pass

    def get_latest_transformation(self) -> dict[str, np.ndarray]:
        with self.lock:
            return self.latest_transformation

    def reset_on(self):
        logger.info(f"{self.log_header}: Reset signal received")
        self.reset = True

    def reset_off(self):
        logger.info(f"{self.log_header}: Reset signal off")
        self.reset = False

    def close(self):
        self.should_exit.set()
        self.sio.disconnect()
        logger.info(f"{self.log_header}: Socketio client is closed.")

        logger.info(f"{self.log_header}: Exiting...")

    def wait_for_data(self):
        logger.info(f"{self.log_header}: Waiting for data from VR device.")
        while self.latest_transformation is None:
            time.sleep(0.1)

        logger.info(f"{self.log_header}: Received data from VR device.")
