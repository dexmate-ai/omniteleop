#!/usr/bin/env python3
"""OpenXR Unity socket client for Quest VR teleoperation.

Handles communication with Unity OpenXR VR applications,
processing Quest controller and hand tracking data.
"""

import numpy as np

from omniteleop.leader.communication.base_socket_client import (
    BaseSocketClient,
    process_hand_skeleton,
    process_pose,
    remove_metacarpal,
)
from omniteleop.leader.communication.proto import quest_teleop_pb2

# Coordinate system transformation matrices
OPERATOR2OPENXR = np.array(
    [
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1],
    ]
)

OPENXR2ROBOT_GLOBAL = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Rotation transformation matrices
ROTATE_90_X = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

ROTATE_180_Z = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

def rotate_head(transformation_matrix: np.ndarray) -> np.ndarray:
    """Apply head rotation transformation.

    Args:
        transformation_matrix: 4x4 transformation matrix

    Returns:
        Rotated transformation matrix
    """
    return transformation_matrix @ ROTATE_90_X @ ROTATE_180_Z

def transform(mat: np.ndarray) -> np.ndarray:
    """Transform OpenXR coordinates to robot coordinates.

    Args:
        mat: 4x4 transformation matrix in OpenXR coordinates

    Returns:
        4x4 transformation matrix in robot coordinates
    """
    result = np.eye(4)
    rotation = OPENXR2ROBOT_GLOBAL @ mat[:3, :3] @ OPERATOR2OPENXR
    position = OPENXR2ROBOT_GLOBAL @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result

def transform_joints(joint_pos: np.ndarray, wrist_pose: np.ndarray) -> np.ndarray:
    """Transform joint positions relative to wrist pose.

    Args:
        joint_pos: Array of joint positions (N, 3)
        wrist_pose: 4x4 wrist transformation matrix

    Returns:
        Transformed joint positions
    """
    # Transform relative to wrist
    transformed_joints = (
        joint_pos[1:] @ wrist_pose[:3, :3]
        - wrist_pose[:3, 3][None, :] @ wrist_pose[:3, :3]
    )
    return transformed_joints @ OPERATOR2OPENXR.T

class OpenXRUnitySocketClient(BaseSocketClient):
    """Socket client for Unity OpenXR Quest applications.

    Handles communication with Unity-based VR applications using
    Quest controllers and hand tracking via SocketIO.
    """

    def __init__(
        self,
        server_url: str,
        rate_limiter_freq: float = 40,
    ):
        """Initialize OpenXR Unity socket client.

        Args:
            server_url: URL of the Unity OpenXR server
            rate_limiter_freq: Rate limiting frequency for data processing
        """
        super().__init__(
            server_url,
            vr_type="quest",
            rate_limiter_freq=rate_limiter_freq,
        )

    def _parse_proto_data(self, data: bytes, tracking_type: str):
        """Parse Quest protobuf data based on tracking type.

        Args:
            data: Raw protobuf data bytes
            tracking_type: Type of tracking data ("hand" or "controller")

        Returns:
            Parsed protobuf message object

        Raises:
            ValueError: If tracking_type is not supported
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        if tracking_type == "hand":
            update = quest_teleop_pb2.HandTransformPair()
        elif tracking_type == "controller":
            update = quest_teleop_pb2.ControllerTransformPair()
        else:
            raise ValueError(f"Unknown tracking type: {tracking_type}")

        update.ParseFromString(data)
        return update

    def _process_transformations(
        self, quest_update, tracking_type: str
    ) -> dict[str, np.ndarray]:
        """Process Quest tracking data into transformation matrices.

        Args:
            quest_update: Parsed Quest protobuf data
            tracking_type: Type of tracking data ("hand" or "controller")

        Returns:
            Dictionary of transformation matrices and sensor data
        """
        # Process basic pose data common to both tracking types
        left_wrist = process_pose(
            quest_update.left_hand.wrist_pose, is_right_handed=False
        )
        right_wrist = process_pose(
            quest_update.right_hand.wrist_pose, is_right_handed=False
        )
        left_elbow = process_pose(quest_update.left_arm, is_right_handed=False)
        right_elbow = process_pose(quest_update.right_arm, is_right_handed=False)
        if tracking_type == "hand":
            left_joints = process_hand_skeleton(
                quest_update.left_hand.skeleton, is_right_handed=False
            )  # (25, 3)
            right_joints = process_hand_skeleton(
                quest_update.right_hand.skeleton, is_right_handed=False
            )  # (25, 3)

            left_joints = transform_joints(left_joints, left_wrist)
            right_joints = transform_joints(right_joints, right_wrist)

            return {
                "left_fingers": remove_metacarpal(left_joints),
                "right_fingers": remove_metacarpal(right_joints),
                "left_wrist": transform(left_wrist),
                "right_wrist": transform(right_wrist),
                "left_elbow": transform(left_elbow),
                "right_elbow": transform(right_elbow),
                "head": rotate_head(
                    transform(process_pose(quest_update.head, is_right_handed=False))
                ),
            }
        elif tracking_type == "controller":
            # Extract controller input data
            left_hand_trigger = quest_update.left_hand.hand_trigger
            left_index_trigger = quest_update.left_hand.index_trigger
            right_hand_trigger = quest_update.right_hand.hand_trigger
            right_index_trigger = quest_update.right_hand.index_trigger
            left_grip_trigger = quest_update.left_hand.grip_trigger
            right_grip_trigger = quest_update.right_hand.grip_trigger
            left_thumbstick_click = quest_update.left_hand.thumbstick_click
            right_thumbstick_click = quest_update.right_hand.thumbstick_click

            left_thumbstick = [
                quest_update.left_hand.thumbstick.x,
                quest_update.left_hand.thumbstick.y,
            ]
            right_thumbstick = [
                quest_update.right_hand.thumbstick.x,
                quest_update.right_hand.thumbstick.y,
            ]

            return {
                # Pose data
                "left_wrist": transform(left_wrist),
                "right_wrist": transform(right_wrist),
                "left_elbow": transform(left_elbow),
                "right_elbow": transform(right_elbow),
                "head": rotate_head(
                    transform(process_pose(quest_update.head, is_right_handed=False))
                ),
                # Controller inputs
                "left_hand_trigger": left_hand_trigger,
                "left_index_trigger": left_index_trigger,
                "right_hand_trigger": right_hand_trigger,
                "right_index_trigger": right_index_trigger,
                "left_grip_trigger": left_grip_trigger,
                "right_grip_trigger": right_grip_trigger,
                "left_thumbstick": left_thumbstick,
                "right_thumbstick": right_thumbstick,
                "left_thumbstick_click": left_thumbstick_click,
                "right_thumbstick_click": right_thumbstick_click,
            }

    @property
    def log_header(self) -> str:
        return f"[{self.log_color}]OpenXR Socket Client[/{self.log_color}]"
