import numpy as np

from dexexo.leader.communication.base_socket_client import (
    BaseSocketClient,
    process_hand_skeleton,
    process_pose,
    remove_metacarpal,
)

from dexexo.leader.communication.proto import avp_teleop_pb2

OPERATOR2AVP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2AVP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)

VISIONPRO2ROBOT_GLOBAL = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
HEAD_YUP2ZUP = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)

def transform_wrist(mat: np.ndarray, right_rot: np.ndarray):
    result = np.eye(4)
    rotation = VISIONPRO2ROBOT_GLOBAL @ mat[:3, :3] @ right_rot
    pos = VISIONPRO2ROBOT_GLOBAL @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result

def rotate_head(R):
    R_x = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    R_rotated = HEAD_YUP2ZUP @ R @ R_x
    return R_rotated

class AvpSocketClient(BaseSocketClient):
    def _parse_proto_data(self, data: bytes, key: str = None):
        """Parse AVP proto data."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        update = avp_teleop_pb2.AvpUpdate()
        update.ParseFromString(data)
        return update

    def _process_transformations(
        self, avp_update, key: str = None
    ) -> dict[str, np.ndarray]:
        left_joints = process_hand_skeleton(avp_update.left_hand.skeleton)
        right_joints = process_hand_skeleton(avp_update.right_hand.skeleton)
        left_joints = left_joints @ OPERATOR2AVP_LEFT
        right_joints = right_joints @ OPERATOR2AVP_RIGHT

        return {
            "head": rotate_head(process_pose(avp_update.head)),
            "left_fingers": remove_metacarpal(left_joints),
            "right_fingers": remove_metacarpal(right_joints),
            "left_wrist": transform_wrist(
                process_pose(avp_update.left_hand.wrist_pose),
                OPERATOR2AVP_LEFT,
            ),
            "right_wrist": transform_wrist(
                process_pose(avp_update.right_hand.wrist_pose),
                OPERATOR2AVP_RIGHT,
            ),
            "has_elbow": False,
        }

    @property
    def log_header(self) -> str:
        return f"[{self.log_color}]Teleop Socket Client[/{self.log_color}]"
