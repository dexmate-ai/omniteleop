"""
Episode data structures for MCAP teleop recording.
These dataclasses define the structure of robot state/action data
and handle conversion to/from protobuf for MCAP serialization.

Supports both single-timestep and batched (T, ...) data via BatchableMixin.
"""

from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Union, Any, Type
import numpy as np
from .metadata import EpisodeMetadata

# Lazy import to avoid circular dependencies - proto classes loaded on first use
_proto_classes = None

def _get_proto_classes():
    """Lazy load protobuf classes."""
    global _proto_classes
    if _proto_classes is None:
        from .proto.robot_pb2 import (
            RobotState as RobotStateProto,
            RobotAction as RobotActionProto,
            ArmState,
            HandState,
            ArmAction,
            HandAction,
            ChassisAction,
            WrenchState as WrenchStateProto,
        )
        from .proto.policy_pb2 import PolicyStepAnnotation as PolicyStepAnnotationProto

        _proto_classes = {
            "RobotState": RobotStateProto,
            "RobotAction": RobotActionProto,
            "ArmState": ArmState,
            "HandState": HandState,
            "ArmAction": ArmAction,
            "HandAction": HandAction,
            "ChassisAction": ChassisAction,
            "WrenchState": WrenchStateProto,
            "PolicyStepAnnotation": PolicyStepAnnotationProto,
        }
    return _proto_classes

def query_proto_msg(msg, field_name: str | List[str]) -> Any:
    if isinstance(field_name, list) and len(field_name) == 1:
        field_name = field_name[0]

    if isinstance(field_name, str):
        if hasattr(msg, field_name):
            return getattr(msg, field_name)
    else:
        if hasattr(msg, field_name[0]):
            return query_proto_msg(getattr(msg, field_name[0]), field_name[1:])
    return None

class BatchableMixin:
    """Mixin for dataclasses that support batching via concat and __getitem__."""

    @classmethod
    def concat(
        cls: Type["BatchableMixin"], items: List["BatchableMixin"]
    ) -> "BatchableMixin":
        """Stack list of single-timestep items into batched."""
        if not items:
            return cls()

        updates = {}
        for f in fields(cls):
            name = f.name

            # Special handling for num_frames
            if name == "num_frames":
                updates[name] = len(items)
                continue

            # Collect values for this field
            values = [getattr(i, name) for i in items]

            # If all None, result is None
            if all(v is None for v in values):
                updates[name] = None
                continue

            # Check type of first non-None value
            first = next(v for v in values if v is not None)

            if hasattr(first, "concat"):
                # Recursive concat for nested Batchables
                updates[name] = first.__class__.concat(values)
            elif isinstance(first, (np.ndarray, list)):
                # Stack arrays
                # Ensure all are arrays (handle None if mixed, though shouldn't happen in valid data)
                valid_values = [v for v in values if v is not None]
                updates[name] = np.stack(valid_values)
            elif isinstance(first, (int, float, np.number)):
                # Stack scalars into array (e.g. timestamps)
                updates[name] = np.array(values)
            else:
                # Fallback for other types (e.g. metadata dict) - usually just take first
                updates[name] = first

        return cls(**updates)

    def __getitem__(self, idx: Union[int, slice]) -> "BatchableMixin":
        """Index or slice into batched data."""
        updates = {}

        # Calculate new num_frames
        if isinstance(idx, slice):
            # Try to find reference length from first array/batchable field
            current_len = 0
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, np.ndarray):
                    current_len = len(val)
                    break
                elif hasattr(val, "__len__") and not isinstance(val, (str, dict)):
                    current_len = len(val)
                    break

            # Dummy slicer to calculate length
            # Creating a range of that length and slicing it is robust
            if current_len > 0:
                _range = range(current_len)[idx]
                updates["num_frames"] = len(_range)
            else:
                updates["num_frames"] = 0
        else:
            updates["num_frames"] = None  # Single timestep

        for f in fields(self):
            name = f.name
            if name == "num_frames":
                continue

            val = getattr(self, name)

            if val is None:
                updates[name] = None
            elif hasattr(val, "__getitem__") and not isinstance(val, (str, dict)):
                # Arrays and nested Batchables
                updates[name] = val[idx]
            else:
                # Metadata, single values that shouldn't be sliced
                updates[name] = val

        return self.__class__(**updates)

    def __len__(self) -> int:
        return self.num_frames if self.num_frames is not None else 1

@dataclass
class JointState(BatchableMixin):
    """Joint state for a single component (arm, hand, etc.).

    Works for both single-timestep (qpos shape: (D,)) and batched (qpos shape: (T, D)).
    """

    qpos: np.ndarray
    qvel: Optional[np.ndarray] = None
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def to_arm_proto(self, proto_arm):
        """Fill an ArmState protobuf message."""
        proto_arm.qpos.extend(self.qpos.astype(np.float32))
        if self.qvel is not None:
            proto_arm.qvel.extend(self.qvel.astype(np.float32))

    def to_hand_proto(self, proto_hand):
        """Fill a HandState protobuf message (no qvel)."""
        proto_hand.qpos.extend(self.qpos.astype(np.float32))

    @classmethod
    def from_arm_proto(cls, proto_arm) -> "JointState":
        """Create from ArmState protobuf message."""
        if proto_arm is None:
            return None
        # Check if qpos is empty (protobuf sub-messages are never None, just empty)
        if not proto_arm.qpos:
            return None
        qpos = np.array(proto_arm.qpos, dtype=np.float32)
        qvel = np.array(proto_arm.qvel, dtype=np.float32) if proto_arm.qvel else None
        return cls(qpos=qpos, qvel=qvel)

    @classmethod
    def from_hand_proto(cls, proto_hand) -> "JointState":
        """Create from HandState protobuf message."""
        if proto_hand is None:
            return None
        # Check if qpos is empty (protobuf sub-messages are never None, just empty)
        if not proto_hand.qpos:
            return None
        return cls(qpos=np.array(proto_hand.qpos, dtype=np.float32))

@dataclass
class WrenchState(BatchableMixin):
    """Force-torque sensor state for a single wrist.

    Works for both single-timestep (force shape: (3,)) and batched (force shape: (T, 3)).
    """

    force: np.ndarray  # (3,) or (T, 3) in N
    torque: np.ndarray  # (3,) or (T, 3) in Nm
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def to_proto(self, proto_wrench):
        """Fill a WrenchState protobuf message."""
        proto_wrench.force.extend(self.force.astype(np.float32))
        proto_wrench.torque.extend(self.torque.astype(np.float32))

    @classmethod
    def from_proto(cls, proto_wrench) -> Optional["WrenchState"]:
        """Create from WrenchState protobuf message."""
        if proto_wrench is None or not proto_wrench.force:
            return None
        return cls(
            force=np.array(proto_wrench.force, dtype=np.float32),
            torque=np.array(proto_wrench.torque, dtype=np.float32),
        )

@dataclass
class RobotState(BatchableMixin):
    """Robot state at a single timestep or batched over time."""

    left_arm: Optional[JointState] = None
    right_arm: Optional[JointState] = None
    left_hand: Optional[JointState] = None
    right_hand: Optional[JointState] = None
    head: Optional[JointState] = None
    torso: Optional[JointState] = None
    left_wrist_wrench: Optional[WrenchState] = None
    right_wrist_wrench: Optional[WrenchState] = None
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def to_proto(self):
        """Convert to RobotState protobuf message."""
        protos = _get_proto_classes()
        msg = protos["RobotState"]()

        if self.left_arm is not None:
            self.left_arm.to_arm_proto(msg.left_arm)
        if self.right_arm is not None:
            self.right_arm.to_arm_proto(msg.right_arm)
        if self.left_hand is not None:
            self.left_hand.to_hand_proto(msg.left_hand)
        if self.right_hand is not None:
            self.right_hand.to_hand_proto(msg.right_hand)
        if self.head is not None:
            self.head.to_hand_proto(msg.head)
        if self.torso is not None:
            self.torso.to_hand_proto(msg.torso)
        if self.left_wrist_wrench is not None:
            self.left_wrist_wrench.to_proto(msg.left_wrist_wrench)
        if self.right_wrist_wrench is not None:
            self.right_wrist_wrench.to_proto(msg.right_wrist_wrench)
        return msg

    @classmethod
    def from_proto(cls, msg) -> "RobotState":
        """Create from RobotState protobuf message."""
        return cls(
            left_arm=JointState.from_arm_proto(query_proto_msg(msg, ["left_arm"])),
            right_arm=JointState.from_arm_proto(query_proto_msg(msg, ["right_arm"])),
            left_hand=JointState.from_hand_proto(query_proto_msg(msg, ["left_hand"])),
            right_hand=JointState.from_hand_proto(query_proto_msg(msg, ["right_hand"])),
            head=JointState.from_hand_proto(query_proto_msg(msg, ["head"])),
            torso=JointState.from_hand_proto(query_proto_msg(msg, ["torso"])),
            left_wrist_wrench=WrenchState.from_proto(
                query_proto_msg(msg, ["left_wrist_wrench"])
            ),
            right_wrist_wrench=WrenchState.from_proto(
                query_proto_msg(msg, ["right_wrist_wrench"])
            ),
        )

@dataclass
class RobotAction(BatchableMixin):
    """Robot action at a single timestep or batched over time."""

    left_arm: Optional[JointState] = None
    right_arm: Optional[JointState] = None
    left_hand: Optional[JointState] = None
    right_hand: Optional[JointState] = None
    head: Optional[JointState] = None
    torso: Optional[JointState] = None
    chassis_vx: Optional[np.ndarray] = None
    chassis_vy: Optional[np.ndarray] = None
    chassis_wz: Optional[np.ndarray] = None
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def to_proto(self):
        """Convert to RobotAction protobuf message."""
        protos = _get_proto_classes()
        msg = protos["RobotAction"]()

        if self.left_arm is not None:
            self.left_arm.to_arm_proto(msg.left_arm)
        if self.right_arm is not None:
            self.right_arm.to_arm_proto(msg.right_arm)
        if self.left_hand is not None:
            self.left_hand.to_hand_proto(msg.left_hand)
        if self.right_hand is not None:
            self.right_hand.to_hand_proto(msg.right_hand)
        if self.head is not None:
            self.head.to_hand_proto(msg.head)
        if self.torso is not None:
            self.torso.to_hand_proto(msg.torso)

        # Chassis (fields are scalars in proto, handle None implicitly as 0.0 or explicit check?)
        # For now, if chassis fields are arrays/scalars, assume they exist if not None.
        # But if they are None, 0.0 is default in proto.
        if self.chassis_vx is not None:
            msg.chassis.vx = float(
                self.chassis_vx.item()
                if isinstance(self.chassis_vx, np.ndarray) and self.chassis_vx.ndim > 0
                else self.chassis_vx
            )
        if self.chassis_vy is not None:
            msg.chassis.vy = float(
                self.chassis_vy.item()
                if isinstance(self.chassis_vy, np.ndarray) and self.chassis_vy.ndim > 0
                else self.chassis_vy
            )
        if self.chassis_wz is not None:
            msg.chassis.wz = float(
                self.chassis_wz.item()
                if isinstance(self.chassis_wz, np.ndarray) and self.chassis_wz.ndim > 0
                else self.chassis_wz
            )

        return msg

    @classmethod
    def from_proto(cls, msg) -> "RobotAction":
        """Create from RobotAction protobuf message."""

        # Helper to create JointState only if qpos is not empty
        def _joint_state_from_qpos(qpos_list):
            if not qpos_list:
                return None
            return JointState(qpos=np.array(qpos_list, dtype=np.float32))

        has_chassis = msg.chassis.vx != 0 or msg.chassis.vy != 0 or msg.chassis.wz != 0
        return cls(
            left_arm=_joint_state_from_qpos(query_proto_msg(msg, ["left_arm", "qpos"])),
            right_arm=_joint_state_from_qpos(
                query_proto_msg(msg, ["right_arm", "qpos"])
            ),
            left_hand=_joint_state_from_qpos(
                query_proto_msg(msg, ["left_hand", "qpos"])
            ),
            right_hand=_joint_state_from_qpos(
                query_proto_msg(msg, ["right_hand", "qpos"])
            ),
            head=_joint_state_from_qpos(query_proto_msg(msg, ["head", "qpos"])),
            torso=_joint_state_from_qpos(query_proto_msg(msg, ["torso", "qpos"])),
            chassis_vx=np.array(
                [query_proto_msg(msg, ["chassis", "vx"])], dtype=np.float32
            )
            if has_chassis
            else None,
            chassis_vy=np.array(
                [query_proto_msg(msg, ["chassis", "vy"])], dtype=np.float32
            )
            if has_chassis
            else None,
            chassis_wz=np.array(
                [query_proto_msg(msg, ["chassis", "wz"])], dtype=np.float32
            )
            if has_chassis
            else None,
        )

@dataclass
class PolicyStepAnnotation(BatchableMixin):
    """Policy inference annotation for a single timestep or batched over time.

    Written to /policy/step topic during policy deployments.
    Not present in teleop recordings.
    """

    inference_begin_ns: Union[int, np.ndarray, None] = None
    inference_end_ns: Union[int, np.ndarray, None] = None
    action_queue_size: Union[int, np.ndarray, None] = None
    policy_control_flag: Union[int, np.ndarray, None] = None  # 1=policy, 0=human/other
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def to_proto(self):
        """Convert to PolicyStepAnnotation protobuf message."""
        protos = _get_proto_classes()
        msg = protos["PolicyStepAnnotation"]()
        if self.inference_begin_ns is not None:
            msg.inference_begin_ns = int(self.inference_begin_ns)
        if self.inference_end_ns is not None:
            msg.inference_end_ns = int(self.inference_end_ns)
        if self.action_queue_size is not None:
            msg.action_queue_size = int(self.action_queue_size)
        if self.policy_control_flag is not None:
            msg.policy_control_flag = int(self.policy_control_flag)
        return msg

    @classmethod
    def from_proto(cls, msg) -> "PolicyStepAnnotation":
        """Create from PolicyStepAnnotation protobuf message."""
        return cls(
            inference_begin_ns=msg.inference_begin_ns,
            inference_end_ns=msg.inference_end_ns,
            action_queue_size=msg.action_queue_size,
            policy_control_flag=msg.policy_control_flag,
        )

@dataclass
class CameraStream:
    """Camera images at a single timestep or batched over time.

    Uses a dict for flexible camera naming (from metadata.available_cameras).
    Single: images[cam_name] shape (H, W, 3)
    Batched: images[cam_name] shape (T, H, W, 3)
    """

    images: Dict[str, np.ndarray] = field(default_factory=dict)
    timestamp_ns: Union[int, np.ndarray, None] = None
    num_frames: Optional[int] = None

    def __getitem__(
        self, key: Union[str, int, slice]
    ) -> Union[np.ndarray, "CameraStream"]:
        """Get camera by name (str) or index/slice into batched (int/slice)."""
        if isinstance(key, str):
            return self.images.get(key)
        else:
            # Index/slice into batched
            indexed_images = {name: arr[key] for name, arr in self.images.items()}
            timestamp_ns = (
                self.timestamp_ns[key] if self.timestamp_ns is not None else None
            )

            if isinstance(key, slice):
                num_frames = (
                    len(next(iter(indexed_images.values()))) if indexed_images else 0
                )
            else:
                num_frames = None

            return CameraStream(
                images=indexed_images, timestamp_ns=timestamp_ns, num_frames=num_frames
            )

    def __setitem__(self, camera_name: str, image: np.ndarray):
        """Set camera image by name."""
        self.images[camera_name] = image

    def __contains__(self, camera_name: str) -> bool:
        return camera_name in self.images

    def __len__(self) -> int:
        return self.num_frames if self.num_frames is not None else 1

    @property
    def available_cameras(self) -> List[str]:
        return list(self.images.keys())

    @classmethod
    def concat(cls, items: List["CameraStream"]) -> "CameraStream":
        """Stack list of single-timestep CameraStreams into batched."""
        if not items:
            return cls()

        camera_names = items[0].available_cameras
        stacked_images = {}
        for name in camera_names:
            stacked_images[name] = np.stack([item.images[name] for item in items])

        timestamp_ns = None
        if items[0].timestamp_ns is not None:
            timestamp_ns = np.array([item.timestamp_ns for item in items])

        return cls(
            images=stacked_images, timestamp_ns=timestamp_ns, num_frames=len(items)
        )

@dataclass
class Episode:
    """Complete episode container with nested batched dataclasses.

    All data aligned to state timestamps (reference).
    """

    state: RobotState
    action: Optional[RobotAction] = None
    images: Optional[CameraStream] = None
    timestamps_ns: Optional[np.ndarray] = None  # Reference timestamps
    metadata: Optional[EpisodeMetadata] = None
    policy_steps: Optional[PolicyStepAnnotation] = (
        None  # Present only in policy deployments
    )

    def __getitem__(self, idx: Union[int, slice]) -> "Episode":
        """Index or slice into episode."""
        return Episode(
            state=self.state[idx],
            action=self.action[idx] if self.action is not None else None,
            images=self.images[idx] if self.images is not None else None,
            timestamps_ns=self.timestamps_ns[idx]
            if self.timestamps_ns is not None
            else None,
            metadata=self.metadata,
            policy_steps=self.policy_steps[idx]
            if self.policy_steps is not None
            else None,
        )

    def __len__(self) -> int:
        return len(self.state)

    @property
    def length(self) -> int:
        """Number of timesteps in episode."""
        return len(self)

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """Convert to flat dict with dot-notation keys for training.

        Format:
        - observation.state.left_arm → state.left_arm.qpos
        - observation.state.left_arm_qvel → state.left_arm.qvel
        - action.left_arm → action.left_arm.qpos
        - observation.images.{camera}_rgb → images[camera]
        """
        result = {}

        # State
        if self.state.left_arm is not None:
            result["observation.state.left_arm.qpos"] = self.state.left_arm.qpos
            if self.state.left_arm.qvel is not None:
                result["observation.state.left_arm.qvel"] = self.state.left_arm.qvel

        if self.state.right_arm is not None:
            result["observation.state.right_arm.qpos"] = self.state.right_arm.qpos
            if self.state.right_arm.qvel is not None:
                result["observation.state.right_arm.qvel"] = self.state.right_arm.qvel

        if self.state.left_hand is not None:
            result["observation.state.left_hand.qpos"] = self.state.left_hand.qpos

        if self.state.right_hand is not None:
            result["observation.state.right_hand.qpos"] = self.state.right_hand.qpos

        if self.state.head is not None:
            result["observation.state.head.qpos"] = self.state.head.qpos

        if self.state.torso is not None:
            result["observation.state.torso.qpos"] = self.state.torso.qpos

        if self.state.left_wrist_wrench is not None:
            result["observation.state.left_wrist_wrench.force"] = (
                self.state.left_wrist_wrench.force
            )
            result["observation.state.left_wrist_wrench.torque"] = (
                self.state.left_wrist_wrench.torque
            )

        if self.state.right_wrist_wrench is not None:
            result["observation.state.right_wrist_wrench.force"] = (
                self.state.right_wrist_wrench.force
            )
            result["observation.state.right_wrist_wrench.torque"] = (
                self.state.right_wrist_wrench.torque
            )

        # Action
        if self.action is not None:
            if self.action.left_arm is not None:
                result["action.left_arm.qpos"] = self.action.left_arm.qpos
            if self.action.right_arm is not None:
                result["action.right_arm.qops"] = self.action.right_arm.qpos
            if self.action.left_hand is not None:
                result["action.left_hand.qpos"] = self.action.left_hand.qpos
            if self.action.right_hand is not None:
                result["action.right_hand.qpos"] = self.action.right_hand.qpos
            if self.action.head is not None:
                result["action.head.qpos"] = self.action.head.qpos
            if self.action.torso is not None:
                result["action.torso.qpos"] = self.action.torso.qpos

        # Images
        if self.images is not None:
            for camera_name, arr in self.images.images.items():
                result[f"observation.images.{camera_name}"] = arr

        return result
