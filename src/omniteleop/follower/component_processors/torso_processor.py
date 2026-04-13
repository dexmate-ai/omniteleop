"""Torso component processor."""

from typing import Dict, Any, List
import numpy as np

from loguru import logger
from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkIKConfig

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand, CommandMode
from omniteleop.common.log_utils import suppress_loguru_module

class TorsoProcessor(BaseComponentProcessor):
    """Processes torso movement commands.

    Supports two modes:
    - Absolute: Direct joint position control
    - Relative: Delta movement using IK to maintain head target frame position
    """

    def __init__(self, side, config, motion_manager, robot_info, teleop_mode):
        """Initialize torso processor.

        Args:
            side: Not used (always None for torso)
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        super().__init__(None, config, motion_manager, robot_info, teleop_mode)

        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints("torso")

        # Setup IK configuration for torso delta movements
        self._setup_torso_ik_config()

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "torso"

    def _setup_torso_ik_config(self) -> None:
        """Configure torso IK parameters for delta movements."""
        self.torso_ik_damping = IKDampingWeightsConfig(
            default=1e-12,
            override={
                "torso_j1": 1e-20,
                "torso_j2": 1e-20,
                "torso_j3": 1e-20,
            },
        )
        self._torso_ik_target_frame = "head_l1"
        self.torso_ik_config = LocalPinkIKConfig(
            damping_weights=self.torso_ik_damping,
            target_frames=[self._torso_ik_target_frame],
        )

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process torso movement command.

        Routes to absolute or relative processing based on command mode.

        Args:
            input_data: Torso input data with 'pos' and 'mode' fields
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        mode = input_data.get("mode", CommandMode.RELATIVE)
        positions = input_data.get("pos", [])

        if not positions:
            return False

        if mode == CommandMode.RELATIVE:
            # Handle as delta movement using IK
            return self._process_torso_delta(positions, command)
        else:
            # Direct absolute positions
            command.output_components["torso"] = {
                "pos": positions,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

    def _process_torso_delta(self, deltas: List[float], command: RobotCommand) -> bool:
        """Compute torso joint positions from Cartesian delta movement.

        Uses IK to solve for torso joint positions that achieve the
        desired Cartesian displacement of the head target frame.

        Args:
            deltas: Cartesian displacement [dx, dy, dz] in meters
            command: RobotCommand to update with joint positions

        Returns:
            True if IK succeeded and positions were set
        """
        if len(deltas) < 3:
            return False

        # Get current torso pose
        torso_pose = self.motion_manager.fk([self._torso_ik_target_frame])[
            self._torso_ik_target_frame
        ]

        # Apply delta movement
        delta_translation = np.array(deltas)
        target_torso_pose = torso_pose.np.copy()
        target_torso_pose[:3, 3] += delta_translation

        # Solve IK
        target_pose = {self._torso_ik_target_frame: target_torso_pose}

        try:
            with suppress_loguru_module("dexmotion", enabled=True):
                joint_solution, has_collision, within_limits = self.motion_manager.ik(
                    target_pose=target_pose,
                    type="pink",
                    custom_config=self.torso_ik_config,
                )

                if not has_collision:
                    self.motion_manager.set_joint_pos(joint_solution)

        except Exception as e:
            logger.error(f"Error solving torso IK: {e}")
            return False

        # Extract torso joint positions from solution
        if joint_solution and within_limits and not has_collision:
            torso_joint_positions = [
                joint_solution[f"torso_j{i + 1}"] for i in range(3)
            ]
            command.output_components["torso"] = {
                "pos": torso_joint_positions,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

        return False

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync torso to robot state.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        torso_pos = robot_joints.get("torso", [])
        if torso_pos:
            self.motion_manager.torso.set_joint_pos(torso_pos)
