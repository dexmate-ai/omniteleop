"""Safety validation for robot commands."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from loguru import logger
from dexmotion.motion_manager import MotionManager
from dexmotion.utils import robot_utils

from omniteleop.follower.input_handlers.base_handler import RobotCommand

@dataclass
class CollisionState:
    """State information for collision checking."""

    has_collision: bool
    joint_positions: np.ndarray
    colliding_links: list

class SafetyValidator:
    """Validates robot commands for safety.

    Performs safety checks including:
    - Emergency stop handling
    - Joint limit enforcement
    - Collision detection
    - Safe IK retry for collision avoidance
    """

    def __init__(self, config: Dict[str, Any], motion_manager: MotionManager):
        """Initialize safety validator.

        Args:
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
        """
        self.config = config
        self.motion_manager = motion_manager

        # Safety parameters
        safety_config = self.config.get("safety", {})
        self.enable_collision_check = safety_config.get("enable_collision_check", True)
        self.collision_padding = safety_config.get("collision_padding", 0.02)

        # Cache joint names for arm components
        self._left_arm_joint_names = [f"L_arm_j{i + 1}" for i in range(7)]
        self._right_arm_joint_names = [f"R_arm_j{i + 1}" for i in range(7)]

    def validate(self, command: RobotCommand) -> None:
        """Apply safety validation to robot command.

        Performs safety checks in order:
        1. Emergency stop check (bypasses other checks)
        2. Joint limit enforcement
        3. Collision detection (if enabled)

        Args:
            command: RobotCommand to validate (modified in-place)

        Side Effects:
            Updates command validity and safety flags
        """
        # Check for emergency stop
        if command.safety_flags.emergency_stop:
            return

        # Apply joint limits if arm components exist
        self._enforce_joint_limits(command)

        # Check for collisions if enabled
        if self.enable_collision_check:
            self._check_collisions(command)

    def _enforce_joint_limits(self, command: RobotCommand) -> None:
        """Clip joint positions to mechanical limits.

        Uses robot model to enforce joint position limits for safety.
        Only processes arm components.

        Args:
            command: RobotCommand with arm positions to validate

        Side Effects:
            - Updates command.output_components with clipped positions
            - Sets command.safety_flags.limits_enforced to True
        """
        # Check if arm components exist
        if (
            "left_arm" not in command.output_components
            or "right_arm" not in command.output_components
        ):
            return

        left_arm_positions = command.output_components["left_arm"]["pos"]
        right_arm_positions = command.output_components["right_arm"]["pos"]

        # Build complete joint position dictionary
        current_positions = self.motion_manager.get_joint_pos_dict()
        updated_positions = current_positions.copy()

        # Update arm positions
        self._update_arm_positions(
            updated_positions, left_arm_positions, right_arm_positions
        )

        # Apply joint limits using robot model
        clipped_positions = robot_utils.clip_joint_positions_to_limits(
            self.motion_manager.pin_robot, updated_positions
        )

        # Update command with clipped positions
        command.output_components["left_arm"]["pos"] = [
            clipped_positions[f"L_arm_j{i + 1}"] for i in range(7)
        ]
        command.output_components["right_arm"]["pos"] = [
            clipped_positions[f"R_arm_j{i + 1}"] for i in range(7)
        ]
        command.safety_flags.limits_enforced = True

    def _update_arm_positions(
        self,
        positions_dict: Dict[str, float],
        left_positions: Optional[List[float]],
        right_positions: Optional[List[float]],
    ) -> None:
        """Update arm joint positions in dictionary.

        Args:
            positions_dict: Dictionary to update
            left_positions: Left arm positions
            right_positions: Right arm positions
        """
        if left_positions is not None:
            for i, pos in enumerate(left_positions):
                positions_dict[f"L_arm_j{i + 1}"] = pos

        if right_positions is not None:
            for i, pos in enumerate(right_positions):
                positions_dict[f"R_arm_j{i + 1}"] = pos

    def _check_collisions(self, command: RobotCommand) -> None:
        """Detect and mitigate potential robot collisions.

        Performs collision checking on commanded positions and attempts
        to find collision-free alternatives using IK if needed.

        Args:
            command: RobotCommand with positions to check

        Side Effects:
            - Sets command.safety_flags.collision_detected if collision found
            - Sets command.valid to False if no safe solution exists
            - Updates motion manager state if collision-free
        """
        if not self._has_arm_components(command):
            return

        # Check for collisions
        collision_state = self._detect_collisions(command)

        if not collision_state.has_collision:
            # Safe to proceed
            self.motion_manager.set_joint_pos(collision_state.joint_positions)
            return

        # Handle collision
        self._handle_collision(command)

    def _has_arm_components(self, command: RobotCommand) -> bool:
        """Check if command contains arm components.

        Args:
            command: RobotCommand to check

        Returns:
            True if both arm components present
        """
        return (
            "left_arm" in command.output_components
            and "right_arm" in command.output_components
        )

    def _detect_collisions(self, command: RobotCommand) -> CollisionState:
        """Detect collisions in commanded configuration.

        Args:
            command: RobotCommand with arm positions

        Returns:
            CollisionState with detection results
        """
        # Extract arm positions
        left_arm_positions = command.output_components["left_arm"]["pos"]
        right_arm_positions = command.output_components["right_arm"]["pos"]

        # Build complete state
        joint_positions = self._build_robot_state(
            left_arm_positions, right_arm_positions
        )

        # Check collisions
        is_collision, colliding_links, _ = robot_utils.check_collisions_at_state(
            robot=self.motion_manager.pin_robot,
            qpos=joint_positions,
            distance_padding=self.collision_padding,
            stop_at_first_collision=True,
        )

        return CollisionState(
            has_collision=is_collision,
            joint_positions=joint_positions,
            colliding_links=colliding_links,
        )

    def _build_robot_state(
        self,
        left_arm_positions: List[float],
        right_arm_positions: List[float],
    ) -> np.ndarray:
        """Build complete robot state for collision checking.

        Merges commanded positions with current robot state.

        Args:
            left_arm_positions: Left arm joint positions (7 DOF)
            right_arm_positions: Right arm joint positions (7 DOF)

        Returns:
            Complete robot joint positions as numpy array
        """
        current_positions = self.motion_manager.get_joint_pos_dict()

        # Update arm positions
        self._update_arm_positions(
            current_positions, left_arm_positions, right_arm_positions
        )

        # Clip joint positions to limits
        current_positions = robot_utils.clip_joint_positions_to_limits(
            self.motion_manager.pin_robot, current_positions
        )

        # Convert to joint position array
        return robot_utils.get_qpos_from_joint_dict(
            self.motion_manager.pin_robot, current_positions
        )

    def _handle_collision(self, command: RobotCommand) -> None:
        """Handle detected collision.

        Args:
            command: RobotCommand to update
        """
        command.safety_flags.collision_detected = True

        if not self._solve_safe_ik(command):
            logger.warning("Failed to find collision-free configuration")
            command.valid = False

    def _solve_safe_ik(self, command: RobotCommand) -> bool:
        """Find collision-free joint configuration using IK.

        Attempts multiple IK solutions with position perturbations
        to escape local minima and find safe configurations.

        Args:
            command: RobotCommand with target positions

        Returns:
            True if safe configuration found, False otherwise

        Algorithm:
            1. Compute target EE poses from joint positions
            2. Solve IK for those poses
            3. If collision detected, perturb targets and retry
            4. Update command with safe positions if found
        """
        left_arm_positions = command.output_components["left_arm"]["pos"]
        right_arm_positions = command.output_components["right_arm"]["pos"]

        # Build complete state
        updated_joint_pos = self._build_robot_state(
            left_arm_positions, right_arm_positions
        )

        # Compute target end-effector poses from joint positions
        target_poses = self.motion_manager.fk(
            ["L_ee", "R_ee"], qpos=updated_joint_pos, update_robot_state=False
        )

        # Extract positions and orientations
        target_positions: Dict[str, np.ndarray] = {}
        target_orientations: Dict[str, np.ndarray] = {}

        for ee_name, pose in target_poses.items():
            if pose is not None:
                target_positions[ee_name] = pose.translation
                # Convert rotation matrix to Euler angles
                rotation = Rotation.from_matrix(pose.rotation)
                target_orientations[ee_name] = rotation.as_euler("xyz", degrees=False)

        # Try IK solving with multiple attempts
        max_attempts = 3
        position_noise_std = 0.005  # 5mm standard deviation for perturbations

        for attempt in range(max_attempts):
            try:
                # Solve IK with Pink solver
                solution, is_collision, within_limits = self.motion_manager.ik(
                    target_pos=target_positions,
                    target_rot=target_orientations,
                    type="pink",
                )

                # Update robot state with the new solution
                if not is_collision:
                    self.motion_manager.set_joint_pos(solution)

                # Check if solution is valid
                if solution is not None and within_limits and not is_collision:
                    # Update command with safe positions
                    command.output_components["left_arm"]["pos"] = [
                        solution[f"L_arm_j{i + 1}"] for i in range(7)
                    ]
                    command.output_components["right_arm"]["pos"] = [
                        solution[f"R_arm_j{i + 1}"] for i in range(7)
                    ]
                    logger.debug(
                        f"Found collision-free configuration on attempt {attempt + 1}"
                    )
                    return True

                # If not last attempt, add small perturbations to targets
                if attempt < max_attempts - 1:
                    for ee_name in target_positions:
                        # Add small random offset to help escape local minima
                        noise = np.random.normal(0, position_noise_std, 3)
                        target_positions[ee_name] = target_positions[ee_name] + noise

            except Exception as e:
                logger.debug(f"IK attempt {attempt + 1} failed: {e}")
                continue

        return False
