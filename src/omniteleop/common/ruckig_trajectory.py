"""Ruckig-based trajectory generators for smooth, jerk-limited motion."""

import numpy as np
import ruckig

class RuckigArmTrajectoryGenerator:
    """Jerk-limited trajectory generator for 7-DOF arm joints."""

    def __init__(
        self,
        init_qpos: np.ndarray,
        control_cycle: float = 0.005,
        safety_factor: float = 3.0,
    ) -> None:
        self.dof = 7

        self.otg = ruckig.Ruckig(self.dof, control_cycle)
        self.inp = ruckig.InputParameter(self.dof)
        self.out = ruckig.OutputParameter(self.dof)

        self.inp.current_position = init_qpos.tolist()
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof

        self.inp.max_velocity = (
            np.deg2rad([180, 180, 220, 220, 220, 220, 220]) / safety_factor
        ).tolist()
        self.inp.max_acceleration = (
            np.deg2rad([600, 600, 600, 600, 600, 600, 600]) / safety_factor
        ).tolist()
        self.inp.max_jerk = (
            np.deg2rad([6000, 6000, 6000, 6000, 6000, 6000, 6000]) / safety_factor
        ).tolist()

    def update(
        self, target_position: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step the trajectory. Optionally set a new target position first."""
        if target_position is not None:
            self.inp.target_position = target_position.tolist()
            self.inp.target_velocity = [0.0] * self.dof
            self.inp.target_acceleration = [0.0] * self.dof

        _ = self.otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)

        return np.array(self.out.new_position), np.array(self.out.new_velocity)

    def reset(self, qpos: np.ndarray) -> None:
        """Reset generator to a new position with zero velocity/acceleration."""
        self.inp.current_position = qpos.tolist()
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof
        self.inp.target_position = qpos.tolist()
        self.inp.target_velocity = [0.0] * self.dof
        self.inp.target_acceleration = [0.0] * self.dof

class RuckigTorsoTrajectoryGenerator:
    """Jerk-limited trajectory generator with conservative limits for torso joints."""

    def __init__(
        self,
        init_qpos: np.ndarray,
        control_cycle: float = 0.005,
        safety_factor: float = 3.0,
    ) -> None:
        self.dof = len(init_qpos)

        self.otg = ruckig.Ruckig(self.dof, control_cycle)
        self.inp = ruckig.InputParameter(self.dof)
        self.out = ruckig.OutputParameter(self.dof)

        self.inp.current_position = init_qpos.tolist()
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof

        self.inp.max_velocity = (np.deg2rad([180] * self.dof) / safety_factor).tolist()
        self.inp.max_acceleration = (
            np.deg2rad([200] * self.dof) / safety_factor
        ).tolist()
        self.inp.max_jerk = (np.deg2rad([3000] * self.dof) / safety_factor).tolist()

    def update(
        self, target_position: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step the trajectory. Optionally set a new target position first."""
        if target_position is not None:
            self.inp.target_position = target_position.tolist()
            self.inp.target_velocity = [0.0] * self.dof
            self.inp.target_acceleration = [0.0] * self.dof

        _ = self.otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)

        return np.array(self.out.new_position), np.array(self.out.new_velocity)

    def reset(self, qpos: np.ndarray) -> None:
        """Reset generator to a new position with zero velocity/acceleration."""
        self.inp.current_position = qpos.tolist()
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof
        self.inp.target_position = qpos.tolist()
        self.inp.target_velocity = [0.0] * self.dof
        self.inp.target_acceleration = [0.0] * self.dof
