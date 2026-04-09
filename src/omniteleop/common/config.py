#!/usr/bin/env python3
"""Configuration loader for the omniteleop teleoperation system.

Loads configuration from YAML file with support for environment variable override.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger
from omniteleop import LIB_PATH

class RobotConfig(dict):
    """Robot configuration loaded from YAML file.

    Inherits from dict to provide direct dictionary access while
    adding a few convenience methods for common operations.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Load configuration from YAML file.

        Args:
            config_path: Optional path to config file.
                        If None, checks ROBOT_CONFIG env var to select config,
                        then defaults to vega_1_f5d6.yaml
        """
        # Determine config path
        if config_path is None:
            # Check ROBOT_CONFIG environment variable to select config file
            robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
            config_filename = f"{robot_config}.yaml"
            config_path = LIB_PATH / "configs" / config_filename

            if not config_path.exists():
                # List available configs
                configs_dir = LIB_PATH / "configs"
                available = [
                    f.stem
                    for f in configs_dir.glob("*.yaml")
                    if f.stem != "vr_robot_config"
                ]
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}\n"
                    f"ROBOT_CONFIG='{robot_config}' is not valid.\n"
                    f"Available configs: {available}\n"
                    "Set ROBOT_CONFIG to one of the available options."
                )

            logger.info(f"Using config for ROBOT_CONFIG={robot_config}: {config_path}")

        self.config_path = Path(config_path)

        # Load YAML and initialize dict
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        super().__init__(data)
        logger.info(f"Configuration loaded from {self.config_path}")

    def get_topic(self, topic_name: str, default: str = None) -> str:
        """Get topic name from config.

        Args:
            topic_name: Name of the topic to get

        Returns:
            Topic name
        """
        return self.get("topics", {}).get(topic_name, default)

    def get_rate(self, rate_name: str, default: float = 40) -> float:
        """Get rate from config.

        Args:
            rate_name: Name of the rate to get
            default: Default rate if not found

        Returns:
            Rate
        """
        return self.get("rates", {}).get(rate_name, default)

    def get_init_pos(self, part: str) -> Dict[str, float]:
        """Get initial arm position from config.

        Args:
            part: "left_arm" or "right_arm" or "head" or "torso" or "left_hand" or "right_hand"

        Returns:
            Initial position
        """
        return self.get("init_pos", {}).get(part, None)

    def get_leader_arms(self) -> Dict[str, Any]:
        """Get leader arms configuration with backward compatibility.

        Returns:
            Leader arms configuration
        """
        # Check new unified structure first
        teleop_mode = self.get("teleop_mode", "exo_joycon")
        if teleop_mode == "exo_joycon":
            handler_config = self.get("input_handlers", {}).get("exo_joycon", {})
            if "leader_arms" in handler_config:
                return handler_config["leader_arms"]

        # Fall back to old structure for backward compatibility
        return self.get("leader_arms", {})

    def get_joycon_config(self) -> Dict[str, Any]:
        """Get JoyCon configuration with backward compatibility.

        Returns:
            JoyCon configuration
        """
        # Check new unified structure first
        teleop_mode = self.get("teleop_mode", "exo_joycon")
        if teleop_mode == "exo_joycon":
            handler_config = self.get("input_handlers", {}).get("exo_joycon", {})
            if "joycon" in handler_config:
                return handler_config["joycon"]

        # Fall back to old structure for backward compatibility
        return self.get("joycon", {})

    def get_vr_server_url(self, server_url: Optional[str] = None) -> str:
        """Get VR server URL from config.

        Args:
            server_url: Optional server URL to use if not found in config

        Returns:
            VR server URL
        """
        handler_config = self.get("input_handlers", {}).get("vr", {})
        return handler_config.get("socket", {}).get("server_url", server_url)

# Global configuration instance (singleton pattern)
_config: Optional[RobotConfig] = None

def get_config(config_path: Optional[Path] = None) -> RobotConfig:
    """Get the robot configuration (loads if needed).

    This is the primary way to access configuration throughout the codebase.
    It automatically loads configuration if not already loaded.

    Priority order for config file:
    1. Explicitly passed config_path (reloads if different)
    2. ROBOT_CONFIG environment variable (e.g., "vega_1_f5d6" -> vega_1_f5d6.yaml)
    3. Default: vega_1_f5d6.yaml

    Args:
        config_path: Optional path to config file (forces reload if provided)

    Returns:
        RobotConfig instance (dict-like object)

    Example:
        from omniteleop.common import get_config

        # Get config (loads based on ROBOT_CONFIG env var)
        config = get_config()

        # Direct dictionary access
        leader_arms = config["leader_arms"]
        motor_ids = config["leader_arms"]["left_arm"]["motor_ids"]

        # Convenience methods
        rate = config.get_rate("control_rate")
        topic = config.get_topic("exo_joints")
    """
    global _config

    # Force reload if config_path is provided
    if config_path is not None:
        _config = RobotConfig(config_path)
    elif _config is None:
        _config = RobotConfig()

    return _config
