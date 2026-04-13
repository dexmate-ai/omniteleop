import os
from typing import Dict, Any

def get_robot_and_hand_types():
    robot_config = os.environ["ROBOT_CONFIG"]
    if "gripper" in robot_config:
        return "gripper", robot_config
    else:
        return "hand_f5d6", robot_config

def has_chassis_action(robot_config: Dict[str, Any]) -> bool:
    return "chassis" in robot_config["filters"]["components"].keys()
