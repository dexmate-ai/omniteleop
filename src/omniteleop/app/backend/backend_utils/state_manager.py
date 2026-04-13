"""Teleop state manager — tracks estop only.

The job of this module is to gate the E-Stop flag so the video publisher
and recording system can react to it.  All other status (robot health,
subprocess liveness) is derived by StateChecker and TeleopApp directly.
"""

import asyncio
import time
from typing import Optional

class StateManager:
    def __init__(self):
        self._estop = False
        self._lock = asyncio.Lock()

    async def estop(self):
        async with self._lock:
            self._estop = True

    async def clear_estop(self):
        async with self._lock:
            self._estop = False

    async def is_estop(self) -> bool:
        async with self._lock:
            return self._estop

    async def get_state_dict(
        self, is_recording: bool = False, episode_id: Optional[str] = None
    ):
        async with self._lock:
            return {
                "timestamp_posix": time.time(),
                "estop": self._estop,
                "is_recording": is_recording,
                "episode_id": episode_id,
            }

# Global singleton
state_manager = StateManager()
