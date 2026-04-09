"""State publisher module - handles state streaming via WebSocket."""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect

class StatePublisher:
    """Handles state streaming."""

    def __init__(self):
        """Initialize state publisher."""
        self.active_connections = set()
        self._get_state_callback = None
        self._on_connection_callback = None
        self._on_disconnection_callback = None

    def set_state_callback(self, callback):
        """Set callback function to get current state."""
        self._get_state_callback = callback

    def set_connection_callback(self, callback):
        """Set callback function to be called when frontend connects."""
        self._on_connection_callback = callback

    def set_disconnection_callback(self, callback):
        """Set callback function to be called when all frontend connections close."""
        self._on_disconnection_callback = callback

    async def add_connection(self, websocket: WebSocket):
        """Add a WebSocket connection for state streaming."""
        await websocket.accept()
        self.active_connections.add(websocket)

    async def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

        # Notify when all connections are closed
        if len(self.active_connections) == 0 and self._on_disconnection_callback:
            self._on_disconnection_callback()

    async def broadcast_state(self):
        """Broadcast state to all connected clients."""
        if not self._get_state_callback:
            return

        state_dict = await self._get_state_callback()
        disconnected = set()

        for ws in self.active_connections:
            try:
                await ws.send_json(state_dict)
            except Exception:
                disconnected.add(ws)

        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.discard(ws)

        # Notify when all connections are closed
        if (
            len(self.active_connections) == 0
            and disconnected
            and self._on_disconnection_callback
        ):
            self._on_disconnection_callback()

    async def start_stream(self, websocket: WebSocket):
        """Start streaming state to a WebSocket."""
        await self.add_connection(websocket)

        # Notify that frontend has connected (triggers 1-second delay before real states)
        if self._on_connection_callback:
            asyncio.create_task(self._on_connection_callback())

        try:
            interval = 1.0 / 30.0  # ~30 messages per second
            while websocket in self.active_connections:
                await self.broadcast_state()
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"State stream error: {e}")
        finally:
            await self.remove_connection(websocket)

# Global state publisher instance
state_publisher = StatePublisher()
