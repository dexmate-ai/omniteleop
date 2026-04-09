"""Video publisher module - handles video streaming via WebSocket."""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Callable

class VideoPublisher:
    """Handles video streaming for cameras."""

    def __init__(self):
        """Initialize video publisher."""
        self.active_streams = {}  # camera_id -> WebSocket
        self._get_frame_callback = None  # Callback to get camera frames

    def set_frame_callback(self, callback: Callable[[str], bytes]):
        """Set callback function to get camera frames.

        Args:
            callback: Function that takes camera_id and returns frame bytes
        """
        self._get_frame_callback = callback

    async def start_stream(self, websocket: WebSocket, camera_id: str):
        """Start streaming video for a camera."""
        if not self._get_frame_callback:
            await websocket.close(code=500, reason="Frame callback not set")
            return

        await websocket.accept()
        self.active_streams[camera_id] = websocket

        interval = 1.0 / 30.0  # ~30 FPS

        try:
            while camera_id in self.active_streams:
                # Get frame from callback (this should not raise send errors)
                try:
                    frame = self._get_frame_callback(camera_id)
                except Exception as e:
                    # Error getting frame from callback
                    error_msg = str(e).lower()
                    # If it's a send/close error, the websocket is likely closed
                    if "send" in error_msg or "close" in error_msg:
                        break
                    # Otherwise, just log and continue
                    print(f"Error getting frame for {camera_id}: {e}")
                    await asyncio.sleep(interval)
                    continue

                # Try to send frame
                if frame:
                    try:
                        await websocket.send_bytes(frame)
                    except (
                        WebSocketDisconnect,
                        RuntimeError,
                        ConnectionError,
                        OSError,
                    ) as e:
                        # WebSocket was closed or connection lost, break out of loop
                        error_msg = str(e).lower()
                        # Don't print error for normal disconnects
                        if "close message" not in error_msg:
                            print(f"WebSocket send error for {camera_id}: {e}")
                        break

                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            # Normal disconnect, no error needed
            pass
        except (RuntimeError, ConnectionError, OSError) as e:
            # WebSocket connection errors - only log if not about closing
            error_msg = str(e).lower()
            if "close message" not in error_msg:
                print(f"Video stream connection error for {camera_id}: {e}")
        except Exception as e:
            # Other errors
            error_msg = str(e).lower()
            if "close message" not in error_msg:
                print(f"Video stream error for {camera_id}: {e}")
        finally:
            # Always remove from active streams
            self.active_streams.pop(camera_id, None)

    async def stop_stream(self, camera_id: str):
        """Stop streaming for a camera."""
        if camera_id in self.active_streams:
            ws = self.active_streams[camera_id]
            try:
                await ws.close()
            except:
                pass
            self.active_streams.pop(camera_id, None)

    async def stop_all_streams(self):
        """Stop all active streams."""
        for camera_id in list(self.active_streams.keys()):
            await self.stop_stream(camera_id)

# Global video publisher instance
video_publisher = VideoPublisher()
