"""
AudioReceiver: accepts raw PCM audio from WebSocket and yields frames.

- Expects PCM 16-bit mono 16kHz.
- Emits fixed-size frames (e.g. 20ms = 640 bytes) for VAD/chunking.
- Runs in async context; does not block event loop.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.config import get_settings


class AudioReceiver:
    """
    Buffers incoming binary WebSocket messages into fixed-size PCM frames.
    Any remainder is kept for the next iteration.
    """

    def __init__(self, frame_bytes: int | None = None) -> None:
        settings = get_settings()
        self._frame_bytes = frame_bytes or settings.FRAME_BYTES
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        """Append raw PCM bytes. Call from WebSocket handler."""
        self._buffer.extend(data)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        """Yield one frame when enough bytes are buffered. Non-blocking."""
        while len(self._buffer) < self._frame_bytes:
            # Let other tasks run; caller typically awaits more WebSocket data
            await asyncio.sleep(0)
        frame = bytes(self._buffer[: self._frame_bytes])
        del self._buffer[: self._frame_bytes]
        return frame

    def drain_frames(self) -> list[bytes]:
        """
        Drain all complete frames from the buffer (e.g. on disconnect).
        Returns list of full frames; remainder stays in buffer.
        """
        out: list[bytes] = []
        while len(self._buffer) >= self._frame_bytes:
            out.append(bytes(self._buffer[: self._frame_bytes]))
            del self._buffer[: self._frame_bytes]
        return out

    def remaining_bytes(self) -> int:
        """Bytes left in buffer (incomplete frame)."""
        return len(self._buffer)
