"""
RollingBuffer: time-based audio window for real-time STT (no silence gating).

Why Whisper was waiting for silence:
- The previous pipeline used AudioChunker that only emitted when silence exceeded
  SILENCE_COMMIT_MS (e.g. 600ms). So no transcription happened until the user
  stopped speaking, causing long delays and only the last 1–2 seconds to appear.

How the rolling buffer solves it:
- We keep a fixed window (e.g. 5s) of the most recent audio and transcribe
  periodically (e.g. every 1s) based on time elapsed, not silence.
- Triggers: buffer_duration >= WINDOW_SIZE and elapsed_since_last_emit >= STEP.
- Partial text appears while the user is speaking; final text is committed by
  segment age (handled in WebSocketManager).
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from app.config import get_settings


class RollingBuffer:
    """
    Maintains a rolling window of PCM frames. Emits (chunk_bytes, chunk_start_sec)
    every STEP_SECONDS when the window is full. No VAD or silence detection.
    """

    def __init__(
        self,
        on_chunk: Callable[[bytes, float], None],
        frame_bytes: int | None = None,
        window_sec: float | None = None,
        step_sec: float | None = None,
        min_chunk_sec: float | None = None,
    ) -> None:
        settings = get_settings()
        self._frame_bytes = frame_bytes or settings.FRAME_BYTES
        self._frame_ms = settings.FRAME_MS
        self._frame_sec = self._frame_ms / 1000.0
        self._frames_per_sec = int(1.0 / self._frame_sec)
        self._window_sec = window_sec if window_sec is not None else settings.STT_WINDOW_SECONDS
        self._step_sec = step_sec if step_sec is not None else settings.STT_STEP_SECONDS
        self._min_chunk_sec = min_chunk_sec if min_chunk_sec is not None else settings.STT_MIN_CHUNK_SECONDS
        self._on_chunk = on_chunk

        self._window_frames = max(1, int(self._frames_per_sec * self._window_sec))
        self._step_frames = max(1, int(self._frames_per_sec * self._step_sec))
        self._min_frames = max(1, int(self._frames_per_sec * self._min_chunk_sec))

        self._buffer: deque[bytes] = deque(maxlen=self._window_frames)
        self._total_frames = 0
        self._last_emit_frame = -1

    def push(self, frame: bytes) -> None:
        """Append one frame. May trigger on_chunk when step interval reached (no silence check)."""
        self._buffer.append(frame)
        self._total_frames += 1

        # Emit when we have at least min_frames and (full window + step since last emit)
        if len(self._buffer) < self._min_frames:
            return
        if len(self._buffer) < self._window_frames:
            return
        if self._last_emit_frame >= 0 and (self._total_frames - self._last_emit_frame) < self._step_frames:
            return

        chunk = b"".join(self._buffer)
        # Chunk start time in session seconds (for segment alignment)
        chunk_start_sec = (self._total_frames - self._window_frames) * self._frame_sec
        self._last_emit_frame = self._total_frames
        self._on_chunk(chunk, chunk_start_sec)

    def flush(self) -> tuple[bytes, float] | None:
        """On disconnect: flush remaining buffer if >= min_frames. Returns (chunk, chunk_start_sec) or None."""
        if len(self._buffer) < self._min_frames:
            return None
        chunk = b"".join(self._buffer)
        chunk_start_sec = (self._total_frames - len(self._buffer)) * self._frame_sec
        return (chunk, chunk_start_sec)
