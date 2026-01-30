"""
AudioChunker: ring buffer that emits overlapping chunks for ASR.

- Frame size: 20ms.
- Chunk size: ~1.5 seconds (configurable).
- Overlap between consecutive chunks: 300ms (configurable).
- Chunks are emitted when silence is detected for > SILENCE_COMMIT_MS (e.g. 600ms)
  to avoid cutting words.

Logic:
1. Push frames into a ring buffer.
2. When we have at least one full chunk of audio AND we detect silence for
   SILENCE_COMMIT_MS, we emit a chunk (with overlap from previous chunk).
3. Overlap is implemented by keeping OVERLAP_MS of audio at the start of the
   next chunk (ring buffer naturally gives us continuity).
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from app.config import get_settings


class AudioChunker:
    """
    Maintains a ring buffer of PCM frames. Emits chunks (bytes) when
    silence duration exceeds SILENCE_COMMIT_MS, with overlap for context.
    """

    def __init__(
        self,
        on_chunk: Callable[[bytes], None],
        frame_bytes: int | None = None,
        chunk_duration_ms: int | None = None,
        overlap_ms: int | None = None,
        silence_commit_ms: int | None = None,
    ) -> None:
        settings = get_settings()
        self._frame_bytes = frame_bytes or settings.FRAME_BYTES
        self._frame_ms = settings.FRAME_MS
        self._chunk_duration_ms = chunk_duration_ms or settings.CHUNK_DURATION_MS
        self._overlap_ms = overlap_ms or settings.OVERLAP_MS
        self._silence_commit_ms = silence_commit_ms or settings.SILENCE_COMMIT_MS
        self._on_chunk = on_chunk

        # Frames per chunk (e.g. 1500/20 = 75)
        self._frames_per_chunk = self._chunk_duration_ms // self._frame_ms
        # Overlap in frames (e.g. 300/20 = 15)
        self._overlap_frames = self._overlap_ms // self._frame_ms
        # Silence commit in frames (e.g. 600/20 = 30)
        self._silence_commit_frames = self._silence_commit_ms // self._frame_ms

        # Ring buffer: list of frames (each is bytes)
        self._buffer: deque[bytes] = deque(maxlen=self._frames_per_chunk + self._overlap_frames)
        # Consecutive silence frame count
        self._silence_frames: int = 0
        # Whether we have ever seen speech in current segment (to avoid emitting empty chunks)
        self._had_speech: bool = False

    def push(self, frame: bytes, is_speech: bool) -> None:
        """
        Push one frame and its VAD result. May trigger on_chunk callback
        when silence duration exceeds threshold after speech.
        """
        self._buffer.append(frame)
        if is_speech:
            self._silence_frames = 0
            self._had_speech = True
        else:
            self._silence_frames += 1

        # Emit chunk when: we have enough frames, we had speech, and silence long enough
        if (
            len(self._buffer) >= self._frames_per_chunk
            and self._had_speech
            and self._silence_frames >= self._silence_commit_frames
        ):
            # Build chunk: use full buffer (includes overlap with next segment)
            chunk = b"".join(self._buffer)
            self._on_chunk(chunk)
            # Reset for next segment: keep last overlap_frames for continuity
            self._had_speech = False
            self._silence_frames = 0
            keep = list(self._buffer)[-self._overlap_frames:]
            self._buffer = deque(keep, maxlen=self._frames_per_chunk + self._overlap_frames)

    def flush(self) -> bytes | None:
        """
        Flush remaining buffer as one chunk (e.g. on disconnect).
        Returns chunk bytes if there was any audio, else None.
        """
        if not self._buffer:
            return None
        chunk = b"".join(self._buffer)
        self._buffer.clear()
        self._had_speech = False
        self._silence_frames = 0
        return chunk
