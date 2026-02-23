"""
TranscriptWriter: session-based, append-only persistence of FINAL transcript segments.

Why append-only is critical:
- Final segments arrive over time; we must never overwrite or truncate.
- Each write appends one line so the full conversation order is preserved.
- Partial transcripts are excluded so the file contains only committed, stable text.

Why partial is excluded:
- Partial text is live/uncertain and may change; writing it would cause duplicates
  and wrong content when the next ASR result revises it. Only final (committed)
  segments are written.
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def _format_timestamp_line(timestamp_ms: int, session_start_ms: int, text: str) -> str:
    """Format one line with optional [MM:SS.ss] prefix (elapsed since session start)."""
    elapsed_sec = (timestamp_ms - session_start_ms) / 1000.0
    mm = int(elapsed_sec // 60)
    ss = elapsed_sec % 60
    return f"[{mm:02d}:{ss:05.2f}] {text}".strip()


def _format_speaker_line(
    text: str,
    timestamp_ms: Optional[int],
    session_start_ms: int,
    add_timestamps: bool,
    speaker_id: Optional[str] = None,
    overlap: bool = False,
) -> str:
    """Format one line with optional [MM:SS.ss], [Speaker A], [overlap] prefix.
    Speaker labels and overlap are approximate; see diarization module limitations."""
    parts: list[str] = []
    if add_timestamps and timestamp_ms is not None:
        elapsed_sec = (timestamp_ms - session_start_ms) / 1000.0
        mm = int(elapsed_sec // 60)
        ss = elapsed_sec % 60
        parts.append(f"[{mm:02d}:{ss:05.2f}]")
    if speaker_id:
        parts.append(f"[{speaker_id}]")
    if overlap:
        parts.append("[overlap]")
    parts.append(text.strip())
    return " ".join(parts)


class TranscriptWriterBase(ABC):
    """Base for session transcript writer. Only final segments are appended; partial is never written."""

    @abstractmethod
    async def start(self) -> None:
        """Open file when WebSocket starts. Call once at session start."""
        ...

    @abstractmethod
    def append_final(
        self,
        text: str,
        timestamp_ms: Optional[int] = None,
        speaker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        overlap: bool = False,
    ) -> None:
        """Append one final segment (one line). Non-blocking; queues write. Partial must never call this."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Flush and close file when WebSocket closes. Safe to call from finally."""
        ...


class NoOpTranscriptWriter(TranscriptWriterBase):
    """When transcript saving is disabled. No file I/O."""

    async def start(self) -> None:
        pass

    def append_final(
        self,
        text: str,
        timestamp_ms: Optional[int] = None,
        speaker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        overlap: bool = False,
    ) -> None:
        pass

    async def close(self) -> None:
        pass


class TranscriptWriter(TranscriptWriterBase):
    """
    One file per session: transcripts/{session_id}.txt.
    Append-only; one line per final segment; optional [MM:SS.ss] prefix.
    Worker task drains queue so we never block the transcription loop.
    """

    def __init__(
        self,
        session_id: str,
        session_start_ms: int,
        transcript_dir: Optional[str] = None,
        add_timestamps: Optional[bool] = None,
    ) -> None:
        settings = get_settings()
        self._session_id = session_id
        self._session_start_ms = session_start_ms
        self._transcript_dir = transcript_dir or settings.TRANSCRIPT_DIR
        self._add_timestamps = add_timestamps if add_timestamps is not None else settings.TRANSCRIPT_ADD_TIMESTAMPS
        self._path = os.path.join(self._transcript_dir, f"{session_id}.txt")
        self._file = None
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._started = False

    async def _worker(self) -> None:
        """Drain queue: write each line (append + newline + flush). None = close. Log errors, never crash."""
        while True:
            line = await self._queue.get()
            if line is None:
                break
            if self._file is None:
                continue
            try:
                self._file.write(line + "\n")
                self._file.flush()
            except OSError as e:
                logger.warning("Transcript write failed for %s: %s", self._path, e)
        try:
            if self._file is not None:
                self._file.close()
        except OSError as e:
            logger.warning("Transcript close failed for %s: %s", self._path, e)
        finally:
            self._file = None

    async def start(self) -> None:
        """Open file when WebSocket starts; start worker so writes are async-safe."""
        if self._started:
            return
        self._started = True
        try:
            os.makedirs(self._transcript_dir, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")
        except OSError as e:
            logger.warning("Transcript file open failed for %s: %s", self._path, e)
        self._worker_task = asyncio.create_task(self._worker())

    def append_final(
        self,
        text: str,
        timestamp_ms: Optional[int] = None,
        speaker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        overlap: bool = False,
    ) -> None:
        """Append one final segment. Partial must never call this. Non-blocking (queue)."""
        text = (text or "").strip()
        if not text:
            return
        if speaker_id is not None or overlap:
            line = _format_speaker_line(
                text, timestamp_ms, self._session_start_ms, self._add_timestamps, speaker_id, overlap
            )
        elif self._add_timestamps and timestamp_ms is not None:
            line = _format_timestamp_line(timestamp_ms, self._session_start_ms, text)
        else:
            line = text
        try:
            self._queue.put_nowait(line)
        except asyncio.QueueFull:
            logger.warning("Transcript queue full for session %s, dropping line", self._session_id)

    async def close(self) -> None:
        """Signal worker to stop and close file. Safe flush to disk."""
        if not self._started or self._worker_task is None:
            return
        try:
            self._queue.put_nowait(None)
            await asyncio.wait_for(self._worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None


def create_transcript_writer(session_id: str, session_start_ms: int) -> TranscriptWriterBase:
    """Create writer when TRANSCRIPT_SAVE_ENABLED is true; else no-op."""
    settings = get_settings()
    if not getattr(settings, "TRANSCRIPT_SAVE_ENABLED", True):
        return NoOpTranscriptWriter()
    return TranscriptWriter(
        session_id=session_id,
        session_start_ms=session_start_ms,
    )
