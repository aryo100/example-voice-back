"""
AudioRecorder: optional per-session recording of WebSocket audio to WAV/MP3.

- When recording disabled: no-op (append/finalize do nothing).
- When enabled: in-memory buffer only; flush ONLY on WebSocket close / session end.
- Gain is applied ONLY to recorded file; STT stream is never modified.
- WAV: one open → write all frames → close once. MP3: write WAV first, then convert.
- Does not depend on ASR. finalize() runs in executor to avoid blocking.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
import wave
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

# PCM contract: signed int16, little-endian, mono, 16kHz
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
NCHANNELS = 1

# RMS threshold for optional silence warning (int16 scale)
RMS_SILENCE_THRESHOLD = 100
RMS_SILENCE_CHUNKS_WARN = 50


class AudioRecorderBase(ABC):
    """Base for session recorder. append() accepts raw PCM; finalize() writes file once and returns path or None."""

    @abstractmethod
    def append(self, data: bytes) -> None:
        """Append validated PCM bytes. Safe from WebSocket receive loop. Does not affect STT stream."""
        ...

    @abstractmethod
    def finalize(self) -> Optional[str]:
        """Flush buffer, write file (WAV or WAV→MP3), close once. Run in executor. Returns path or None."""
        ...


class NoOpAudioRecorder(AudioRecorderBase):
    """Recorder when recording disabled. No buffer, no file I/O, STT unchanged."""

    def append(self, data: bytes) -> None:
        pass

    def finalize(self) -> Optional[str]:
        return None


def _validate_pcm_chunk(data: bytes) -> bool:
    """PCM contract: length divisible by 2 (int16), non-empty. Returns True if valid."""
    if len(data) == 0:
        logger.debug("Recording: dropped empty chunk")
        return False
    if len(data) % 2 != 0:
        logger.warning("Recording: dropped malformed chunk (length %d not divisible by 2)", len(data))
        return False
    return True


def _apply_gain_to_pcm(pcm_bytes: bytes, gain: float) -> bytes:
    """
    Apply gain to PCM int16 samples and clamp to [-32768, 32767].
    Used ONLY for recording buffer; STT stream is never passed here.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    scaled = (samples.astype(np.float32) * gain).clip(-32768, 32767).astype(np.int16)
    return scaled.tobytes()


def _rms_int16(pcm_bytes: bytes) -> float:
    """RMS of int16 samples (for optional silence logging)."""
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def _write_wav_sync(pcm_bytes: bytes, out_path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Write PCM to WAV: one open, set header once, write all frames, close once.
    Buffering is required so we write a single contiguous file; never open/close per chunk.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with wave.open(out_path, "wb") as wav:
        wav.setnchannels(NCHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)


def _wav_to_mp3_sync(wav_path: str, mp3_path: str, bitrate: str) -> None:
    """Convert WAV file to MP3 after recording ends. Blocking, single-threaded. Never stream PCM into encoder."""
    from pydub import AudioSegment

    segment = AudioSegment.from_wav(wav_path)
    segment.export(mp3_path, format="mp3", bitrate=bitrate)


class AudioRecorder(AudioRecorderBase):
    """
    One WebSocket session = one in-memory buffer. Append validated, gained PCM only.
    Flush ONLY on finalize() (WebSocket close / session end). No file open per chunk.
    """

    def __init__(self, session_id: str | None = None) -> None:
        settings = get_settings()
        self._session_id = session_id or uuid.uuid4().hex[:12]
        # Single buffer for whole session; flush only when session ends
        self._buffer = bytearray()
        self._sample_rate = getattr(settings, "SAMPLE_RATE", SAMPLE_RATE)
        self._record_dir = getattr(settings, "BACKEND_RECORD_DIR", None) or getattr(settings, "AUDIO_RECORD_DIR", "./recordings")
        self._format = getattr(settings, "BACKEND_RECORD_FORMAT", "wav")
        self._bitrate = getattr(settings, "BACKEND_RECORD_BITRATE", "128k")
        self._gain = max(1.0, min(4.0, getattr(settings, "BACKEND_RECORD_GAIN", 2.0)))
        self._finalized = False
        self._dropped_chunks = 0
        self._low_rms_count = 0

    def append(self, data: bytes) -> None:
        if self._finalized:
            return
        if not _validate_pcm_chunk(data):
            self._dropped_chunks += 1
            return
        # Gain applied ONLY to copy we store; STT path never sees this
        gained = _apply_gain_to_pcm(data, self._gain)
        self._buffer.extend(gained)
        # Optional: warn if many chunks are very quiet (helps debug frontend)
        rms = _rms_int16(data)
        if rms < RMS_SILENCE_THRESHOLD:
            self._low_rms_count += 1
            if self._low_rms_count == RMS_SILENCE_CHUNKS_WARN:
                logger.warning(
                    "Recording: %d consecutive chunks with RMS < %s (possible frontend silence)",
                    RMS_SILENCE_CHUNKS_WARN,
                    RMS_SILENCE_THRESHOLD,
                )
        else:
            self._low_rms_count = 0

    def finalize(self) -> Optional[str]:
        if self._finalized:
            return None
        self._finalized = True
        if len(self._buffer) == 0:
            if self._dropped_chunks:
                logger.debug("Recording: no data to write (all chunks dropped: %d)", self._dropped_chunks)
            return None
        settings = get_settings()
        os.makedirs(self._record_dir, exist_ok=True)
        ts = int(time.time())
        base = f"session_{self._session_id}_{ts}"

        # Always write WAV first: single open, single write, single close
        wav_path = os.path.join(self._record_dir, f"{base}.wav")
        _write_wav_sync(bytes(self._buffer), wav_path, self._sample_rate)

        if self._format == "mp3":
            mp3_path = os.path.join(self._record_dir, f"{base}.mp3")
            _wav_to_mp3_sync(wav_path, mp3_path, self._bitrate)
            try:
                os.remove(wav_path)
            except OSError:
                pass
            return mp3_path
        return wav_path


def create_audio_recorder() -> AudioRecorderBase:
    """Create recorder when ENABLE_AUDIO_RECORDING or ENABLE_BACKEND_RECORDING is true. Disabled by default."""
    settings = get_settings()
    if getattr(settings, "ENABLE_BACKEND_RECORDING", False) or getattr(settings, "ENABLE_AUDIO_RECORDING", False):
        return AudioRecorder()
    return NoOpAudioRecorder()
