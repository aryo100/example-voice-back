"""
LocalWhisperEngine: Whisper-compatible ASR using faster-whisper.

- Model loaded ONCE at startup (singleton, injected at construction).
- PARTIAL: lower beam size, faster decode, may change.
- FINAL: higher beam size, stable decode, word_timestamps when available.
- Audio: float32 mono [-1, 1]; PCM bytes converted in pipeline.
- Runs in executor so event loop stays responsive.
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.asr.base import ASREngine, ASRResult, SegmentTimestamp, WordTimestamp
from app.config import get_settings

# Type for shared WhisperModel (loaded at startup)
WhisperModelT = Any


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM 16-bit mono bytes to float32 [-1.0, 1.0]. Keeps overlap audio."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


class LocalWhisperEngine(ASREngine):
    """
    Local Whisper via faster-whisper. Uses shared model (singleton).
    transcribe() is async; heavy work runs in executor.
    """

    def __init__(self, model: WhisperModelT | None = None) -> None:
        """
        model: shared WhisperModel instance (loaded at app startup).
        If None, engine will not work until model is set (e.g. for lazy init).
        """
        self._model = model

    def _transcribe_sync(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        """
        Synchronous transcribe; run from executor.
        PARTIAL: beam_size=1, no word_timestamps, faster.
        FINAL: beam_size=5, word_timestamps=True, stable.
        """
        if self._model is None:
            return ASRResult(text="", confidence=0.0, word_timestamps=None, segments=None, is_final=is_final)

        settings = get_settings()
        beam_size = settings.LOCAL_WHISPER_BEAM_SIZE_FINAL if is_final else settings.LOCAL_WHISPER_BEAM_SIZE_PARTIAL
        word_timestamps = is_final and settings.LOCAL_WHISPER_WORD_TIMESTAMPS

        segments, _ = self._model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
            condition_on_previous_text=is_final,  # False for partial = faster
            word_timestamps=word_timestamps,
        )

        parts: list[str] = []
        word_ts: list[WordTimestamp] = []
        seg_ts: list[SegmentTimestamp] = []

        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)
                seg_ts.append(SegmentTimestamp(start=seg.start, end=seg.end, text=t))
            if word_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    word_ts.append(
                        WordTimestamp(
                            word=w.word or "",
                            start=w.start,
                            end=w.end,
                        )
                    )

        text = " ".join(parts).strip() if parts else ""
        confidence = 1.0 if text else 0.0
        return ASRResult(
            text=text,
            confidence=confidence,
            word_timestamps=word_ts if word_ts else None,
            segments=seg_ts if seg_ts else None,
            is_final=is_final,
        )

    async def transcribe(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        """Run _transcribe_sync in executor so event loop is not blocked."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio,
            is_final,
        )

    @property
    def sample_rate(self) -> int:
        return get_settings().SAMPLE_RATE
