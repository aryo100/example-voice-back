"""
ASREngine: abstract interface for Whisper-compatible ASR.

Implementations: LocalWhisperEngine (faster-whisper), CloudflareWhisperEngine.
All run heavy work in executor to avoid blocking the event loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class WordTimestamp:
    """Single word with start/end in seconds."""

    word: str
    start: float
    end: float


@dataclass
class SegmentTimestamp:
    """One segment: start/end in seconds, text."""

    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    """Result of one ASR transcribe call."""

    text: str
    confidence: float  # 0.0–1.0 estimate
    word_timestamps: list[WordTimestamp] | None = None  # if available
    segments: list[SegmentTimestamp] | None = None  # for rolling-buffer commitment
    is_final: bool = True  # True when stable (after silence)


# Backward-compat alias
TranscriptResult = ASRResult


class ASREngine(ABC):
    """
    Abstract ASR engine. Accepts float32 mono audio (normalized [-1, 1]).
    transcribe() is async; implementations may run sync work in executor.
    """

    @abstractmethod
    async def transcribe(self, audio: "np.ndarray", is_final: bool) -> ASRResult:
        """
        Transcribe one chunk of audio.
        - is_final=False: partial (faster decode, may change).
        - is_final=True: final (stable decode, word_timestamps if available).
        Must not block event loop; run heavy work in executor.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected sample rate (e.g. 16000)."""
        ...
