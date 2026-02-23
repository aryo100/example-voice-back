"""
TTS engine interface. Implementations: Edge TTS (local), Cloudflare (later).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Abstract TTS. synthesize(text) returns (audio_bytes, mime_type)."""

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """
        Convert text to speech. Returns (raw_audio_bytes, mime_type).
        e.g. (mp3_bytes, "audio/mpeg") or (wav_bytes, "audio/wav").
        """
        ...

    @property
    @abstractmethod
    def format(self) -> str:
        """MIME type of output, e.g. 'audio/mpeg'."""
        ...
