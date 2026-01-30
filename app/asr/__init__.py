"""ASR: swappable Whisper-compatible engines."""
from .base import ASREngine, ASRResult, SegmentTimestamp, TranscriptResult, WordTimestamp
from .local_whisper import LocalWhisperEngine, pcm_bytes_to_float32
from .cloudflare import CloudflareWhisperEngine

__all__ = [
    "ASREngine",
    "ASRResult",
    "SegmentTimestamp",
    "TranscriptResult",
    "WordTimestamp",
    "LocalWhisperEngine",
    "CloudflareWhisperEngine",
    "pcm_bytes_to_float32",
]
