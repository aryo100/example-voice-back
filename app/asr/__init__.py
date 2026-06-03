"""ASR: swappable Whisper-compatible engines."""
from .base import ASREngine, ASRResult, SegmentTimestamp, TranscriptResult, WordTimestamp
from .local_whisper import LocalWhisperEngine, pcm_bytes_to_float32
from .cloudflare import CloudflareWhisperEngine
from .coqui import (
    CoquiSTTEngine,
    is_coqui_auto_mode,
    load_coqui_auto_whisper,
    load_coqui_model,
)

__all__ = [
    "ASREngine",
    "ASRResult",
    "SegmentTimestamp",
    "TranscriptResult",
    "WordTimestamp",
    "LocalWhisperEngine",
    "CloudflareWhisperEngine",
    "CoquiSTTEngine",
    "load_coqui_model",
    "load_coqui_auto_whisper",
    "is_coqui_auto_mode",
    "pcm_bytes_to_float32",
]
