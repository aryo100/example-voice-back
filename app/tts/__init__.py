"""
TTS: text-to-speech for assistant replies.

- edge: Edge TTS (local / free, Microsoft).
- cloudflare: Cloudflare TTS (to be added when using Cloudflare).
"""
from __future__ import annotations

from app.tts.base import TTSEngine
from app.tts.edge_tts import EdgeTTSEngine
from app.tts.service import synthesize_speech, get_tts_engine

__all__ = ["TTSEngine", "EdgeTTSEngine", "synthesize_speech", "get_tts_engine"]
