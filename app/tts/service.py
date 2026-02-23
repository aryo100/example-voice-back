"""
TTS service: pilih engine dari config.
- Lokal: edge (Edge TTS).
- Nanti pakai Cloudflare: cloudflare (TTS dari Cloudflare).
- none: matikan TTS.
synthesize_speech(text) -> (audio_base64 | None, mime_type).
"""
from __future__ import annotations

import base64
import logging
from typing import Tuple

from app.config import get_settings
from app.tts.base import TTSEngine
from app.tts.edge_tts import EdgeTTSEngine

logger = logging.getLogger(__name__)

def get_tts_engine() -> TTSEngine | None:
    """Return TTS engine from config (edge / cloudflare / none)."""
    settings = get_settings()
    backend = (getattr(settings, "TTS_BACKEND", "") or "edge").strip().lower()
    if backend == "none" or backend == "":
        return None
    if backend == "cloudflare":
        # TODO: return CloudflareTTSEngine() when implemented
        logger.warning("TTS_BACKEND=cloudflare not implemented yet; use edge")
        return None
    if backend == "edge":
        voice = getattr(settings, "TTS_EDGE_VOICE", "") or ""
        return EdgeTTSEngine(voice=voice or None)
    logger.warning("Unknown TTS_BACKEND=%s; use edge or cloudflare", backend)
    return None


async def synthesize_speech(text: str) -> Tuple[str | None, str]:
    """
    Generate TTS audio for text. Returns (base64_audio, mime_type).
    If TTS disabled or fails, returns (None, "").
    """
    engine = get_tts_engine()
    if not (text or "").strip():
        return None, ""
    if not engine:
        logger.info("TTS disabled (TTS_BACKEND=none or unknown); audio=null")
        return None, ""
    try:
        audio_bytes, mime = await engine.synthesize(text)
        if not audio_bytes:
            logger.warning("TTS returned no audio (engine=%s); check logs for 403/network", type(engine).__name__)
            return None, mime or "audio/mpeg"
        return base64.b64encode(audio_bytes).decode("ascii"), mime
    except Exception as e:
        logger.warning("TTS synthesize failed: %s", e)
        return None, ""
