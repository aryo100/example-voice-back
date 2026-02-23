"""
Edge TTS engine (Microsoft Edge online TTS). Untuk local; gratis, tidak butuh API key.
Jika dapat 403 dari Microsoft: cek jaringan/region atau set TTS_BACKEND=none.
"""
from __future__ import annotations

import logging
import re
from typing import List

from app.tts.base import TTSEngine

logger = logging.getLogger(__name__)

# Default voice: Indonesia atau en-US
DEFAULT_VOICE = "id-ID-ArdiNeural"  # Indonesia; fallback en-US-GuyNeural

# Chunk panjang teks agar tidak kena batas/403; gabung MP3 per chunk
MAX_CHARS_PER_CHUNK = 800


def _split_text_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Pecah teks jadi potongan ~max_chars, pecah di batas kalimat bila bisa."""
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    # Pecah di . ! ? \n
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    current: List[str] = []
    current_len = 0
    for p in parts:
        if current_len + len(p) + 1 <= max_chars:
            current.append(p)
            current_len += len(p) + 1
        else:
            if current:
                chunks.append(" ".join(current))
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                current = []
                current_len = 0
            else:
                current = [p]
                current_len = len(p) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class EdgeTTSEngine(TTSEngine):
    """TTS via edge-tts (Microsoft Edge). Output: MP3."""

    def __init__(self, voice: str | None = None) -> None:
        self._voice = (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE

    @property
    def format(self) -> str:
        return "audio/mpeg"

    async def _synthesize_one(self, text: str) -> bytes:
        """Satu request ke Edge TTS; return raw MP3 bytes atau b''."""
        try:
            import edge_tts
        except ImportError as err:
            logger.error("edge-tts not installed: %s", err)
            return b""
        communicate = edge_tts.Communicate(text.strip(), self._voice)
        chunks: List[bytes] = []
        try:
            async for chunk in communicate.stream():
                if isinstance(chunk, dict) and chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if data:
                        chunks.append(data)
        except Exception as e:
            logger.error(
                "Edge TTS stream failed (403 = region/network?): %s",
                e,
                exc_info=False,
            )
            return b""
        out = b"".join(chunks)
        if not out:
            logger.warning("Edge TTS returned no audio for %d chars", len(text))
        return out

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if not (text or "").strip():
            return b"", self.format
        text_stripped = text.strip()
        if len(text_stripped) <= MAX_CHARS_PER_CHUNK:
            audio = await self._synthesize_one(text_stripped)
            return audio, self.format
        # Teks panjang: pecah per chunk, synthesize, gabung MP3
        text_chunks = _split_text_chunks(text_stripped, MAX_CHARS_PER_CHUNK)
        all_bytes: List[bytes] = []
        for i, seg in enumerate(text_chunks):
            if not seg.strip():
                continue
            seg_audio = await self._synthesize_one(seg)
            if not seg_audio:
                break
            all_bytes.append(seg_audio)
        return b"".join(all_bytes), self.format
