"""
CloudflareWhisperEngine: Whisper via Cloudflare Workers AI.

Accepts float32 audio; converts to PCM bytes, wraps as WAV for API.
Runs HTTP call in executor to avoid blocking event loop.
"""
from __future__ import annotations

import asyncio
import io
import logging
import wave

import numpy as np
import httpx

from app.asr.base import ASREngine, ASRResult
from app.config import get_settings

logger = logging.getLogger(__name__)

# Cloudflare expects encoded audio (WAV/MP3 bytes), not raw PCM s16le.
_MIN_PCM_BYTES = 3200  # ~100 ms @ 16 kHz mono 16-bit


def _float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] to PCM 16-bit mono bytes."""
    samples = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return samples.tobytes()


def _pcm_bytes_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono in a WAV container (required by Workers AI Whisper)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


def _parse_whisper_response(data: dict) -> str:
    result = data.get("result", data)
    if isinstance(result, dict):
        return (result.get("text") or result.get("transcript") or "").strip()
    if isinstance(result, str):
        return result.strip()
    return ""


def _empty_result() -> ASRResult:
    return ASRResult(
        text="",
        confidence=0.0,
        word_timestamps=None,
        segments=None,
        is_final=True,
    )


def _sync_transcribe_cloudflare(pcm_bytes: bytes, sample_rate: int) -> ASRResult:
    """Blocking HTTP call; run in executor."""
    settings = get_settings()
    account_id = settings.CLOUDFLARE_ACCOUNT_ID
    token = settings.CLOUDFLARE_API_TOKEN
    if not account_id or not token:
        logger.warning("Cloudflare Whisper: CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN not set")
        return _empty_result()

    if len(pcm_bytes) < _MIN_PCM_BYTES:
        logger.debug("Cloudflare Whisper: skip short chunk (%d bytes)", len(pcm_bytes))
        return _empty_result()

    wav_bytes = _pcm_bytes_to_wav_bytes(pcm_bytes, sample_rate)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper"
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=60.0) as client:
        # Documented curl style: raw audio body (--data-binary @file.wav)
        resp = client.post(
            url,
            headers={**headers, "Content-Type": "audio/wav"},
            content=wav_bytes,
        )
        if resp.status_code != 200:
            # Workers binding style: JSON array of file bytes
            resp = client.post(url, headers=headers, json={"audio": list(wav_bytes)})

    if resp.status_code != 200:
        detail = resp.text[:800] if resp.text else ""
        logger.warning(
            "Cloudflare Whisper failed: HTTP %s (pcm=%d wav=%d) %s",
            resp.status_code,
            len(pcm_bytes),
            len(wav_bytes),
            detail,
        )
        return _empty_result()

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("Cloudflare Whisper: invalid JSON response: %s", e)
        return _empty_result()

    if not data.get("success", True) and data.get("errors"):
        logger.warning("Cloudflare Whisper API errors: %s", data.get("errors"))
        return _empty_result()

    text = _parse_whisper_response(data)
    return ASRResult(
        text=text,
        confidence=1.0 if text else 0.0,
        word_timestamps=None,
        segments=None,
        is_final=True,
    )


class CloudflareWhisperEngine(ASREngine):
    """
    Remote Whisper via Cloudflare Workers AI.
    async transcribe() runs HTTP in executor; no partial/final distinction.
    """

    async def transcribe(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        """Convert audio to PCM, wrap as WAV, run HTTP in executor. is_final ignored (one pass)."""
        pcm_bytes = _float32_to_pcm_bytes(audio)
        sample_rate = get_settings().SAMPLE_RATE
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _sync_transcribe_cloudflare,
            pcm_bytes,
            sample_rate,
        )

    @property
    def sample_rate(self) -> int:
        return get_settings().SAMPLE_RATE
