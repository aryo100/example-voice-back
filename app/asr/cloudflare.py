"""
CloudflareWhisperEngine: Whisper via Cloudflare Workers AI.

Accepts float32 audio; converts to PCM bytes for API.
Runs HTTP call in executor to avoid blocking event loop.
"""
from __future__ import annotations

import asyncio

import numpy as np
import httpx

from app.asr.base import ASREngine, ASRResult
from app.config import get_settings


def _float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] to PCM 16-bit mono bytes."""
    samples = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return samples.tobytes()


def _sync_transcribe_cloudflare(pcm_bytes: bytes) -> ASRResult:
    """Blocking HTTP call; run in executor."""
    settings = get_settings()
    account_id = settings.CLOUDFLARE_ACCOUNT_ID
    token = settings.CLOUDFLARE_API_TOKEN
    if not account_id or not token:
        return ASRResult(text="", confidence=0.0, word_timestamps=None, segments=None, is_final=True)

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"audio": list(pcm_bytes)}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        return ASRResult(text="", confidence=0.0, word_timestamps=None, segments=None, is_final=True)

    data = resp.json()
    result = data.get("result", data)
    if isinstance(result, dict):
        text = result.get("text", result.get("transcript", ""))
    elif isinstance(result, str):
        text = result
    else:
        text = ""
    text = (text or "").strip()
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
        """Convert audio to PCM, run HTTP in executor. is_final ignored (one pass)."""
        pcm_bytes = _float32_to_pcm_bytes(audio)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _sync_transcribe_cloudflare,
            pcm_bytes,
        )

    @property
    def sample_rate(self) -> int:
        return get_settings().SAMPLE_RATE
