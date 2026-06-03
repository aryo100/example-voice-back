"""
CoquiSTTEngine: Coqui STT (id/en) or faster-whisper auto language (mixed Indo + English).

- COQUI_STT_LANG=id|en: Coqui TFLite + scorer (auto-download on first use).
- COQUI_STT_LANG=auto: faster-whisper with language=None (per-chunk auto-detect), same as reference script.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from app.asr.base import ASREngine, ASRResult, SegmentTimestamp
from app.asr.coqui_models import ensure_coqui_model
from app.config import get_settings

logger = logging.getLogger(__name__)

CoquiModelT = Any
WhisperModelT = Any


def _coqui_lang() -> str:
    return (get_settings().COQUI_STT_LANG or "id").strip().lower()


def is_coqui_auto_mode(lang: str | None = None) -> bool:
    return (lang or _coqui_lang()) == "auto"


def load_coqui_auto_whisper(
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModelT:
    """Load faster-whisper for COQUI_STT_LANG=auto (bilingual / mixed conversation)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as err:
        raise ImportError(
            "faster-whisper is required for COQUI_STT_LANG=auto. "
            "Install with: pip install faster-whisper"
        ) from err

    settings = get_settings()
    name = model_name or settings.COQUI_AUTO_WHISPER_MODEL
    dev = device or settings.LOCAL_WHISPER_DEVICE
    ctype = compute_type or settings.LOCAL_WHISPER_COMPUTE_TYPE
    model = WhisperModel(name, device=dev, compute_type=ctype)
    logger.info(
        "Coqui auto mode: faster-whisper loaded model=%s device=%s compute_type=%s (language=auto)",
        name,
        dev,
        ctype,
    )
    return model


def load_coqui_model(lang: str | None = None, model_dir: str | None = None) -> CoquiModelT:
    """Load Coqui STT Model with external scorer. Downloads weights if needed."""
    language = (lang or _coqui_lang()).strip().lower()
    if language == "auto":
        raise ValueError(
            "COQUI_STT_LANG=auto uses faster-whisper, not Coqui. Call load_coqui_auto_whisper() instead."
        )

    try:
        from stt import Model
    except ImportError as err:
        raise ImportError(
            "stt (Coqui STT) is required for ASR_BACKEND=coqui with COQUI_STT_LANG=id|en. "
            "Install with: pip install 'numpy<2' stt==1.4.0 (Python 3.10, Linux recommended)"
        ) from err

    settings = get_settings()
    directory = model_dir or settings.COQUI_MODEL_DIR
    model_path, scorer_path = ensure_coqui_model(language, directory)

    model = Model(str(model_path))
    model.enableExternalScorer(str(scorer_path))
    logger.info(
        "Coqui STT loaded lang=%s sample_rate=%s model=%s",
        language,
        model.sampleRate(),
        model_path,
    )
    return model


def _resample_float32(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or len(audio) == 0:
        return audio
    duration = len(audio) / from_sr
    new_len = max(1, int(round(duration * to_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def _float32_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767.0).clip(-32768, 32767).astype(np.int16)


class CoquiSTTEngine(ASREngine):
    """
    ASR when ASR_BACKEND=coqui.
    - id/en: shared Coqui Model (singleton).
    - auto: shared faster-whisper with language=None per chunk.
    """

    def __init__(
        self,
        model: CoquiModelT | None = None,
        whisper_model: WhisperModelT | None = None,
    ) -> None:
        self._model = model
        self._whisper = whisper_model

    def _transcribe_coqui_sync(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        if self._model is None:
            return ASRResult(
                text="",
                confidence=0.0,
                word_timestamps=None,
                segments=None,
                is_final=is_final,
            )

        pipeline_sr = get_settings().SAMPLE_RATE
        model_sr = int(self._model.sampleRate())
        chunk = _resample_float32(audio, pipeline_sr, model_sr)
        pcm = _float32_to_int16(chunk)

        if len(pcm) < model_sr // 10:
            return ASRResult(
                text="",
                confidence=0.0,
                word_timestamps=None,
                segments=None,
                is_final=is_final,
            )

        try:
            text = (self._model.stt(pcm) or "").strip()
        except Exception as e:
            logger.warning("Coqui STT failed: %s", e)
            text = ""

        return ASRResult(
            text=text,
            confidence=1.0 if text else 0.0,
            word_timestamps=None,
            segments=None,
            is_final=is_final,
        )

    def _transcribe_whisper_auto_sync(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        if self._whisper is None:
            return ASRResult(
                text="",
                confidence=0.0,
                word_timestamps=None,
                segments=None,
                is_final=is_final,
            )

        settings = get_settings()
        beam_size = (
            settings.LOCAL_WHISPER_BEAM_SIZE_FINAL
            if is_final
            else settings.LOCAL_WHISPER_BEAM_SIZE_PARTIAL
        )

        segments_iter, info = self._whisper.transcribe(
            audio,
            language=None,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
            condition_on_previous_text=is_final,
        )

        detected = getattr(info, "language", None) or "?"
        parts: list[str] = []
        seg_ts: list[SegmentTimestamp] = []
        for seg in segments_iter:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)
                seg_ts.append(SegmentTimestamp(start=seg.start, end=seg.end, text=t))

        text = " ".join(parts).strip() if parts else ""
        if text and detected != "?":
            logger.debug("Coqui auto: detected_lang=%s chunk_len=%.2fs", detected, len(audio) / settings.SAMPLE_RATE)

        return ASRResult(
            text=text,
            confidence=1.0 if text else 0.0,
            word_timestamps=None,
            segments=seg_ts if seg_ts else None,
            is_final=is_final,
        )

    def _transcribe_sync(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        if self._whisper is not None:
            return self._transcribe_whisper_auto_sync(audio, is_final)
        return self._transcribe_coqui_sync(audio, is_final)

    async def transcribe(self, audio: np.ndarray, is_final: bool) -> ASRResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio,
            is_final,
        )

    @property
    def sample_rate(self) -> int:
        if self._model is not None:
            return int(self._model.sampleRate())
        return get_settings().SAMPLE_RATE
