"""
VADProcessor: Voice Activity Detection on 20ms PCM frames.

Uses webrtcvad (aggressiveness 0–3). Detects speech vs silence per frame.
Caller uses this to decide when to commit chunks (e.g. after >600ms silence).
"""
from __future__ import annotations

import webrtcvad
from app.config import get_settings


class VADProcessor:
    """
    Wraps webrtcvad. Frame must be exactly 10, 20, or 30 ms of 16 kHz mono PCM.
    We use 20ms frames (320 samples = 640 bytes).
    """

    def __init__(self, aggressiveness: int = 2) -> None:
        """
        aggressiveness: 0 (least aggressive) to 3 (most aggressive).
        Higher = more frames classified as silence.
        """
        self._vad = webrtcvad.Vad(aggressiveness)
        settings = get_settings()
        self._frame_bytes = settings.FRAME_BYTES
        self._frame_ms = settings.FRAME_MS
        self._sample_rate = settings.SAMPLE_RATE

    def is_speech(self, frame: bytes) -> bool:
        """
        Returns True if frame contains speech, False if silence/noise.
        frame must be exactly FRAME_BYTES (e.g. 640 for 20ms @ 16kHz).
        """
        if len(frame) != self._frame_bytes:
            return False
        return self._vad.is_speech(frame, self._sample_rate)

    @property
    def frame_ms(self) -> int:
        return self._frame_ms

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes
