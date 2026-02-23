"""
TranscriptMerger: produces partial (real-time) and final (committed) transcripts.

- PARTIAL: sent as soon as we have interim ASR for current chunk; may change.
- FINAL: sent when chunk is committed after silence (>600ms); stable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class TranscriptMessage:
    """
    Message sent to client over WebSocket.

    Partial transcripts may not include speaker info.
    Final transcripts must be speaker-tagged when diarization is enabled;
    speaker assignment may be revised on finalization.
    """

    type: str  # "partial" | "final"
    text: str
    confidence: float
    timestamp: int  # unix_ms
    word_timestamps: list[dict] | None = None  # [{"word": str, "start": float, "end": float}]
    # Speaker-aware (final only; partial may omit)
    speaker_id: str | None = None  # e.g. "Speaker A", "Speaker B"
    start_time: float | None = None  # segment start, session-relative seconds
    end_time: float | None = None   # segment end, session-relative seconds
    overlap: bool = False          # True when overlapping speech detected


def _unix_ms() -> int:
    return int(time.time() * 1000)


class TranscriptMerger:
    """
    Receives ASR results and invokes callback with TranscriptMessage.
    Chunk results are final; streaming partials can be emitted when
    we have interim results (e.g. from streaming ASR). Here we treat
    chunk result as final; partial can be same text until final.
    """

    def __init__(self, on_message: Callable[[TranscriptMessage], None]) -> None:
        self._on_message = on_message
        self._pending_partial: str = ""

    def on_partial_result(self, text: str, confidence: float = 1.0) -> None:
        """Emit partial transcript (real-time, may change)."""
        self._pending_partial = text
        self._on_message(
            TranscriptMessage(
                type="partial",
                text=text,
                confidence=confidence,
                timestamp=_unix_ms(),
            )
        )

    def on_final_result(self, text: str, confidence: float = 1.0) -> None:
        """Emit final transcript (committed after silence)."""
        self._on_message(
            TranscriptMessage(
                type="final",
                text=text,
                confidence=confidence,
                timestamp=_unix_ms(),
            )
        )
        self._pending_partial = ""

    @property
    def pending_partial(self) -> str:
        return self._pending_partial
