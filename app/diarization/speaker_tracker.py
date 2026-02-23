"""
Speaker tracking for diarization-only enhancement.

- Assigns speaker labels (Speaker A, Speaker B) consistent within session_id.
- No real identity inference; labels are session-local.
- Overlap: when two speakers in same time range (from future diarization), mark segment
  and prefer dominant speaker. Current implementation uses gap-based alternation only.

Limitations (MUST be kept in sync with product behavior):
- Overlapping speech may be partially lost on single-channel input; we do not separate audio.
- Speaker labels are approximate; no attempt to infer real identities.
- Accuracy depends on mic quality and distance.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Session state: last assigned speaker index (0, 1, ...), last segment end time (sec), optional embeddings.
_session_state: dict[str, dict[str, Any]] = {}

# Speaker label prefix; indices map to Speaker A, Speaker B, ...
SPEAKER_PREFIX = "Speaker "


def _speaker_id(index: int) -> str:
    """Stable label for speaker index: Speaker A, Speaker B, ..."""
    return f"{SPEAKER_PREFIX}{chr(65 + (index % 26))}"


def get_speaker_tracker() -> "SpeakerTracker":
    return SpeakerTracker()


class SpeakerTracker:
    """
    Assigns speaker_id to segments by gap-based alternation.
    Persists last speaker and last end time per session for consistency.
    Speaker embeddings may be stored per session for future embedding-based assignment.
    """

    def assign(
        self,
        session_id: str,
        start_time: float,
        end_time: float,
        overlap_hint: bool = False,
    ) -> tuple[str, bool]:
        """
        Assign speaker_id and overlap for a segment.

        - session_id: same as transcript session; labels are consistent within it.
        - start_time, end_time: segment times in session seconds.
        - overlap_hint: True if caller detected overlap (e.g. from diarization); we mark segment as overlap.
        Returns (speaker_id, overlap).
        """
        settings = get_settings()
        enabled = getattr(settings, "DIARIZATION_ENABLED", True)
        gap_sec = getattr(settings, "DIARIZATION_SPEAKER_GAP_SEC", 0.5)
        max_speakers = getattr(settings, "DIARIZATION_MAX_SPEAKERS", 2)

        if not enabled:
            return (None, False)

        state = _session_state.setdefault(
            session_id,
            {"last_speaker_index": 0, "last_end_time": 0.0, "speaker_embeddings": []},
        )
        last_idx = state["last_speaker_index"]
        last_end = state["last_end_time"]

        # Gap-based alternation: if segment starts well after last segment ended, switch speaker.
        gap = start_time - last_end
        if gap >= gap_sec and last_end > 0:
            next_idx = (last_idx + 1) % max_speakers
            state["last_speaker_index"] = next_idx
            last_idx = next_idx

        state["last_end_time"] = end_time
        speaker_id = _speaker_id(last_idx)
        overlap = overlap_hint
        return (speaker_id, overlap)

    def get_embeddings(self, session_id: str) -> list[Any]:
        """Return stored speaker embeddings for session (for future embedding-based clustering)."""
        state = _session_state.get(session_id)
        if not state:
            return []
        return state.get("speaker_embeddings", [])

    def set_embeddings(self, session_id: str, embeddings: list[Any]) -> None:
        """Persist speaker embeddings per session for better consistency across segments."""
        state = _session_state.setdefault(
            session_id,
            {"last_speaker_index": 0, "last_end_time": 0.0, "speaker_embeddings": []},
        )
        state["speaker_embeddings"] = embeddings
