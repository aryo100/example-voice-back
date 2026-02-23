"""
Speaker-aware transcription (diarization only).

- No audio separation; no multi-channel input.
- Assigns speaker labels (Speaker A, Speaker B) consistent within session_id.
- Overlap handling: mark segments as overlapping; prefer dominant speaker; optional [overlap] annotation.

Limitations (see speaker_tracker.py and models.py):
- Overlapping speech may be partially lost on single-channel input.
- Speaker labels are approximate; no real identity inference.
- Accuracy depends on mic quality and distance.
"""
from __future__ import annotations

from app.diarization.models import SpeakerSegment
from app.diarization.speaker_tracker import SpeakerTracker, get_speaker_tracker

__all__ = ["SpeakerSegment", "SpeakerTracker", "get_speaker_tracker"]
