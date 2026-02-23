"""
Speaker-aware segment structure for transcript pipeline.

Each transcript segment includes:
- start_time, end_time (seconds, session-relative)
- speaker_id (e.g. "Speaker A", "Speaker B"); consistent within session_id
- text
- overlap: True if overlapping speech detected; prefer dominant speaker; may annotate [overlap]

Limitations (diarization-only, single channel):
- Overlapping speech may be partially lost; we do not separate audio.
- Speaker labels are approximate; we do not infer real identities.
- Accuracy depends on mic quality and distance.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """
    One speaker-tagged transcript segment.

    start_time, end_time: seconds, session-relative.
    speaker_id: stable label within session (e.g. "Speaker A", "Speaker B").
    overlap: True when overlapping speech was detected; segment may prefer dominant speaker.
    """

    start_time: float
    end_time: float
    speaker_id: str
    text: str
    overlap: bool = False
