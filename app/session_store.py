"""
In-memory session store for chat. session_id is generated on the backend (WebSocket).
Transcript is updated only by WebSocket; chat only reads transcript_text as knowledge.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

# session_id -> {
#   "messages": [...],
#   "transcript_text": str,            # transcript_raw: full text; never sent to TTS, never rewritten by chat
#   "transcript_finalized": bool,
#   "created_at": float,
#   "global_summary": str,             # part of transcript_summary (compressed context for LLM only)
#   "segment_summaries": list[str],   # optional
#   "compressed_rolling_summary": str, # part of transcript_summary when transcript is large
#   "transcript_summary_hash": str,   # invalidates summary when transcript changes
# }
_session_store: dict[str, dict[str, Any]] = {}


def generate_session_id() -> str:
    """Generate a new session_id (UUID hex, 12 chars). Backend only."""
    return uuid.uuid4().hex[:12]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return session dict or None if not found."""
    return _session_store.get(session_id)


def set_session(session_id: str, data: dict[str, Any]) -> None:
    """Store or overwrite session."""
    _session_store[session_id] = data


def delete_session(session_id: str) -> bool:
    """Remove session from store. Return True if it existed."""
    if session_id in _session_store:
        del _session_store[session_id]
        return True
    return False


def ensure_session_for_websocket(session_id: str) -> None:
    """Create session if not exists. Called by WebSocket when connection starts. Transcript updated only by WebSocket."""
    if session_id not in _session_store:
        _session_store[session_id] = {
            "messages": [],
            "transcript_text": "",
            "transcript_finalized": False,
            "created_at": time.time(),
        }


def ensure_session_from_transcript_file(session_id: str, transcript_text: str) -> dict[str, Any] | None:
    """
    If session is not in memory but transcript_text is provided (e.g. loaded from transcripts/{session_id}.txt),
    create session in memory with that transcript and return it. Return None if transcript_text is empty.
    Used by chat endpoint so session_id that matches a transcript file works after server restart.
    """
    if session_id in _session_store:
        return _session_store[session_id]
    if not (transcript_text or "").strip():
        return None
    _session_store[session_id] = {
        "messages": [],
        "transcript_text": transcript_text.strip(),
        "transcript_finalized": True,
        "created_at": time.time(),
    }
    return _session_store[session_id]


def update_session_transcript(
    session_id: str,
    transcript_text: str,
    transcript_finalized: bool = False,
) -> None:
    """Update transcript for session. Only to be called by WebSocket transcript handler; chat must NOT call this."""
    s = _session_store.get(session_id)
    if s is None:
        return
    s["transcript_text"] = transcript_text
    s["transcript_finalized"] = transcript_finalized


def session_store() -> dict[str, dict[str, Any]]:
    """Return the underlying store (read-only view for debugging)."""
    return _session_store
