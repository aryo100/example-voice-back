"""
In-memory session store for chat. session_id is generated on the backend (WebSocket).
Transcript is updated only by WebSocket; chat only reads transcript_text as knowledge.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

# session_id -> { "messages": [...], "transcript_text": str, "transcript_finalized": bool, "created_at": float }
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
