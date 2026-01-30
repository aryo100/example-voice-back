"""
Schemas for unified chat API (CHAT MODE vs REFINE MODE).

CHAT MODE: default; conversational assistant, does not modify transcript.
REFINE MODE: one-time only; first interaction for a session may refine transcript.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.refine import RefineResponse


class ChatRequest(BaseModel):
    """Request body for POST /api/chat. session_id generated on backend for first message."""

    message: str | None = Field(None, description="User message (required unless action=reset)")
    session_id: str | None = Field(
        None,
        description="Absent for first message (backend generates it); required for follow-up and reset",
    )
    transcript: str | None = Field(
        None,
        description="Raw transcript text for first message only; backend refines it once",
    )
    is_first_message: bool | None = Field(
        None,
        description="True = first message: session_id null, backend generates session_id and refines transcript once",
    )
    action: str | None = Field(None, description="Optional: 'reset' to clear session (requires session_id)")
    segments: list[dict] | None = Field(
        None,
        description="Optional raw transcript segments for first-time refine (alternative to transcript text)",
    )


class ChatResponse(BaseModel):
    """Response body for POST /api/chat. session_id + reply for chat; reply only for reset."""

    session_id: str | None = Field(None, description="Generated on first message; same for follow-ups")
    reply: str = Field("", description="AI reply (conversational)")
    mode: str | None = Field(None, description="Legacy: chat | refine")
    message: str | None = Field(None, description="Legacy alias for reply")
    refinement: RefineResponse | None = Field(None, description="Legacy: present only when mode=refine")
