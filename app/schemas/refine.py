"""
Schemas for context-aware re-transcription (refine-transcript) API.

Inputs: user question, raw transcript segments (timestamps + confidence),
optional audio reference (last N seconds or description), domain hint.
Output: structured JSON per segment (original, refined, confidence, justification).
Original transcript is never overwritten; refined is a revision layer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RefineSegmentInput(BaseModel):
    """One raw transcript segment as input to the refine API."""

    segment_id: str = Field(..., description="Unique id for this segment (e.g. index or uuid)")
    text: str = Field(..., description="Original STT text")
    start_sec: float = Field(0.0, description="Start time in seconds")
    end_sec: float = Field(0.0, description="End time in seconds")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="STT confidence 0–1")


class RefineRequest(BaseModel):
    """Request body for POST /api/refine-transcript."""

    user_question: str = Field(..., description="User question providing semantic context")
    segments: list[RefineSegmentInput] = Field(..., description="Raw transcript segments to analyze")
    audio_reference_seconds: float | None = Field(
        None,
        description="Last N seconds of audio this transcript refers to (hint for context)",
    )
    audio_description: str | None = Field(
        None,
        description="Optional short description of audio (e.g. 'gadget unboxing') when no file",
    )
    domain_hint: str | None = Field(
        None,
        description="Domain hint: e.g. gadget review, tutorial, Indonesian naturalness",
    )
    session_id: str | None = Field(
        None,
        description="Optional session id to store refinement as revision layer",
    )


class RefineSegmentOutput(BaseModel):
    """One segment in the refine API response (structured JSON as specified)."""

    segment_id: str
    original_text: str
    refined_text: str
    confidence_before: float = Field(..., ge=0.0, le=1.0)
    confidence_after: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field("", description="Short explanation when correction is applied")


class RefineResponse(BaseModel):
    """Response body for POST /api/refine-transcript."""

    segments: list[RefineSegmentOutput] = Field(..., description="Per-segment refinement result")
    session_id: str | None = Field(None, description="Set when revision layer was stored")
