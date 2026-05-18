"""
Schemas for context-aware re-transcription (refine-transcript) API.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RefineSegmentInput(BaseModel):
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
        description="Optional short description of audio when no file",
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
    segment_id: str
    original_text: str
    refined_text: str
    confidence_before: float = Field(..., ge=0.0, le=1.0)
    confidence_after: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field("", description="Short explanation when correction is applied")


class RefineResponse(BaseModel):
    segments: list[RefineSegmentOutput] = Field(..., description="Per-segment refinement result")
    session_id: str | None = Field(None, description="Set when revision layer was stored")
