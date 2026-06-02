"""SQLAlchemy models for PostgreSQL transcript storage."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TranscriptSession(Base):
    """
    One row per WebSocket session.
    Transcript lines and refinements are JSON; audio lives in object storage (URL only here).
    """

    __tablename__ = "transcript_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    lines: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    refinements: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    recording_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    recording_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recording_content_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recording_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )
