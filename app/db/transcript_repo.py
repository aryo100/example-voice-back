"""PostgreSQL CRUD for transcript sessions."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.db.models import TranscriptSession
from app.db.postgres import get_session


async def append_line(session_id: str, line: str) -> None:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None:
            row = TranscriptSession(session_id=session_id, lines=[line])
            db.add(row)
        else:
            row.lines = list(row.lines or []) + [line]
        await db.commit()


async def get_content(session_id: str) -> str | None:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None:
            return None
        if row.lines:
            text = "\n".join(str(x) for x in row.lines).strip()
            return text or None
        raw = (row.content or "").strip()
        return raw or None


async def set_content(session_id: str, content: str) -> None:
    text = (content or "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()] if text else []
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None:
            row = TranscriptSession(session_id=session_id, content=text, lines=lines)
            db.add(row)
        else:
            row.content = text
            row.lines = lines
        await db.commit()


async def save_refinements(session_id: str, payload: list[dict[str, Any]]) -> None:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None:
            row = TranscriptSession(session_id=session_id, lines=[], refinements=payload)
            db.add(row)
        else:
            row.refinements = payload
        await db.commit()


async def get_refinements(session_id: str) -> list[dict[str, Any]] | None:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None or not row.refinements:
            return None
        ref = row.refinements
        return ref if isinstance(ref, list) else None


async def has_refinements(session_id: str) -> bool:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        return bool(row and row.refinements)


async def set_recording_metadata(
    session_id: str,
    *,
    url: str,
    filename: str,
    content_type: str,
    size: int,
) -> None:
    async with get_session() as db:
        row = await db.get(TranscriptSession, session_id)
        if row is None:
            row = TranscriptSession(session_id=session_id, lines=[])
            db.add(row)
        row.recording_url = url
        row.recording_filename = filename
        row.recording_content_type = content_type
        row.recording_size = size
        await db.commit()


async def get_recording_url(session_id: str) -> str | None:
    async with get_session() as db:
        result = await db.execute(
            select(TranscriptSession.recording_url).where(
                TranscriptSession.session_id == session_id
            )
        )
        url = result.scalar_one_or_none()
        return (url or "").strip() or None
