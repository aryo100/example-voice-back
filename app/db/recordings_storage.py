"""Upload session recordings to cloud object storage; persist URL in PostgreSQL."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import transcript_storage_postgresql
from app.db import transcript_repo
from app.storage.object_storage import object_storage_enabled, upload_recording_sync

logger = logging.getLogger(__name__)


async def save_session_recording(
    session_id: str,
    audio_bytes: bytes,
    *,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    """
    Upload audio to Supabase/Firebase Storage and save the URL in PostgreSQL.
    Must run on the FastAPI event loop (not from a thread pool).
    Returns {url, filename, content_type, size, object_path}.
    """
    if not transcript_storage_postgresql():
        raise RuntimeError("PostgreSQL transcript storage is not enabled")
    if not object_storage_enabled():
        raise RuntimeError("OBJECT_STORAGE_BACKEND must be supabase or firebase for cloud recordings")

    object_path = f"{session_id}/{filename}"
    loop = asyncio.get_running_loop()
    url = await loop.run_in_executor(
        None,
        upload_recording_sync,
        object_path,
        audio_bytes,
        content_type,
    )
    meta = {
        "url": url,
        "filename": filename,
        "content_type": content_type,
        "size": len(audio_bytes),
        "object_path": object_path,
    }
    await transcript_repo.set_recording_metadata(
        session_id,
        url=url,
        filename=filename,
        content_type=content_type,
        size=len(audio_bytes),
    )
    logger.info(
        "Recording uploaded session_id=%s url=%s size=%d",
        session_id,
        url,
        len(audio_bytes),
    )
    return meta
