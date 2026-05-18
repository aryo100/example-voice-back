"""Session audio recordings in MongoDB GridFS (when USE_DATABASE / TRANSCRIPT_STORAGE=mongodb)."""
from __future__ import annotations

import io
import logging
from typing import Any

from app.config import get_settings, transcript_storage_mongodb

logger = logging.getLogger(__name__)


def _recordings_bucket_name() -> str:
    return getattr(get_settings(), "MONGODB_RECORDINGS_BUCKET", "recordings")


def _transcripts_collection_name() -> str:
    return getattr(get_settings(), "MONGODB_TRANSCRIPTS_COLLECTION", "transcripts")


async def save_session_recording(
    session_id: str,
    audio_bytes: bytes,
    *,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    """
    Upload audio to GridFS and link metadata on the transcript document.
    Returns {file_id, filename, content_type, size, bucket}.
    """
    from motor.motor_asyncio import AsyncIOMotorGridFSBucket

    from app.db.mongo import get_mongo_db

    db = get_mongo_db()
    bucket_name = _recordings_bucket_name()
    bucket = AsyncIOMotorGridFSBucket(db, bucket_name=bucket_name)

    file_id = await bucket.upload_from_stream(
        filename,
        io.BytesIO(audio_bytes),
        metadata={
            "session_id": session_id,
            "content_type": content_type,
        },
    )
    meta = {
        "file_id": str(file_id),
        "filename": filename,
        "content_type": content_type,
        "size": len(audio_bytes),
        "bucket": bucket_name,
    }
    await db[_transcripts_collection_name()].update_one(
        {"session_id": session_id},
        {"$set": {"session_id": session_id, "recording": meta}},
        upsert=True,
    )
    logger.info(
        "Recording saved to GridFS bucket=%s session_id=%s file_id=%s size=%d",
        bucket_name,
        session_id,
        file_id,
        len(audio_bytes),
    )
    return meta


def save_session_recording_sync(
    session_id: str,
    audio_bytes: bytes,
    *,
    filename: str,
    content_type: str,
) -> dict[str, Any] | None:
    """Blocking wrapper for finalize() in thread pool."""
    if not transcript_storage_mongodb():
        return None
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            save_session_recording(
                session_id,
                audio_bytes,
                filename=filename,
                content_type=content_type,
            )
        )
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            save_session_recording(
                session_id,
                audio_bytes,
                filename=filename,
                content_type=content_type,
            ),
        ).result(timeout=120)
