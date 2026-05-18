"""Async MongoDB client (Motor). Initialized in FastAPI lifespan when transcript storage is mongodb."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.config import get_settings, transcript_storage_mongodb

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: Any = None


def get_mongo_client() -> "AsyncIOMotorClient":
    if _client is None:
        raise RuntimeError("MongoDB client not initialized (lifespan not run or TRANSCRIPT_STORAGE=file)")
    return _client


def get_mongo_db() -> "AsyncIOMotorDatabase":
    settings = get_settings()
    return get_mongo_client()[settings.MONGODB_DATABASE]


async def connect_mongo() -> None:
    global _client
    if not transcript_storage_mongodb():
        return
    from motor.motor_asyncio import AsyncIOMotorClient

    settings = get_settings()
    uri = (settings.MONGODB_URI or "").strip()
    if not uri:
        raise ValueError("MONGODB_URI is required when TRANSCRIPT_STORAGE=mongodb")
    _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    await _client.admin.command("ping")
    coll = _client[settings.MONGODB_DATABASE][settings.MONGODB_TRANSCRIPTS_COLLECTION]
    await coll.create_index("session_id", unique=True)
    logger.info("MongoDB connected (database=%s)", settings.MONGODB_DATABASE)


async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")
