"""
Transcript persistence: file (default) or MongoDB (TRANSCRIPT_STORAGE=mongodb).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.config import get_settings, transcript_storage_mongodb

logger = logging.getLogger(__name__)


def _transcript_dir() -> str:
    return getattr(get_settings(), "TRANSCRIPT_DIR", "./transcripts")


def _mongo_collection_name() -> str:
    return getattr(get_settings(), "MONGODB_TRANSCRIPTS_COLLECTION", "transcripts")


def _file_path(session_id: str) -> str:
    return os.path.join(_transcript_dir(), f"{session_id}.txt")


def _refinements_file_path(session_id: str) -> str:
    return os.path.join(_transcript_dir(), f"{session_id}_refinements.json")


# --- File ---


def _file_append_line(session_id: str, line: str) -> None:
    path = _file_path(session_id)
    os.makedirs(_transcript_dir(), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _file_get_content(session_id: str) -> str | None:
    path = _file_path(session_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    except OSError as e:
        logger.warning("Could not read transcript file %s: %s", path, e)
        return None


def _file_set_content(session_id: str, content: str) -> None:
    path = _file_path(session_id)
    os.makedirs(_transcript_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((content or "").strip())
        if content and not content.endswith("\n"):
            f.write("\n")


def _file_save_refinements(session_id: str, payload: list[dict[str, Any]]) -> None:
    path = _refinements_file_path(session_id)
    os.makedirs(_transcript_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _file_get_refinements(session_id: str) -> list[dict[str, Any]] | None:
    path = _refinements_file_path(session_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read refinements %s: %s", path, e)
        return None


def _file_has_refinements(session_id: str) -> bool:
    return os.path.isfile(_refinements_file_path(session_id))


# --- MongoDB ---


async def _mongo_append_line(session_id: str, line: str) -> None:
    from app.db.mongo import get_mongo_db

    await get_mongo_db()[_mongo_collection_name()].update_one(
        {"session_id": session_id},
        {"$push": {"lines": line}, "$set": {"session_id": session_id}},
        upsert=True,
    )


async def _mongo_get_content(session_id: str) -> str | None:
    from app.db.mongo import get_mongo_db

    doc = await get_mongo_db()[_mongo_collection_name()].find_one(
        {"session_id": session_id},
        projection={"lines": 1, "content": 1},
    )
    if not doc:
        return None
    if doc.get("lines"):
        text = "\n".join(str(x) for x in doc["lines"]).strip()
        return text or None
    raw = (doc.get("content") or "").strip()
    return raw or None


async def _mongo_set_content(session_id: str, content: str) -> None:
    from app.db.mongo import get_mongo_db

    text = (content or "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()] if text else []
    await get_mongo_db()[_mongo_collection_name()].update_one(
        {"session_id": session_id},
        {"$set": {"session_id": session_id, "content": text, "lines": lines}},
        upsert=True,
    )


async def _mongo_save_refinements(session_id: str, payload: list[dict[str, Any]]) -> None:
    from app.db.mongo import get_mongo_db

    await get_mongo_db()[_mongo_collection_name()].update_one(
        {"session_id": session_id},
        {"$set": {"session_id": session_id, "refinements": payload}},
        upsert=True,
    )


async def _mongo_get_refinements(session_id: str) -> list[dict[str, Any]] | None:
    from app.db.mongo import get_mongo_db

    doc = await get_mongo_db()[_mongo_collection_name()].find_one(
        {"session_id": session_id},
        projection={"refinements": 1},
    )
    if not doc:
        return None
    ref = doc.get("refinements")
    return ref if isinstance(ref, list) else None


async def _mongo_has_refinements(session_id: str) -> bool:
    from app.db.mongo import get_mongo_db

    doc = await get_mongo_db()[_mongo_collection_name()].find_one(
        {"session_id": session_id},
        projection={"refinements": 1},
    )
    return bool(doc and doc.get("refinements"))


# --- Public ---


async def append_transcript_line(session_id: str, line: str) -> None:
    if transcript_storage_mongodb():
        await _mongo_append_line(session_id, line)
    else:
        _file_append_line(session_id, line)


async def get_transcript_content(session_id: str) -> str | None:
    if transcript_storage_mongodb():
        return await _mongo_get_content(session_id)
    return _file_get_content(session_id)


async def set_transcript_content(session_id: str, content: str) -> None:
    if transcript_storage_mongodb():
        await _mongo_set_content(session_id, content)
    else:
        _file_set_content(session_id, content)


async def save_refinement_layer(session_id: str, payload: list[dict[str, Any]]) -> None:
    if transcript_storage_mongodb():
        await _mongo_save_refinements(session_id, payload)
    else:
        _file_save_refinements(session_id, payload)


async def get_refinement_layer(session_id: str) -> list[dict[str, Any]] | None:
    if transcript_storage_mongodb():
        return await _mongo_get_refinements(session_id)
    return _file_get_refinements(session_id)


async def session_has_refinements(session_id: str) -> bool:
    if transcript_storage_mongodb():
        return await _mongo_has_refinements(session_id)
    return _file_has_refinements(session_id)


async def build_chat_transcript_context(session_id: str) -> str | None:
    """Prefer refinement layer, else raw transcript."""
    ref = await get_refinement_layer(session_id)
    if ref:
        lines = [
            item.get("refined_text", item.get("original_text", ""))
            for item in ref
            if isinstance(item, dict)
        ]
        joined = "\n".join(x for x in lines if x)
        if joined.strip():
            return joined.strip()
    return await get_transcript_content(session_id)
