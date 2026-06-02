"""
Transcript persistence: file (default) or PostgreSQL (TRANSCRIPT_STORAGE=postgresql).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.config import get_settings, transcript_storage_postgresql

logger = logging.getLogger(__name__)


def _transcript_dir() -> str:
    return getattr(get_settings(), "TRANSCRIPT_DIR", "./transcripts")


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


# --- Public ---


async def append_transcript_line(session_id: str, line: str) -> None:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        await transcript_repo.append_line(session_id, line)
    else:
        _file_append_line(session_id, line)


async def get_transcript_content(session_id: str) -> str | None:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        return await transcript_repo.get_content(session_id)
    return _file_get_content(session_id)


async def set_transcript_content(session_id: str, content: str) -> None:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        await transcript_repo.set_content(session_id, content)
    else:
        _file_set_content(session_id, content)


async def save_refinement_layer(session_id: str, payload: list[dict[str, Any]]) -> None:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        await transcript_repo.save_refinements(session_id, payload)
    else:
        _file_save_refinements(session_id, payload)


async def get_refinement_layer(session_id: str) -> list[dict[str, Any]] | None:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        return await transcript_repo.get_refinements(session_id)
    return _file_get_refinements(session_id)


async def session_has_refinements(session_id: str) -> bool:
    if transcript_storage_postgresql():
        from app.db import transcript_repo

        return await transcript_repo.has_refinements(session_id)
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
