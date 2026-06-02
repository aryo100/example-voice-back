"""Async PostgreSQL client (SQLAlchemy 2 + asyncpg). Initialized in FastAPI lifespan."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings, transcript_storage_postgresql
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _database_url() -> str:
    settings = get_settings()
    url = (getattr(settings, "DATABASE_URL", "") or "").strip()
    if not url:
        raise ValueError("DATABASE_URL is required when TRANSCRIPT_STORAGE=postgresql")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("PostgreSQL engine not initialized (lifespan not run or TRANSCRIPT_STORAGE=file)")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("PostgreSQL session factory not initialized")
    async with _session_factory() as session:
        yield session


async def connect_postgres() -> None:
    global _engine, _session_factory
    if not transcript_storage_postgresql():
        return

    url = _database_url()
    _engine = create_async_engine(url, pool_pre_ping=True, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with get_session() as session:
        await session.execute(text("SELECT 1"))

    logger.info("PostgreSQL connected")


async def close_postgres() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("PostgreSQL connection closed")
    _engine = None
    _session_factory = None
