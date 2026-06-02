"""
Cloud object storage for session audio recordings.

Backends: Supabase Storage, Firebase Storage.
PostgreSQL stores only the public/signed URL returned after upload.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from app.config import get_settings, object_storage_enabled as _object_storage_enabled_cfg

logger = logging.getLogger(__name__)


class ObjectStorageBackend(ABC):
    @abstractmethod
    def upload(self, object_path: str, data: bytes, content_type: str) -> str:
        """Upload bytes and return a URL suitable for storing in PostgreSQL."""

    @abstractmethod
    def delete(self, object_path: str) -> None:
        """Best-effort delete (optional maintenance)."""


class SupabaseStorageBackend(ObjectStorageBackend):
    def __init__(self) -> None:
        settings = get_settings()
        url = (settings.SUPABASE_URL or "").strip().rstrip("/")
        key = (settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for Supabase storage")
        self._bucket = (settings.SUPABASE_STORAGE_BUCKET or "recordings").strip()
        self._prefix = (settings.SUPABASE_STORAGE_PREFIX or "sessions").strip().strip("/")
        from supabase import create_client

        self._client = create_client(url, key)

    def _full_path(self, object_path: str) -> str:
        path = object_path.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def upload(self, object_path: str, data: bytes, content_type: str) -> str:
        path = self._full_path(object_path)
        bucket = self._client.storage.from_(self._bucket)
        bucket.upload(
            path,
            data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return bucket.get_public_url(path)

    def delete(self, object_path: str) -> None:
        path = self._full_path(object_path)
        self._client.storage.from_(self._bucket).remove([path])


class FirebaseStorageBackend(ObjectStorageBackend):
    def __init__(self) -> None:
        settings = get_settings()
        bucket_name = (settings.FIREBASE_STORAGE_BUCKET or "").strip()
        if not bucket_name:
            raise ValueError("FIREBASE_STORAGE_BUCKET is required for Firebase storage")
        self._bucket_name = bucket_name
        self._prefix = (settings.FIREBASE_STORAGE_PREFIX or "sessions").strip().strip("/")
        self._make_public = getattr(settings, "FIREBASE_STORAGE_MAKE_PUBLIC", True)

        import firebase_admin
        from firebase_admin import credentials

        if not firebase_admin._apps:
            cred_path = (settings.FIREBASE_CREDENTIALS_PATH or "").strip()
            cred_json = (settings.FIREBASE_CREDENTIALS_JSON or "").strip()
            if cred_json:
                cred = credentials.Certificate(json.loads(cred_json))
            elif cred_path and os.path.isfile(cred_path):
                cred = credentials.Certificate(cred_path)
            else:
                raise ValueError(
                    "Set FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON for Firebase storage"
                )
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    def _full_path(self, object_path: str) -> str:
        path = object_path.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def upload(self, object_path: str, data: bytes, content_type: str) -> str:
        from firebase_admin import storage

        path = self._full_path(object_path)
        bucket = storage.bucket(self._bucket_name)
        blob = bucket.blob(path)
        blob.upload_from_string(data, content_type=content_type)
        if self._make_public:
            blob.make_public()
            return blob.public_url
        from datetime import timedelta

        return blob.generate_signed_url(expiration=timedelta(days=7))

    def delete(self, object_path: str) -> None:
        from firebase_admin import storage

        path = self._full_path(object_path)
        storage.bucket(self._bucket_name).blob(path).delete()


_backend: ObjectStorageBackend | None = None


def get_object_storage() -> ObjectStorageBackend:
    """Return configured object storage backend (singleton)."""
    global _backend
    if _backend is not None:
        return _backend

    settings = get_settings()
    backend = (getattr(settings, "OBJECT_STORAGE_BACKEND", "none") or "none").strip().lower()
    if backend == "supabase":
        _backend = SupabaseStorageBackend()
    elif backend == "firebase":
        _backend = FirebaseStorageBackend()
    elif backend in ("none", ""):
        raise ValueError(
            "OBJECT_STORAGE_BACKEND must be 'supabase' or 'firebase' when uploading recordings to cloud storage"
        )
    else:
        raise ValueError(f"Unknown OBJECT_STORAGE_BACKEND: {backend}")
    return _backend


def object_storage_enabled() -> bool:
    return _object_storage_enabled_cfg()


def upload_recording_sync(object_path: str, data: bytes, content_type: str) -> str:
    """Blocking upload for use from recorder finalize() in a thread pool."""
    return get_object_storage().upload(object_path, data, content_type)


def upload_recording_metadata() -> dict[str, Any]:
    """Return storage backend label for health/logging."""
    settings = get_settings()
    return {
        "backend": (getattr(settings, "OBJECT_STORAGE_BACKEND", "none") or "none").strip().lower(),
    }
