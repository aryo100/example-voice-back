"""Coqui STT model registry and download (GitHub STT-models releases)."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class _CoquiModelSpec(TypedDict):
    release: str
    model: str
    scorer: str


COQUI_MODELS: dict[str, _CoquiModelSpec] = {
    "en": {
        "release": "english/coqui/v1.0.0-large-vocab",
        "model": "model.tflite",
        "scorer": "large_vocabulary.scorer",
    },
    "id": {
        "release": "indonesian/itml/v0.1.1",
        "model": "model.tflite",
        "scorer": "Indonesian-digits-yesno.scorer",
    },
}


def _download_url(release: str, filename: str) -> str:
    return (
        "https://github.com/coqui-ai/STT-models/releases/download/"
        f"{release}/{filename}"
    )


def ensure_coqui_model(lang: str, model_dir: str | Path) -> tuple[Path, Path]:
    """
    Download Coqui STT model + scorer if missing.
    Returns (model_path, scorer_path).
    """
    lang = (lang or "id").strip().lower()
    if lang not in COQUI_MODELS:
        raise ValueError(f"Unsupported COQUI_STT_LANG={lang!r}. Use: {', '.join(COQUI_MODELS)}")

    cfg = COQUI_MODELS[lang]
    subdir = Path(model_dir) / lang
    subdir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for key in ("model", "scorer"):
        filename = cfg[key]
        dest = subdir / filename
        if not dest.exists():
            url = _download_url(cfg["release"], filename)
            logger.info("Downloading Coqui STT model (%s): %s", lang, filename)
            urllib.request.urlretrieve(url, dest)
        paths[key] = dest

    return paths["model"], paths["scorer"]
