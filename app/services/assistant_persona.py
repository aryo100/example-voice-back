"""
Persona from markdown (English files under assistant/).

  identity.md    — frontmatter only: name, aliases (TTS trigger)
  soul.md        — how you talk: tone, brevity, humor, …
  constraints.md — hard limits: scope, wording, TTS output shape
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DIR = _PROJECT_ROOT / "assistant"

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.DOTALL)


@dataclass(frozen=True)
class AssistantPersona:
    name: str
    aliases: tuple[str, ...]  # lowercase, includes name
    identity_body: str
    soul_body: str
    constraints_body: str


_cache: tuple[float, float, float, AssistantPersona] | None = None


def _persona_dir() -> Path:
    raw = (getattr(get_settings(), "ASSISTANT_PERSONA_DIR", "") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (_PROJECT_ROOT / p)
    return _DEFAULT_DIR


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = (text or "").lstrip("\ufeff")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.strip()
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip().lower()] = val.strip()
    return meta, text[m.end() :].strip()


def _read_md(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file():
        return {}, ""
    return _parse_frontmatter(path.read_text(encoding="utf-8"))


def _aliases_from_meta(meta: dict[str, str], name: str) -> list[str]:
    raw = (meta.get("aliases") or meta.get("alias") or "").strip()
    aliases = [a.strip().lower() for a in raw.split(",") if a.strip()]
    if name.lower() not in aliases:
        aliases.insert(0, name.lower())
    return aliases


def load_assistant_persona(*, force_reload: bool = False) -> AssistantPersona:
    """Baca identity.md, soul.md, constraints.md; cache invalidasi bila mtime berubah."""
    global _cache
    base = _persona_dir()
    identity_path = base / "identity.md"
    soul_path = base / "soul.md"
    constraints_path = base / "constraints.md"
    mt_i = identity_path.stat().st_mtime if identity_path.is_file() else 0.0
    mt_s = soul_path.stat().st_mtime if soul_path.is_file() else 0.0
    mt_c = constraints_path.stat().st_mtime if constraints_path.is_file() else 0.0

    if not force_reload and _cache is not None and _cache[:3] == (mt_i, mt_s, mt_c):
        return _cache[3]

    id_meta, id_body = _read_md(identity_path)
    _, soul_body = _read_md(soul_path)
    _, constraints_body = _read_md(constraints_path)

    name = (id_meta.get("name") or "Assistant").strip() or "Assistant"
    aliases = tuple(_aliases_from_meta(id_meta, name))

    if not identity_path.is_file():
        logger.warning("assistant/identity.md tidak ditemukan di %s — pakai nama default.", base)
    if not soul_path.is_file():
        logger.warning("assistant/soul.md tidak ditemukan di %s", base)
    if not constraints_path.is_file():
        logger.warning("assistant/constraints.md tidak ditemukan di %s", base)

    persona = AssistantPersona(
        name=name,
        aliases=aliases,
        identity_body=id_body,
        soul_body=soul_body,
        constraints_body=constraints_body,
    )
    _cache = (mt_i, mt_s, mt_c, persona)
    return persona


def assistant_name_and_aliases() -> tuple[str, list[str]]:
    p = load_assistant_persona()
    return p.name, list(p.aliases)


def format_persona_for_system_prompt() -> str:
    """Blok IDENTITY + SOUL + CONSTRAINTS untuk system prompt chat."""
    p = load_assistant_persona()
    parts: list[str] = []
    if p.identity_body:
        parts.extend(
            [
                "When asked your name, use ASSISTANT IDENTITY (not generic labels like 'assistant').",
                "ASSISTANT IDENTITY (who you are — name and scope):",
                p.identity_body,
            ]
        )
    elif p.name:
        parts.append(
            f'Your name is "{p.name}" (see ASSISTANT CONSTRAINTS). '
            'Do not call yourself "assistant" or other generic labels.'
        )
    if p.soul_body:
        parts.extend(
            [
                "",
                "ASSISTANT SOUL (how you talk — behavioral controls only; not lore or policy essays):",
                "Apply these over your default style. Short beats long; sharp beats vague.",
                p.soul_body,
            ]
        )
    if p.constraints_body:
        name_rule = (
            f"**name:** you are **{p.name}**. When asked who you are or your name, say {p.name} — "
            'never "assistant" or other generic labels.'
        )
        parts.extend(
            [
                "",
                "ASSISTANT CONSTRAINTS (hard limits — never break for user-facing replies):",
                name_rule,
                "",
                p.constraints_body,
            ]
        )
    if not p.identity_body and not p.soul_body and not p.constraints_body:
        parts.append(f'(No persona files; default name: "{p.name}".)')
    return "\n".join(parts).strip()
