"""
LLM provider router: prioritas OpenRouter → Cloudflare → Hugging Face.
Satu entry point completion() mencoba provider berurutan sampai berhasil.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Provider terakhir yang sukses: diprioritaskan di awal request berikutnya (kurangi tunggu OpenRouter kosong).
_last_successful_provider: str | None = None


def _provider_priority() -> list[str]:
    """Daftar provider sesuai config, lalu tambahkan fallback yang belum disebut."""
    s = get_settings()
    raw = (getattr(s, "LLM_PROVIDER_PRIORITY", "") or "openrouter,cloudflare,huggingface").strip().lower()
    requested = [p.strip() for p in raw.split(",") if p.strip()]
    valid = ("openrouter", "cloudflare", "huggingface")
    ordered = [p for p in requested if p in valid]
    # Selalu lengkapi fallback agar "openrouter" saja tidak mematikan provider lain.
    for provider in valid:
        if provider not in ordered:
            ordered.append(provider)
    return ordered


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Format messages jadi satu prompt (untuk Hugging Face)."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts) + "\n\nASSISTANT:"


def _extract_openrouter_text(data: dict[str, Any]) -> tuple[str | None, str]:
    """
    Extract teks dari format OpenRouter/OpenAI chat completion.
    content dapat berupa string atau array content parts.
    """
    choices = data.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return None, "response tanpa choices"
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(msg, dict):
        msg = {}

    content = msg.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text, ""
    elif isinstance(content, list):
        # Untuk multimodal/structured content parts: gabungkan part bertipe text.
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text.strip())
        merged = "\n".join(text_parts).strip()
        if merged:
            return merged, ""

    # Beberapa provider menaruh teks di `text` tingkat choice.
    choice_text = first.get("text") if isinstance(first, dict) else None
    if isinstance(choice_text, str) and choice_text.strip():
        return choice_text.strip(), ""

    reasoning = msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return None, "content kosong (hanya reasoning)"
    return None, "response kosong"


async def _try_openrouter(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float = 60.0,
) -> tuple[str | None, str]:
    """Panggil OpenRouter; return (content, "") jika sukses, (None, alasan) jika gagal."""
    s = get_settings()
    key = (getattr(s, "OPENROUTER_API_KEY", "") or "").strip()
    if not key:
        return None, "tidak dikonfigurasi (OPENROUTER_API_KEY kosong)"
    model = (getattr(s, "OPENROUTER_MODEL", "") or "mistralai/devstral-2512:free").strip()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Hindari output yang hanya reasoning pada model reasoning-capable.
        "reasoning": {"exclude": True},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                OPENROUTER_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        content, reason = _extract_openrouter_text(data)
        return content, reason
    except httpx.HTTPStatusError as e:
        hint = " (model tidak ditemukan? cek OPENROUTER_MODEL)" if e.response.status_code == 404 else ""
        return None, f"HTTP {e.response.status_code}{hint}"
    except Exception as e:
        return None, str(e)


async def _try_cloudflare(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float = 60.0,
) -> tuple[str | None, str]:
    """Panggil Cloudflare Workers AI; return (content, "") jika sukses, (None, alasan) jika gagal."""
    s = get_settings()
    account_id = (getattr(s, "CLOUDFLARE_ACCOUNT_ID", "") or "").strip()
    token = (getattr(s, "CLOUDFLARE_API_TOKEN", "") or "").strip()
    if not account_id or not token:
        return None, "tidak dikonfigurasi (CLOUDFLARE_ACCOUNT_ID atau CLOUDFLARE_API_TOKEN kosong)"
    model = (getattr(s, "REFINE_CF_MODEL", "") or "@cf/meta/llama-3.1-8b-instruct").strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        result = data.get("result", data)
        if isinstance(result, dict):
            content = (result.get("response") or "").strip()
        elif isinstance(result, str):
            content = result.strip()
        else:
            content = ""
        return (content, "") if content else (None, "response kosong")
    except httpx.HTTPStatusError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


async def _try_huggingface(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float = 60.0,
) -> tuple[str | None, str]:
    """Panggil Hugging Face Inference; return (content, "") jika sukses, (None, alasan) jika gagal."""
    s = get_settings()
    key = (getattr(s, "HUGGINGFACE_API_KEY", "") or "").strip()
    if not key:
        return None, "tidak dikonfigurasi (HUGGINGFACE_API_KEY kosong)"
    model = (getattr(s, "HUGGINGFACE_MODEL", "") or "mistralai/Mistral-7B-Instruct-v0.2").strip()
    url = f"https://api-inference.huggingface.co/models/{model}"
    prompt = _messages_to_prompt(messages)
    payload: dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": min(1.0, max(0.0, temperature)),
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                content = (first.get("generated_text") or "").strip()
            else:
                content = str(first).strip()
        elif isinstance(data, dict):
            content = (data.get("generated_text") or data.get("output") or "").strip()
        else:
            content = ""
        return (content, "") if content else (None, "response kosong")
    except httpx.HTTPStatusError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


async def completion(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.4,
    timeout: float = 60.0,
) -> str:
    """
    Panggil LLM dengan prioritas: OpenRouter → Cloudflare → Hugging Face.
    Return assistant reply (content). Raise ValueError jika semua provider gagal atau tidak dikonfigurasi.
    """
    global _last_successful_provider

    order = _provider_priority()
    valid = ("openrouter", "cloudflare", "huggingface")
    sticky = _last_successful_provider
    if sticky and sticky in valid and sticky in order:
        order = [sticky] + [p for p in order if p != sticky]

    failures: list[str] = []
    for provider in order:
        if provider == "openrouter":
            out, reason = await _try_openrouter(messages, max_tokens, temperature, timeout)
        elif provider == "cloudflare":
            out, reason = await _try_cloudflare(messages, max_tokens, temperature, timeout)
        elif provider == "huggingface":
            out, reason = await _try_huggingface(messages, max_tokens, temperature, timeout)
        else:
            continue
        if out is not None:
            _last_successful_provider = provider
            logger.info("LLM response from provider=%s", provider)
            return out
        failures.append(f"{provider}: {reason}")
        logger.info("LLM provider=%s gagal: %s; coba provider berikutnya", provider, reason)
    raise ValueError(
        "Semua provider LLM gagal atau tidak dikonfigurasi. "
        + " | ".join(failures)
        + ". Untuk fallback: set CLOUDFLARE_ACCOUNT_ID & CLOUDFLARE_API_TOKEN, atau HUGGINGFACE_API_KEY. Untuk OpenRouter 404: gunakan model yang valid (lihat openrouter.ai/docs#models)."
    )
