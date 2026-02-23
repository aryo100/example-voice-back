"""
Chat context aggregation: transcript as knowledge, not full input.

- Build a short GLOBAL SUMMARY of the entire transcript (~300–500 tokens).
- Optional SEGMENT SUMMARIES (short summaries per chunk of transcript).
- Select a SMALL RAW TRANSCRIPT WINDOW only if relevant to the user question.
- Never send full transcript to the LLM.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# --- Config defaults (can be overridden in Settings) ---
CHAT_GLOBAL_SUMMARY_MAX_WORDS = 80  # ~100 tokens
CHAT_SEGMENT_SUMMARIES_COUNT = 5
CHAT_SNIPPET_MAX_CHARS = 600  # ~10–20 sec of text
CHAT_RECENT_LINES = 8  # always consider last N lines for "recent" window
# Iterative compression loop
CHAT_COMPRESSION_THRESHOLD_CHARS = 12000
CHAT_CHUNK_CHARS = 6000
CHAT_ROLLING_MAX_CHARS = 4000


def _transcript_hash(transcript_text: str) -> str:
    """Stable hash of transcript for cache invalidation (summary recompute when transcript changes)."""
    return hashlib.sha256((transcript_text or "").strip().encode("utf-8")).hexdigest()[:16]


def _get_cloudflare_auth(settings: Settings) -> tuple[str, str]:
    """Return (account_id, token) for Workers AI."""
    account_id = (getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "") or "").strip()
    token = (getattr(settings, "CLOUDFLARE_API_TOKEN", "") or "").strip()
    return account_id, token


_SUMMARIZE_SYSTEM = """You are a summarizer. Given a voice transcript, output exactly two parts in plain text.

RULES:
- Do NOT invent or add information not in the transcript.
- The transcript may be incomplete or noisy; summarize only what is there.
- Use at most 80 words for the global summary.
- Use 3 to 5 bullet points for segment summaries (each bullet = one part of the conversation).
- Output format exactly:

GLOBAL SUMMARY:
<one short paragraph, at most 80 words>

SEGMENT SUMMARIES:
1. <short phrase or sentence>
2. ...
3. ...
(3 to 5 bullets only)

Do not output JSON or markdown code blocks. Use the labels above literally."""


def _parse_summary_response(raw: str) -> dict[str, Any]:
    """Parse LLM summary response into global_summary and segment_summaries."""
    raw = (raw or "").strip()
    global_summary = ""
    segment_summaries: list[str] = []

    # GLOBAL SUMMARY: ... (until SEGMENT SUMMARIES: or end)
    gs_match = re.search(r"GLOBAL SUMMARY:\s*\n(.*?)(?=SEGMENT SUMMARIES:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if gs_match:
        global_summary = gs_match.group(1).strip()
        global_summary = re.sub(r"\n+", " ", global_summary)[:600]

    # SEGMENT SUMMARIES: 1. ... 2. ...
    ss_match = re.search(r"SEGMENT SUMMARIES:\s*\n(.*)", raw, re.DOTALL | re.IGNORECASE)
    if ss_match:
        block = ss_match.group(1).strip()
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Numbered list: "1. text" or "- text"
            m = re.match(r"^(?:\d+\.|\-)\s*(.+)", line)
            if m:
                segment_summaries.append(m.group(1).strip())
            else:
                segment_summaries.append(line)
        segment_summaries = segment_summaries[:CHAT_SEGMENT_SUMMARIES_COUNT]

    return {"global_summary": global_summary or "No summary.", "segment_summaries": segment_summaries}


# --- Iterative context compression loop ---

_BATCH_SUMMARIZE_SYSTEM = """You are a summarizer. Given ONE batch of a voice transcript, output a compact summary.

RULES:
- Do NOT invent or add information not in the text.
- Preserve: technical terms, names, steps, numbers, and clear ASR corrections.
- Use bullet points or short paragraphs.
- Keep the summary under 300 words.
- Output ONLY the summary, no labels or headers.
- Order of information must match the original transcript order."""

_RE_SUMMARIZE_ROLLING_SYSTEM = """You are a summarizer. Given accumulated summaries from multiple transcript batches, compress them into one compact summary.

RULES:
- Do NOT invent or add information not in the text.
- Preserve: technical terms, names, steps, numbers.
- Use bullet points or short paragraphs.
- Keep the result under 400 words.
- Output ONLY the compressed summary, no labels or headers.
- Preserve chronological order of events."""


def _chunk_transcript(transcript_text: str, chunk_chars: int) -> list[str]:
    """Split transcript into chunks by char limit, never breaking mid-line. Order preserved."""
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return []
    lines = transcript_text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        line_len = len(line_stripped) + 1
        if current_len + line_len > chunk_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line_stripped)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


async def _call_llm_summary(system_prompt: str, user_content: str, max_output_tokens: int = 512) -> str:
    """Single LLM call for summarization. Returns raw response text."""
    settings = get_settings()
    account_id, token = _get_cloudflare_auth(settings)
    if not account_id or not token:
        return ""
    model = getattr(settings, "REFINE_CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("LLM summary call failed: %s", e)
        return ""
    result = data.get("result", data)
    content = (result.get("response", "") if isinstance(result, dict) else (result if isinstance(result, str) else "")) or ""
    return (content or "").strip()


async def _summarize_batch(chunk: str) -> str:
    """Summarize one transcript batch. Preserve terms, names, steps. Returns compact text."""
    if not (chunk or "").strip():
        return ""
    return await _call_llm_summary(_BATCH_SUMMARIZE_SYSTEM, f"Summarize this transcript batch:\n\n{chunk}")


async def _re_summarize_rolling(rolling_text: str) -> str:
    """Re-compress accumulated rolling summary when it grows too large."""
    if not (rolling_text or "").strip():
        return ""
    return await _call_llm_summary(_RE_SUMMARIZE_ROLLING_SYSTEM, f"Compress these accumulated summaries:\n\n{rolling_text}", max_output_tokens=1024)


async def run_compression_loop(transcript_text: str) -> dict[str, Any]:
    """
    Iterative context compression: chunk transcript → summarize each batch → merge into rolling summary;
    re-summarize rolling summary when it exceeds ROLLING_MAX_CHARS. Never send full transcript at once.
    Returns {"global_summary": str, "compressed_rolling_summary": str, "segment_summaries": list} for compatibility.
    Cached per session by caller using transcript_hash; re-run only when new transcript data arrives.
    """
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"global_summary": "(No transcript yet.)", "compressed_rolling_summary": "", "segment_summaries": []}

    settings = get_settings()
    threshold = getattr(settings, "CHAT_COMPRESSION_THRESHOLD_CHARS", CHAT_COMPRESSION_THRESHOLD_CHARS)
    chunk_chars = getattr(settings, "CHAT_CHUNK_CHARS", CHAT_CHUNK_CHARS)
    rolling_max = getattr(settings, "CHAT_ROLLING_MAX_CHARS", CHAT_ROLLING_MAX_CHARS)

    chunks = _chunk_transcript(transcript_text, chunk_chars)
    if not chunks:
        return {"global_summary": "(No content.)", "compressed_rolling_summary": "", "segment_summaries": []}

    logger.info("Compression loop: %s chunks (chunk_chars=%s)", len(chunks), chunk_chars)
    rolling: list[str] = []
    for i, chunk in enumerate(chunks):
        batch_summary = await _summarize_batch(chunk)
        if not batch_summary:
            continue
        rolling.append(batch_summary)
        merged = "\n\n".join(rolling)
        if len(merged) > rolling_max:
            compressed = await _re_summarize_rolling(merged)
            if compressed:
                rolling = [compressed]
            # else keep rolling as-is to avoid loss
    compressed_rolling_summary = "\n\n".join(rolling)

    # Global summary: one short pass over first chunk or over compressed if single block
    if len(compressed_rolling_summary) > 800:
        global_summary = await _call_llm_summary(
            "In at most 80 words, summarize the following in one short paragraph. Do not invent information.",
            compressed_rolling_summary[:4000],
            max_output_tokens=150,
        )
        global_summary = global_summary or "Long transcript; see detailed compressed summary below."
    else:
        global_summary = compressed_rolling_summary[:500] if compressed_rolling_summary else "See compressed summary below."

    return {
        "global_summary": global_summary or "Long transcript; see compressed summary below.",
        "compressed_rolling_summary": compressed_rolling_summary,
        "segment_summaries": [],  # rolled into compressed_rolling_summary
    }


async def summarize_transcript_for_chat(transcript_text: str) -> dict[str, Any]:
    """
    Produce global summary + segment summaries (or compressed rolling summary when transcript is long).
    When transcript exceeds CHAT_COMPRESSION_THRESHOLD_CHARS, use iterative compression loop;
    otherwise one LLM call. Returns {"global_summary", "segment_summaries", "compressed_rolling_summary"}.
    Transcript is never sent in full; we send only summaries.
    """
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"global_summary": "(No transcript yet.)", "segment_summaries": [], "compressed_rolling_summary": ""}

    settings = get_settings()
    threshold = getattr(settings, "CHAT_COMPRESSION_THRESHOLD_CHARS", CHAT_COMPRESSION_THRESHOLD_CHARS)
    if len(transcript_text) > threshold:
        result = await run_compression_loop(transcript_text)
        return {
            "global_summary": result.get("global_summary", ""),
            "segment_summaries": result.get("segment_summaries", []),
            "compressed_rolling_summary": result.get("compressed_rolling_summary", ""),
        }

    account_id, token = _get_cloudflare_auth(settings)
    if not account_id or not token:
        logger.warning("Cloudflare auth missing; returning heuristic summary")
        # Fallback: first 80 words as "summary"
        words = transcript_text.split()[:CHAT_GLOBAL_SUMMARY_MAX_WORDS]
        return {"global_summary": " ".join(words) + ("..." if len(transcript_text.split()) > CHAT_GLOBAL_SUMMARY_MAX_WORDS else ""), "segment_summaries": [], "compressed_rolling_summary": ""}

    model = getattr(settings, "REFINE_CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")
    max_tokens = getattr(settings, "REFINE_CHAT_MAX_TOKENS", 2048)
    # Truncate transcript for summary call to stay within context
    max_chars_for_summary = 12000
    transcript_for_summary = transcript_text if len(transcript_text) <= max_chars_for_summary else transcript_text[:max_chars_for_summary] + "\n[... truncated ...]"

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "messages": [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user", "content": f"Summarize this voice transcript:\n\n{transcript_for_summary}"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Summarization failed: %s; using heuristic", e)
        words = transcript_text.split()[:CHAT_GLOBAL_SUMMARY_MAX_WORDS]
        return {"global_summary": " ".join(words) + ("..." if len(transcript_text.split()) > CHAT_GLOBAL_SUMMARY_MAX_WORDS else ""), "segment_summaries": [], "compressed_rolling_summary": ""}

    result = data.get("result", data)
    content = (result.get("response", "") if isinstance(result, dict) else (result if isinstance(result, str) else "")) or ""
    out = _parse_summary_response(content)
    out["compressed_rolling_summary"] = ""
    return out


def select_relevant_snippets(transcript_text: str, user_question: str, max_chars: int = CHAT_SNIPPET_MAX_CHARS) -> str:
    """
    Select a small raw transcript window relevant to the user question.
    - Lines that contain any word from the question (case-insensitive).
    - Last RECENT_LINES lines (recent context).
    - No duplicates, no overlapping ranges; total length <= max_chars.
    """
    transcript_text = (transcript_text or "").strip()
    user_question = (user_question or "").strip()
    if not transcript_text:
        return ""

    lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Words from question (min 2 chars to avoid noise)
    words = set(re.findall(r"[a-zA-Z0-9\u00c0-\u024f]{2,}", user_question.lower()))
    if not words:
        # No keywords: return only recent window
        recent = lines[-CHAT_RECENT_LINES:]
        text = "\n".join(recent)
        return text[:max_chars] if len(text) > max_chars else text

    # Indices of lines that match any keyword
    matching_indices: set[int] = set()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(w in line_lower for w in words):
            matching_indices.add(i)

    # Add recent lines
    for i in range(max(0, len(lines) - CHAT_RECENT_LINES), len(lines)):
        matching_indices.add(i)

    # Build snippet: include matching lines and recent, in order, no duplicate lines, cap total chars
    ordered_indices = sorted(matching_indices)
    snippet_lines: list[str] = []
    seen: set[str] = set()
    for idx in ordered_indices:
        line = lines[idx]
        if line in seen:
            continue
        seen.add(line)
        snippet_lines.append(line)
        if sum(len(ln) + 1 for ln in snippet_lines) > max_chars:
            break
    text = "\n".join(snippet_lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def build_chat_system_prompt(
    session_id: str,
    global_summary: str,
    segment_summaries: list[str],
    relevant_snippets: str,
    compressed_rolling_summary: str = "",
) -> str:
    """
    Build compact system prompt for chat: knowledge = global summary + (compressed rolling or segment summaries) + optional raw window.
    Model rules: transcript may be incomplete/noisy; do not hallucinate; global understanding first.
    """
    parts = [
        "You are a conversational assistant for a voice streaming app. You have access to a SUMMARY of what the user has said (from speech-to-text), not the full transcript.",
        "- If the transcript includes speaker labels (e.g. Speaker A, Speaker B), you may reference them: 'Speaker A said...', 'When Speaker B interrupted...'. Labels are session-local and do not imply real identities.",
        "",
        "RULES:",
        "- The transcript may be incomplete or noisy. Do NOT invent or assume missing content.",
        "- Prefer global understanding first; give precise details only when evidence exists.",
        "- You may reinterpret wording only if strongly implied by context; do not change facts.",
        "- Do NOT output JSON or structured data; reply in plain, natural language.",
        "- Do NOT refine or modify the transcript; you only answer questions about it.",
        "",
        f"Session: {session_id}",
        "",
        "GLOBAL SUMMARY OF TRANSCRIPT:",
        global_summary or "(none)",
    ]
    if (compressed_rolling_summary or "").strip():
        parts.append("")
        parts.append("DETAILED COMPRESSED SUMMARY (from full transcript, chronological):")
        parts.append(compressed_rolling_summary.strip())
    if segment_summaries:
        parts.append("")
        parts.append("SEGMENT SUMMARIES (parts of the conversation):")
        for i, s in enumerate(segment_summaries, 1):
            parts.append(f"  {i}. {s}")
    if relevant_snippets.strip():
        parts.append("")
        parts.append("RELEVANT TRANSCRIPT EXCERPT (raw, for this question only):")
        parts.append(relevant_snippets.strip())
    parts.append("")
    parts.append("Answer the user's question based on the above. If the answer is not in the summary or excerpt, say so.")
    return "\n".join(parts)


def build_chat_system_prompt_for_tts(transcript_summary: str) -> str:
    """
    LLM prompt template for conversational AI with transcript as background knowledge.
    Transcript summary is used as context only; never spoken. Replies are suitable for TTS.
    Do NOT include timestamps or speaker labels in the model output.
    """
    return (
        "You are a conversational AI assistant.\n"
        "You have access to transcript context as background knowledge.\n"
        "Do NOT repeat transcript unless asked.\n"
        "Respond naturally for spoken output.\n"
        "Do NOT include timestamps or speaker labels.\n\n"
        "CONTEXT:\n"
        "Transcript summary:\n"
        '"""\n'
        f"{transcript_summary or '(No transcript yet.)'}\n"
        '"""'
    )


def ensure_session_summary(session: dict[str, Any], transcript_text: str) -> dict[str, Any]:
    """
    Ensure session has valid global_summary and segment_summaries for the current transcript.
    If transcript_hash changed or summary missing, returns session with summary fields to be filled by caller
    (caller must call summarize_transcript_for_chat and set_session).
    Returns session dict (possibly with stale summary); caller checks transcript_hash vs current_hash.
    """
    current_hash = _transcript_hash(transcript_text)
    stored_hash = session.get("transcript_summary_hash") or ""
    if stored_hash == current_hash and (session.get("global_summary") or session.get("segment_summaries") is not None):
        return session
    session["transcript_summary_hash"] = current_hash
    # Clear so caller knows to recompute
    session["global_summary"] = session.get("global_summary") or ""
    session["segment_summaries"] = session.get("segment_summaries") if stored_hash == current_hash else []
    return session
