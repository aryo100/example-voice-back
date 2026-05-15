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

from app.config import get_settings
from app.services.llm_provider import completion as llm_completion

logger = logging.getLogger(__name__)

# --- Config defaults (can be overridden in config) ---
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

    # GLOBAL SUMMARY: ... (until SEGMENT SUMMARIES: or end). Batas dari config agar full context tidak terpotong.
    gs_match = re.search(r"GLOBAL SUMMARY:\s*\n(.*?)(?=SEGMENT SUMMARIES:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if gs_match:
        global_summary = gs_match.group(1).strip()
        global_summary = re.sub(r"\n+", " ", global_summary)
        max_chars = getattr(get_settings(), "CHAT_GLOBAL_SUMMARY_MAX_CHARS", 4000)
        if len(global_summary) > max_chars:
            global_summary = global_summary[:max_chars] + "..."

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

    if not (global_summary or "").strip() and (raw or "").strip():
        # Model sering tidak mengikuti label GLOBAL SUMMARY: — pakai teks respons apa adanya (dibatasi).
        fallback = re.sub(r"\s+", " ", raw).strip()
        max_chars = getattr(get_settings(), "CHAT_GLOBAL_SUMMARY_MAX_CHARS", 4000)
        if len(fallback) > max_chars:
            fallback = fallback[:max_chars] + "..."
        if fallback:
            global_summary = fallback

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
    """Single LLM call for summarization (OpenRouter → Cloudflare → Hugging Face). Returns raw response text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return await llm_completion(messages, max_tokens=max_output_tokens, temperature=0.2, timeout=45.0)


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


def _heuristic_global_summary(transcript_text: str) -> str:
    words = transcript_text.split()[:CHAT_GLOBAL_SUMMARY_MAX_WORDS]
    return " ".join(words) + ("..." if len(transcript_text.split()) > CHAT_GLOBAL_SUMMARY_MAX_WORDS else "")


async def run_compression_loop(
    transcript_text: str,
    *,
    max_batch_calls: int | None = None,
    chunk_chars_override: int | None = None,
) -> dict[str, Any]:
    """
    Iterative context compression: chunk transcript → summarize each batch → merge into rolling summary;
    re-summarize rolling summary when it exceeds ROLLING_MAX_CHARS. Never send full transcript at once.
    Returns {"global_summary": str, "compressed_rolling_summary": str, "segment_summaries": list} for compatibility.
    Cached per session by caller using transcript_hash; re-run only when new transcript data arrives.

    max_batch_calls: batasi jumlah batch LLM (mis. saat /api/session/activate).
    chunk_chars_override: ukuran chunk sementara (mis. chunk lebih besar saat activate).
    """
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"global_summary": "(No transcript yet.)", "compressed_rolling_summary": "", "segment_summaries": []}

    settings = get_settings()
    chunk_chars = chunk_chars_override or getattr(settings, "CHAT_CHUNK_CHARS", CHAT_CHUNK_CHARS)
    rolling_max = getattr(settings, "CHAT_ROLLING_MAX_CHARS", CHAT_ROLLING_MAX_CHARS)

    chunks = _chunk_transcript(transcript_text, chunk_chars)
    if not chunks:
        return {"global_summary": "(No content.)", "compressed_rolling_summary": "", "segment_summaries": []}

    total_chunks = len(chunks)
    if max_batch_calls is not None:
        chunks = chunks[: max(0, max_batch_calls)]
        if total_chunks > len(chunks):
            logger.info(
                "Compression loop: memproses %s/%s batch (batas aktivasi cepat; chunk_chars=%s)",
                len(chunks),
                total_chunks,
                chunk_chars,
            )
    else:
        logger.info("Compression loop: %s chunks (chunk_chars=%s)", len(chunks), chunk_chars)

    rolling: list[str] = []
    stopped_all_providers_failed = False
    for i, chunk in enumerate(chunks):
        try:
            batch_summary = await _summarize_batch(chunk)
        except ValueError as e:
            logger.warning(
                "Compression loop: semua provider LLM gagal pada batch %s/%s; hentikan. %s",
                i + 1,
                len(chunks),
                e,
            )
            stopped_all_providers_failed = True
            break
        if not batch_summary:
            continue
        rolling.append(batch_summary)
        merged = "\n\n".join(rolling)
        if len(merged) > rolling_max:
            try:
                compressed = await _re_summarize_rolling(merged)
            except ValueError as e:
                logger.warning("Compression loop: re-summarize rolling gagal (semua provider); lanjut tanpa kompresi. %s", e)
                compressed = ""
            if compressed:
                rolling = [compressed]
            # else keep rolling as-is to avoid loss
    compressed_rolling_summary = "\n\n".join(rolling)
    if stopped_all_providers_failed and not compressed_rolling_summary.strip():
        return {
            "global_summary": _heuristic_global_summary(transcript_text) or "(Ringkasan tidak tersedia; semua provider LLM gagal.)",
            "compressed_rolling_summary": "",
            "segment_summaries": [],
        }
    if max_batch_calls is not None and total_chunks > len(chunks) and compressed_rolling_summary.strip():
        compressed_rolling_summary += (
            "\n\n[Catatan: ringkasan aktivasi dibatasi; sebagian transcript belum diproses LLM.]"
        )

    # Global summary: one short pass over first chunk or over compressed if single block
    max_global_chars = getattr(get_settings(), "CHAT_GLOBAL_SUMMARY_MAX_CHARS", 4000)
    if len(compressed_rolling_summary) > 800:
        try:
            global_summary = await _call_llm_summary(
                "In at most 80 words, summarize the following in one short paragraph. Do not invent information.",
                compressed_rolling_summary[:4000],
                max_output_tokens=150,
            )
            global_summary = global_summary or "Long transcript; see detailed compressed summary below."
        except ValueError as e:
            logger.warning("Compression loop: global summary LLM gagal; pakai ringkasan heuristik. %s", e)
            global_summary = _heuristic_global_summary(compressed_rolling_summary) or "Long transcript; see detailed compressed summary below."
    else:
        global_summary = (compressed_rolling_summary[:max_global_chars] if compressed_rolling_summary else "See compressed summary below.")

    return {
        "global_summary": global_summary or "Long transcript; see compressed summary below.",
        "compressed_rolling_summary": compressed_rolling_summary,
        "segment_summaries": [],  # rolled into compressed_rolling_summary
    }


async def summarize_transcript_for_chat(transcript_text: str, *, activate: bool = False) -> dict[str, Any]:
    """
    Produce global summary + segment summaries (or compressed rolling summary when transcript is long).
    When transcript exceeds CHAT_COMPRESSION_THRESHOLD_CHARS, use iterative compression loop;
    otherwise one LLM call. Returns {"global_summary", "segment_summaries", "compressed_rolling_summary"}.
    Transcript is never sent in full; we send only summaries.

    activate=True (mis. dari /api/session/activate): batasi batch LLM + chunk lebih besar agar respons cepat.
    """
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"global_summary": "(No transcript yet.)", "segment_summaries": [], "compressed_rolling_summary": ""}

    settings = get_settings()
    threshold = getattr(settings, "CHAT_COMPRESSION_THRESHOLD_CHARS", CHAT_COMPRESSION_THRESHOLD_CHARS)
    if len(transcript_text) > threshold:
        if activate:
            max_b = int(getattr(settings, "CHAT_ACTIVATE_MAX_COMPRESSION_BATCHES", 6) or 6)
            chunk_ovr = int(getattr(settings, "CHAT_ACTIVATE_CHUNK_CHARS", 0) or 0) or None
            result = await run_compression_loop(
                transcript_text,
                max_batch_calls=max(1, max_b),
                chunk_chars_override=chunk_ovr,
            )
        else:
            result = await run_compression_loop(transcript_text)
        return {
            "global_summary": result.get("global_summary", ""),
            "segment_summaries": result.get("segment_summaries", []),
            "compressed_rolling_summary": result.get("compressed_rolling_summary", ""),
        }

    # Short path: one LLM call (OpenRouter → Cloudflare → Hugging Face)
    max_tokens = getattr(settings, "REFINE_CHAT_MAX_TOKENS", 2048)
    max_chars_for_summary = 12000
    transcript_for_summary = transcript_text if len(transcript_text) <= max_chars_for_summary else transcript_text[:max_chars_for_summary] + "\n[... truncated ...]"
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Summarize this voice transcript:\n\n{transcript_for_summary}"},
    ]
    try:
        content = await llm_completion(messages, max_tokens=max_tokens, temperature=0.2, timeout=45.0)
        out = _parse_summary_response(content)
        out["compressed_rolling_summary"] = ""
        return out
    except ValueError:
        logger.warning("No LLM provider available; using heuristic summary")
    except Exception as e:
        logger.warning("Summarization failed: %s; using heuristic", e)
    words = transcript_text.split()[:CHAT_GLOBAL_SUMMARY_MAX_WORDS]
    return {
        "global_summary": " ".join(words) + ("..." if len(transcript_text.split()) > CHAT_GLOBAL_SUMMARY_MAX_WORDS else ""),
        "segment_summaries": [],
        "compressed_rolling_summary": "",
    }


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


def build_llm_compact_knowledge_system_prompt(knowledge_text: str) -> str:
    """
    System prompt untuk CHAT_LLM_COMPACT_KNOWLEDGE: isi knowledge tanpa label CONTEXT / Transcript summary.
    """
    body = (knowledge_text or "").strip() or "(No knowledge yet.)"
    return (
        "You are a conversational AI assistant.\n"
        "You have access to session knowledge derived from voice transcripts (may be partial or summarized).\n"
        "Do NOT repeat content verbatim unless asked.\n"
        "Respond naturally for spoken output.\n"
        "Do NOT include timestamps or speaker labels.\n\n"
        '"""\n'
        f"{body}\n"
        '"""'
    )


def build_chat_knowledge_base_system_prompt(extra_instructions: str = "") -> str:
    """
    Pesan system pertama saat CHAT_KNOWLEDGE_MULTI_SYSTEM: instruksi umum (tanpa isi transkrip).
    Pesan system berikutnya = segmen transkrip per chunk baris.
    """
    extra = (extra_instructions or "").strip()
    parts = [
        "You are a conversational AI assistant for a voice session.",
        "The following additional system messages are consecutive transcript segments (read-only knowledge).",
        "Each segment header states the inclusive line range (1-based, file order).",
        "Answer the user using this knowledge; do not invent facts not supported by the transcript.",
        "Respond naturally for spoken output.",
        "Do NOT include timestamps or speaker labels in your reply unless the user explicitly asks.",
    ]
    if extra:
        parts.extend(["", "ADDITIONAL INSTRUCTIONS:", extra])
    return "\n".join(parts)


def build_transcript_knowledge_system_messages(
    transcript_text: str,
    *,
    lines_per_chunk: int = 50,
    max_chunks: int = 40,
    max_chars_per_chunk: int = 12000,
) -> list[dict[str, str]]:
    """
    Satu dict per chunk: {"role": "system", "content": "Transkrip dialog baris X–Y:\\n..."}.
    Urutan kronologis; memudahkan provider yang membatasi ukuran satu body besar.
    """
    text = (transcript_text or "").strip()
    if not text:
        return [{"role": "system", "content": "(Tidak ada transkrip di session.)"}]

    lines = text.splitlines()
    if lines_per_chunk < 1:
        lines_per_chunk = 50
    if max_chunks < 1:
        max_chunks = 40
    if max_chars_per_chunk < 500:
        max_chars_per_chunk = 500

    out: list[dict[str, str]] = []
    total_line_groups = (len(lines) + lines_per_chunk - 1) // lines_per_chunk
    capped_groups = min(total_line_groups, max_chunks)
    if total_line_groups > max_chunks:
        logger.warning(
            "Transcript knowledge: %s baris → %s chunk (dibatasi CHAT_KNOWLEDGE_MAX_CHUNKS=%s)",
            len(lines),
            total_line_groups,
            max_chunks,
        )

    for g in range(capped_groups):
        i0 = g * lines_per_chunk
        chunk_lines = lines[i0 : i0 + lines_per_chunk]
        start = i0 + 1
        end = i0 + len(chunk_lines)
        body = "\n".join(chunk_lines)
        if len(body) > max_chars_per_chunk:
            body = body[: max_chars_per_chunk - 40] + "\n[... segmen dipotong (CHAT_KNOWLEDGE_MAX_CHARS_PER_CHUNK) ...]"
        content = f"Transkrip dialog baris {start}–{end}:\n{body}"
        out.append({"role": "system", "content": content})
    return out


def summary_field_invalid_for_llm(global_summary: str | None) -> bool:
    """True bila ringkasan session tidak boleh dipakai sebagai satu-satunya knowledge LLM."""
    gs = (global_summary or "").strip()
    if not gs:
        return True
    bad = {
        "no summary.",
        "(no transcript yet.)",
        "(transcript summary unavailable.)",
    }
    return gs.lower() in bad


def _merge_global_and_compressed(global_summary: str | None, compressed_rolling: str | None) -> str:
    gs = (global_summary or "").strip()
    if gs.lower() in ("no summary.", "(no transcript yet.)"):
        gs = ""
    cr = (compressed_rolling or "").strip()
    if gs and cr:
        return f"{gs}\n\n{cr}"
    return gs or cr or ""


def build_chat_system_prompt_llm_compact(
    *,
    parts: str,
    global_summary: str,
    compressed_rolling: str,
    relevant_snippets: str,
    extra_instructions: str = "",
    transcript_text: str | None = None,
    user_question: str | None = None,
    knowledge_omit_summary: bool = False,
) -> str:
    """
    System prompt untuk mode compact (hindari payload besar).

    parts:
    - both: ringkasan global + compressed + cuplikan relevan (default).
    - summary_only: ringkasan + compressed saja (tanpa blok excerpt ganda).
    - excerpt_only: tanpa ringkasan global; hanya cuplikan relevan + instruksi tambahan (paling ringan).

    knowledge_omit_summary=True (CHAT_LLM_COMPACT_KNOWLEDGE): tidak mengirim ringkasan session, cuplikan,
    atau teks transkrip terpotong ke LLM — hanya instruksi peran + CHAT_LLM_EXTRA_SYSTEM_PROMPT.
    """
    extra = (extra_instructions or "").strip()

    if knowledge_omit_summary:
        lines = [
            "You are a conversational AI assistant for a voice session.",
            "A transcript exists for this session but is not included in this request: no excerpts, no truncated transcript lines, and no summary text from the transcript.",
            "Answer from the user's message and general reasoning; if the answer depends on transcript content you were not given, say so briefly.",
            "Respond naturally for spoken output.",
            "Do NOT include timestamps or speaker labels in your reply.",
        ]
        if extra:
            lines.extend(["", "ADDITIONAL INSTRUCTIONS:", extra])
        return "\n".join(lines)

    if parts == "excerpt_only":
        body = (relevant_snippets or "").strip() or "(Tidak ada cuplikan relevan untuk pertanyaan ini.)"
        lines = [
            "You are a conversational AI assistant.",
            "You only have a short relevant excerpt from a longer voice transcript (not the full transcript).",
            "Answer the user's question using the excerpt; if it is not enough to answer, say so briefly.",
            "Respond naturally for spoken output.",
            "Do NOT include timestamps or speaker labels.",
        ]
        if extra:
            lines.extend(["", "ADDITIONAL INSTRUCTIONS:", extra])
        lines.extend(["", "RELEVANT EXCERPT:", '"""', body, '"""'])
        return "\n".join(lines)

    transcript_summary = _merge_global_and_compressed(global_summary, compressed_rolling)
    used_wide_fallback = False
    if summary_field_invalid_for_llm(transcript_summary) and (transcript_text or "").strip():
        fb = int(getattr(get_settings(), "CHAT_LLM_KNOWLEDGE_FALLBACK_SNIPPET_CHARS", 12000) or 12000)
        uq = (user_question or "").strip()
        digest = select_relevant_snippets(transcript_text or "", uq, max_chars=max(400, fb))
        if digest.strip():
            transcript_summary = (
                "Bounded transcript context (excerpt from session, not the full file):\n" + digest.strip()
            )
            used_wide_fallback = True

    system_prompt = build_llm_compact_knowledge_system_prompt(transcript_text or "(No knowledge yet.)")
    logger.info("system_prompt: %s", system_prompt)
    if parts != "summary_only" and (relevant_snippets or "").strip() and not used_wide_fallback:
        system_prompt = system_prompt + "\n\nRelevant excerpt (for this question only):\n" + relevant_snippets.strip()
    if extra:
        system_prompt = system_prompt + "\n\nADDITIONAL INSTRUCTIONS:\n" + extra
    return system_prompt


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
