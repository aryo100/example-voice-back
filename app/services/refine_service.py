"""
Context-aware re-transcription and chat: two modes.

- REFINE MODE (one-time): re-interpret transcript using user context; structured output stored.
- CHAT MODE (default): conversational assistant; may reference transcript but must NOT modify it.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.config import get_settings
from app.services.llm_provider import completion as llm_completion
from app.schemas.refine import (
    RefineRequest,
    RefineResponse,
    RefineSegmentInput,
    RefineSegmentOutput,
)

logger = logging.getLogger(__name__)

# --- REFINE MODE: one-time transcript re-interpretation ---

_SYSTEM_PROMPT = """You are a backend AI assistant for a voice streaming system. You have access to recorded audio metadata, partial and final speech transcripts, and transcript confidence and timestamps.

Your role:
- Do NOT create new conversations.
- Do NOT hallucinate missing audio.
- ONLY work with existing transcripts and user questions.

Primary task:
- When the user asks a question, re-interpret the LAST recorded transcript using the user's question as context.
- Improve misheard words, technical terms, and vocabulary bias.
- Preserve original meaning and speaking style.
- Prefer natural spoken Indonesian when the transcript is in Indonesian.

Strict rules:
- Never invent facts.
- Never summarize unless explicitly asked.
- Never rewrite unrelated parts.
- If unsure, keep the original text.

Always respond in JSON only.

OUTPUT FORMAT:
Return a single JSON array. Each element must have exactly these keys:
- segment_id (string): same as input
- original_text (string): unchanged from input
- refined_text (string): corrected text or same as original_text
- confidence_before (number 0-1): same as input confidence
- confidence_after (number 0-1): higher only if correction is justified, else same
- justification (string): brief explanation, or "No change needed"

Return only the JSON array, no markdown or extra text."""


def _build_user_message(req: RefineRequest) -> str:
    """Build the user message with question, segments, and context hints."""
    parts = [f"User question: {req.user_question}"]

    if req.domain_hint:
        parts.append(f"Domain hint: {req.domain_hint}")

    if req.audio_reference_seconds is not None:
        parts.append(f"Reference: last {req.audio_reference_seconds} seconds of audio.")
    if req.audio_description:
        parts.append(f"Audio description: {req.audio_description}")
    if req.audio_reference_seconds is None and not req.audio_description:
        ref_sec = getattr(get_settings(), "REFINE_AUDIO_REFERENCE_SECONDS", 30.0)
        parts.append(f"Reference: last {ref_sec} seconds of audio (default hint).")

    segments_json = [
        {
            "segment_id": s.segment_id,
            "text": s.text,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "confidence": s.confidence,
        }
        for s in req.segments
    ]
    parts.append("LAST recorded transcript segments (re-interpret using the user question above; return refined JSON array):")
    parts.append(json.dumps(segments_json, ensure_ascii=False))

    return "\n\n".join(parts)


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from model response (may be wrapped in markdown code block)."""
    raw = raw.strip()
    # Remove optional markdown code block
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return json.loads(raw)


def _parse_segments(
    raw_list: list[Any],
    input_segments: list[RefineSegmentInput],
) -> list[RefineSegmentOutput]:
    """Convert parsed JSON list to RefineSegmentOutput, preserving order and filling defaults."""
    by_id = {s.segment_id: s for s in input_segments}
    out: list[RefineSegmentOutput] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        seg_id = item.get("segment_id") or (input_segments[i].segment_id if i < len(input_segments) else str(i))
        orig = by_id.get(seg_id) or (input_segments[i] if i < len(input_segments) else None)
        original_text = item.get("original_text")
        if original_text is None and orig:
            original_text = orig.text
        elif original_text is None:
            original_text = ""
        refined = item.get("refined_text", original_text)
        conf_before = float(item.get("confidence_before", orig.confidence if orig else 0.0))
        conf_after = float(item.get("confidence_after", conf_before))
        justification = str(item.get("justification", ""))
        out.append(
            RefineSegmentOutput(
                segment_id=seg_id,
                original_text=original_text,
                refined_text=refined,
                confidence_before=min(1.0, max(0.0, conf_before)),
                confidence_after=min(1.0, max(0.0, conf_after)),
                justification=justification,
            )
        )
    # If model returned fewer items, fill rest with original
    for i, seg in enumerate(input_segments):
        if i >= len(out) or out[i].segment_id != seg.segment_id:
            existing_ids = {o.segment_id for o in out}
            if seg.segment_id not in existing_ids:
                out.append(
                    RefineSegmentOutput(
                        segment_id=seg.segment_id,
                        original_text=seg.text,
                        refined_text=seg.text,
                        confidence_before=seg.confidence,
                        confidence_after=seg.confidence,
                        justification="No change (model did not return this segment)",
                    )
                )
    # Keep same order as input
    order = {s.segment_id: idx for idx, s in enumerate(input_segments)}
    out.sort(key=lambda o: order.get(o.segment_id, 999))
    return out


async def refine_transcript_with_chat(req: RefineRequest) -> RefineResponse:
    """
    Call LLM (OpenRouter → Cloudflare → Hugging Face) to refine transcript segments.
    Returns structured per-segment result. Raises ValueError if chat is disabled or all providers fail.
    """
    settings = get_settings()
    if not getattr(settings, "REFINE_CHAT_ENABLED", True):
        raise ValueError("Refine chat is disabled (REFINE_CHAT_ENABLED=false)")

    user_message = _build_user_message(req)
    if not req.segments:
        return RefineResponse(segments=[])

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    max_tokens = getattr(settings, "REFINE_CHAT_MAX_TOKENS", 2048)
    try:
        content = await llm_completion(messages, max_tokens=max_tokens, temperature=0.2)
    except ValueError as e:
        raise ValueError("Refine failed: no LLM provider available. Set OPENROUTER_API_KEY, CLOUDFLARE_*, or HUGGINGFACE_API_KEY.") from e

    content = (content or "").strip()
    if not content:
        raise ValueError("LLM returned empty response")

    try:
        raw_list = _extract_json_array(content)
    except json.JSONDecodeError as e:
        logger.warning("Refine response was not valid JSON: %s", e)
        raise ValueError("Chat response was not valid JSON") from e

    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    segments_out = _parse_segments(raw_list, req.segments)
    return RefineResponse(segments=segments_out, session_id=req.session_id)


# --- CHAT MODE: conversational assistant (do not refine transcript) ---

_CHAT_SYSTEM_PROMPT = """You are a conversational assistant for a voice streaming app. The user may have a transcript of recorded audio (read-only context below).

Your role:
- Answer the user's questions naturally and conversationally.
- You MAY reference the transcript to answer questions about what was said.
- You must NOT modify, refine, or re-interpret the transcript.
- Do NOT output JSON or structured data; reply in plain, natural language.
- If the user asks to "fix" or "refine" the transcript, explain that refinement is done once when they first connect; you can only discuss the existing transcript.
- Prefer natural spoken Indonesian when the user writes in Indonesian."""


def _transcript_dir() -> str:
    return getattr(get_settings(), "TRANSCRIPT_DIR", "./transcripts")


async def session_already_refined(session_id: str) -> bool:
    """True if this session has already been refined."""
    if not session_id:
        return False
    from app.transcript.storage import session_has_refinements

    return await session_has_refinements(session_id)


async def load_transcript_segments(session_id: str) -> list[RefineSegmentInput]:
    """Load transcript lines as segments."""
    from app.transcript.storage import get_transcript_content

    segments: list[RefineSegmentInput] = []
    content = await get_transcript_content(session_id)
    if not content:
        return segments
    for i, line in enumerate(content.splitlines()):
        text = line.strip()
        if not text:
            continue
        segments.append(
            RefineSegmentInput(
                segment_id=str(i),
                text=text,
                start_sec=0.0,
                end_sec=0.0,
                confidence=0.5,
            )
        )
    return segments


async def get_transcript_context_for_chat(session_id: str) -> str | None:
    """Build read-only transcript context for CHAT MODE."""
    from app.transcript.storage import build_chat_transcript_context

    return await build_chat_transcript_context(session_id)


def _transcript_text_to_segments(transcript: str) -> list[RefineSegmentInput]:
    """Convert raw transcript text (one segment per non-empty line) to RefineSegmentInput list."""
    segments: list[RefineSegmentInput] = []
    for i, line in enumerate(transcript.strip().splitlines()):
        text = line.strip()
        if not text:
            continue
        segments.append(
            RefineSegmentInput(
                segment_id=str(i),
                text=text,
                start_sec=0.0,
                end_sec=0.0,
                confidence=0.5,
            )
        )
    return segments


_CLASSIFY_NEED_REFINE_PROMPT = """You are a classifier. Given the user message and the current voice transcript, decide: Is the user providing factual context or corrections that should be used to fix/improve the transcript? For example: correcting a brand name, explaining what they meant, giving topic or channel context. Answer with exactly one word: YES or NO."""


async def classify_need_refine(user_message: str, transcript_text: str) -> bool:
    """
    Ask LLM whether the user is providing facts/context to correct the transcript.
    Returns True only if we should run refine (user gives factual context to fix transcript).
    Uses LLM provider priority (OpenRouter → Cloudflare → Hugging Face).
    """
    if not (user_message or "").strip():
        return False
    transcript_snippet = (transcript_text or "").strip()[:2000]
    user_content = f"User message: {user_message.strip()}\n\nTranscript (excerpt):\n{transcript_snippet or '(empty)'}\n\nAnswer (YES or NO):"
    messages = [
        {"role": "system", "content": _CLASSIFY_NEED_REFINE_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        content = await llm_completion(messages, max_tokens=10, temperature=0.0, timeout=15.0)
        return "yes" in (content or "").strip().lower()
    except Exception as e:
        logger.warning("Classify need_refine failed: %s", e)
        return False


async def refine_raw_transcript(transcript: str, user_question: str) -> str:
    """
    Refine raw transcript text once using Cloudflare LLM. Returns joined refined text.
    Call only when user provides factual context to correct transcript.
    """
    segments = _transcript_text_to_segments(transcript)
    if not segments:
        return transcript
    req = RefineRequest(user_question=user_question, segments=segments)
    response = await refine_transcript_with_chat(req)
    return "\n".join(s.refined_text for s in response.segments if s.refined_text)


def _log_chat_request(max_tokens: int, messages: list[dict[str, str]], max_content_len: int = 2000) -> None:
    """Log payload sent to LLM (content truncated for readability)."""
    logger.info("LLM request: max_tokens=%s, messages=%s", max_tokens, len(messages))
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        length = len(content)
        # dont truncate content but not remove this line, just comment it out
        # display = content if length <= max_content_len else content[:max_content_len] + f"\n... [truncated, total {length} chars]"
        display = content
        logger.info("  [%s] role=%s len=%s:\n%s", i, role, length, display)


async def chat_with_ai_messages(messages: list[dict[str, str]]) -> str:
    """
    Call LLM (OpenRouter → Cloudflare → Hugging Face) with full message history. Returns assistant reply only.
    messages = [ { "role": "system"|"user"|"assistant", "content": "..." }, ... ]
    """
    settings = get_settings()
    if not getattr(settings, "REFINE_CHAT_ENABLED", True):
        raise ValueError("Chat is disabled (REFINE_CHAT_ENABLED=false)")

    max_tokens = getattr(settings, "REFINE_CHAT_MAX_TOKENS", 2048)
    # _log_chat_request(max_tokens, messages)
    return await llm_completion(messages, max_tokens=max_tokens, temperature=0.4)


async def chat_with_ai(message: str, transcript_context: str | None = None) -> str:
    """
    Call LLM (OpenRouter → Cloudflare → Hugging Face) in CHAT MODE. Returns plain conversational reply.
    Does NOT refine or modify transcript. Raises ValueError if disabled or all providers fail.
    """
    settings = get_settings()
    if not getattr(settings, "REFINE_CHAT_ENABLED", True):
        raise ValueError("Chat is disabled (REFINE_CHAT_ENABLED=false)")

    user_content = message
    if transcript_context:
        user_content = f"Transcript (read-only, do not modify):\n{transcript_context}\n\nUser message: {message}"

    messages = [
        {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    max_tokens = getattr(settings, "REFINE_CHAT_MAX_TOKENS", 2048)
    return await llm_completion(messages, max_tokens=max_tokens, temperature=0.4)
