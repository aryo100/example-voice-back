"""
FastAPI app: WebSocket endpoint for real-time speech-to-text;
HTTP API: chat (CHAT MODE default, REFINE MODE one-time) and refine-transcript.

Client sends binary PCM 16-bit mono 16kHz. Server responds with JSON:
{ "type": "partial" | "final", "text": "...", "confidence": 0.0-1.0, "timestamp": unix_ms }
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.asr.base import ASREngine
from app.asr.local_whisper import LocalWhisperEngine
from app.asr.cloudflare import CloudflareWhisperEngine
from app.websocket_manager import WebSocketManager, WebSocketManagerWithAssistant
from app.schemas.refine import RefineRequest, RefineResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.session_store import get_session, set_session, delete_session, ensure_session_from_transcript_file
from app.services.refine_service import (
    refine_transcript_with_chat,
    chat_with_ai_messages,
    session_already_refined,
    get_transcript_context_for_chat,
)
from app.tts.service import synthesize_speech
from app.services.chat_context import (
    summarize_transcript_for_chat,
    select_relevant_snippets,
    build_chat_system_prompt,
    build_chat_system_prompt_for_tts,
    _transcript_hash,
)

logger = logging.getLogger(__name__)


def _assistant_name_and_aliases(settings: Any) -> tuple[str, list[str]]:
    """Return (assistant_name, list of aliases including name). Used for message and transcript checks."""
    name = (getattr(settings, "ASSISTANT_NAME", "") or "Salam").strip()
    aliases_str = (getattr(settings, "ASSISTANT_NAME_ALIASES", "") or "").strip()
    aliases = [a.strip().lower() for a in aliases_str.split(",") if a.strip()]
    if name.lower() not in aliases:
        aliases.insert(0, name.lower())
    return name, aliases


def _is_invoking_assistant(user_message: str, assistant_name: str, name_aliases: list[str] | None = None) -> bool:
    """True bila user memanggil asisten dengan nama atau alias (e.g. 'Salam' atau 'Salam, apa tadi dibahas?')."""
    if not (user_message or "").strip():
        return False
    msg_lower = user_message.strip().lower()
    names_to_check = [assistant_name.strip().lower()]
    if name_aliases:
        names_to_check = name_aliases
    elif assistant_name.strip():
        names_to_check = [assistant_name.strip().lower()]
    for n in names_to_check:
        if not n:
            continue
        if msg_lower == n:
            return True
        if msg_lower.startswith(n):
            rest = msg_lower[len(n) :].lstrip()
            if not rest or rest.startswith(",") or rest.startswith(" ") or rest.startswith("!"):
                return True
        if msg_lower.startswith("hei ") and msg_lower[4:].startswith(n):
            return True
        if msg_lower.startswith("hai ") and msg_lower[4:].startswith(n):
            return True
    return False


def _transcript_invokes_assistant(transcript_text: str, name_aliases: list[str]) -> bool:
    """True bila transcript berisi nama asisten atau alias (e.g. seseorang bilang 'halo salam')."""
    if not (transcript_text or "").strip() or not name_aliases:
        return False
    transcript_lower = transcript_text.strip().lower()
    for n in name_aliases:
        if not n:
            continue
        if n in transcript_lower:
            return True
    return False


def _configure_logging() -> None:
    """Set log level and optional file from config. Log file = LOG_FILE (kosong = hanya konsol)."""
    settings = get_settings()
    level_name = (getattr(settings, "LOG_LEVEL", "") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = (getattr(settings, "LOG_FILE", "") or "").strip()

    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Pastikan ada handler konsol (uvicorn mungkin sudah set)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setLevel(level)
            if not h.formatter:
                h.setFormatter(fmt)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            logger.info("Log file: %s", os.path.abspath(log_file))
        except OSError as e:
            logger.warning("Cannot open log file %s: %s", log_file, e)


# Set in lifespan so WebSocket route can get engine without Request
_current_app: FastAPI | None = None


def get_asr_engine(app: FastAPI | None = None) -> ASREngine:
    """Return ASR engine based on config. Local uses singleton model from app.state."""
    a = app or _current_app
    if a is None:
        raise RuntimeError("App not initialized (lifespan not run?)")
    settings = get_settings()
    if settings.ASR_BACKEND == "cloudflare":
        return CloudflareWhisperEngine()
    model = getattr(a.state, "whisper_model", None)
    return LocalWhisperEngine(model=model)


def _load_whisper_model():
    """Load faster-whisper model once. Called at startup when ASR_BACKEND=local."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as err:
        raise ImportError(
            "faster-whisper is required for ASR_BACKEND=local. "
            "Install with: pip install faster-whisper"
        ) from err
    settings = get_settings()
    return WhisperModel(
        settings.LOCAL_WHISPER_MODEL,
        device=settings.LOCAL_WHISPER_DEVICE,
        compute_type=settings.LOCAL_WHISPER_COMPUTE_TYPE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _current_app
    _current_app = app
    _configure_logging()
    # Load Whisper model once at startup when using local backend (singleton)
    settings = get_settings()
    if settings.ASR_BACKEND == "local":
        app.state.whisper_model = _load_whisper_model()
    else:
        app.state.whisper_model = None
    yield
    # Shutdown: no explicit cleanup needed for faster-whisper
    app.state.whisper_model = None
    _current_app = None


app = FastAPI(
    title="Real-time Speech-to-Text",
    description="WebSocket streaming ASR with VAD and chunking",
    lifespan=lifespan,
)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """
    WebSocket: client sends raw PCM 16-bit mono 16kHz (binary).
    Server sends JSON: { type, text, confidence, timestamp [, word_timestamps] }.
    Hanya transcript; tidak ada assistant_reply.
    """
    await websocket.accept()
    engine = get_asr_engine()
    manager = WebSocketManager(websocket, engine)
    try:
        await manager.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/transcribe-with-assistant")
async def websocket_transcribe_with_assistant(websocket: WebSocket) -> None:
    """
    WebSocket sama seperti /ws/transcribe, plus: bila transcript memanggil asisten (e.g. Salam),
    server kirim dulu { type: "assistant_processing", message: "Menunggu respons..." }, lalu setelah LLM + TTS selesai
    kirim { type: "assistant_reply", text, trigger_audio, audio?, audio_mime }. Client tampilkan waiting sampai assistant_reply.
    """
    await websocket.accept()
    engine = get_asr_engine()
    manager = WebSocketManagerWithAssistant(websocket, engine)
    try:
        await manager.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _save_refinement_layer(session_id: str, response: RefineResponse) -> None:
    """Save refinement as revision layer: transcripts/{session_id}_refinements.json (original unchanged)."""
    settings = get_settings()
    transcript_dir = getattr(settings, "TRANSCRIPT_DIR", "./transcripts")
    path = os.path.join(transcript_dir, f"{session_id}_refinements.json")
    try:
        os.makedirs(transcript_dir, exist_ok=True)
        payload = [s.model_dump() for s in response.segments]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Refinement layer saved: %s", path)
    except OSError as e:
        logger.warning("Failed to save refinement layer %s: %s", path, e)




@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with session. Transcript is knowledge only (transcript_summary); never spoken, never sent to TTS.
    On first chat: build transcript summary if missing. On subsequent: reuse summary + chat history.
    Chat works even when transcript WebSocket is closed (session hydrated from file if needed).
    Transcript is NEVER rewritten unless explicitly requested (e.g. /api/refine-transcript).
    Returns: session_id, text (assistant reply for TTS), audio (null unless TTS implemented).
    """
    action = (request.action or "").strip() or None
    session_id = (request.session_id or "").strip() or None
    message = (request.message or "").strip() if request.message else ""

    # --- Reset (optional) ---
    if action == "reset":
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for reset")
        delete_session(session_id)
        return ChatResponse(session_id=None, text="Session reset.", reply="Session reset.")

    # --- Message and session_id required for chat ---
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required (get it from WebSocket)")

    session = get_session(session_id)
    if not session:
        # Hydrate session from transcript file if it exists (e.g. after server restart)
        transcript_from_file = get_transcript_context_for_chat(session_id)
        session = ensure_session_from_transcript_file(session_id, transcript_from_file or "")
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found; connect WebSocket first or use a session_id that has a transcript file in transcripts/",
        )

    transcript_text = session.get("transcript_text") or ""
    current_hash = _transcript_hash(transcript_text)
    stored_hash = session.get("transcript_summary_hash") or ""
    need_summary = stored_hash != current_hash or not (session.get("global_summary") or "").strip()

    if need_summary and transcript_text.strip():
        try:
            summary_result = await summarize_transcript_for_chat(transcript_text)
            session["global_summary"] = summary_result.get("global_summary", "")
            session["segment_summaries"] = summary_result.get("segment_summaries", [])
            session["compressed_rolling_summary"] = summary_result.get("compressed_rolling_summary", "")
            session["transcript_summary_hash"] = current_hash
        except Exception as e:
            logger.warning("Summary compute failed: %s; using fallback", e)
            session["global_summary"] = "(Transcript summary unavailable.)"
            session["segment_summaries"] = []
            session["compressed_rolling_summary"] = ""
            session["transcript_summary_hash"] = current_hash
    elif need_summary:
        session["global_summary"] = "(No transcript yet.)"
        session["segment_summaries"] = []
        session["compressed_rolling_summary"] = ""
        session["transcript_summary_hash"] = current_hash

    # transcript_raw = session transcript_text (full); transcript_summary = compressed context for LLM only
    global_summary = session.get("global_summary") or "(No transcript yet.)"
    compressed_rolling = session.get("compressed_rolling_summary") or ""
    transcript_summary = global_summary.strip()
    if compressed_rolling.strip():
        transcript_summary = transcript_summary + "\n\n" + compressed_rolling.strip()
    # Optional: add relevant snippet for this message (keeps context compact)
    snippet_max = getattr(get_settings(), "CHAT_SNIPPET_MAX_CHARS", 600)
    relevant_snippets = select_relevant_snippets(transcript_text, message, max_chars=snippet_max)
    system_prompt = build_chat_system_prompt_for_tts(transcript_summary)
    # If we have snippets for this question, append to system so model can reference details
    if relevant_snippets.strip():
        system_prompt = system_prompt + "\n\nRelevant excerpt (for this question only):\n" + relevant_snippets.strip()

    chat_history = list(session.get("messages", []))
    settings = get_settings()
    max_history = getattr(settings, "CHAT_HISTORY_MAX_MESSAGES", 20)
    if len(chat_history) > max_history:
        chat_history = chat_history[-max_history:]
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        *chat_history,
        {"role": "user", "content": message},
    ]
    try:
        reply = await chat_with_ai_messages(messages_for_llm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=502, detail="Chat failed")

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})
    session["messages"] = chat_history
    set_session(session_id, session)
    # Trigger audio: bila user memanggil nama asisten ATAU transcript (session atau file) berisi nama/alias (e.g. Salam).
    assistant_name, name_aliases = _assistant_name_and_aliases(settings)
    trigger_from_message = _is_invoking_assistant(message, assistant_name, name_aliases)
    trigger_from_session = _transcript_invokes_assistant(transcript_text, name_aliases)
    # Fallback: cek juga isi file transcript (session in-memory bisa tertinggal satu baris)
    transcript_from_file = get_transcript_context_for_chat(session_id) or ""
    trigger_from_file = _transcript_invokes_assistant(transcript_from_file, name_aliases)
    trigger_audio = trigger_from_message or trigger_from_session or trigger_from_file
    # TTS: lokal pakai Edge TTS; nanti Cloudflare pakai TTS dari sana
    audio_b64 = None
    if trigger_audio:
        audio_b64, _ = await synthesize_speech(reply)

    return ChatResponse(
        session_id=session_id,
        text=reply,
        audio=audio_b64,
        trigger_audio=trigger_audio,
        reply=reply,
        message=reply,
        mode="chat",
    )


@app.post("/api/refine-transcript", response_model=RefineResponse)
async def refine_transcript(request: RefineRequest) -> RefineResponse:
    """
    One-time transcript refinement (explicit). Refuses if session already refined.

    Inputs: user question, raw transcript segments (timestamps + confidence),
    optional audio reference, domain hint, session_id.
    Output: per-segment refined text, confidence, justification. Stored as revision layer.
    """
    if request.session_id and session_already_refined(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="Transcript for this session has already been refined; use /api/chat for conversation",
        )
    try:
        response = await refine_transcript_with_chat(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Refine transcript failed: %s", e)
        raise HTTPException(status_code=502, detail="Refine chat failed")

    if request.session_id:
        _save_refinement_layer(request.session_id, response)

    return response
