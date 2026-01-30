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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.asr.base import ASREngine
from app.asr.local_whisper import LocalWhisperEngine
from app.asr.cloudflare import CloudflareWhisperEngine
from app.websocket_manager import WebSocketManager
from app.schemas.refine import RefineRequest, RefineResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.session_store import get_session, set_session, delete_session
from app.services.refine_service import (
    refine_transcript_with_chat,
    chat_with_ai_messages,
    session_already_refined,
    classify_need_refine,
    refine_raw_transcript,
)

logger = logging.getLogger(__name__)

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


def _build_transcript_system_prompt(transcript_text: str) -> str:
    """System prompt for chat: transcript as factual background knowledge. Do not change during session."""
    return (
        "You are a conversational AI.\n"
        "The following is a transcript of what the user has spoken recently.\n"
        "Treat this transcript as factual background knowledge.\n"
        "Do NOT repeat it unless asked.\n"
        "Transcript:\n"
        f"{transcript_text.strip() if transcript_text else '(none yet)'}"
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with session. Transcript is knowledge context (updated only by WebSocket).

    Flow: get session by session_id → get transcript_text → build system prompt (index 0) →
    prepend system prompt, append session.messages (chat history), append current user message →
    send to LLM → append user + assistant to session.messages. Transcript is NOT modified by chat.
    Reset: action=reset + session_id → delete session.
    """
    action = (request.action or "").strip() or None
    session_id = (request.session_id or "").strip() or None
    message = (request.message or "").strip() if request.message else ""

    # --- Reset (optional) ---
    if action == "reset":
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for reset")
        delete_session(session_id)
        return ChatResponse(session_id=None, reply="Session reset.")

    # --- Message and session_id required for chat ---
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required (get it from WebSocket)")

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found; connect WebSocket first to create session")

    transcript_text = session.get("transcript_text") or ""
    # Optional: if AI thinks user is giving facts to correct transcript, refine once and update session
    if transcript_text.strip() and not session.get("transcript_refined_with_context"):
        try:
            need_refine = await classify_need_refine(message, transcript_text)
            if need_refine:
                refined = await refine_raw_transcript(transcript_text, message)
                if refined.strip():
                    session["transcript_text"] = refined
                    session["transcript_refined_with_context"] = True
                    set_session(session_id, session)
                    transcript_text = refined
        except Exception as e:
            logger.warning("Conditional refine failed: %s", e)

    system_prompt = _build_transcript_system_prompt(transcript_text)
    # Message flow: [ system(transcript knowledge), ...session.messages, user(current) ]
    chat_history = list(session.get("messages", []))
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
    return ChatResponse(session_id=session_id, reply=reply, mode="chat", message=reply)


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
