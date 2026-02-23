"""
WebSocketManager: orchestrates audio pipeline and ASR.

Rolling windows: Audio chunks overlap (e.g. 5s window, 1s step). Whisper returns segments
with start/end relative to each chunk. Only timestamps determine uniqueness—we must never
re-commit a segment (segment.end in session time <= committed_until_time → skip).
Segment stability: We commit a segment as FINAL only when it is "behind" the commit horizon
(current_audio_time - COMMIT_DELAY). Segments after the horizon stay PARTIAL until a later
chunk pushes the horizon forward. This prevents duplication and preserves order.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

import numpy as np

from fastapi import WebSocket

from app.audio import (
    AudioChunker,
    AudioRecorderBase,
    AudioReceiver,
    RollingBuffer,
    VADProcessor,
    create_audio_recorder,
)
from app.asr.base import ASREngine
from app.config import get_settings
from app.session_store import ensure_session_for_websocket, get_session, set_session, update_session_transcript
from app.tts.service import synthesize_speech
from app.transcript.merger import TranscriptMessage
from app.transcript.writer import TranscriptWriterBase, create_transcript_writer
from app.diarization import get_speaker_tracker

logger = logging.getLogger(__name__)


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM 16-bit mono bytes to float32 [-1.0, 1.0]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _normalize_commit_text(text: str) -> str:
    """Before commit: strip repeated whitespace, remove trailing punctuation duplication.
    Jangan hilangkan duplikat kata (e.g. 'Salam Salam') agar pemanggilan asisten berkali-kali tetap terdeteksi."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?,;:])\1+", r"\1", text)
    return text.strip()


# Min overlap length (chars) so we don't match tiny fragments like " the " across segments
_MIN_OVERLAP_CHARS = 12


def _normalize_for_overlap(s: str) -> str:
    """Lowercase, collapse spaces, normalize punctuation so '. then' and ', then' match."""
    s = re.sub(r"\s+", " ", s.strip()).lower()
    s = re.sub(r"[.,;:!?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _text_to_append(accumulated_lines: list[str], new_segment: str) -> str | None:
    """
    Dedupe overlapping segments: compare with full transcript so far, not just last segment.
    Each chunk re-transcribes overlapping audio, so segment boundaries differ—we need to find
    overlap with the *end* of accumulated text and append only the new part.
    - accumulated_lines: lines we've already written (joined for comparison).
    - new_segment: raw segment from Whisper.
    Returns the text to append (suffix only, or full if no overlap), or None to skip (duplicate).
    """
    new = _normalize_commit_text(new_segment)
    if not new:
        return None
    accumulated = " ".join(accumulated_lines)
    acc_norm = _normalize_for_overlap(accumulated)
    new_norm = _normalize_for_overlap(new)
    if not acc_norm:
        return new
    if new_norm == acc_norm:
        return None
    if acc_norm.endswith(new_norm) or new_norm in acc_norm:
        return None
    if acc_norm and new_norm.startswith(acc_norm):
        suffix_norm = new_norm[len(acc_norm) :].strip()
        if not suffix_norm:
            return None
        return _normalize_commit_text(suffix_norm) or None
    # Longest suffix of accumulated that equals prefix of new (case-insensitive, min length)
    overlap_len = 0
    for L in range(min(len(acc_norm), len(new_norm)), _MIN_OVERLAP_CHARS - 1, -1):
        if acc_norm[-L:] == new_norm[:L]:
            overlap_len = L
            break
    if overlap_len >= _MIN_OVERLAP_CHARS:
        overlap_str = new_norm[:overlap_len]
        suffix = new_norm[overlap_len:].strip()
        if not suffix:
            return None
        # Preserve casing: find where overlap ends in original new and take rest
        match = re.search(re.escape(overlap_str), new, re.IGNORECASE)
        if match:
            suffix_original = new[match.end() :].strip()
            return _normalize_commit_text(suffix_original) or None
        return _normalize_commit_text(suffix) or None
    # Fallback: longest prefix of new_norm that appears *anywhere* in acc_norm
    # (Whisper sometimes sends segments that start mid-sentence, so suffix-of-acc != prefix-of-new)
    for L in range(min(len(new_norm), len(acc_norm)), _MIN_OVERLAP_CHARS - 1, -1):
        # Only split at word boundary so word count matches original
        if L < len(new_norm) and new_norm[L] != " ":
            continue
        if new_norm[:L] in acc_norm:
            suffix_norm = new_norm[L:].strip()
            if not suffix_norm:
                return None
            overlap_words = len(new_norm[:L].split())
            new_words = new.split()
            if overlap_words >= len(new_words):
                return None
            suffix_original = " ".join(new_words[overlap_words:]).strip()
            return _normalize_commit_text(suffix_original) or None
    return new


def _format_final_line_for_session(text: str, speaker_id: str | None, overlap: bool) -> str:
    """Format one final line for session transcript (chat sees speaker labels). Semua transcript huruf kecil."""
    if not text:
        return ""
    content = (text or "").strip().lower()
    if not content:
        return ""
    if speaker_id:
        prefix = f"[{speaker_id}] "
        if overlap:
            prefix += "[overlap] "
        return prefix + content
    return content


def _message_to_json(msg: TranscriptMessage) -> str:
    payload: dict = {
        "type": msg.type,
        "text": msg.text,
        "confidence": msg.confidence,
        "timestamp": msg.timestamp,
    }
    if msg.word_timestamps:
        payload["word_timestamps"] = msg.word_timestamps
    # Speaker-aware (final only)
    if msg.speaker_id is not None:
        payload["speaker_id"] = msg.speaker_id
    if msg.start_time is not None:
        payload["start_time"] = msg.start_time
    if msg.end_time is not None:
        payload["end_time"] = msg.end_time
    if msg.overlap:
        payload["overlap"] = True
    return json.dumps(payload)


class WebSocketManager:
    """
    One WebSocket = one transcription session. Rolling buffer mode: partial text
    appears while user speaks; final text is append-only and never overwritten.
    """

    def __init__(self, websocket: WebSocket, engine: ASREngine) -> None:
        self._ws = websocket
        self._engine = engine
        settings = get_settings()
        self._receiver = AudioReceiver()
        self._vad = VADProcessor(aggressiveness=2)
        self._use_rolling = getattr(settings, "STT_USE_ROLLING_BUFFER", True)
        self._chunk_queue: asyncio.Queue[tuple[bytes, float]] = asyncio.Queue()
        self._consumer_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._recorder: AudioRecorderBase = create_audio_recorder()

        # One session ID per WebSocket; used for transcript file and optional recording
        self._session_id = uuid.uuid4().hex[:12]
        self._transcript_writer: TranscriptWriterBase | None = None

        # Segment-aware commit: only timestamps determine uniqueness; append-only final text.
        self._committed_until_time: float = 0.0  # session seconds; segments ending before this are skipped
        self._final_text: list[str] = []  # accumulated lines written (for overlap dedup with full transcript)
        self._chunker: AudioChunker | None = None
        self._rolling_buffer: RollingBuffer | None = None
        self._speaker_tracker = get_speaker_tracker()

    def _after_final_commit(self, line_for_session: str) -> None:
        """Override di subclass untuk kirim assistant_reply via WS. Base: no-op."""
        pass

    async def _send_message(self, msg: TranscriptMessage) -> None:
        if self._closed:
            return
        try:
            await self._ws.send_text(_message_to_json(msg))
        except Exception:
            self._closed = True

    def _on_rolling_chunk(self, chunk: bytes, chunk_start_sec: float) -> None:
        """Called by RollingBuffer on time-based trigger (no silence)."""
        try:
            self._chunk_queue.put_nowait((chunk, chunk_start_sec))
        except asyncio.QueueFull:
            pass

    def _on_chunker_chunk(self, chunk: bytes) -> None:
        """Legacy: called by AudioChunker when silence exceeded."""
        try:
            self._chunk_queue.put_nowait((chunk, 0.0))
        except asyncio.QueueFull:
            pass

    async def _asr_consumer(self) -> None:
        """
        Consume (chunk, chunk_start_sec). Run ASR once; commit segments by timestamp only.
        Rolling windows overlap—only segment.end in session time determines uniqueness.
        Commit horizon: segments ending before (current_audio_time - COMMIT_DELAY) are stable → FINAL.
        """
        settings = get_settings()
        sample_rate = settings.SAMPLE_RATE
        chunk_duration_sec = lambda b: len(b) / (sample_rate * 2)  # PCM 16-bit mono
        commit_delay = getattr(settings, "STT_COMMIT_AGE_SECONDS", 2.0)
        # Epsilon: absorb timestamp jitter so overlapping windows don't commit same segment twice
        commit_skip_epsilon = 0.05

        while True:
            try:
                item = await asyncio.wait_for(self._chunk_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._closed:
                    break
                continue
            if self._closed and self._chunk_queue.empty():
                break
            chunk, chunk_start_sec = item
            if not chunk:
                continue

            audio = _pcm_bytes_to_float32(chunk)
            chunk_dur = chunk_duration_sec(chunk)
            ts_ms = int(time.time() * 1000)

            # End of this chunk in session time; commit horizon = segments before this are stable
            current_audio_time = chunk_start_sec + chunk_dur
            commit_horizon = current_audio_time - commit_delay

            result = await self._engine.transcribe(audio, is_final=True)
            if self._closed:
                break

            # Process Whisper segments in order; only timestamps determine commit/skip.
            if result.segments:
                partial_parts: list[str] = []
                for seg in result.segments:
                    segment_start_session = chunk_start_sec + seg.start
                    segment_end_session = chunk_start_sec + seg.end
                    # Skip: already committed (rolling windows re-send same segment; epsilon absorbs jitter)
                    if segment_end_session <= self._committed_until_time + commit_skip_epsilon:
                        continue
                    # Commit as FINAL if: (a) segment fully behind horizon, (b) segment starts before horizon,
                    # or (c) session ended (flush chunk)—commit all remaining so tail is not lost
                    behind_horizon = segment_end_session <= commit_horizon
                    starts_before_horizon = segment_start_session < commit_horizon
                    is_session_ended = self._closed
                    if behind_horizon or starts_before_horizon or is_session_ended:
                        raw = _normalize_commit_text(seg.text)
                        if not raw:
                            continue
                        # Only append the new part; compare with full transcript so far (not just last segment)
                        text = _text_to_append(self._final_text, raw)
                        self._committed_until_time = max(
                            self._committed_until_time, segment_end_session
                        )
                        if text is None:
                            continue
                        # Speaker-aware: assign speaker_id and overlap (diarization-only; no audio separation)
                        speaker_id, overlap = self._speaker_tracker.assign(
                            self._session_id,
                            segment_start_session,
                            segment_end_session,
                            overlap_hint=False,
                        )
                        text_low = (text or "").strip().lower()
                        line_for_session = _format_final_line_for_session(text_low, speaker_id, overlap)
                        self._final_text.append(line_for_session)
                        await self._send_message(
                            TranscriptMessage(
                                type="final",
                                text=text_low,
                                confidence=result.confidence,
                                timestamp=ts_ms,
                                word_timestamps=None,
                                speaker_id=speaker_id,
                                start_time=segment_start_session,
                                end_time=segment_end_session,
                                overlap=overlap,
                            )
                        )
                        if self._transcript_writer:
                            self._transcript_writer.append_final(
                                text_low,
                                ts_ms,
                                speaker_id=speaker_id,
                                start_time=segment_start_session,
                                end_time=segment_end_session,
                                overlap=overlap,
                            )
                        # Update session transcript (only WebSocket updates transcript; chat only reads)
                        update_session_transcript(
                            self._session_id,
                            "\n".join(self._final_text),
                            transcript_finalized=False,
                        )
                        self._after_final_commit(line_for_session)
                    else:
                        t = (seg.text or "").strip()
                        if t:
                            partial_parts.append(t)

                partial_text = " ".join(partial_parts).strip()
                if partial_text:
                    await self._send_message(
                        TranscriptMessage(
                            type="partial",
                            text=partial_text,
                            confidence=result.confidence,
                            timestamp=ts_ms,
                            word_timestamps=None,
                        )
                    )
            else:
                # No segments (e.g. Cloudflare): one chunk = one final; no rolling deduplication
                if result.text:
                    text = _normalize_commit_text(result.text)
                    await self._send_message(
                        TranscriptMessage(
                            type="partial",
                            text=text,
                            confidence=result.confidence,
                            timestamp=ts_ms,
                            word_timestamps=None,
                        )
                    )
                    # Speaker-aware: use chunk time range for segment times
                    segment_start_session = chunk_start_sec
                    segment_end_session = chunk_start_sec + chunk_dur
                    speaker_id, overlap = self._speaker_tracker.assign(
                        self._session_id,
                        segment_start_session,
                        segment_end_session,
                        overlap_hint=False,
                    )
                    text_low = (text or "").strip().lower()
                    line_for_session = _format_final_line_for_session(text_low, speaker_id, overlap)
                    self._final_text.append(line_for_session)
                    await self._send_message(
                        TranscriptMessage(
                            type="final",
                            text=text_low,
                            confidence=result.confidence,
                            timestamp=ts_ms,
                            word_timestamps=None,
                            speaker_id=speaker_id,
                            start_time=segment_start_session,
                            end_time=segment_end_session,
                            overlap=overlap,
                        )
                    )
                    if self._transcript_writer:
                        self._transcript_writer.append_final(
                            text_low,
                            ts_ms,
                            speaker_id=speaker_id,
                            start_time=segment_start_session,
                            end_time=segment_end_session,
                            overlap=overlap,
                        )
                    update_session_transcript(
                        self._session_id,
                        "\n".join(self._final_text),
                        transcript_finalized=False,
                    )
                    self._after_final_commit(line_for_session)

    async def run(self) -> None:
        """Main loop: receive binary, feed rolling buffer or chunker, run consumer."""
        settings = get_settings()
        use_rolling = getattr(settings, "STT_USE_ROLLING_BUFFER", True)
        session_start_ms = int(time.time() * 1000)
        self._transcript_writer = create_transcript_writer(self._session_id, session_start_ms)
        await self._transcript_writer.start()
        ensure_session_for_websocket(self._session_id)
        try:
            await self._ws.send_text(json.dumps({"type": "session", "session_id": self._session_id}))
        except Exception:
            pass

        if use_rolling:
            self._rolling_buffer = RollingBuffer(on_chunk=self._on_rolling_chunk)
        else:
            self._chunker = AudioChunker(on_chunk=self._on_chunker_chunk)

        self._consumer_task = asyncio.create_task(self._asr_consumer())

        try:
            while not self._closed:
                try:
                    msg = await self._ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    data = msg.get("bytes")
                    if data is None:
                        continue
                except Exception:
                    break
                self._receiver.feed(data)
                self._recorder.append(data)
                for frame in self._receiver.drain_frames():
                    if use_rolling:
                        self._rolling_buffer.push(frame)
                    else:
                        is_speech = self._vad.is_speech(frame)
                        self._chunker.push(frame, is_speech)
        finally:
            self._closed = True
            # Flush remaining audio into queue so consumer can process it
            if use_rolling and self._rolling_buffer:
                flushed = self._rolling_buffer.flush()
                if flushed:
                    chunk, start = flushed
                    try:
                        self._chunk_queue.put_nowait((chunk, start))
                    except asyncio.QueueFull:
                        pass
            else:
                if self._chunker:
                    flushed = self._chunker.flush()
                    if flushed:
                        try:
                            self._chunk_queue.put_nowait((flushed, 0.0))
                        except asyncio.QueueFull:
                            pass
            # Drain queue so all chunks (including flush) are transcribed; avoid short timeout losing early chunks
            if self._consumer_task:
                try:
                    await asyncio.wait_for(self._consumer_task, timeout=300.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self._consumer_task.cancel()
                    try:
                        await self._consumer_task
                    except asyncio.CancelledError:
                        pass
            if self._transcript_writer:
                await self._transcript_writer.close()
            update_session_transcript(
                self._session_id,
                "\n".join(self._final_text),
                transcript_finalized=True,
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._recorder.finalize)


# --- Subclass untuk route dengan assistant reply via WS ---
def _transcript_contains_assistant(transcript_text: str) -> bool:
    """True bila transcript berisi nama asisten atau alias (e.g. Salam). Dipakai di WS dengan assistant."""
    if not (transcript_text or "").strip():
        return False
    settings = get_settings()
    name = (getattr(settings, "ASSISTANT_NAME", "") or "Salam").strip().lower()
    aliases_str = (getattr(settings, "ASSISTANT_NAME_ALIASES", "") or "").strip()
    aliases = [a.strip().lower() for a in aliases_str.split(",") if a.strip()]
    if name and name not in aliases:
        aliases.insert(0, name)
    transcript_lower = transcript_text.strip().lower()
    return any(n and n in transcript_lower for n in aliases)


class WebSocketManagerWithAssistant(WebSocketManager):
    """
    Sama seperti WebSocketManager, tapi saat transcript memanggil asisten (e.g. Salam),
    kirim assistant_reply via WebSocket (type: assistant_reply, text, trigger_audio).
    Pakai route /ws/transcribe-with-assistant.
    """

    def __init__(self, websocket: WebSocket, engine: ASREngine) -> None:
        super().__init__(websocket, engine)
        self._last_assistant_trigger_time: float = 0.0

    def _after_final_commit(self, line_for_session: str) -> None:
        if not _transcript_contains_assistant(line_for_session):
            return
        debounce_sec = getattr(get_settings(), "ASSISTANT_WS_DEBOUNCE_SEC", 15.0)
        if time.time() - self._last_assistant_trigger_time < debounce_sec:
            return
        self._last_assistant_trigger_time = time.time()
        asyncio.create_task(self._maybe_send_assistant_reply_ws(line_for_session))

    async def _maybe_send_assistant_reply_ws(self, user_message: str) -> None:
        if self._closed or not (user_message or "").strip():
            return
        try:
            # Kirim waiting dulu; client tampilkan "Menunggu respons..." sampai assistant_reply (setelah LLM + TTS)
            try:
                await self._ws.send_text(
                    json.dumps({"type": "assistant_processing", "message": "Menunggu respons..."})
                )
            except Exception:
                pass
            from app.services.chat_context import (
                summarize_transcript_for_chat,
                build_chat_system_prompt_for_tts,
                _transcript_hash,
            )
            from app.services.refine_service import chat_with_ai_messages

            session = get_session(self._session_id)
            if not session:
                return
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
                    set_session(self._session_id, session)
                except Exception:
                    pass
            global_summary = session.get("global_summary") or "(No transcript yet.)"
            compressed_rolling = session.get("compressed_rolling_summary") or ""
            transcript_summary = global_summary.strip()
            if compressed_rolling.strip():
                transcript_summary = transcript_summary + "\n\n" + compressed_rolling.strip()
            system_prompt = build_chat_system_prompt_for_tts(transcript_summary)
            messages_for_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message.strip()},
            ]
            reply = await chat_with_ai_messages(messages_for_llm)
            if self._closed:
                return
            # TTS: lokal pakai Edge TTS; nanti Cloudflare pakai TTS dari sana
            audio_b64, audio_mime = await synthesize_speech(reply)
            if not audio_b64:
                logger.warning(
                    "assistant_reply: TTS failed (audio=null). Cek log: 403 = Edge TTS diblok (region/jaringan)? Set TTS_BACKEND=none untuk non-audio."
                )
            payload = json.dumps({
                "type": "assistant_reply",
                "text": reply,
                "trigger_audio": True,
                "audio": audio_b64,
                "audio_mime": audio_mime or "audio/mpeg",
            })
            await self._ws.send_text(payload)
        except Exception as e:
            logger.exception("assistant_reply failed: %s", e)
