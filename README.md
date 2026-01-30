# Real-Time Speech-to-Text Backend

Python 3.11 backend for **real-time speech-to-text** over WebSocket. Accepts continuous PCM audio, runs VAD and chunking with overlap, and returns partial and final transcripts via a swappable Whisper-compatible ASR engine.

## Stack

- **Transport**: WebSocket (no HTTP polling)
- **Framework**: FastAPI
- **Audio**: PCM 16-bit mono, 16 kHz
- **VAD**: webrtcvad (or optional Silero-VAD)
- **ASR**: Whisper-compatible (local: faster-whisper, remote: Cloudflare Workers AI)
- **Optional**: Per-session audio recording to MP3 (env toggle; see docs/AUDIO_RECORDING_NOTES.md)

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Unix
pip install -r requirements.txt

# Optional: for local ASR
pip install faster-whisper

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **WebSocket**: `ws://localhost:8000/ws/transcribe`
- **Health**: `GET /health`
- **Refine transcript**: `POST /api/refine-transcript` — context-aware re-transcription using chat AI (see [docs/REFINE_TRANSCRIPT.md](docs/REFINE_TRANSCRIPT.md))

## Streaming Logic: Chunking + VAD

### Overview

1. **Client** sends raw PCM frames (binary) over WebSocket.
2. **AudioReceiver** buffers bytes and yields fixed **20 ms frames** (640 bytes @ 16 kHz).
3. **VADProcessor** labels each frame as speech or silence (webrtcvad).
4. **AudioChunker** keeps a **ring buffer** of frames and emits **chunks** when:
   - There is at least **~1.5 s** of audio in the buffer, and
   - **Silence** has been detected for **>600 ms** (configurable).
5. Chunks are sent to **ASREngine** (in a thread pool) to avoid blocking the event loop.
6. Server sends **partial** (real-time, may change) and **final** (committed after silence) JSON messages.

### Why This Design

- **20 ms frames**: Matches webrtcvad and keeps latency low.
- **~1.5 s chunks**: Enough context for Whisper; not so long that latency explodes.
- **300 ms overlap**: Adjacent chunks share 300 ms of audio so words at boundaries are not cut.
- **Commit after 600 ms silence**: Avoids cutting words mid-utterance; only commit when the user has paused.

### Ring Buffer and Overlap

- **Frame size**: 20 ms → 320 samples → 640 bytes.
- **Chunk size**: 1500 ms → 75 frames.
- **Overlap**: 300 ms → 15 frames shared with the next chunk.
- **Silence commit**: 600 ms → 30 consecutive silence frames before emitting a chunk.

After a chunk is emitted, the chunker keeps the **last 300 ms** (15 frames) in the buffer so the next chunk starts with that overlap. This reduces word-boundary artifacts.

### Message Flow

```
Client (binary PCM) → WebSocket → AudioReceiver → VADProcessor → AudioChunker
                                                                      ↓
                                                            on_chunk(bytes)
                                                                      ↓
                                                            ASR queue → executor
                                                                      ↓
                                                            ASREngine.transcribe()
                                                                      ↓
Server → Client (JSON): partial then final { type, text, confidence, timestamp }
```

## WebSocket Protocol

### Client → Server

- **Binary**: Raw PCM 16-bit mono 16 kHz. Send in any chunk size; the server buffers and frames internally.

### Server → Client (JSON)

```json
{
  "type": "partial",
  "text": "",
  "confidence": 0.0,
  "timestamp": 1706544000000
}
```

```json
{
  "type": "final",
  "text": "Hello world",
  "confidence": 1.0,
  "timestamp": 1706544000123,
  "word_timestamps": [{"word": "Hello", "start": 0.0, "end": 0.2}, {"word": "world", "start": 0.2, "end": 0.5}]
}
```

- **partial**: Sent while processing; text may be empty or interim (faster decode, lower beam).
- **final**: Committed transcript for the chunk (after silence); may include **word_timestamps** when using local Whisper.
- **timestamp**: Unix time in milliseconds.

## Refine transcript (context-aware re-transcription)

**POST** `/api/refine-transcript` lets chat AI reinterpret recent transcript segments using a user question and domain hint. Original STT is never overwritten; refined text is a revision layer. Uses **Cloudflare Workers AI** (free tier), no OpenAI/GPT. Inputs: `user_question`, `segments` (with timestamps & confidence), optional `audio_reference_seconds` / `audio_description`, `domain_hint`, optional `session_id` (to save refinement as `{session_id}_refinements.json`). Output: per-segment `original_text`, `refined_text`, `confidence_before` / `confidence_after`, `justification`. Requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (same as Cloudflare ASR). See **docs/REFINE_TRANSCRIPT.md**.

## Configuration (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_BACKEND` | `local` | `local` or `cloudflare` |
| `SAMPLE_RATE` | 16000 | PCM sample rate |
| `FRAME_MS` | 20 | Frame duration (ms) |
| `CHUNK_DURATION_MS` | 1500 | Chunk length before overlap |
| `OVERLAP_MS` | 300 | Overlap between chunks |
| `SILENCE_COMMIT_MS` | 600 | Silence duration before committing chunk |
| `LOCAL_WHISPER_MODEL` | `base` | `base` \| `small` \| `medium` \| `large-v3` |
| `LOCAL_WHISPER_DEVICE` | `cpu` | `cpu` \| `cuda` |
| `LOCAL_WHISPER_COMPUTE_TYPE` | `int8` | `int8` \| `float16` |
| `LOCAL_WHISPER_BEAM_SIZE_PARTIAL` | `1` | Beam size for partial (faster) |
| `LOCAL_WHISPER_BEAM_SIZE_FINAL` | `5` | Beam size for final (stable) |
| `LOCAL_WHISPER_WORD_TIMESTAMPS` | `true` | Emit word timings on final |
| `CLOUDFLARE_ACCOUNT_ID` | - | For Cloudflare ASR |
| `CLOUDFLARE_API_TOKEN` | - | For Cloudflare ASR |
| `ENABLE_AUDIO_RECORDING` | `false` | Save one MP3 per WebSocket session |
| `AUDIO_RECORD_DIR` | `./recordings` | Directory for recorded MP3s |
| `AUDIO_RECORD_BITRATE` | `128k` | MP3 bitrate |

## Folder Structure

```
app/
├── main.py              # FastAPI app, WebSocket route
├── config.py            # Settings (pydantic-settings)
├── websocket_manager.py # WebSocketManager: pipeline + ASR consumer
├── audio/
│   ├── receiver.py      # AudioReceiver: buffer → 20ms frames
│   ├── vad.py           # VADProcessor: webrtcvad
│   ├── chunker.py       # AudioChunker: ring buffer, overlap, silence commit
│   └── recorder.py      # Optional AudioRecorder: PCM → MP3 per session
├── asr/
│   ├── base.py          # ASREngine abstract, TranscriptResult
│   ├── local_whisper.py # LocalWhisperEngine (faster-whisper)
│   └── cloudflare.py    # CloudflareWhisperEngine (Workers AI)
└── transcript/
    └── merger.py        # TranscriptMerger: partial vs final messages

recordings/               # When ENABLE_AUDIO_RECORDING=true (one MP3 per session)
```

## Local Whisper engine (where it plugs in)

- **Model**: Loaded **once at startup** in `app/main.py` lifespan when `ASR_BACKEND=local`; stored in `app.state.whisper_model`.
- **Engine**: `LocalWhisperEngine(model=...)` is created per WebSocket in the `/ws/transcribe` route via `get_asr_engine()`.
- **Pipeline**: `WebSocketManager` converts PCM bytes → float32 numpy, then calls `await engine.transcribe(audio, is_final=False)` (partial) and `await engine.transcribe(audio, is_final=True)` (final). Heavy work runs in executor so the event loop stays responsive.
- **Config**: See `.env.example`; model name, device, compute_type, and beam sizes are configurable.

See **docs/LOCAL_WHISPER_NOTES.md** for performance tradeoffs (beam size, model size, device, partial vs final).

## Swapping ASR

Implement `ASREngine` (see `app/asr/base.py`):

- `async transcribe(audio: np.ndarray, is_final: bool) -> ASRResult`
- `sample_rate: int`

Then in `app/main.py`, extend `get_asr_engine()` to return your implementation. ASR must not block the event loop (run sync work in executor).

## Non-Functional

- **Async**: Receiving, VAD, and chunking are sync on small buffers; ASR runs in `run_in_executor`.
- **Low latency**: 20 ms frames, 1.5 s chunks, commit on 600 ms silence.
- **No HTTP upload**: Only WebSocket; no short-recording HTTP endpoints.
