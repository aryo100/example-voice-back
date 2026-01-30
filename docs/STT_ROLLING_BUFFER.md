# Real-Time STT: Rolling Buffer

## Why transcription was delayed (silence-based pipeline)

Previously, chunks were emitted only when **silence** exceeded ~600ms (SILENCE_COMMIT_MS). As a result:

- No transcription happened until the user stopped speaking.
- Long speech produced only the **last 1–2 seconds** (whatever fit in the final chunk after silence).
- Earlier content was never sent to ASR.

## How the rolling buffer fixes it

- We keep a **fixed window** (e.g. 5s) of the most recent audio.
- We transcribe **on a timer** (e.g. every 1s), not on silence.
- **Partial** text = newest segment(s) (can change).
- **Final** text = segments older than COMMIT_AGE (append-only, never overwritten).

So partial text appears while the user is speaking, and the full transcript is built from committed segments.

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_USE_ROLLING_BUFFER` | `true` | Use rolling buffer; if `false`, legacy silence-based chunker |
| `STT_WINDOW_SECONDS` | `5.0` | Buffer window (2–5s typical) |
| `STT_STEP_SECONDS` | `1.0` | Transcribe every N seconds |
| `STT_MIN_CHUNK_SECONDS` | `0.5` | Do not transcribe chunks < 500ms (reduce hallucination) |
| `STT_COMMIT_AGE_SECONDS` | `2.0` | Commit segments older than N s; newer stay as partial |

## Triggers

Transcription runs when:

- Buffer duration ≥ `STT_WINDOW_SECONDS`, and
- Time since last emit ≥ `STT_STEP_SECONDS`.

**Not** when silence is detected. Silence is optional (e.g. for VAD elsewhere); it does not gate emission.

## Segment-aware commit strategy

Rolling windows overlap; **only segment timestamps** determine uniqueness. We never replace or re-send entire transcript—only append new finals.

- **committed_until_time** (session seconds): Segments ending at or before this are **skipped** (already committed; rolling windows may re-send the same segment).
- **current_audio_time** = end of this chunk in session time = `chunk_start_sec + chunk_duration`.
- **Commit horizon** = `current_audio_time - STT_COMMIT_AGE_SECONDS`. Segments ending **before** the horizon are stable → **FINAL**; segments after the horizon stay **PARTIAL** until a later chunk pushes the horizon forward.

For each Whisper segment (in chronological order):

1. `segment_end_session = chunk_start_sec + segment.end`
2. If `segment_end_session <= committed_until_time` → **SKIP** (no duplicate).
3. If `segment_end_session <= commit_horizon` → **COMMIT** as FINAL (one message per segment); then `committed_until_time = max(committed_until_time, segment_end_session)`.
4. Else → **PARTIAL** (may change in next window).

Text is normalized before commit (strip repeated whitespace, remove trailing punctuation duplication). One FINAL message per committed segment; one PARTIAL message with concatenated partial segments.

## WebSocket messages (unchanged)

- `{ "type": "partial", "text": "...", "confidence", "timestamp" }` — live text.
- `{ "type": "final", "text": "...", ... }` — committed slice; frontend appends to transcript.

One WebSocket = one session; final text is accumulated and never overwritten.
