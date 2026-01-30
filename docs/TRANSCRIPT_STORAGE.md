# Session Transcript Storage

All **final** transcript segments are appended to a single `.txt` file per WebSocket session. Partial text is never written.

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIPT_SAVE_ENABLED` | `true` | Save final segments to file |
| `TRANSCRIPT_DIR` | `./transcripts` | Directory for session files (auto-created) |
| `TRANSCRIPT_ADD_TIMESTAMPS` | `false` | Prefix each line with `[MM:SS.ss]` (elapsed since session start) |

## File layout

- **Path**: `transcripts/{session_id}.txt`
- **Session ID**: Unique per WebSocket (e.g. 12-char hex).
- **Content**: One line per final segment; append-only; correct order.

Example (no timestamps):

```
halo saya ingin bertanya
tentang arsitektur backend
dan real time speech to text
```

With `TRANSCRIPT_ADD_TIMESTAMPS=true`:

```
[00:01.20] halo saya ingin bertanya
[00:04.85] tentang arsitektur backend
[00:08.10] dan real time speech to text
```

## Why append-only

- Final segments arrive over time; overwriting would lose earlier content.
- Each write appends one line so the full conversation order is preserved.
- No in-memory buffer of the whole transcript; we write as each final is produced.

## Why partial is excluded

- Partial text is live/uncertain and may change on the next ASR result.
- Writing it would create duplicates and wrong lines when the engine revises.
- Only **final** (committed) segments are written.

## Lifecycle

- **Open**: When WebSocket starts (`run()` → `writer.start()`).
- **Write**: On each `type: "final"` message → `writer.append_final(text, timestamp_ms)` (queued; does not block audio).
- **Close**: When client disconnects / WebSocket closes → `writer.close()` (flush, close file).

## Async-safe

- Writes are queued; a worker task drains the queue and writes to disk.
- The transcription loop only enqueues; it never blocks on file I/O.
- Failures (e.g. disk full) are logged; the WebSocket and transcription keep running.
