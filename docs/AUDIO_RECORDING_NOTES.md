# Optional Backend Audio Recording

Per-WebSocket-session recording of incoming PCM to WAV or MP3. Independent of ASR; disabled by default; can be enabled via env.

## Config (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUDIO_RECORDING` / `ENABLE_BACKEND_RECORDING` | `false` | Set `true` to save one file per session |
| `BACKEND_RECORD_FORMAT` | `wav` | `wav` or `mp3` (WAV preferred: no re-encode, safe lifecycle) |
| `BACKEND_RECORD_DIR` | `./recordings` | Directory for files |
| `BACKEND_RECORD_GAIN` | `2.0` | Gain 1.5–4.0 applied **only** to recorded file; STT stream unchanged |
| `BACKEND_RECORD_BITRATE` | `128k` | For MP3 only |

## Why buffering is required

PCM chunks must **not** be written to file one-by-one. We maintain a single in-memory buffer (bytearray) and append validated chunks. The file is written **only once** when the WebSocket closes (finalize). This avoids:

- Multiple open/close per chunk (corrupt headers, truncation)
- Inconsistent chunk timing and lost frames
- WAV/MP3 header written multiple times

## Why gain is applied

Incoming PCM from the frontend is often very low amplitude (e.g. browser mic). Gain (1.5x–4.0x) is applied **only** to the copy we store in the recording buffer. The STT path (receiver → VAD → chunker → ASR) always receives the **raw** bytes; STT is unchanged. Recorded files are then clearly audible after clamping to int16.

## Why WAV is preferred

- **Lifecycle**: One open → set header once → write all frames → close once. No streaming into an encoder.
- **MP3**: We first write a full WAV, then convert WAV → MP3 **after** recording ends (blocking, single-threaded in executor). We never stream PCM directly into the MP3 encoder.

## Behavior

- **Disabled**: No-op recorder; no file I/O, no buffering; STT still works.
- **Enabled**: One buffer per WebSocket connection. Chunks are validated (length divisible by 2, non-empty); invalid chunks are dropped and logged. Gain is applied to the buffer copy only. On disconnect, we write WAV (or WAV then convert to MP3), then close. One file per session.

## PCM contract

- Signed int16, little-endian, mono, 16 kHz.
- Chunk length must be divisible by 2; empty chunks are rejected. Dropped/malformed frames are logged.

## Optional: silence logging

RMS is computed per chunk. If many consecutive chunks have RMS below a threshold, a warning is logged to help debug frontend silence issues.

## Where it plugs in

1. **WebSocketManager**: `create_audio_recorder()` (no-op or real from env).
2. **On binary message**: After `receiver.feed(data)`, call `recorder.append(data)` (raw data still goes to STT; recorder gets validated + gained copy internally).
3. **On disconnect**: `await loop.run_in_executor(None, self._recorder.finalize)` so file I/O does not block the event loop.

## Session isolation

One WebSocket connection = one audio buffer. No global buffers. Buffer is flushed and discarded on disconnect after writing the file.
