# Local Whisper Engine: Performance Tradeoffs

## Model loaded once at startup (singleton)

The faster-whisper model is loaded in **app/main.py** lifespan when `ASR_BACKEND=local` and stored in `app.state.whisper_model`. Every WebSocket connection gets the same model instance. This avoids loading the model per connection and keeps memory and startup cost predictable.

## Streaming behavior: partial vs final

- **PARTIAL** (`is_final=False`): Lower beam size (default 1), faster decode, text may change. Used for real-time feedback.
- **FINAL** (`is_final=True`): Higher beam size (default 5), stable decode, word_timestamps when enabled. Used after silence commit.

The pipeline runs **two passes** per chunk for local Whisper: one partial, one final. Cloudflare uses a single pass (no partial/final distinction).

## Config and tradeoffs

| Setting | Lower latency / faster | Higher quality / stable |
|--------|------------------------|--------------------------|
| **LOCAL_WHISPER_MODEL** | `base` | `small`, `medium`, `large-v3` |
| **LOCAL_WHISPER_DEVICE** | `cpu` | `cuda` (if GPU) |
| **LOCAL_WHISPER_COMPUTE_TYPE** | `int8` (CPU-friendly) | `float16` (needs CUDA) |
| **LOCAL_WHISPER_BEAM_SIZE_PARTIAL** | `1` | 2–3 (slower) |
| **LOCAL_WHISPER_BEAM_SIZE_FINAL** | 3–5 | 5 (default) |
| **LOCAL_WHISPER_WORD_TIMESTAMPS** | `false` (faster) | `true` |

- **int8** on CPU: good balance of speed and quality; **float16** on CUDA: faster and often better quality if you have a GPU.
- **beam_size=1** (partial): greedy decode, minimal latency; **beam_size=5** (final): better accuracy, more compute.
- **word_timestamps=true**: adds a bit of cost; set `false` if you don’t need word-level timing.

## Audio handling

- PCM 16-bit mono 16 kHz bytes are converted to **float32 in [-1.0, 1.0]** in `WebSocketManager` (`_pcm_bytes_to_float32`). Overlap audio is kept; chunks are not resampled.
- `LocalWhisperEngine` receives **np.ndarray (float32)** and runs in an executor so the event loop is not blocked.

## Where to plug in

1. **Startup**: `app/main.py` → `lifespan` → `_load_whisper_model()` → `app.state.whisper_model`.
2. **Per connection**: `get_asr_engine()` → `LocalWhisperEngine(model=app.state.whisper_model)`.
3. **Per chunk**: `WebSocketManager._asr_consumer()` → `_pcm_bytes_to_float32(chunk)` → `await engine.transcribe(audio, is_final=False)` then `await engine.transcribe(audio, is_final=True)`.

## Confidence

faster-whisper does not expose a confidence score. The engine uses **1.0** when text is non-empty and **0.0** when empty. For real confidence you’d need a separate model or logprobs from the decoder.
