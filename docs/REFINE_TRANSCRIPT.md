# Context-Aware Re-Transcription and Chat

The AI operates in **two modes**:

- **CHAT MODE (default)**: Conversational assistant. May reference the transcript but **does not** modify or refine it. Use **POST /api/chat** for normal two-way conversation.
- **REFINE MODE (one-time)**: Re-interprets the transcript using user context; structured output is stored. Triggered only when the transcript has **never** been refined before and this is the **first** interaction for that session (via `/api/chat` with `session_id` and segments). After refinement, all further interactions are CHAT MODE.

Rules: Do not refine during normal chat; do not repeatedly refine the same transcript; chat responses are conversational; refinement output is structured and stored once per session.

## Chat endpoint

**POST** `/api/chat`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User message |
| `session_id` | string \| null | When set, first interaction may trigger one-time refine |
| `segments` | array \| null | Optional raw transcript segments for first-time refine (else loaded from file) |

Response: `{ "mode": "chat" \| "refine", "message": "...", "refinement": RefineResponse \| null }`. When `mode=refine`, refinement is stored and subsequent calls are chat-only.

## Refine endpoint (explicit, one-time)

**POST** `/api/refine-transcript`

Refuses with 400 if the session has already been refined (use `/api/chat` for conversation).

## Inputs

| Field | Type | Description |
|-------|------|-------------|
| `user_question` | string | User question providing semantic context |
| `segments` | array | Raw transcript segments (see below) |
| `audio_reference_seconds` | float \| null | Last N seconds of audio (optional hint) |
| `audio_description` | string \| null | Short description of audio when no file (e.g. "gadget unboxing") |
| `domain_hint` | string \| null | e.g. "gadget review", "tutorial", "Indonesian naturalness" |
| `session_id` | string \| null | If set, refinement is saved as `{session_id}_refinements.json` in transcript dir |

Each **segment**:

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | string | Unique id (e.g. index or uuid) |
| `text` | string | Original STT text |
| `start_sec` | float | Start time (seconds) |
| `end_sec` | float | End time (seconds) |
| `confidence` | float | STT confidence 0–1 |

## Output

JSON with one object per segment:

```json
{
  "segments": [
    {
      "segment_id": "...",
      "original_text": "...",
      "refined_text": "...",
      "confidence_before": 0.42,
      "confidence_after": 0.83,
      "justification": "Based on gadget review context..."
    }
  ],
  "session_id": "abc123"
}
```

- **original_text**: unchanged from input (preserved).
- **refined_text**: corrected text or same as original if no change.
- **confidence_after**: increased only when correction is justified.
- **justification**: short explanation; "No change needed" when unchanged.

## Rules (enforced in prompt)

- Do **not** overwrite original transcript.
- Treat refined transcript as a revision layer.
- If no correction needed, return original.
- Keep language consistent with audio.
- **Never** hallucinate content not present in audio.

## Use cases

- Fix brand names, channel names, technical terms.
- Improve Indonesian (or other language) naturalness.

## Configuration

Refine uses **Cloudflare Workers AI** (free tier). No OpenAI or paid LLMs.

| Env | Default | Description |
|-----|---------|-------------|
| `REFINE_CHAT_ENABLED` | `true` | Enable refine endpoint |
| `REFINE_CF_MODEL` | `@cf/meta/llama-3.1-8b-instruct` | Workers AI text generation model |
| `REFINE_CHAT_MAX_TOKENS` | `2048` | Max response tokens |
| `REFINE_AUDIO_REFERENCE_SECONDS` | `30.0` | Default "last N seconds" hint when not in request |
| `CLOUDFLARE_ACCOUNT_ID` | (required) | Cloudflare account ID (same as ASR) |
| `CLOUDFLARE_API_TOKEN` | (required) | Cloudflare API token (same as ASR) |

## Revision layer storage

When `session_id` is provided, the response is saved to:

`{TRANSCRIPT_DIR}/{session_id}_refinements.json`

Original transcript remains in `{session_id}.txt`. Refined text is stored as a separate revision file (array of segment objects).

## Validation

- Original transcript is preserved in response and on disk.
- Refined transcript should match user intent and domain.
- Confidence increases only when the model provides a strong justification.
