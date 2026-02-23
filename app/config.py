"""Application configuration. Loads from env vars."""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """App settings. Override via environment variables."""

    # Audio: PCM 16-bit mono, 16kHz
    SAMPLE_RATE: int = 16000
    SAMPLE_WIDTH: int = 2  # 16-bit
    CHANNELS: int = 1

    # Frame: 20ms @ 16kHz = 320 samples = 640 bytes
    FRAME_MS: int = 20
    FRAME_BYTES: int = 640  # 320 * 2

    # Chunking (legacy: used only when STT_USE_ROLLING_BUFFER=false)
    CHUNK_DURATION_MS: int = 1500
    OVERLAP_MS: int = 300
    SILENCE_COMMIT_MS: int = 600

    # Real-time STT: rolling buffer (no silence gating). When true, transcription is time-based.
    STT_USE_ROLLING_BUFFER: bool = True
    STT_WINDOW_SECONDS: float = 5.0  # buffer window (2–5s typical)
    STT_STEP_SECONDS: float = 1.0  # transcribe every N seconds
    STT_MIN_CHUNK_SECONDS: float = 0.5  # do not transcribe chunks < 500ms (prevent hallucination)
    STT_COMMIT_AGE_SECONDS: float = 2.0  # commit horizon: segments ending before (current_audio_time - this) → FINAL

    # ASR backend: "local" | "cloudflare"
    ASR_BACKEND: Literal["local", "cloudflare"] = "local"

    # Cloudflare Workers AI: ASR (when ASR_BACKEND=cloudflare) and refine-transcript (LLM)
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""

    # Local Whisper (when ASR_BACKEND=local) — model loaded once at startup
    LOCAL_WHISPER_MODEL: str = "base"  # base | small | medium | large-v3
    LOCAL_WHISPER_DEVICE: Literal["cpu", "cuda"] = "cpu"
    LOCAL_WHISPER_COMPUTE_TYPE: Literal["int8", "float16"] = "int8"
    # Partial (faster): lower beam. Final (stable): higher beam.
    LOCAL_WHISPER_BEAM_SIZE_PARTIAL: int = 1
    LOCAL_WHISPER_BEAM_SIZE_FINAL: int = 5
    LOCAL_WHISPER_WORD_TIMESTAMPS: bool = True  # only on final

    # Optional backend recording (per WebSocket session → one file). Disabled by default.
    ENABLE_AUDIO_RECORDING: bool = False  # alias for ENABLE_BACKEND_RECORDING
    ENABLE_BACKEND_RECORDING: bool = False
    BACKEND_RECORD_FORMAT: Literal["wav", "mp3"] = "wav"  # WAV preferred: no re-encode, safe lifecycle
    BACKEND_RECORD_DIR: str = "./recordings"
    BACKEND_RECORD_BITRATE: str = "128k"  # for MP3 only
    # Gain applied ONLY to recorded file (1.5–4.0). STT stream is unchanged.
    BACKEND_RECORD_GAIN: float = 2.0
    # Legacy alias
    AUDIO_RECORD_DIR: str = "./recordings"
    AUDIO_RECORD_BITRATE: str = "128k"

    # Session transcript storage: one .txt per WebSocket, append-only (final segments only).
    TRANSCRIPT_SAVE_ENABLED: bool = True
    TRANSCRIPT_DIR: str = "./transcripts"
    TRANSCRIPT_ADD_TIMESTAMPS: bool = False  # prefix each line with [MM:SS.ss]

    # Speaker-aware transcription (diarization only; no audio separation, single channel).
    DIARIZATION_ENABLED: bool = True
    DIARIZATION_SPEAKER_GAP_SEC: float = 0.5  # gap (sec) to alternate speaker
    DIARIZATION_MAX_SPEAKERS: int = 2

    # Context-aware re-transcription: Cloudflare Workers AI (free tier), no OpenAI/GPT.
    REFINE_CHAT_ENABLED: bool = True
    REFINE_CF_MODEL: str = "@cf/meta/llama-3.1-8b-instruct"  # Workers AI text generation
    REFINE_CHAT_MAX_TOKENS: int = 2048
    REFINE_AUDIO_REFERENCE_SECONDS: float = 30.0  # "last N seconds" context hint for prompt

    # Chat context aggregation: transcript as knowledge, not full input.
    CHAT_GLOBAL_SUMMARY_MAX_WORDS: int = 80  # ~300 tokens max for global summary
    CHAT_SNIPPET_MAX_CHARS: int = 600  # raw transcript window only if relevant (~10–20 sec)
    CHAT_HISTORY_MAX_MESSAGES: int = 20  # recent messages to include (keeps token usage low)
    # Iterative compression: when transcript exceeds this (chars), use chunk loop instead of single summary.
    CHAT_COMPRESSION_THRESHOLD_CHARS: int = 12000
    CHAT_CHUNK_CHARS: int = 6000  # per-batch chunk size (~1500 tokens)
    CHAT_ROLLING_MAX_CHARS: int = 4000  # re-summarize rolling summary when it exceeds this

    # Asisten (nama untuk trigger audio/TTS): bila user memanggil nama ini atau nama ada di transcript, respons bisa diputar sebagai audio.
    ASSISTANT_NAME: str = "Salam"
    # Alias nama (ASR kadang menulis variasi); dipakai untuk deteksi di pesan dan transcript. Comma-separated.
    ASSISTANT_NAME_ALIASES: str = ""
    # Debounce (detik): jangan kirim assistant_reply via WS lebih sering dari ini setelah pemanggilan nama.
    ASSISTANT_WS_DEBOUNCE_SEC: float = 15.0

    # TTS (untuk balasan asisten): edge = Edge TTS (local), cloudflare = Cloudflare (nanti), none = matikan.
    TTS_BACKEND: str = "edge"
    TTS_EDGE_VOICE: str = "id-ID-ArdiNeural"  # voice Edge TTS, e.g. id-ID-ArdiNeural, en-US-GuyNeural

    # Logging: level (DEBUG, INFO, WARNING, ERROR); file path = simpan log ke file (kosong = hanya konsol).
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = ""  # contoh: "logs/app.log" — kosong = tidak simpan ke file

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
