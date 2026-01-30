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

    # Context-aware re-transcription: Cloudflare Workers AI (free tier), no OpenAI/GPT.
    REFINE_CHAT_ENABLED: bool = True
    REFINE_CF_MODEL: str = "@cf/meta/llama-3.1-8b-instruct"  # Workers AI text generation
    REFINE_CHAT_MAX_TOKENS: int = 2048
    REFINE_AUDIO_REFERENCE_SECONDS: float = 30.0  # "last N seconds" context hint for prompt

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
