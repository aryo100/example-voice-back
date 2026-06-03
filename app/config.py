"""Application configuration. Loads from env vars."""
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


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

    # ASR backend: "local" | "cloudflare" | "coqui"
    ASR_BACKEND: Literal["local", "cloudflare", "coqui"] = "local"

    # Cloudflare Workers AI: ASR (when ASR_BACKEND=cloudflare) and refine-transcript (LLM)
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""

    # Coqui STT (when ASR_BACKEND=coqui) — id | en | auto
    # auto = faster-whisper language=None (Indonesian + English in one session)
    COQUI_STT_LANG: Literal["id", "en", "auto"] = "auto"
    COQUI_MODEL_DIR: str = "./coqui_stt_models"
    COQUI_AUTO_WHISPER_MODEL: str = "small"  # tiny | base | small | medium | large-v3

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

    # Session transcript storage: file (./transcripts) or postgresql (TRANSCRIPT_STORAGE / USE_DATABASE).
    USE_DATABASE: bool = False
    TRANSCRIPT_STORAGE: Literal["file", "postgresql"] = "file"
    TRANSCRIPT_SAVE_ENABLED: bool = True
    TRANSCRIPT_DIR: str = "./transcripts"
    TRANSCRIPT_ADD_TIMESTAMPS: bool = False

    # PostgreSQL (when TRANSCRIPT_STORAGE=postgresql)
    # asyncpg driver: postgresql+asyncpg://user:pass@host:5432/dbname
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/voice_back"

    # Cloud object storage for session audio (URL stored in PostgreSQL only).
    # none = local file (./recordings) when not using database; supabase | firebase for cloud.
    OBJECT_STORAGE_BACKEND: Literal["none", "supabase", "firebase"] = "none"
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_STORAGE_BUCKET: str = "recordings"
    SUPABASE_STORAGE_PREFIX: str = "sessions"
    FIREBASE_CREDENTIALS_PATH: str = ""
    FIREBASE_CREDENTIALS_JSON: str = ""  # inline service account JSON (Render/HF Spaces)
    FIREBASE_STORAGE_BUCKET: str = ""
    FIREBASE_STORAGE_PREFIX: str = "sessions"
    FIREBASE_STORAGE_MAKE_PUBLIC: bool = True

    # Speaker-aware transcription (diarization only; no audio separation, single channel).
    DIARIZATION_ENABLED: bool = True
    DIARIZATION_SPEAKER_GAP_SEC: float = 0.5  # gap (sec) to alternate speaker
    DIARIZATION_MAX_SPEAKERS: int = 2

    # Context-aware re-transcription: LLM via OpenRouter (prioritas 1), Cloudflare (2), Hugging Face (3).
    REFINE_CHAT_ENABLED: bool = True
    # Prioritas provider: comma-separated, e.g. "openrouter,cloudflare,huggingface"
    LLM_PROVIDER_PRIORITY: str = "openrouter,cloudflare,huggingface"
    # OpenRouter (prioritas 1)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-3.5-turbo"  # atau meta-llama/llama-3.1-8b-instruct, dll.
    # Cloudflare Workers AI (prioritas 2)
    REFINE_CF_MODEL: str = "@cf/meta/llama-3.1-8b-instruct"  # Workers AI text generation
    # Hugging Face Inference (prioritas 3)
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # atau model chat lain
    REFINE_CHAT_MAX_TOKENS: int = 2048
    REFINE_AUDIO_REFERENCE_SECONDS: float = 30.0  # "last N seconds" context hint for prompt

    # Chat context aggregation: transcript as knowledge, not full input.
    CHAT_GLOBAL_SUMMARY_MAX_WORDS: int = 80  # ~300 tokens max for global summary
    CHAT_GLOBAL_SUMMARY_MAX_CHARS: int = 4000  # max chars untuk parsed global_summary (jangan potong terlalu pendek saat full context)
    CHAT_SNIPPET_MAX_CHARS: int = 1600  # raw transcript window only if relevant (~10–20 sec)
    CHAT_HISTORY_MAX_MESSAGES: int = 20  # recent messages to include (keeps token usage low)
    # Bila true: konteks LLM = full transcript (raw, sampai CHAT_ONLY_RELEVANT_SNIPPET_MAX_CHARS). Bila false: full summary + optional snippet.
    CHAT_USE_ONLY_RELEVANT_SNIPPET: bool = False
    # Bila true: prompt LLM tanpa teks transkrip (tidak ringkasan, tidak cuplikan, tidak baris terpotong); hanya instruksi + CHAT_LLM_EXTRA_SYSTEM_PROMPT.
    # Transcript penuh tetap di session/file. CHAT_USE_ONLY_RELEVANT_SNIPPET hanya mengatur jalur bila flag ini false.
    CHAT_LLM_COMPACT_KNOWLEDGE: bool = False
    # Isi prompt saat compact knowledge false: both | summary_only | excerpt_only.
    CHAT_LLM_COMPACT_CONTEXT_PARTS: str = "both"
    # Batasi cuplikan fallback ke LLM (bukan file penuh) bila ringkasan global tidak terbaca / summary_only.
    CHAT_LLM_KNOWLEDGE_FALLBACK_SNIPPET_CHARS: int = 12000
    # Saat true: max karakter transcript yang dikirim (50000 = praktis full transcript).
    CHAT_ONLY_RELEVANT_SNIPPET_MAX_CHARS: int = 50000
    # Iterative compression: when transcript exceeds this (chars), use chunk loop instead of single summary.
    CHAT_COMPRESSION_THRESHOLD_CHARS: int = 12000
    CHAT_CHUNK_CHARS: int = 6000  # per-batch chunk size (~1500 tokens)
    CHAT_ROLLING_MAX_CHARS: int = 4000  # re-summarize rolling summary when it exceeds this
    # Transkrip sebagai beberapa pesan role "system" berurutan (chunk per N baris); mengurangi satu blob besar / 413.
    CHAT_KNOWLEDGE_MULTI_SYSTEM: bool = False
    CHAT_KNOWLEDGE_CHUNK_LINES: int = 50
    CHAT_KNOWLEDGE_MAX_CHUNKS: int = 40
    CHAT_KNOWLEDGE_MAX_CHARS_PER_CHUNK: int = 12000

    # Folder persona asisten (identity.md + soul.md). Relatif ke root proyek atau path absolut.
    ASSISTANT_PERSONA_DIR: str = "assistant"
    # Debounce (detik): jangan kirim assistant_reply via WS lebih sering dari ini setelah pemanggilan nama.
    ASSISTANT_WS_DEBOUNCE_SEC: float = 15.0

    # TTS (untuk balasan asisten): edge = Edge TTS (local), cloudflare = Cloudflare (nanti), none = matikan.
    TTS_BACKEND: str = "edge"
    TTS_EDGE_VOICE: str = "id-ID-ArdiNeural"  # voice Edge TTS, e.g. id-ID-ArdiNeural, en-US-GuyNeural

    # Logging: level (DEBUG, INFO, WARNING, ERROR); file path = simpan log ke file (kosong = hanya konsol).
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = ""  # contoh: "logs/app.log" — kosong = tidak simpan ke file

    @field_validator("CHAT_USE_ONLY_RELEVANT_SNIPPET", mode="before")
    @classmethod
    def _coerce_chat_use_only_relevant(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

    @field_validator("CHAT_LLM_COMPACT_KNOWLEDGE", mode="before")
    @classmethod
    def _coerce_chat_llm_compact_knowledge(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

    @field_validator("CHAT_KNOWLEDGE_MULTI_SYSTEM", mode="before")
    @classmethod
    def _coerce_chat_knowledge_multi_system(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

    @field_validator("TRANSCRIPT_STORAGE", mode="before")
    @classmethod
    def _normalize_transcript_storage(cls, v):
        if v is None:
            return "file"
        s = str(v).strip().lower()
        if s in ("postgres", "postgresql", "pgsql", "db", "database"):
            return "postgresql"
        return "file"

    @field_validator("USE_DATABASE", mode="before")
    @classmethod
    def _coerce_use_database(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

    @model_validator(mode="after")
    def _apply_use_database(self) -> Self:
        if self.USE_DATABASE:
            self.TRANSCRIPT_STORAGE = "postgresql"
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()


def chat_use_only_relevant_snippet() -> bool:
    """
    Baca CHAT_USE_ONLY_RELEVANT_SNIPPET sebagai bool. Normalisasi eksplisit agar string dari env
    (e.g. 'false') tidak dianggap truthy. Pakai ini di main.py dan websocket_manager.
    """
    s = get_settings()
    v = getattr(s, "CHAT_USE_ONLY_RELEVANT_SNIPPET", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def chat_llm_compact_knowledge() -> bool:
    """Baca CHAT_LLM_COMPACT_KNOWLEDGE (prompt LLM tanpa isi transkrip/cuplikan/ringkasan teks)."""
    s = get_settings()
    v = getattr(s, "CHAT_LLM_COMPACT_KNOWLEDGE", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def chat_llm_compact_context_parts() -> str:
    """both | summary_only | excerpt_only — bagian konteks yang dikirim ke LLM saat mode compact."""
    s = get_settings()
    raw = (getattr(s, "CHAT_LLM_COMPACT_CONTEXT_PARTS", None) or "both").strip().lower().replace("-", "_")
    if raw in ("excerpt_only", "excerpt"):
        return "excerpt_only"
    if raw in ("summary_only", "summary"):
        return "summary_only"
    return "both"


def chat_knowledge_multi_system() -> bool:
    """Transkrip sebagai beberapa pesan system berurutan (chunk baris)."""
    s = get_settings()
    v = getattr(s, "CHAT_KNOWLEDGE_MULTI_SYSTEM", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def transcript_storage_postgresql() -> bool:
    """True when transcripts are stored in PostgreSQL."""
    s = get_settings()
    raw = getattr(s, "TRANSCRIPT_STORAGE", "file")
    if isinstance(raw, str):
        return raw.strip().lower() in ("postgresql", "postgres", "pgsql", "db", "database")
    return False


def object_storage_enabled() -> bool:
    """True when cloud object storage is configured for recordings."""
    backend = (getattr(get_settings(), "OBJECT_STORAGE_BACKEND", "none") or "none").strip().lower()
    return backend in ("supabase", "firebase")


def assistant_name_and_aliases() -> tuple[str, list[str]]:
    """Nama & alias dari assistant/identity.md (frontmatter) — untuk deteksi TTS."""
    from app.services.assistant_persona import assistant_name_and_aliases as _from_persona

    return _from_persona()
