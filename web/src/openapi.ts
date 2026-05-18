/**
 * OpenAPI 3.0 spec for the Python backend (documentation only — served from this dev server).
 */

export const API_TITLE = "Voice STT & Assistant API";
export const API_VERSION = "1.0.0";

export type OpenApiDocument = Record<string, unknown>;

export function buildOpenApiDocument(apiBaseUrl: string): OpenApiDocument {
  const wsBase = apiBaseUrl.replace(/^http/, "ws");

  return {
    openapi: "3.0.3",
    info: {
      title: API_TITLE,
      version: API_VERSION,
      description: [
        "Real-time speech-to-text over WebSocket plus HTTP APIs for session activation, chat, and transcript refinement.",
        "",
        "## Typical flow",
        "1. Stream audio on `WS /ws/transcribe` or `WS /ws/transcribe-with-assistant` (PCM s16le mono 16 kHz).",
        "2. `POST /api/session/activate` with `session_id` or custom `transcript`.",
        "3. `POST /api/chat` with the same `session_id` and `message`.",
        "",
        "## WebSocket (not callable from Swagger)",
        `- \`${wsBase}/ws/transcribe\` — ASR only`,
        `- \`${wsBase}/ws/transcribe-with-assistant\` — ASR + optional \`assistant_reply\` (TTS)`,
        "",
        "Server → client JSON: `partial` | `final` (`text`, `confidence`, `timestamp`); assistant route adds `assistant_processing`, `assistant_reply`.",
        "",
        "## Persona",
        "Assistant behavior: `assistant/identity.md`, `soul.md`, `constraints.md` in the backend repo.",
      ].join("\n"),
    },
    servers: [{ url: apiBaseUrl, description: "Python FastAPI backend" }],
    tags: [
      { name: "health", description: "Liveness" },
      { name: "session", description: "Activate session / load transcript" },
      { name: "chat", description: "Conversational assistant (CHAT MODE)" },
      { name: "refine", description: "One-time refinement (REFINE MODE)" },
    ],
    paths: {
      "/health": {
        get: {
          tags: ["health"],
          summary: "Health check",
          responses: {
            "200": {
              description: "OK",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/HealthResponse" },
                  example: { status: "ok" },
                },
              },
            },
          },
        },
      },
      "/api/session/activate": {
        post: {
          tags: ["session"],
          summary: "Activate session from file or custom transcript",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/ActivateSessionRequest" },
                examples: {
                  fromFile: { value: { session_id: "c6b6cf6ff6be" } },
                  custom: {
                    value: {
                      session_id: "my-demo",
                      transcript: "[Speaker A] Halo\n[Speaker B] Hai",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Session ready for chat",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/ActivateSessionResponse" },
                },
              },
            },
            "400": { $ref: "#/components/responses/BadRequest" },
            "404": { $ref: "#/components/responses/NotFound" },
            "502": { $ref: "#/components/responses/BadGateway" },
          },
        },
      },
      "/api/chat": {
        post: {
          tags: ["chat"],
          summary: "Chat with session (heard dialog as knowledge)",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/ChatRequest" },
                example: {
                  session_id: "c6b6cf6ff6be",
                  message: "Salam, apa yang dibahas tadi?",
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Assistant reply",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/ChatResponse" },
                },
              },
            },
            "400": { $ref: "#/components/responses/BadRequest" },
            "404": { $ref: "#/components/responses/NotFound" },
            "502": { $ref: "#/components/responses/BadGateway" },
          },
        },
      },
      "/api/refine-transcript": {
        post: {
          tags: ["refine"],
          summary: "One-time transcript refinement",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/RefineRequest" },
              },
            },
          },
          responses: {
            "200": {
              description: "Per-segment refinement",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/RefineResponse" },
                },
              },
            },
            "400": { $ref: "#/components/responses/BadRequest" },
            "502": { $ref: "#/components/responses/BadGateway" },
          },
        },
      },
    },
    components: {
      responses: {
        BadRequest: {
          description: "Invalid request",
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/ErrorResponse" },
            },
          },
        },
        NotFound: {
          description: "Not found",
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/ErrorResponse" },
            },
          },
        },
        BadGateway: {
          description: "Upstream LLM failure",
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/ErrorResponse" },
            },
          },
        },
      },
      schemas: {
        ErrorResponse: {
          type: "object",
          required: ["detail"],
          properties: { detail: { type: "string" } },
        },
        HealthResponse: {
          type: "object",
          properties: { status: { type: "string", example: "ok" } },
        },
        ActivateSessionRequest: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description: "Load transcripts/{session_id}.txt when used alone",
            },
            transcript: { type: "string", description: "Custom dialog text" },
          },
        },
        ActivateSessionResponse: {
          type: "object",
          required: ["session_id", "transcript_length", "summary_ready"],
          properties: {
            session_id: { type: "string" },
            transcript_length: { type: "integer" },
            summary_ready: { type: "boolean" },
          },
        },
        ChatRequest: {
          type: "object",
          properties: {
            message: { type: "string" },
            session_id: { type: "string" },
            action: { type: "string", enum: ["reset"] },
            transcript: { type: "string" },
            is_first_message: { type: "boolean" },
            segments: { type: "array", items: { type: "object" } },
          },
        },
        ChatResponse: {
          type: "object",
          properties: {
            session_id: { type: "string", nullable: true },
            text: { type: "string" },
            audio: { type: "string", nullable: true, description: "Base64 TTS" },
            trigger_audio: { type: "boolean" },
            reply: { type: "string", nullable: true },
            mode: { type: "string", nullable: true },
            message: { type: "string", nullable: true },
            refinement: { type: "object", nullable: true },
          },
        },
        RefineSegmentInput: {
          type: "object",
          required: ["segment_id", "text"],
          properties: {
            segment_id: { type: "string" },
            text: { type: "string" },
            start_sec: { type: "number", default: 0 },
            end_sec: { type: "number", default: 0 },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
        },
        RefineRequest: {
          type: "object",
          required: ["user_question", "segments"],
          properties: {
            user_question: { type: "string" },
            segments: {
              type: "array",
              items: { $ref: "#/components/schemas/RefineSegmentInput" },
            },
            audio_reference_seconds: { type: "number", nullable: true },
            audio_description: { type: "string", nullable: true },
            domain_hint: { type: "string", nullable: true },
            session_id: { type: "string", nullable: true },
          },
        },
        RefineSegmentOutput: {
          type: "object",
          properties: {
            segment_id: { type: "string" },
            original_text: { type: "string" },
            refined_text: { type: "string" },
            confidence_before: { type: "number" },
            confidence_after: { type: "number" },
            justification: { type: "string" },
          },
        },
        RefineResponse: {
          type: "object",
          required: ["segments"],
          properties: {
            segments: {
              type: "array",
              items: { $ref: "#/components/schemas/RefineSegmentOutput" },
            },
            session_id: { type: "string", nullable: true },
          },
        },
        AsrOutboundMessage: {
          type: "object",
          description: "WebSocket server → client (ASR)",
          properties: {
            type: { type: "string", enum: ["partial", "final"] },
            text: { type: "string" },
            confidence: { type: "number" },
            timestamp: { type: "integer", description: "Unix ms" },
          },
        },
        AssistantProcessingMessage: {
          type: "object",
          properties: {
            type: { type: "string", enum: ["assistant_processing"] },
            message: { type: "string" },
          },
        },
        AssistantReplyMessage: {
          type: "object",
          properties: {
            type: { type: "string", enum: ["assistant_reply"] },
            text: { type: "string" },
            trigger_audio: { type: "boolean" },
            audio: { type: "string", nullable: true },
            audio_mime: { type: "string", nullable: true },
          },
        },
      },
    },
  };
}
