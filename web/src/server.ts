/**
 * Dev server: Swagger UI + OpenAPI JSON + static stream_test.html + API proxy to backend.
 *
 *   npm install
 *   npm run dev
 *
 *   http://localhost:8080/docs        — Swagger UI
 *   http://localhost:8080/openapi.json
 *   http://localhost:8080/stream_test.html
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import swaggerUi from "swagger-ui-express";
import { buildOpenApiDocument } from "./openapi.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(__dirname, "..");

const PORT = Number(process.env.PORT ?? 8080);
const BACKEND = (process.env.BACKEND_URL ?? "http://localhost:8000").replace(/\/$/, "");

const app = express();
app.use(express.json({ limit: "2mb" }));

const openApiDoc = buildOpenApiDocument(BACKEND);

app.get("/openapi.json", (_req, res) => {
  res.json(openApiDoc);
});

app.use(
  "/docs",
  swaggerUi.serve,
  swaggerUi.setup(undefined, {
    swaggerOptions: { url: "/openapi.json" },
    customSiteTitle: "Voice STT & Assistant API",
  }),
);

app.get("/", (_req, res) => {
  res.redirect("/docs");
});

/** Proxy API calls to Python backend (avoids CORS from static page). */
async function proxyToBackend(
  req: express.Request,
  res: express.Response,
  backendPath: string,
): Promise<void> {
  const url = `${BACKEND}${backendPath}`;
  try {
    const init: RequestInit = {
      method: req.method,
      headers: { "Content-Type": "application/json" },
    };
    if (req.method !== "GET" && req.method !== "HEAD") {
      init.body = JSON.stringify(req.body ?? {});
    }
    const upstream = await fetch(url, init);
    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/json");
    res.send(text);
  } catch (err) {
    res.status(502).json({
      detail: err instanceof Error ? err.message : "Proxy error",
    });
  }
}

app.post("/api/chat", (req, res) => void proxyToBackend(req, res, "/api/chat"));
app.post("/api/session/activate", (req, res) =>
  void proxyToBackend(req, res, "/api/session/activate"),
);
app.post("/api/refine-transcript", (req, res) =>
  void proxyToBackend(req, res, "/api/refine-transcript"),
);

app.use(express.static(WEB_ROOT));

app.listen(PORT, () => {
  console.log(`Web dev server http://localhost:${PORT}`);
  console.log(`  Swagger UI  http://localhost:${PORT}/docs`);
  console.log(`  OpenAPI     http://localhost:${PORT}/openapi.json`);
  console.log(`  Stream test http://localhost:${PORT}/stream_test.html`);
  console.log(`  API proxy   -> ${BACKEND}`);
});
