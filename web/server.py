#!/usr/bin/env python3
"""
Dev server: serve stream_test.html dan proxy /api/chat ke backend.
Menghindari CORS saat buka dari file:// (origin null).

Cara pakai:
  1. Jalankan backend di http://localhost:8000 (WebSocket + /api/chat).
  2. Di folder ini: python server.py
  3. Buka http://localhost:8080/stream_test.html

Request ke http://localhost:8080/api/chat akan di-forward ke http://localhost:8000/api/chat.
"""
import http.server
import urllib.request
import json
import sys

BACKEND = "http://localhost:8000"
PORT = 8080


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            req = urllib.request.Request(
                BACKEND + "/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    resp_body = r.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp_body)
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/") or "stream_test.html"
        if path == "stream_test.html" or path == "":
            path = "stream_test.html"
        try:
            with open(path, "rb") as f:
                self.send_response(200)
                if path.endswith(".html"):
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"Serving at http://localhost:{PORT}/stream_test.html")
    print(f"API /api/chat proxied to {BACKEND}")
    http.server.HTTPServer(("", PORT), ProxyHandler).serve_forever()
