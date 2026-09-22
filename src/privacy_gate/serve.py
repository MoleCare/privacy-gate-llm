# SPDX-License-Identifier: Apache-2.0
"""A loopback HTTP sidecar, so any language can use the gate.

    privacy-gate serve                      # the bundled head, Ollama on 127.0.0.1:11434, port 8231
    privacy-gate serve --backend local      # the in-process encoder

    curl -s localhost:8231/check -d '{"text":"her biopsy is booked for the 20th"}'
    {"hold": true, "score": 6.3272, "threshold": 0.1209, "margin": 6.2063}

Standard library only, and bound to 127.0.0.1. That is deliberate: this exists so a Java, TypeScript or Go
service can ask the question without reimplementing anything, not as a shared network service.

**It has no authentication.** Do not bind it to a routable address. Anything that can reach it can submit
text and read the verdict, and the text submitted here is by definition the text you were unsure about.
`--host` will let you change the bind address, and you should not.

When the embedding endpoint cannot be reached the sidecar answers 503 with `"hold": true`: a gate that
cannot reach its model must not say "safe to send".
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .gate import Gate

MAX_BODY = 1 << 20  # 1 MiB. Prompts are text; anything larger is a mistake.
DEFAULT_PORT = 8231


def make_handler(gate: Gate) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "privacy-gate"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "ok": True,
                        "threshold": gate.threshold,
                        "embedding_model": gate.embedding_model,
                        "backend": gate.embedder.name,
                        "head_sha256": gate.head_sha256,
                    },
                )
            else:
                self._send(404, {"error": "try POST /check"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/check":
                self._send(404, {"error": "try POST /check"})
                return
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                self._send(400, {"error": "bad content-length"})
                return
            if length <= 0 or length > MAX_BODY:
                self._send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
                return
            try:
                request = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self._send(400, {"error": "body must be JSON"})
                return

            texts = request.get("texts")
            single = texts is None
            if single:
                text = request.get("text")
                if not isinstance(text, str):
                    self._send(400, {"error": 'send {"text": "..."} or {"texts": [...]}'})
                    return
                texts = [text]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                self._send(400, {"error": "texts must be a list of strings"})
                return

            try:
                decisions = gate.decide_many(texts)
            except Exception as error:  # the embedding endpoint is the fragile part
                # Fail closed. A gate that cannot reach its model must not answer "safe to send".
                self._send(
                    503,
                    {
                        "error": f"cannot score: {type(error).__name__}",
                        "detail": str(error)[:200],
                        "hold": True,
                        "reason": "gate unavailable, failing closed",
                    },
                )
                return

            results = [
                {
                    "hold": d.hold,
                    "score": round(d.score, 4),
                    "threshold": round(d.threshold, 4),
                    "margin": round(d.margin, 4),
                }
                for d in decisions
            ]
            self._send(200, results[0] if single else {"results": results})

        def log_message(self, fmt: str, *args: object) -> None:
            # The default handler logs the request line. Nothing sensitive is in a URL here, but staying
            # quiet by default keeps prompts out of any terminal scrollback that gets pasted somewhere.
            if getattr(self.server, "verbose", False):
                super().log_message(fmt, *args)

    return Handler


def serve(gate: Gate, host: str = "127.0.0.1", port: int = DEFAULT_PORT, verbose: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(gate))
    server.verbose = verbose  # type: ignore[attr-defined]
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: bound to {host}: this sidecar has no authentication. Anything that can reach it can "
              "submit text and read the verdict.", flush=True)
    print(f"privacy-gate: {gate.embedder.name} backend, {gate.embedding_model}, threshold {gate.threshold}, "
          f"listening on http://{host}:{port} (POST /check, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
