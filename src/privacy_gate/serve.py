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
cannot reach its model must not say "safe to send". The reason is logged on the server, not sent to the client:
an exception message can carry an internal URL.

Every error is JSON with a stable `code` (`not_found`, `bad_content_length`, `body_too_large`, `bad_json`,
`bad_request`, `backend_unavailable`) next to a human-readable `error`.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .gate import Gate

MAX_BODY = 1 << 20  # 1 MiB. Prompts are text; anything larger is a mistake.
DEFAULT_PORT = 8231
REQUEST_TIMEOUT = 10.0  # seconds a client may stall on a socket before its thread is let go

log = logging.getLogger(__name__)


class SidecarServer(ThreadingHTTPServer):
    """The threading server with the one setting the handler reads: whether to log request lines."""

    def __init__(
        self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], verbose: bool = False
    ) -> None:
        self.verbose = verbose
        super().__init__(address, handler)


def make_handler(gate: Gate) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "privacy-gate"
        timeout = REQUEST_TIMEOUT  # applied to the socket in setup(), so a slow client cannot hold a thread

        def _error(self, status: int, code: str, message: str, **extra: object) -> None:
            self._send(status, {"error": message, "code": code, **extra})

        def _send(self, status: int, payload: dict[str, Any]) -> None:
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
                self._error(404, "not_found", "try POST /check")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/check":
                self._error(404, "not_found", "try POST /check")
                return
            raw_length = (self.headers.get("content-length") or "").strip()
            if not (raw_length.isascii() and raw_length.isdigit()):  # missing, negative, or not an integer
                self._error(400, "bad_content_length", "content-length must be a non-negative integer")
                return
            length = int(raw_length)
            if length == 0:
                self._error(400, "bad_request", 'send {"text": "..."} or {"texts": [...]}')
                return
            if length > MAX_BODY:
                self._error(413, "body_too_large", f"body must be 1..{MAX_BODY} bytes")
                self.close_connection = True  # the unread body must not be parsed as the next request
                return
            try:
                raw = self.rfile.read(length)
            except OSError:  # includes the socket timeout: nobody is left to answer
                self.close_connection = True
                return
            try:
                request = json.loads(raw)
            except ValueError:  # JSONDecodeError, and UnicodeDecodeError for bytes that are not UTF-8
                self._error(400, "bad_json", "body must be JSON")
                return
            if not isinstance(request, dict):
                self._error(
                    400, "bad_request", 'body must be a JSON object: {"text": "..."} or {"texts": [...]}'
                )
                return

            texts = request.get("texts")
            single = texts is None
            if single:
                text = request.get("text")
                if not isinstance(text, str):
                    self._error(400, "bad_request", 'send {"text": "..."} or {"texts": [...]}')
                    return
                texts = [text]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                self._error(400, "bad_request", "texts must be a list of strings")
                return

            try:
                decisions = gate.decide_many(texts)
            except Exception as error:  # the embedding endpoint is the fragile part
                # The message can name an internal URL, so it goes to the server log only. The texts are not logged.
                log.error("cannot score: %s: %s", type(error).__name__, str(error)[:500])
                # Fail closed. A gate that cannot reach its model must not answer "safe to send".
                self._error(
                    503,
                    "backend_unavailable",
                    "cannot score",
                    hold=True,
                    reason="gate unavailable, failing closed",
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
    server = SidecarServer((host, port), make_handler(gate), verbose=verbose)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: bound to {host}: this sidecar has no authentication. Anything that can reach it can "
            "submit text and read the verdict.",
            flush=True,
        )
    print(
        f"privacy-gate: {gate.embedder.name} backend, {gate.embedding_model}, threshold {gate.threshold}, "
        f"listening on http://{host}:{port} (POST /check, GET /health)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
