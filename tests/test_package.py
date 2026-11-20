# SPDX-License-Identifier: Apache-2.0
"""The package as a user meets it: the bundled head, the three backends, the command line and the sidecar.
No network and no model: every backend is faked at the boundary it talks through."""

from __future__ import annotations

import ast
import io
import json
import logging
import math
import socket
import subprocess
import sys
import threading
import types
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

from privacy_gate import backends, cli, ollama
from privacy_gate.gate import Gate, bundled_head_path
from privacy_gate.serve import MAX_BODY, SidecarServer, make_handler

DIMS = 1024


def unit(seed: float) -> list[float]:
    """A unit vector that scores differently for different seeds."""
    raw = [math.sin(seed * (i + 1)) for i in range(DIMS)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


class FakeEmbedder:
    name = "fake"

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise ollama.OllamaError("no model here")
        return [unit(1.0 + len(t)) for t in texts]


class TestBundledHead(unittest.TestCase):
    def test_load_without_a_path_uses_the_shipped_head(self) -> None:
        gate = Gate.load(embedder=FakeEmbedder())
        self.assertEqual(gate.head_path, bundled_head_path())
        self.assertEqual(len(gate.model.weights), DIMS)
        self.assertEqual(len(gate.head_sha256), 64)
        self.assertGreater(gate.threshold, 0.0)
        self.assertEqual(gate.meta["embedding_model"], "bge-m3")
        self.assertNotIn("weights", gate.meta)

    def test_old_call_shape_still_works(self) -> None:
        gate = Gate.load("model/head-v1.json", url="http://example.test:11434")
        self.assertIsInstance(gate.embedder, backends.OllamaEmbedder)
        self.assertEqual(gate.embedder.url, "http://example.test:11434")
        self.assertEqual(gate.embedder.model, "bge-m3")

    def test_decides_through_the_embedder(self) -> None:
        fake = FakeEmbedder()
        gate = Gate.load(embedder=fake)
        decisions = gate.decide_many(["one", "two texts"])
        self.assertEqual(fake.calls, [["one", "two texts"]])
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0].margin, decisions[0].score - gate.threshold)

    def test_refuses_an_unnormalised_vector(self) -> None:
        class Raw:
            name = "raw"

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.5] * DIMS for _ in texts]

        with self.assertRaises(ValueError):
            Gate.load(embedder=Raw()).decide("x")


class TestOllamaBackend(unittest.TestCase):
    def test_posts_to_api_embed(self) -> None:
        with mock.patch.object(ollama, "_post", return_value={"embeddings": [unit(1), unit(2)]}) as post:
            vectors = backends.OllamaEmbedder("http://box:11434/", "bge-m3").embed(["a", "b"])
        self.assertEqual(len(vectors), 2)
        url, path, payload, _timeout = post.call_args[0]
        self.assertEqual((url, path), ("http://box:11434", "/api/embed"))
        self.assertEqual(payload["input"], ["a", "b"])
        self.assertEqual(payload["model"], "bge-m3")

    def test_options_reach_ollama_but_never_num_ctx(self) -> None:
        with mock.patch.object(ollama, "_post", return_value={"embeddings": [unit(1)]}) as post:
            backends.OllamaEmbedder(options={"num_gpu": 0, "num_ctx": 4096}).embed(["a"])
        self.assertEqual(post.call_args[0][2]["options"], {"num_gpu": 0})
        with mock.patch.object(ollama, "_post", return_value={"embeddings": [unit(1)]}) as post:
            backends.OllamaEmbedder().embed(["a"])
        self.assertNotIn("options", post.call_args[0][2])

    def test_wrong_count_is_an_error(self) -> None:
        with (
            mock.patch.object(ollama, "_post", return_value={"embeddings": [unit(1)]}),
            self.assertRaises(ollama.OllamaError),
        ):
            backends.OllamaEmbedder().embed(["a", "b"])


class TestOpenAIBackend(unittest.TestCase):
    def _serve(self, payload: dict, status: int = 200):
        captured = {}

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            if status != 200:
                raise urllib.error.HTTPError(request.full_url, status, "nope", {}, io.BytesIO(b"denied"))
            return Response(json.dumps(payload).encode())

        return captured, fake_urlopen

    def test_posts_v1_embeddings_in_order_and_normalises(self) -> None:
        half = [0.5 * v for v in unit(3)]  # not unit length: must be normalised
        captured, fake = self._serve(
            {"data": [{"index": 1, "embedding": unit(2)}, {"index": 0, "embedding": half}]}
        )
        with (
            mock.patch.object(urllib.request, "urlopen", fake),
            mock.patch.dict("os.environ", {"PRIVACY_GATE_API_KEY": "k"}),
        ):
            vectors = backends.OpenAIEmbedder("http://gw:11500", "bge-m3").embed(["a", "b"])
        self.assertEqual(captured["url"], "http://gw:11500/v1/embeddings")
        self.assertEqual(captured["headers"]["authorization"], "Bearer k")
        self.assertEqual(
            captured["body"], {"model": "bge-m3", "input": ["a", "b"], "encoding_format": "float"}
        )
        self.assertAlmostEqual(
            math.sqrt(sum(v * v for v in vectors[0])), 1.0, places=6
        )  # index 0 first, unit
        self.assertAlmostEqual(
            sum(a * b for a, b in zip(vectors[1], unit(2))), 1.0, places=9
        )  # index 1 second

    def test_http_error_is_a_gate_error(self) -> None:
        _captured, fake = self._serve({}, status=401)
        with (
            mock.patch.object(urllib.request, "urlopen", fake),
            mock.patch.dict("os.environ", {}, clear=True),
            self.assertRaises(ollama.OllamaError) as ctx,
        ):
            backends.OpenAIEmbedder("http://gw:11500").embed(["a"])
        self.assertIn("401", str(ctx.exception))

    def test_zero_vector_is_refused(self) -> None:
        with self.assertRaises(ollama.OllamaError):
            backends.normalise([0.0] * 4)


class TestLocalBackend(unittest.TestCase):
    def test_uses_sentence_transformers_with_normalisation_on(self) -> None:
        calls = {}

        class FakeST:
            def __init__(self, model_id, device=None):
                calls["model"], calls["device"] = model_id, device

            def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True):
                calls["normalize"] = normalize_embeddings
                return [unit(i + 1) for i, _ in enumerate(texts)]

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeST
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            embedder = backends.make_embedder("local")
            gate = Gate.load(embedder=embedder)
            vectors = embedder.embed(["a", "b"])
        self.assertEqual(calls["model"], "BAAI/bge-m3")  # from the head file
        self.assertTrue(calls["normalize"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(gate.embedder.name, "local")

    def test_missing_dependency_says_how_to_install(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"sentence_transformers": None}),
            self.assertRaises(ollama.OllamaError) as ctx,
        ):
            backends.LocalEmbedder().embed(["a"])
        self.assertIn("privacy-gate-llm[local]", str(ctx.exception))

    def test_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            backends.make_embedder("cloud")


class TestCli(unittest.TestCase):
    def run_cli(self, argv: list[str], embedder=None, stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(backends, "make_embedder", return_value=embedder or FakeEmbedder()),
            mock.patch.object(sys, "stdin", io.StringIO(stdin)),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_check_text_argument(self) -> None:
        code, out, _ = self.run_cli(["check", "her biopsy", "is booked"])
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^(hold|send)  score=-?\d+\.\d{4} threshold=\d+\.\d{4} margin=-?\d+\.\d{4}\n$")

    def test_check_json_and_stdin_lines(self) -> None:
        code, out, _ = self.run_cli(["check", "--stdin", "--json"], stdin="one\n\ntwo\n")
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"hold", "score", "threshold", "margin"})

    def test_fail_on_hold_exit_code(self) -> None:
        class AlwaysHold(FakeEmbedder):
            def embed(self, texts):
                gate = Gate.load(embedder=FakeEmbedder())
                # the vector that scores highest among a few seeds, so the fake result is a hold
                best = max((unit(s) for s in (1.0, 2.0, 3.0, 5.0, 8.0, 13.0)), key=gate.model.score)
                return [best for _ in texts]

        code, out, _ = self.run_cli(["check", "--fail-on-hold", "x"], embedder=AlwaysHold())
        self.assertIn(code, (0, cli.EXIT_HOLD))
        self.assertEqual(code == cli.EXIT_HOLD, out.startswith("hold"))

    def test_cannot_score_fails_closed_with_exit_1(self) -> None:
        code, _out, err = self.run_cli(["check", "x"], embedder=FakeEmbedder(fail=True))
        self.assertEqual(code, 1)
        self.assertIn("failing closed", err)

    def test_info_prints_the_hash(self) -> None:
        code, out, _ = self.run_cli(["info"])
        self.assertEqual(code, 0)
        info = json.loads(out)
        self.assertEqual(len(info["head_sha256"]), 64)
        self.assertEqual(info["dimensions"], DIMS)
        self.assertEqual(info["ruleset"]["id"], "privacy-v2")
        self.assertEqual(len(info["ruleset"]["sha256"]), 64)
        self.assertEqual(info["ruleset"]["rules"], 17)

    def test_no_text_is_a_usage_error(self) -> None:
        code, _out, err = self.run_cli(["check"], stdin="")
        self.assertEqual(code, 2)
        self.assertIn("give text", err)


class LeakyBackend(FakeEmbedder):
    """A backend whose error names an internal address, as a real connection error would."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ollama.OllamaError("cannot reach http://internal-gpu.test:11434/api/embed: connection refused")


class TestSidecar(unittest.TestCase):
    def _server(self, embedder, timeout: float | None = None) -> tuple[SidecarServer, HTTPConnection]:
        handler = make_handler(Gate.load(embedder=embedder))
        if timeout is not None:
            handler.timeout = timeout
        server = SidecarServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)

    def _post(self, conn: HTTPConnection, body: bytes | dict | list | str | int | None) -> tuple[int, dict]:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        conn.request("POST", "/check", data, {"content-type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.getheader("content-type"), "application/json")
        return response.status, json.loads(response.read())

    def _raw(self, server: SidecarServer, request: bytes) -> tuple[int, dict]:
        """Send bytes http.client would refuse to send, and read one JSON response."""
        with socket.create_connection(server.server_address, timeout=5) as sock:
            sock.sendall(request)
            chunks = []
            while chunk := sock.recv(65536):
                chunks.append(chunk)
        head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(body)

    def test_single_and_batch(self) -> None:
        _server, conn = self._server(FakeEmbedder())
        status, body = self._post(conn, {"text": "her biopsy"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"hold", "score", "threshold", "margin"})
        status, body = self._post(conn, {"texts": ["a", "b", "c"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["results"]), 3)

    def test_health_names_the_backend_and_the_head(self) -> None:
        _server, conn = self._server(FakeEmbedder())
        conn.request("GET", "/health")
        body = json.loads(conn.getresponse().read())
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend"], "fake")
        self.assertEqual(len(body["head_sha256"]), 64)

    def test_unreachable_model_fails_closed(self) -> None:
        _server, conn = self._server(FakeEmbedder(fail=True))
        with self.assertLogs("privacy_gate.serve", logging.ERROR):
            status, body = self._post(conn, {"text": "x"})
        self.assertEqual(status, 503)
        self.assertTrue(body["hold"])
        self.assertEqual(body["code"], "backend_unavailable")

    def test_backend_error_is_logged_not_sent(self) -> None:
        _server, conn = self._server(LeakyBackend())
        with self.assertLogs("privacy_gate.serve", logging.ERROR) as logs:
            status, body = self._post(conn, {"text": "her biopsy"})
        self.assertEqual(status, 503)
        self.assertIs(body["hold"], True)
        self.assertEqual(body["code"], "backend_unavailable")
        self.assertEqual(body["reason"], "gate unavailable, failing closed")
        self.assertNotIn("internal-gpu", json.dumps(body))
        self.assertNotIn("OllamaError", json.dumps(body))
        self.assertIn("internal-gpu.test", logs.output[0])
        self.assertNotIn("biopsy", logs.output[0])

    def test_bad_bodies(self) -> None:
        _server, conn = self._server(FakeEmbedder())
        self.assertEqual(
            self._post(conn, b"not json"), (400, {"error": "body must be JSON", "code": "bad_json"})
        )
        self.assertEqual(self._post(conn, b"\xff\xfe not utf-8")[1]["code"], "bad_json")
        self.assertEqual(self._post(conn, {"texts": "one"})[0], 400)
        self.assertEqual(self._post(conn, {"texts": ["a", 1]})[0], 400)
        self.assertEqual(self._post(conn, {"text": 7})[0], 400)
        self.assertEqual(self._post(conn, {})[0], 400)

    def test_json_that_is_not_an_object(self) -> None:
        _server, conn = self._server(FakeEmbedder())
        for body in (["her biopsy"], 42, "her biopsy", None, True):
            with self.subTest(body=body):
                status, payload = self._post(conn, body)
                self.assertEqual(status, 400)
                self.assertEqual(payload["code"], "bad_request")
        # the connection is still usable afterwards: nothing crashed
        self.assertEqual(self._post(conn, {"text": "fine"})[0], 200)

    def test_oversized_body(self) -> None:
        server, _conn = self._server(FakeEmbedder())
        status, body = self._raw(
            server, f"POST /check HTTP/1.1\r\nHost: x\r\nContent-Length: {MAX_BODY + 1}\r\n\r\n{{}}".encode()
        )
        self.assertEqual((status, body["code"]), (413, "body_too_large"))

    def test_bad_content_length(self) -> None:
        server, _conn = self._server(FakeEmbedder())
        for value in ("abc", "-5", "1.5", "", "\xb2"):
            with self.subTest(value=value):
                status, body = self._raw(
                    server,
                    f"POST /check HTTP/1.1\r\nHost: x\r\nContent-Length: {value}\r\n\r\n".encode("latin-1"),
                )
                self.assertEqual((status, body["code"]), (400, "bad_content_length"))
        status, body = self._raw(server, b"POST /check HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        self.assertEqual((status, body["code"]), (400, "bad_content_length"))

    def test_unknown_paths(self) -> None:
        _server, conn = self._server(FakeEmbedder())
        conn.request("POST", "/nope", b"{}")
        response = conn.getresponse()
        self.assertEqual((response.status, json.loads(response.read())["code"]), (404, "not_found"))
        conn.request("GET", "/check")
        response = conn.getresponse()
        self.assertEqual((response.status, json.loads(response.read())["code"]), (404, "not_found"))

    def test_slow_client_is_let_go(self) -> None:
        self.assertEqual(make_handler(Gate.load(embedder=FakeEmbedder())).timeout, 10.0)
        server, _conn = self._server(FakeEmbedder(), timeout=0.2)
        with socket.create_connection(server.server_address, timeout=5) as sock:
            # promise ten bytes and send none: the server must close the socket, not wait forever
            sock.sendall(b"POST /check HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n")
            self.assertEqual(sock.recv(1), b"")


class TestRuntimeOnly(unittest.TestCase):
    """The wheel ships the runtime and nothing a maintainer only needs to retrain or measure the head."""

    PACKAGE = Path(__file__).resolve().parent.parent / "src" / "privacy_gate"
    MOVED = ("calibrate", "evaluate", "embed", "dataset", "labels", "prompt", "features")

    def test_training_modules_live_outside_the_package(self) -> None:
        for name in self.MOVED:
            self.assertFalse((self.PACKAGE / f"{name}.py").exists(), f"{name}.py is back in src/privacy_gate")

    def test_the_package_never_imports_training(self) -> None:
        for path in self.PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    self.assertNotEqual(name.split(".")[0], "training", f"{path.name} imports {name}")

    def test_python_dash_m_is_the_cli(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "privacy_gate", "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("serve", done.stdout)


if __name__ == "__main__":
    unittest.main()
