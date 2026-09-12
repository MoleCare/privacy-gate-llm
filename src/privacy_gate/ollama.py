# SPDX-License-Identifier: Apache-2.0
"""A very small Ollama client. Standard library only, so this runs on the server
with no install step.

`num_ctx` is deliberately never sent. Setting it makes Ollama re-load the model
under a second cache entry, which on a shared box evicts whatever the owner has
pinned. The same rule is enforced in the model factory for the same reason.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_URL = "http://127.0.0.1:11434"

#: Ollama accepts a JSON schema here and constrains decoding to match it, so the
#: model cannot answer with anything but one of the four labels.
LABEL_SCHEMA = {"type": "string", "enum": ["CLEAN", "HEALTH", "SECRET", "PII"]}


@dataclass(frozen=True)
class Reply:
    content: str
    seconds: float
    eval_count: int


class OllamaError(RuntimeError):
    pass


def _post(url: str, path: str, payload: dict, timeout: float, attempts: int = 4) -> dict:
    """POST with backoff.

    yserver is shared. Another session can load a 50 GB model at any moment,
    which evicts this one and makes the next call time out or fail. Losing a
    whole evaluation to one transient failure is the mistake java-dsa-harness
    already paid for, so every call retries before the run gives up.
    """
    data = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url + path,
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
            if attempt < attempts - 1:
                time.sleep(5 * 2**attempt)
        except urllib.error.HTTPError as error:  # pragma: no cover - network path
            raise OllamaError(f"{path} returned {error.code}: {error.read()[:400]!r}") from error
    raise OllamaError(f"{path} failed after {attempts} attempts: {last}")


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    url: str = DEFAULT_URL,
    schema: dict | None = None,
    keep_alive: str | int = "30m",
    # Generous, because the first call of a run may have to evict a 20 GB model
    # and load this one. That took 145 s on yserver; generation itself is in
    # milliseconds. A tight timeout here fails the run for a queueing reason.
    timeout: float = 900.0,
    predict: int = 8,
) -> Reply:
    """One greedy classification. Thinking is off: this is a lookup, not a puzzle,
    and a thinking model would spend hundreds of tokens to say one word."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0, "seed": 0, "num_predict": predict},
    }
    if schema is not None:
        payload["format"] = schema
    started = time.monotonic()
    body = _post(url, "/api/chat", payload, timeout)
    return Reply(
        content=body.get("message", {}).get("content", ""),
        seconds=time.monotonic() - started,
        eval_count=int(body.get("eval_count") or 0),
    )


def resident(url: str = DEFAULT_URL) -> list[dict]:
    """What is loaded right now, so a run can put the box back as it found it."""
    request = urllib.request.Request(url + "/api/ps")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read()).get("models", [])


def pin(model: str, url: str = DEFAULT_URL) -> None:
    """Load a model and keep it loaded, the way `keep_alive: -1` does."""
    _post(url, "/api/chat", {"model": model, "messages": [], "keep_alive": -1}, 600)
