# SPDX-License-Identifier: Apache-2.0
"""The Ollama transport the embedders use. Standard library only, so this runs on the server
with no install step.

The chat client the calibration and evaluation runs use (`chat`, `Reply`, `resident`, `pin`) is a
maintainer tool and lives in `training/chat.py` in the repository, outside the wheel.

`num_ctx` is deliberately never sent. Setting it makes Ollama re-load the model
under a second cache entry, which on a shared box evicts whatever the owner has
pinned. The same rule is enforced in the model factory for the same reason.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


def _post(url: str, path: str, payload: dict[str, Any], timeout: float, attempts: int = 4) -> dict[str, Any]:
    """POST with backoff.

    An inference host is usually shared. Another client can load a 50 GB model
    at any moment, which evicts this one and makes the next call time out or
    fail. Losing a whole evaluation to one transient failure is a mistake worth
    paying for only once, so every call retries before the run gives up.
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
                body: dict[str, Any] = json.loads(response.read())
                return body
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
            if attempt < attempts - 1:
                time.sleep(5 * 2**attempt)
        except urllib.error.HTTPError as error:  # pragma: no cover - network path
            raise OllamaError(f"{path} returned {error.code}: {error.read()[:400]!r}") from error
    raise OllamaError(f"{path} failed after {attempts} attempts: {last}")
