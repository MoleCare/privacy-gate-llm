# SPDX-License-Identifier: Apache-2.0
"""privacy-gate as a LiteLLM proxy guardrail: keep sensitive prompts on a local model, or block them.

The gate reads every text of the request (system, user and assistant messages, tool outputs) and, when any
of them must stay on this machine, either **routes** the request to a local model or **blocks** it. It may
only ever add a hold; a `send` is not an assurance the text is safe, so keep whatever deterministic
guardrails you already run in front of it.

    # proxy config.yaml
    guardrails:
      - guardrail_name: privacy-gate
        litellm_params:
          guardrail: privacy_gate_guardrail.PrivacyGateGuardrail
          mode: pre_call
          default_on: true
          action: route            # or block
          local_model: ollama/qwen3-coder:30b   # where a held request goes (action: route)
          backend: ollama          # ollama | openai | local
          url: http://127.0.0.1:11434
          min_length: 12           # texts shorter than this are not scored

    pip install "privacy-gate @ git+https://github.com/MoleCare/privacy-gate-llm"
    # plus the [local] extra for the in-process encoder; plain `pip install privacy-gate` once on PyPI
    litellm --config config.yaml

A held request carries `metadata.privacy_gate = {"hold": true, "score": ..., "margin": ...}` so the log
shows why it went local. When the encoder cannot be reached the gate **fails closed**: with `action: route`
the request goes local; with `action: block` it is refused.
"""

from __future__ import annotations

from typing import Any

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pragma: no cover - the proxy has it; the tests stub it
    class CustomGuardrail:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

from privacy_gate import backends
from privacy_gate.gate import Gate


class PrivacyGateGuardrail(CustomGuardrail):
    def __init__(
        self,
        action: str = "route",
        local_model: str | None = None,
        backend: str = "ollama",
        url: str | None = None,
        model: str | None = None,
        head: str | None = None,
        min_length: int = 12,
        gate: Gate | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if action not in ("route", "block"):
            raise ValueError("action must be 'route' or 'block'")
        if action == "route" and not local_model:
            raise ValueError("action: route needs local_model")
        self.action, self.local_model, self.min_length = action, local_model, min_length
        self.gate = gate or Gate.load(head, embedder=backends.make_embedder(backend, url, model))

    @staticmethod
    def texts_of(data: dict) -> list[str]:
        """Every string the model would see: message contents, text parts of multimodal content, tool results."""
        out: list[str] = []
        for message in data.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                out += [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            for call in message.get("tool_calls") or []:
                args = (call.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    out.append(args)
        if isinstance(data.get("prompt"), str):
            out.append(data["prompt"])
        if isinstance(data.get("input"), str):
            out.append(data["input"])
        return [t for t in out if t and len(t.strip()) >= 1]

    def verdict(self, data: dict) -> dict:
        texts = [t for t in self.texts_of(data) if len(t) >= self.min_length]
        if not texts:
            return {"hold": False, "scored": 0}
        try:
            decisions = self.gate.decide_many(texts)
        except Exception as error:  # noqa: BLE001 - unreachable encoder: fail closed
            return {"hold": True, "scored": 0, "error": f"{type(error).__name__}: {str(error)[:120]}", "reason": "gate unavailable, failing closed"}
        worst = max(decisions, key=lambda d: d.score)
        return {"hold": any(d.hold for d in decisions), "scored": len(texts), "score": round(worst.score, 4),
                "margin": round(worst.margin, 4), "threshold": round(worst.threshold, 4)}

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> dict:
        result = self.verdict(data)
        data.setdefault("metadata", {})["privacy_gate"] = result
        if not result["hold"]:
            return data
        if self.action == "block":
            raise PermissionError(
                f"privacy-gate: this request must stay on this machine (score {result.get('score')}, "
                f"threshold {result.get('threshold')}); it was not sent"
            )
        data["metadata"]["privacy_gate"]["routed_from"] = data.get("model")
        data["model"] = self.local_model
        return data
