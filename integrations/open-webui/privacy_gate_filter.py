"""
title: privacy-gate
author: MoleCare
author_url: https://github.com/MoleCare/privacy-gate-llm
version: 1.0.0
license: Apache-2.0
description: Keep sensitive chats on a local model. A 1,024-weight head on bge-m3 catches health data, credentials and personal data written as plain English; when it holds, the chat is switched to a local model (or refused).
requirements: privacy-gate
"""

# SPDX-License-Identifier: Apache-2.0
# An Open WebUI filter function: Workspace -> Functions -> + -> paste this file. Turn it on for the models
# that may leave the machine (the cloud ones); the local ones need it not at all.

from __future__ import annotations

from pydantic import BaseModel, Field

from privacy_gate import backends
from privacy_gate.gate import Gate


class Filter:
    class Valves(BaseModel):
        action: str = Field(default="route", description="route: switch to local_model on a hold; block: refuse")
        local_model: str = Field(default="qwen3-coder:30b", description="the model a held chat is switched to")
        backend: str = Field(default="ollama", description="ollama | openai | local")
        url: str = Field(default="http://127.0.0.1:11434", description="the embedding endpoint")
        min_length: int = Field(default=12, description="shorter messages are not scored")
        last_messages: int = Field(default=3, description="how many of the latest messages to score")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._gate: Gate | None = None

    def _get_gate(self) -> Gate:
        if self._gate is None:
            self._gate = Gate.load(embedder=backends.make_embedder(self.valves.backend, self.valves.url))
        return self._gate

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        messages = body.get("messages") or []
        texts = []
        for message in messages[-self.valves.last_messages:]:
            content = message.get("content")
            if isinstance(content, str) and len(content) >= self.valves.min_length:
                texts.append(content)
            elif isinstance(content, list):
                texts += [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                          and len(p.get("text", "")) >= self.valves.min_length]
        if not texts:
            return body
        try:
            decisions = self._get_gate().decide_many(texts)
            hold, worst = any(d.hold for d in decisions), max(d.score for d in decisions)
        except Exception as error:  # noqa: BLE001 - unreachable encoder: fail closed
            hold, worst = True, float("nan")
            body.setdefault("metadata", {})["privacy_gate_error"] = f"{type(error).__name__}"
        if not hold:
            return body
        if self.valves.action == "block":
            raise Exception(f"privacy-gate: this must stay on this machine (score {worst:.2f}); not sent.")
        body.setdefault("metadata", {})["privacy_gate"] = {"hold": True, "score": worst, "routed_from": body.get("model")}
        body["model"] = self.valves.local_model
        return body
