# Using the gate inside tools you already run

Each of these is a few dozen lines on top of `pip install "privacy-gate @ git+https://github.com/MoleCare/privacy-gate-llm"` (PyPI release pending). The rule is the same everywhere: the
gate may only ever **add** a hold. Keep the deterministic checks you already have in front of it.

| Where | File | What a hold does |
|---|---|---|
| **LiteLLM proxy** | [`litellm/privacy_gate_guardrail.py`](litellm/privacy_gate_guardrail.py) | `action: route` switches the request to `local_model`; `action: block` refuses it. Every message text and tool argument is scored; the verdict is written to `metadata.privacy_gate` |
| **Open WebUI** | [`open-webui/privacy_gate_filter.py`](open-webui/privacy_gate_filter.py) | a filter function: on a hold the chat is switched to a local model, or refused |
| **GitHub Actions** | [`../action.yml`](../action.yml) | fails a pull request whose added lines look like health data, credentials or personal data written as prose, with an annotation on each line |
| **Any language** | `privacy-gate serve` | a loopback sidecar: `POST /check {"text": ...}`; see [`../docs/INTEGRATION.md`](../docs/INTEGRATION.md) |

When the encoder cannot be reached, every integration **fails closed**: a held verdict, never a silent send.

## LiteLLM

```yaml
guardrails:
  - guardrail_name: privacy-gate
    litellm_params:
      guardrail: privacy_gate_guardrail.PrivacyGateGuardrail
      mode: pre_call
      default_on: true
      action: route
      local_model: ollama/qwen3-coder:30b
      backend: ollama
      url: http://127.0.0.1:11434
```

Put the file next to `config.yaml` (or on `PYTHONPATH`), install the package as above, start the proxy. A request
whose text must stay goes to `local_model` instead of the model it asked for, and the log line shows why.

## Open WebUI

Workspace → Functions → **+** → paste the file. Enable it on the models that leave the machine. The valves set
the action, the local model and the embedding endpoint.

## GitHub Actions

```yaml
- uses: MoleCare/privacy-gate-llm@v1
  with:
    backend: local          # downloads BAAI/bge-m3 once per runner cache
```

Scans the lines a pull request adds (text files only, one line at a time, lines of 30 characters or more),
annotates each hold, and fails the job. A hold in a fixture or a doc example is exactly what the check is for:
a placeholder that looks real is the kind of text that should not be in a public repository.
