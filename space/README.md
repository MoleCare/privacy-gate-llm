---
title: privacy-gate
emoji: 🚧
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Catches health data, credentials and PII written as prose
models:
- YauhenBichel/privacy-gate-llm
- BAAI/bge-m3
tags:
- privacy
- pii-detection
- phi-detection
- guardrails
- llm-safety
- llm-router
---

# privacy-gate

**Must this text stay on this machine?** Type a sentence and see whether the gate would hold it: a
1,024-weight logistic head on [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) that catches health data,
credentials and personal data written as ordinary English, which regex rules cannot see. It may only ever *add*
a hold behind the rules you already have.

This Space runs the real gate on its CPU (`sentence-transformers`, about a second a sentence) with the head
shipped in `pip install privacy-gate`. **Nothing typed here is stored.** The third tab is the gold-set explorer:
147 evaluation examples with five-fold cross-validated scores, precomputed and committed.

- Code, data and journal: [github.com/MoleCare/privacy-gate-llm](https://github.com/MoleCare/privacy-gate-llm)
- Model: [YauhenBichel/privacy-gate-llm](https://huggingface.co/YauhenBichel/privacy-gate-llm)

A `send` is not an assurance that text is safe. Not production-validated. Not a medical device.
