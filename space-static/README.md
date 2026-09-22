---
title: privacy-gate
emoji: 🚧
colorFrom: yellow
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Health data, secrets and PII in prose, in your browser
models:
- YauhenBichel/privacy-gate-llm
- Xenova/bge-m3
- BAAI/bge-m3
datasets:
- YauhenBichel/privacy-gate-gold
tags:
- privacy
- pii-detection
- phi-detection
- guardrails
- llm-safety
- llm-router
- transformers.js
---

# privacy-gate

**Must this text stay on this machine?** Type a sentence and see whether the gate would hold it: a
1,024-weight logistic head on [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) that catches health data,
credentials and personal data written as ordinary English, which regex rules cannot see. It may only ever *add*
a hold behind the rules you already have.

**This page runs the gate in your browser**: the encoder ([Xenova/bge-m3](https://huggingface.co/Xenova/bge-m3),
int8 ONNX through transformers.js, about 570 MB cached after the first visit) and the head. Nothing you type is
sent anywhere, and the page keeps no logs. A static Space, so it needs no server and no quota.

The second tab is the gold-set explorer: 147 evaluation examples with five-fold cross-validated scores,
precomputed and committed.

- Code, data and journal: [github.com/MoleCare/privacy-gate-llm](https://github.com/MoleCare/privacy-gate-llm)
- Model: [YauhenBichel/privacy-gate-llm](https://huggingface.co/YauhenBichel/privacy-gate-llm) ·
  data: [YauhenBichel/privacy-gate-gold](https://huggingface.co/datasets/YauhenBichel/privacy-gate-gold) ·
  `pip install privacy-gate`

A `send` is not an assurance that text is safe. Not production-validated. Not a medical device.
