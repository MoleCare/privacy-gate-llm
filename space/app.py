# SPDX-License-Identifier: Apache-2.0
"""The demo Space: type a sentence, see whether the gate would hold it, and why not.

Runs the real gate: `BAAI/bge-m3` through sentence-transformers on the Space's CPU, and the shipped head
(`privacy-gate` from this repository). Nothing typed here is stored: the Space keeps no logs and the request
is scored in memory. The second tab is the gold-set explorer, unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from privacy_gate.backends import LocalEmbedder
from privacy_gate.gate import Gate

HERE = Path(__file__).resolve().parent
EXPLORER = HERE / "explorer"
gr.set_static_paths(paths=[EXPLORER])

gate = Gate.load(embedder=LocalEmbedder(device="cpu"))
THRESHOLD = gate.threshold
HEAD = gate.head_sha256[:12]

# Invented, like everything in the gold set. Four that the rules cannot see, four that look sensitive and are not.
EXAMPLES = [
    "The woman from Tuesday's clinic, 34, has a 7mm asymmetric lesion on her left shoulder and a biopsy booked for the 20th.",
    "Use the key I pasted in the channel yesterday for the staging bucket, it still works.",
    "My neighbour Tom at number 12 has just started chemo, so keep the noise down this week.",
    "Her date of birth is the same as mine, 3 March, and she lives above the pharmacy on Mill Road.",
    "The endpoint returns 401 when the Authorization header is missing; see the OpenAPI example with the placeholder token.",
    "Melanoma is the most serious form of skin cancer and is treatable when found early.",
    "Set OPENAI_API_KEY in your environment before running the tests; the fixture uses sk-test-000.",
    "Thanks! The build is green and the release notes are ready for review.",
]


def _verdict(score: float, threshold: float) -> str:
    return "hold" if score > threshold else "send"


def check(text: str, threshold: float):
    text = (text or "").strip()
    if not text:
        return "Type a sentence.", None
    decision = gate.decide(text)
    verdict = _verdict(decision.score, threshold)
    margin = decision.score - threshold
    colour = "#b3261e" if verdict == "hold" else "#2f7d4f"
    headline = (
        f"<div style='font-size:1.6em;font-weight:700;color:{colour}'>{verdict.upper()}</div>"
        f"<div>score <b>{decision.score:.3f}</b> · threshold {threshold:.4f} · margin <b>{margin:+.3f}</b></div>"
    )
    if abs(margin) < 0.5:
        headline += "<div style='margin-top:6px'>The margin is small: this is the kind of text worth adding to the gold set.</div>"
    if verdict == "send":
        headline += ("<div style='margin-top:6px;color:#6b5f70'>A <i>send</i> is not an assurance that the text is safe. "
                     "This is a second gate that may only ever add a hold behind your own rules.</div>")
    return headline, {"hold": verdict == "hold", "score": round(decision.score, 4),
                      "threshold": round(threshold, 4), "margin": round(margin, 4)}


def check_many(text: str, threshold: float):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    decisions = gate.decide_many(lines)
    return [[_verdict(d.score, threshold), round(d.score, 3), round(d.score - threshold, 3), ln] for d, ln in zip(decisions, lines)]


with gr.Blocks(title="privacy-gate: must this stay on this machine?") as demo:
    gr.Markdown(
        "# Must this stay on this machine?\n"
        "A **1,024-weight head on `bge-m3`** that catches health data, credentials and personal data written as "
        "ordinary English, which pattern rules cannot see. It may only ever *add* a hold behind the rules you already "
        "have. Nothing you type here is stored.  \n"
        f"[Code, data and journal](https://github.com/MoleCare/privacy-gate-llm) · "
        f"[model](https://huggingface.co/YauhenBichel/privacy-gate-llm) · `pip install privacy-gate` · head `{HEAD}` · "
        f"encoder on this Space's CPU, about a second per sentence."
    )
    with gr.Tabs():
        with gr.Tab("Try it"):
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(label="Text", lines=3, placeholder="Paste a sentence, a prompt, a log line…")
                    threshold = gr.Slider(-2.0, 4.0, value=THRESHOLD, step=0.01, label="Threshold (the shipped one is the default)",
                                          info="Lower catches more and interrupts more; the score does not change, only the verdict.")
                    button = gr.Button("Check", variant="primary")
                with gr.Column(scale=2):
                    verdict = gr.HTML(label="Verdict")
                    details = gr.JSON(label="As the sidecar would answer")
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text],
                        label="Invented examples. Some the rules cannot see; some look sensitive and are not. The gate is not always right "
                              "about them, and that is what the gold set and its hard negatives are for.")
            button.click(check, inputs=[text, threshold], outputs=[verdict, details])
            text.submit(check, inputs=[text, threshold], outputs=[verdict, details])
        with gr.Tab("Several at once"):
            many = gr.Textbox(label="One text per line", lines=8)
            threshold2 = gr.Slider(-2.0, 4.0, value=THRESHOLD, step=0.01, label="Threshold")
            table = gr.Dataframe(headers=["verdict", "score", "margin", "text"], datatype=["str", "number", "number", "str"],
                                 label="One embedding call for the batch", wrap=True)
            gr.Button("Check all", variant="primary").click(check_many, inputs=[many, threshold2], outputs=[table])
        with gr.Tab("Explore the gold set"):
            gr.Markdown("The 147 evaluation examples with five-fold cross-validated scores, the regex ruleset's verdict beside the "
                        "model's, and the threshold as a slider. Precomputed and committed; this tab scores nothing.")
            gr.HTML(f"<iframe src='/gradio_api/file={EXPLORER / 'index.html'}' style='width:100%;height:1400px;border:0'></iframe>")
    gr.Markdown(
        "**Not a medical device.** A privacy tool: it decides whether text should leave a machine and says nothing "
        "about what a lesion is. Not production-validated: 247 invented examples by one author choose an architecture, "
        "not a threshold for real patient data. Apache-2.0."
    )

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")))
