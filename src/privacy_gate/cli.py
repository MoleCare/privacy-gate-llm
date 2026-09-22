# SPDX-License-Identifier: Apache-2.0
"""The command line: check text, or serve the sidecar.

    privacy-gate check "her biopsy is booked for the 20th"
    hold  score=6.3272 threshold=0.1209 margin=6.2063

    privacy-gate check --json "the build is green"
    {"hold": false, "score": -2.41, "threshold": 0.1209, "margin": -2.53}

    cat prompts.txt | privacy-gate check --stdin --json     # one text per line, one JSON line each
    privacy-gate check --fail-on-hold < text.txt            # exit 3 when anything must stay: for CI

    privacy-gate serve [--port 8231] [--backend ollama|openai|local] [--url ...] [--model ...]

Exit codes: 0 sent, 3 held (with --fail-on-hold), 2 usage, 1 the gate could not score (it fails closed:
treat 1 as hold).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import backends
from .gate import Gate

EXIT_HOLD = 3


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--head", help="head file (default: the one shipped with this version)")
    parser.add_argument("--backend", choices=backends.BACKENDS, default="ollama",
                        help="where the embedding comes from (default: ollama)")
    parser.add_argument("--url", help="the embedding endpoint (ollama and openai backends)")
    parser.add_argument("--model", help="the embedding model name at that endpoint, or the Hugging Face id for local")
    parser.add_argument("--api-key", help="openai backend: a bearer token (or PRIVACY_GATE_API_KEY in the environment)")


def _gate(args: argparse.Namespace) -> Gate:
    embedder = backends.make_embedder(args.backend, args.url, args.model, args.api_key)
    return Gate.load(args.head, embedder=embedder)


def _format(decision, as_json: bool) -> str:  # noqa: ANN001
    if as_json:
        return json.dumps({"hold": decision.hold, "score": round(decision.score, 4),
                           "threshold": round(decision.threshold, 4), "margin": round(decision.margin, 4)})
    verdict = "hold" if decision.hold else "send"
    return f"{verdict}  score={decision.score:.4f} threshold={decision.threshold:.4f} margin={decision.margin:.4f}"


def cmd_check(args: argparse.Namespace) -> int:
    if args.stdin:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        texts = [sys.stdin.read().strip()]
        if not texts[0]:
            print("privacy-gate check: give text as an argument, on stdin, or --stdin for one per line", file=sys.stderr)
            return 2
    try:
        gate = _gate(args)
        decisions = gate.decide_many(texts)
    except Exception as error:  # noqa: BLE001 - every failure means "could not score", and that is a hold
        print(f"privacy-gate: cannot score ({type(error).__name__}): {error}", file=sys.stderr)
        print("privacy-gate: failing closed; treat as hold", file=sys.stderr)
        return 1
    for decision in decisions:
        print(_format(decision, args.json))
    if args.fail_on_hold and any(d.hold for d in decisions):
        return EXIT_HOLD
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve import serve

    try:
        gate = _gate(args)
    except Exception as error:  # noqa: BLE001
        print(f"privacy-gate serve: cannot load ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    serve(gate, host=args.host, port=args.port, verbose=args.verbose)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    gate = Gate.load(args.head, embedder=backends.make_embedder("ollama"))
    print(json.dumps({"head": str(gate.head_path), "head_sha256": gate.head_sha256, "threshold": gate.threshold,
                      "embedding_model": gate.embedding_model, "dimensions": len(gate.model.weights),
                      **{k: gate.meta[k] for k in ("trained_on", "cross_validated_auc", "embedding_model_hf") if k in gate.meta}},
                     indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="privacy-gate", description="Must this text stay on this machine?")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="score text: hold or send")
    check.add_argument("text", nargs="*", help="the text (or give it on stdin)")
    check.add_argument("--stdin", action="store_true", help="one text per line on stdin")
    check.add_argument("--json", action="store_true", help="one JSON object per text")
    check.add_argument("--fail-on-hold", action="store_true", help=f"exit {EXIT_HOLD} when any text must stay")
    _add_backend_options(check)
    check.set_defaults(func=cmd_check)

    serve = sub.add_parser("serve", help="the loopback HTTP sidecar")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8231)
    serve.add_argument("--verbose", action="store_true", help="log request lines (off by default, on purpose)")
    _add_backend_options(serve)
    serve.set_defaults(func=cmd_serve)

    info = sub.add_parser("info", help="which head, its hash, threshold and model")
    info.add_argument("--head")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
