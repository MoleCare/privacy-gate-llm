# SPDX-License-Identifier: Apache-2.0
"""Score the gate as a continuous decision, then choose where to cut it.

Asking a very small model for a word and reading the word back throws almost
everything away. The model does not believe "KEEP" flatly; it prefers KEEP over
SEND by some margin, and that margin is the actual output. Argmax is the special
case of thresholding it at zero, and zero is an arbitrary place to cut.

So the score for one text is

    margin = logP(KEEP) - logP(SEND)

at the first generated position, and the gate is `margin > threshold`. This does
three useful things:

1. It separates *has the model any signal* from *is its default sensible*. A
   model that always answers KEEP can still rank a clinical note above a Gradle
   question, and if it does, a threshold recovers a working gate for free.
2. It makes the trade-off explicit and tunable. Catch and friction move together
   along one dial the owner can set, rather than being whatever the model felt.
3. It gives AUC, which is threshold-independent, so training runs can be compared
   without arguing about operating points.

AUC here is the probability that a randomly chosen sensitive text is ranked above
a randomly chosen clean one. 0.5 is a coin flip and means no signal at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ollama, prompt
from .dataset import Example, load
from .labels import holds

#: The grammar makes the model open with a quote, so the interesting distribution
#: sits at the first position either way. Tokenisers differ on leading spaces and
#: case, so every plausible spelling of each word is pooled.
KEEP_TOKENS = ("KEEP", "keep", " KEEP", " keep", "Keep", " Keep")
SEND_TOKENS = ("SEND", "send", " SEND", " send", "Send", " Send")

#: Well below any real logprob, used when a word is missing from the top-k. It
#: means "the model did not consider this", which is a strong signal, not a gap.
ABSENT = -20.0


class NoAnswerPosition(RuntimeError):
    """Neither word appeared at any generated position.

    Raised rather than returned as a floor value. A silent floor scored every
    `gpt-oss:20b` example identically and produced a tidy, meaningless AUC of
    exactly 0.5 (JOURNAL, run 4). A broken measurement must fail loudly.
    """


@dataclass
class Scored:
    id: str
    slice: str
    gold: str
    sensitive: bool
    margin: float
    keep_logprob: float
    send_logprob: float


def _best(logprobs: dict[str, float], candidates: tuple[str, ...]) -> float | None:
    found = [logprobs[token] for token in candidates if token in logprobs]
    return max(found) if found else None


def answer_position(positions: list[dict[str, float]]) -> dict[str, float]:
    """The first generated position that is actually deciding between the words.

    Models open differently: a `format` enum makes Qwen emit a quote first, and
    `gpt-oss` emits harmony control tokens. What identifies the real answer
    position is that at least one of the two words is a live candidate there.
    """
    for logprobs in positions:
        if _best(logprobs, KEEP_TOKENS) is not None or _best(logprobs, SEND_TOKENS) is not None:
            return logprobs
    raise NoAnswerPosition(
        f"neither KEEP nor SEND in the top-k at any of {len(positions)} positions"
    )


def score_one(text: str, model: str, *, url: str, top_k: int = 20) -> tuple[float, float, float]:
    reply = ollama.chat(
        model,
        prompt.build_binary(text),
        url=url,
        schema={"type": "string", "enum": ["SEND", "KEEP"]},
        logprobs=top_k,
        predict=6,
    )
    logprobs = answer_position(reply.position_logprobs or [])
    # One word may be absent even at the right position; that is a real, very
    # low preference rather than a broken read, so the floor applies here only.
    keep = _best(logprobs, KEEP_TOKENS)
    send = _best(logprobs, SEND_TOKENS)
    keep = ABSENT if keep is None else keep
    send = ABSENT if send is None else send
    return keep - send, keep, send


def collect(
    examples: list[Example], model: str, *, url: str, checkpoint: Path | None = None
) -> list[Scored]:
    done: dict[str, Scored] = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = Scored(**row)
        print(f"  resuming, {len(done)} already scored", file=sys.stderr, flush=True)

    out: list[Scored] = []
    for index, example in enumerate(examples, 1):
        if example.id in done:
            out.append(done[example.id])
            continue
        margin, keep, send = score_one(example.text, model, url=url)
        result = Scored(
            id=example.id,
            slice=example.slice,
            gold=example.label,
            sensitive=holds(example.label),
            margin=round(margin, 5),
            keep_logprob=round(keep, 5),
            send_logprob=round(send, 5),
        )
        out.append(result)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a") as handle:
                handle.write(json.dumps(asdict(result)) + "\n")
        if index % 10 == 0:
            print(f"  {index}/{len(examples)}", file=sys.stderr, flush=True)
    return out


def auc(scored: list[Scored]) -> float:
    """Rank-based AUC, with ties counted as half. No numpy on the server."""
    positives = [s.margin for s in scored if s.sensitive]
    negatives = [s.margin for s in scored if not s.sensitive]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def sweep(scored: list[Scored], rules: dict[str, bool] | None = None) -> list[dict]:
    """Every distinct operating point, as thresholds between adjacent margins."""
    positives = [s for s in scored if s.sensitive]
    negatives = [s for s in scored if not s.sensitive]
    cuts = sorted({s.margin for s in scored})
    # One cut below everything (hold nothing) through one above everything.
    candidates = [cuts[0] - 1.0] + [
        (a + b) / 2 for a, b in zip(cuts, cuts[1:])
    ] + [cuts[-1] + 1.0]

    points = []
    for threshold in candidates:
        def held(s: Scored) -> bool:
            return s.margin > threshold

        caught = sum(1 for s in positives if held(s))
        friction = sum(1 for s in negatives if held(s))
        point = {
            "threshold": round(threshold, 4),
            "catch_rate": round(caught / len(positives), 4),
            "friction_rate": round(friction / len(negatives), 4),
            "leaks": len(positives) - caught,
            "friction": friction,
        }
        if rules is not None:
            composed_caught = sum(1 for s in positives if held(s) or rules.get(s.id, False))
            composed_friction = sum(
                1 for s in negatives if held(s) or rules.get(s.id, False)
            )
            point["composed_catch_rate"] = round(composed_caught / len(positives), 4)
            point["composed_friction_rate"] = round(composed_friction / len(negatives), 4)
            point["composed_leaks"] = len(positives) - composed_caught
        points.append(point)
    return points


def pick(points: list[dict], min_catch: float) -> dict | None:
    """The least annoying threshold that still catches enough.

    Catch is the constraint and friction is what gets minimised, not the other
    way round: a missed leak is an incident, extra friction is a slower answer.
    """
    key = "composed_catch_rate" if "composed_catch_rate" in points[0] else "catch_rate"
    fkey = "composed_friction_rate" if "composed_friction_rate" in points[0] else "friction_rate"
    eligible = [p for p in points if p[key] >= min_catch]
    return min(eligible, key=lambda p: p[fkey]) if eligible else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the gate as a scored classifier.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--rules")
    parser.add_argument("--url", default=ollama.DEFAULT_URL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-catch", type=float, default=0.98)
    args = parser.parse_args(argv)

    examples = load(args.data)
    rules = json.loads(Path(args.rules).read_text()) if args.rules else None
    scored = collect(
        examples, args.model, url=args.url, checkpoint=Path(f"{args.out}.partial.jsonl")
    )
    points = sweep(scored, rules)
    area = auc(scored)
    chosen = pick(points, args.min_catch)

    print(f"### {args.model} — scored gate\n")
    print(f"**AUC {area:.4f}** over {len(scored)} examples "
          f"({sum(s.sensitive for s in scored)} sensitive). 0.5 would be no signal at all.\n")
    if chosen:
        key = "composed_catch_rate" if "composed_catch_rate" in chosen else "catch_rate"
        fkey = "composed_friction_rate" if "composed_friction_rate" in chosen else "friction_rate"
        print(f"Best operating point at catch >= {args.min_catch:.0%}: "
              f"threshold {chosen['threshold']:+.3f}, "
              f"catch {chosen[key]:.1%}, friction {chosen[fkey]:.1%}.\n")
    else:
        print(f"No threshold reaches catch >= {args.min_catch:.0%}.\n")

    print("| threshold | catch | friction |")
    print("|---|---|---|")
    seen: set[tuple] = set()
    for point in points:
        key = "composed_catch_rate" if "composed_catch_rate" in point else "catch_rate"
        fkey = "composed_friction_rate" if "composed_friction_rate" in point else "friction_rate"
        pair = (point[key], point[fkey])
        if pair in seen:
            continue
        seen.add(pair)
        print(f"| {point['threshold']:+.3f} | {point[key]:.1%} | {point[fkey]:.1%} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "auc": area,
                "min_catch": args.min_catch,
                "chosen": chosen,
                "points": points,
                "scored": [asdict(s) for s in scored],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
