#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Wait for a shared inference host to go idle, then run the measurements that
# need it, and stop. Drives the host over ssh.
#
# Why wait rather than queue: Ollama commonly runs with a single slot (-np 1),
# so a busy host serialises everything behind whatever is already running. When
# this was written, an unrelated GPU job kept the host loaded for hours and every
# request that needed a model swap timed out or returned 500. Adding more work
# only makes both jobs slower.
#
#   ./scripts/run-when-free.sh [max_wait_minutes]
#
# Both steps checkpoint per example, so an interrupted run resumes for free.

set -uo pipefail

HOST=${HOST:?set HOST to the ssh alias of your inference host}
REMOTE=${REMOTE:-'~/privacy-gate-llm'}
MAX_WAIT_MIN=${1:-180}
# Chat model for the ceiling. Avoid gemma4:31b: it hangs when asked for
# logprobs, suspected to be its 256k vocabulary.
CEILING_MODEL=${CEILING_MODEL:-qwen3-coder:30b}

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

idle() {
    ssh -o ConnectTimeout=20 "$HOST" '
        load=$(cut -d" " -f1 /proc/loadavg | cut -d. -f1)
        [ "$load" -lt "${IDLE_LOAD:-6}" ]
    ' 2>/dev/null
}

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
log "waiting for $HOST to go idle (load < ${IDLE_LOAD:-6}), up to ${MAX_WAIT_MIN}m"
until idle; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "still busy after ${MAX_WAIT_MIN}m, giving up without running anything"
        exit 75   # EX_TEMPFAIL: try again later, nothing was done
    fi
    sleep 60
done
log "idle, starting"

ssh -o ConnectTimeout=20 "$HOST" "cd $REMOTE && git -C . rev-parse --short HEAD 2>/dev/null" >/dev/null 2>&1

log "step 1/2 embeddings (bge-m3) and the logistic head"
ssh "$HOST" "bash -lc 'cd $REMOTE && PYTHONPATH=src python3 -m privacy_gate.embed \
    --data data/gold.jsonl --out runs/gold-bge-m3.jsonl'" \
  && ssh "$HOST" "bash -lc 'cd $REMOTE && PYTHONPATH=src python3 -m privacy_gate.head \
    --embeddings runs/gold-bge-m3.jsonl --rules data/rules-baseline.json \
    --out runs/head-bge-m3.json'" | tee runs/head-bge-m3.md
log "step 1 exit ${PIPESTATUS[0]:-?}"

log "step 2/2 chat ceiling ($CEILING_MODEL)"
ssh "$HOST" "bash -lc 'cd $REMOTE && PYTHONPATH=src python3 -m privacy_gate.calibrate \
    --model $CEILING_MODEL --rules data/rules-baseline.json \
    --out runs/cal-ceiling.json'" | tee runs/cal-ceiling.md
log "step 2 exit ${PIPESTATUS[0]:-?}"

log "done. Copy the numbers into docs/JOURNAL.md runs 4 and 5."
