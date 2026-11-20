#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Smoke-test a built privacy-gate image: it runs as non-root, has CPU-only torch, answers `info`, becomes healthy
# by its own HEALTHCHECK, and scores a /check request with no network. Used by the PR image job and by the
# release workflow before it pushes. Usage: .github/scripts/smoke-image.sh <image>
set -euo pipefail

image="${1:?usage: smoke-image.sh <image>}"
name="pg-smoke-$$"
port="${SMOKE_PORT:-8231}"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && docker inspect "$name" >/dev/null 2>&1; then
    echo "--- container log"
    docker logs "$name" 2>&1 | tail -n 50 || true
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

fail() { echo "::error::$*"; exit 1; }

echo "--- image $image"
docker image inspect "$image" --format 'size {{.Size}} bytes, user {{.Config.User}}'

user="$(docker image inspect "$image" --format '{{.Config.User}}')"
case "${user%%:*}" in
  "" | 0 | root) fail "the image runs as root (USER is '${user}')" ;;
esac
uid="$(docker run --rm --entrypoint id "$image" -u)"
[ "$uid" != 0 ] || fail "the container runs as uid 0"
echo "non-root: uid $uid"

docker run --rm --network none --entrypoint python "$image" -c \
  "import torch; assert torch.version.cuda is None, torch.version.cuda; print('torch', torch.__version__, 'CPU only')"

# The CLI answers with no network, and its exit code is checked by `set -e`.
docker run --rm --network none --entrypoint privacy-gate "$image" info

docker run -d --name "$name" -p "127.0.0.1:${port}:8231" "$image" >/dev/null

# Wait for the image's own HEALTHCHECK (start period 60 s, interval 30 s) to report healthy.
health=starting
for _ in $(seq 1 60); do
  health="$(docker inspect "$name" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}')"
  running="$(docker inspect "$name" --format '{{.State.Running}}')"
  [ "$running" = true ] || fail "the container exited: $(docker inspect "$name" --format '{{.State.ExitCode}}')"
  [ "$health" = healthy ] && break
  [ "$health" = unhealthy ] && fail "HEALTHCHECK reports unhealthy"
  sleep 5
done
[ "$health" = healthy ] || fail "not healthy after 300 s (status: $health)"
echo "HEALTHCHECK: healthy"

curl -fsS "http://127.0.0.1:${port}/health"
echo

# Scores one invented sentence: proves the bundled encoder loads offline and the head answers. A 503 here is
# the fail-closed `backend_unavailable` answer, which means the encoder did not load.
response="$(curl -fsS "http://127.0.0.1:${port}/check" -d '{"text":"the quarterly report is due on friday"}')"
echo "$response"
echo "$response" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert isinstance(r.get("hold"), bool) and isinstance(r.get("score"), float), r
'
echo "smoke test passed"
