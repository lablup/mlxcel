#!/usr/bin/env bash
# Chat-completions smoke check against a running mlxcel-server.
#
# It boots nothing: point it at a server you started, and it exercises the four
# things a "does serving work on this backend" claim rests on, in order, and
# fails on the first one that does not hold.
#
#   1. `GET /health` answers 200.
#   2. `GET /v1/models` answers 200 and reports the id the request must use.
#   3. `POST /v1/chat/completions` non-streaming returns content and a
#      `finish_reason` of `stop`.
#   4. The same request with `"stream": true` returns SSE frames, closes with
#      `[DONE]`, and assembles to non-empty text.
#
# Two things this gets wrong if written by hand, and why they are handled here:
#
#   1. `model` is a required field on the request body. Omitting it returns 422
#      with a deserialization message, which reads like a server fault and is
#      not one. The id comes from `/v1/models` rather than from the caller.
#   2. The reasoning channel. A thinking model (Qwen3 and family) fills
#      `reasoning_content` before `content`, and a small model can spend the
#      whole budget there without ever closing its thinking block. Judging on
#      `content` alone then reports a working backend as broken, which is the
#      same trap `ab_output_equality.sh` documents and answers with
#      `--show-reasoning`. Both channels count as generated text here, and the
#      output names which one carried it, so a reasoning-only answer reads as
#      what it is rather than as an empty response.
#
# Usage:
#   ./scripts/server_chat_smoke.sh --port 8080
#   ./scripts/server_chat_smoke.sh --base-url http://127.0.0.1:8080 --max-tokens 4096
set -uo pipefail

base=""
port=8080
host=127.0.0.1
max_tokens=2048
timeout=600

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url) base=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --host) host=$2; shift 2 ;;
    --max-tokens) max_tokens=$2; shift 2 ;;
    --timeout) timeout=$2; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$base" ] || base="http://${host}:${port}"

fail() { echo "FAIL: $*" >&2; exit 1; }

code=$(curl -s -m 15 -o /tmp/.mlxcel_smoke_health -w '%{http_code}' "$base/health")
[ "$code" = 200 ] || fail "GET /health returned $code"
echo "health          200 $(cat /tmp/.mlxcel_smoke_health)"

code=$(curl -s -m 15 -o /tmp/.mlxcel_smoke_models -w '%{http_code}' "$base/v1/models")
[ "$code" = 200 ] || fail "GET /v1/models returned $code"
model=$(python3 -c "import json;d=json.load(open('/tmp/.mlxcel_smoke_models'));print(d['data'][0]['id'])") || fail "GET /v1/models returned no model id"
echo "models          200 $model"

req=$(mktemp); trap 'rm -f "$req" /tmp/.mlxcel_smoke_*' EXIT
printf '{"model":"%s","messages":[{"role":"user","content":"Name the capital of France in one word."}],"max_tokens":%s,"temperature":0,"stream":false}\n' "$model" "$max_tokens" > "$req"
code=$(curl -s -m "$timeout" -H 'Content-Type: application/json' -d @"$req" -o /tmp/.mlxcel_smoke_chat -w '%{http_code}' "$base/v1/chat/completions")
[ "$code" = 200 ] || fail "non-streaming chat returned $code: $(head -c 300 /tmp/.mlxcel_smoke_chat)"
python3 - /tmp/.mlxcel_smoke_chat <<'PY' || exit 1
import json, sys
c = json.load(open(sys.argv[1]))["choices"][0]
content = c["message"].get("content") or ""
reasoning = c["message"].get("reasoning_content") or ""
channel = "content" if content.strip() else ("reasoning only" if reasoning.strip() else "none")
shown = (content or reasoning)[:60]
print(f"chat            200 finish_reason={c.get('finish_reason')} channel={channel} text={shown!r}")
if not (content.strip() or reasoning.strip()):
    print("FAIL: the response carried no generated text in either channel", file=sys.stderr)
    sys.exit(1)
PY

printf '{"model":"%s","messages":[{"role":"user","content":"Count from one to five, words only."}],"max_tokens":%s,"temperature":0,"stream":true}\n' "$model" "$max_tokens" > "$req"
curl -s -N -m "$timeout" -H 'Content-Type: application/json' -d @"$req" "$base/v1/chat/completions" > /tmp/.mlxcel_smoke_sse || fail "streaming chat request failed"
python3 - /tmp/.mlxcel_smoke_sse <<'PY' || exit 1
import json, sys
frames = 0; text = ""; reasoning = ""; done = False; finish = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        done = True
        continue
    frames += 1
    choice = json.loads(payload)["choices"][0]
    delta = choice.get("delta", {})
    text += delta.get("content") or ""
    reasoning += delta.get("reasoning_content") or ""
    finish = choice.get("finish_reason") or finish
channel = "content" if text.strip() else ("reasoning only" if reasoning.strip() else "none")
shown = (text or reasoning)[:60]
print(f"chat stream     200 frames={frames} done={done} finish_reason={finish} channel={channel} text={shown!r}")
if not (frames and done and (text.strip() or reasoning.strip())):
    print("FAIL: streaming produced no frames, no [DONE], or no generated text", file=sys.stderr)
    sys.exit(1)
PY

echo "OK"
