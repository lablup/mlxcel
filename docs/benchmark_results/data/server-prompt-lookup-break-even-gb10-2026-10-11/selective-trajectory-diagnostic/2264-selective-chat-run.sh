#!/usr/bin/env bash
set -euo pipefail
cd /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2264
D=docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11
OUT=/tmp/mlxcel-auto-plan
[[ ! -e "$OUT/2264-selective-chat-client.jsonl" ]]
if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then exit 1; fi
{
 date -u +%FT%TZ
 git rev-parse HEAD
 sha256sum target/test-fast/mlxcel-server "$OUT/2264-prompt-lookup-trajectory-after-fix.patch"
} > "$OUT/2264-selective-chat-environment.txt"
MLXCEL_PROMPT_LOOKUP_TRACE=1 target/test-fast/mlxcel-server -m models/mlx/qwen3-1.7b-4bit --port 18080 --metrics --draft-kind prompt-lookup > "$OUT/2264-selective-chat-server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((i=0;i<240;i++)); do
 kill -0 "$server_pid" 2>/dev/null || exit 1
 if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then ready=1; break; fi
 sleep 1
done
[[ $ready == 1 ]]
python3 -I -B "$D/m_client.py" 18080 copy 1 1 32 > "$OUT/2264-selective-chat-warmup-copy.jsonl"
python3 -I -B "$D/m_client.py" 18080 chat 1 1 32 > "$OUT/2264-selective-chat-warmup-chat.jsonl"
python3 -I -B "$D/m_chat.py" 18080 256 > "$OUT/2264-selective-chat-client.jsonl"
