#!/usr/bin/env bash
# Server-path companion to verify.sh (issue #3033): one greedy streaming chat
# request per MLXCEL_METAL_ATTENTION_CHUNK value against `mlxcel serve`, with
# the server's peak memory footprint from `/usr/bin/time -l` (read after the
# server exits), TTFT from server_ttft.py, and the number of SDPA calls the
# Metal gate chunked.
#
# Usage, from the repository root after `cargo build --release --bin mlxcel`:
#   scripts/bench/metal_sdpa_chunk/server_verify.sh <out_dir> <model> [prompt_tokens]
# Environment: MLXCEL_BIN, MODELS_DIR, PORT (default 8110), GEN_TOKENS (48).
set -euo pipefail

OUT_DIR=${1:?usage: server_verify.sh <out_dir> <model> [prompt_tokens]}
MODEL=${2:?model directory name}
TOKENS=${3:-28000}
MLXCEL_BIN=${MLXCEL_BIN:-target/release/mlxcel}
MODELS_DIR=${MODELS_DIR:-models}
PORT=${PORT:-8110}
GEN_TOKENS=${GEN_TOKENS:-48}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT_DIR"
if pgrep -x mlxcel-server >/dev/null || pgrep -x mlxcel >/dev/null; then
  echo "another mlxcel process is running; refusing to share the GPU" >&2
  exit 1
fi
prompt_file="$OUT_DIR/prompt_${TOKENS}.txt"
[ -f "$prompt_file" ] || python3 "$HERE/make_prompt.py" "$TOKENS" > "$prompt_file"

for metal_chunk in 0 1; do
  tag="$OUT_DIR/server_${MODEL}_${TOKENS}_mc${metal_chunk}"
  MLXCEL_METAL_ATTENTION_CHUNK="$metal_chunk" RUST_LOG=mlxcel_core::layers=debug NO_COLOR=1 \
    /usr/bin/time -l "$MLXCEL_BIN" serve -m "$MODELS_DIR/$MODEL" --host 127.0.0.1 --port "$PORT" --parallel 1 \
    > "$tag.server.out" 2> "$tag.server.err" &
  time_pid=$!
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  python3 "$HERE/server_ttft.py" "$PORT" "$prompt_file" "$GEN_TOKENS" "$MODEL" > "$tag.json"
  pkill -TERM -x mlxcel || true
  wait "$time_pid" || true
  peak=$(awk '/peak memory footprint/ {print $1}' "$tag.server.err" | tail -1)
  chunked=$(grep -c 'query-chunking' "$tag.server.err" || true)
  summary=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("ttft_s=%.2f prompt_tokens=%s" % (d["ttft_s"], d["prompt_tokens"]))' "$tag.json")
  echo "$(basename "$tag"): peak=$peak chunked_calls=$chunked $summary"
done
python3 - "$OUT_DIR/server_${MODEL}_${TOKENS}_mc0.json" "$OUT_DIR/server_${MODEL}_${TOKENS}_mc1.json" <<'EOF'
import json, sys
a, b = (json.load(open(p))["text"] for p in sys.argv[1:3])
print("completion identical" if a == b else "COMPLETION DIFFERS")
EOF
