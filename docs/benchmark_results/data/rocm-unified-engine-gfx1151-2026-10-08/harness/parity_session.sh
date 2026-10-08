#!/usr/bin/env bash
# ROCm parity session for issue #2192: per model, (1) the ADR 0007 harness
# `mlxcel-engine-parity` (DirectEngine vs server dense vs server paged, greedy
# and seeded, prompt-cache miss vs hit), (2) `mlxcel generate` vs `mlxcel run`
# vs a dense-storage `mlxcel serve` on one raw greedy prompt, (3) the
# teacher-forced decode-step trace of a dense server against a paged server
# (paged_vs_dense_trace.py + scripts/compare_logit_traces.py).
#
# Usage: parity_session.sh REPO_ROOT OUT_DIR PROMPT_FILE MODELS_DIR model...
# Run under scripts/rocm_gpu_guard.sh. Servers are stopped by the pid this
# script recorded, never by pattern.
set -uo pipefail
ROOT="$1"; OUT="$2"; PROMPT_FILE="$3"; MODELS_DIR="$4"; shift 4
mkdir -p "$OUT"
BIN="$ROOT/target/release"
N_TOKENS="${N_TOKENS:-64}"
TRACE_N="${TRACE_N:-128}"
PORT_A="${PORT_A:-18601}"
PORT_B="${PORT_B:-18602}"
# `$(cat)` strips trailing newlines; the server requests strip them too so
# every arm sees the same prompt text.
# Below 4096 visible KV tokens a lone paged request decodes through the
# gather-then-SDPA fallback (MLXCEL_PAGED_V2_MIN_KV_TOKENS), which is the
# dense arithmetic again. PAGED_FLOOR=0 (the default here) lowers the floor so
# the paged arms run the HIP paged v2 kernel on the short prompt, the path a
# long-context request takes; PAGED_FLOOR=default keeps the shipped floor.
PAGED_FLOOR="${PAGED_FLOOR:-0}"
if [[ "$PAGED_FLOOR" != default ]]; then export MLXCEL_PAGED_V2_MIN_KV_TOKENS="$PAGED_FLOOR"; fi
PROMPT="$(cat "$PROMPT_FILE")"
PROMPT_JSON_FILE="$OUT/prompt.txt"; printf '%s' "$PROMPT" > "$PROMPT_JSON_FILE"

start_server() { # model storage port logfile -> pid
  local model="$1" storage="$2" port="$3" log="$4"
  # An explicit paged backend needs more than one slot (`effective_decode_storage_backend`
  # falls back to dense at max_batch_size 1); one request at a time keeps B=1.
  local slots=1; [[ "$storage" == paged ]] && slots=2
  RUST_LOG=info "$BIN/mlxcel" serve -m "$MODELS_DIR/$model" --parallel "$slots" \
    --decode-storage-backend "$storage" --no-prompt-cache \
    --host 127.0.0.1 --port "$port" > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { echo "$pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo ""; return 1; }
    sleep 1
  done
  echo ""; return 1
}

stop_server() { # pid
  [[ -n "$1" ]] || return 0
  kill "$1" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null || true
}

generated_text() { # extract the continuation `generate` / `run` printed
  # Everything after the "Generating..." line up to the "[Generated" line.
  awk '/^Generating\.\.\./{on=1; next} /^\[Generated /{on=0} on' "$1"
}

for model in "$@"; do
  M="$OUT/$model"; mkdir -p "$M"
  echo "=== $model ($(date -u +%H:%M:%S))"

  # (1) ADR 0007 harness.
  MLXCEL_SDPA_DETERMINISTIC=1 "$BIN/mlxcel-engine-parity" --model "$MODELS_DIR/$model" \
    -n "$N_TOKENS" --json "$M/engine_parity.json" > "$M/engine_parity.txt" 2> "$M/engine_parity.err"
  echo "engine-parity rc=$?" | tee -a "$M/engine_parity.txt"

  # (2) CLI clients on one raw greedy prompt.
  "$BIN/mlxcel" generate -m "$MODELS_DIR/$model" -p "$PROMPT" -n "$N_TOKENS" --temp 0 --no-chat-template \
    > "$M/cli_generate.out" 2> "$M/cli_generate.err"
  "$BIN/mlxcel" run "$MODELS_DIR/$model" -p "$PROMPT" -n "$N_TOKENS" --temp 0 --no-chat-template \
    > "$M/cli_run.out" 2> "$M/cli_run.err"
  generated_text "$M/cli_generate.out" > "$M/cli_generate.text"
  generated_text "$M/cli_run.out" > "$M/cli_run.text"

  # (3) dense server: reference trace and the raw greedy completion.
  pid_a=$(start_server "$model" dense "$PORT_A" "$M/server_dense.log")
  if [[ -z "$pid_a" ]]; then echo "dense server failed to start"; continue; fi
  curl -s "http://127.0.0.1:$PORT_A/completion" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": open(sys.argv[1]).read(), "n_predict": int(sys.argv[2]), "temperature": 0, "cache_prompt": False, "return_tokens": True}))' "$PROMPT_JSON_FILE" "$N_TOKENS")" \
    > "$M/server_dense_completion.json"
  python3 "$ROOT/docs/benchmark_results/data/rocm-unified-engine-gfx1151-2026-10-08/harness/paged_vs_dense_trace.py" \
    --url "http://127.0.0.1:$PORT_A" --prompt-file "$PROMPT_JSON_FILE" --n "$TRACE_N" --n-probs 10 \
    --out "$M/trace_dense.tsv" --label dense 2> "$M/trace_dense.err"
  curl -s "http://127.0.0.1:$PORT_A/metrics" > "$M/server_dense_metrics.txt"
  stop_server "$pid_a"

  # (3b) paged server: candidate trace and the raw greedy completion.
  pid_b=$(start_server "$model" paged "$PORT_B" "$M/server_paged.log")
  if [[ -z "$pid_b" ]]; then echo "paged server failed to start"; continue; fi
  curl -s "http://127.0.0.1:$PORT_B/completion" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": open(sys.argv[1]).read(), "n_predict": int(sys.argv[2]), "temperature": 0, "cache_prompt": False, "return_tokens": True}))' "$PROMPT_JSON_FILE" "$N_TOKENS")" \
    > "$M/server_paged_completion.json"
  python3 "$ROOT/docs/benchmark_results/data/rocm-unified-engine-gfx1151-2026-10-08/harness/paged_vs_dense_trace.py" \
    --url "http://127.0.0.1:$PORT_B" --prompt-file "$PROMPT_JSON_FILE" --n "$TRACE_N" --n-probs 10 \
    --ref "$M/trace_dense.tsv" --out "$M/trace_paged.tsv" --label paged 2> "$M/trace_paged.err"
  curl -s "http://127.0.0.1:$PORT_B/metrics" > "$M/server_paged_metrics.txt"
  stop_server "$pid_b"

  python3 "$ROOT/scripts/compare_logit_traces.py" "$M/trace_dense.decode_only.tsv" "$M/trace_paged.decode_only.tsv" \
    > "$M/compare_dense_vs_paged.txt" 2>&1 || true
  python3 "$ROOT/scripts/compare_logit_traces.py" "$M/trace_dense.tsv" "$M/trace_paged.tsv" \
    > "$M/compare_dense_vs_paged_all_rows.txt" 2>&1 || true

  # Summary lines.
  {
    echo "model: $model"
    python3 - "$M" <<'PY'
import json, sys, pathlib
m = pathlib.Path(sys.argv[1])
def content(p):
    try:
        return json.loads((m / p).read_text()).get("content")
    except Exception as e:
        return f"<error {e}>"
# `generate` echoes the raw prompt before the continuation; `run` prints the
# continuation only, so `run`'s text must be the tail of `generate`'s.
gen = (m / "cli_generate.text").read_text().rstrip("\n")
run = (m / "cli_run.text").read_text().rstrip("\n")
d = content("server_dense_completion.json"); pg = content("server_paged_completion.json")
print("generate vs run text:", "identical" if run and gen.endswith(run) else "DIFFER")
print("run vs server dense text:", "identical" if d is not None and run.strip() == d.strip() else "DIFFER")
print("server dense vs server paged text:", "identical" if d == pg else "DIFFER")
PY
    grep -h "first_divergence\|restarts" "$M/trace_paged.tsv" | head -2
    grep -h "top-1 disagreement\|decided positions" "$M/compare_dense_vs_paged.txt" | head -2
    grep -h "paged decode\|fused v2\|decode storage\|storage" "$M/server_paged.log" | sort | uniq -c | sort -rn | head -5
    grep -h "paged_decode\|lookahead_steps_total\|decode_storage" "$M/server_paged_metrics.txt" | grep -v "^#" | head -8
  } | tee "$M/summary.txt"
done
echo "session done ($(date -u +%H:%M:%S))"
