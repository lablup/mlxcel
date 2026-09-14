#!/usr/bin/env bash
# Counter-based A/B gate for the batched paged decode path.
#
# `scripts/ab_output_equality.sh` answers "does the arm still produce the same
# text". That is the right question everywhere the text can move. It is the
# wrong question here, and this script exists because the difference is not
# obvious from the outside.
#
# `mlxcel generate` never reaches the paged bridges at all. The production route
# is
#
#   models/llama3.rs -> cache::paged_batch_decode_attention
#     -> PagedBlockPool::paged_decode_batched
#       -> launch_v2 / launch_cascade
#         -> paged_attention_decode_v2_partial, paged_attention_merge_states
#
# and every gate on it decides between the fused kernel and a gather fallback
# that computes the same thing more slowly. So a change that silently disables
# the fused path produces byte-identical output and a dead kernel, and an output
# comparison reports EQUAL and calls it a pass. Issue #1803 added exactly such a
# gate at the head of `paged_decode_batched`, which is what this was written for.
#
# The verdict is therefore the counters the server exports, not the text. The
# text is still compared, because a difference there would matter too.
#
# Four things have to be right or the run is vacuous. Each of them cost a re-run
# when the Metal gate for #1803 was measured; see
# `docs/benchmark_results/kernel-backend-kind-metal-m1ultra-2026-09-13.md`.
#
#   1. The paged backend has to actually be selected. `--decode-storage-backend
#      paged` does it. `--kv-unified` does NOT: that flag is read by the
#      KV-budget code, never by the batch scheduler, whose choice is made in
#      `effective_decode_storage_backend`.
#   2. The dispatch floor has to be bypassed. Below 4096 visible KV tokens the
#      planner declines and BOTH sides take gather, so the run compares gather
#      against gather. `MLXCEL_PAGED_ATTENTION_NATIVE=1` forces the fused path
#      for every shape the kernel can actually serve; the kernel's own
#      structural declines still apply, so this cannot manufacture a launch that
#      would be rejected.
#   3. `/metrics` is off unless `--metrics` is passed, and `curl -f` fails
#      silently, so a missing counter file looks like a zero rather than an
#      error. This script fails loudly instead.
#   4. The discriminator is `gather_fallbacks`, not `declines`. A gate inside
#      `paged_decode_batched` returns `Ok((None, ...))`, which the caller folds
#      into `gather_fallbacks`. `declines` counts only rejections made before
#      the pool is touched, and it is not exported to the gauges at all.
#
# The `report_once` log line is corroboration, not the verdict: it prints once
# per outcome KIND, and `NotServable` is a single kind covering every reason, so
# a batched prefill rejected as "not a single-token decode step" can take that
# slot and a later rejection will never print.
#
# The model has to be one whose family opts into `supports_paged_decode_backend()`
# (llama3, qwen3, qwen3_5, gemma3, helium, llama4). Qwen3 MoE does not.
#
# Usage:
#   ./scripts/paged_decode_counter_ab.sh --baseline target/release/mlxcel.before \
#                                        --arm target/release/mlxcel \
#                                        --model models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
#
# Exit status: 0 both sides launched the fused path the same number of times
# with no fallbacks and produced the same text; 1 the counters disagree, a side
# never launched the fused path, or the text differs; 2 the run could not be
# made meaningful (server never came up, metrics unavailable).
set -uo pipefail

BASELINE_BIN=""
ARM_BIN="./target/release/mlxcel"
MODEL=""
PORT=18803
NREQ=4
MAX_TOKENS=64
MAX_BATCH=4
PROMPT="Explain mixture-of-experts routing in two sentences."
OUT_DIR=""

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE_BIN="$2"; shift 2 ;;
    --arm) ARM_BIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --requests) NREQ="$2"; shift 2 ;;
    --max-batch-size) MAX_BATCH="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --prompt-file) PROMPT="$(cat "$2")"; shift 2 ;;
    -n|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$BASELINE_BIN" ]] || { echo "error: --baseline is required" >&2; exit 1; }
[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; exit 1; }
for bin in "$BASELINE_BIN" "$ARM_BIN"; do
  [[ -x "$bin" ]] || { echo "error: not an executable: $bin" >&2; exit 1; }
done
if cmp -s "$BASELINE_BIN" "$ARM_BIN"; then
  echo "error: --baseline and --arm hold identical bytes; the arm was probably never rebuilt." >&2
  echo "       Check the build log for the translation units you edited." >&2
  exit 1
fi

[[ -n "$OUT_DIR" ]] || OUT_DIR="$(mktemp -d -t paged_decode_counter_ab)"
mkdir -p "$OUT_DIR"

echo "baseline: $BASELINE_BIN"
echo "arm:      $ARM_BIN"
echo "model:    $MODEL"
echo "requests: $NREQ x $MAX_TOKENS tokens, temperature 0, batch cap $MAX_BATCH"
echo "outputs:  $OUT_DIR"
echo

# One side: start the server, drive it, scrape the counters, stop it.
run_side() {
  local label="$1" bin="$2"
  local dir="$OUT_DIR/$label"
  rm -rf "$dir"; mkdir -p "$dir"

  RUST_LOG=info MLXCEL_DEBUG_KERNEL_BACKEND=1 MLXCEL_PAGED_ATTENTION_NATIVE=1 \
    "$bin" serve -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
      --decode-storage-backend paged --max-batch-size "$MAX_BATCH" --metrics \
      > "$dir/server.log" 2>&1 &
  local pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" -o "$dir/health.json" 2>/dev/null; then
      ready=1; break
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [[ $ready -ne 1 ]]; then
    echo "[$label] server never became ready; see $dir/server.log" >&2
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 2
  fi

  local model_name
  model_name="$(basename "$MODEL")"
  local i
  for i in $(seq 1 "$NREQ"); do
    if ! jq -n --arg m "$model_name" --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
         '{model:$m, prompt:$p, max_tokens:$n, temperature:0, stream:false}' \
         | curl -fsS "http://127.0.0.1:$PORT/v1/completions" \
             -H 'Content-Type: application/json' --data-binary @- \
             -o "$dir/resp-$i.json"; then
      echo "[$label] request $i failed" >&2
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      return 2
    fi
  done

  if ! curl -fsS "http://127.0.0.1:$PORT/metrics" -o "$dir/metrics.txt"; then
    echo "[$label] /metrics unavailable even with --metrics; cannot judge this run" >&2
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 2
  fi

  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

  for i in $(seq 1 "$NREQ"); do
    jq -r '.choices[0].text' "$dir/resp-$i.json" >> "$dir/texts.txt"
  done
  return 0
}

# `mlxcel_paged_decode_launches_total{path="NAME"} VALUE` -> VALUE, or empty.
counter() {
  local dir="$1" path="$2"
  awk -v p="$path" '
    $0 ~ "^mlxcel_paged_decode_launches_total\\{path=\"" p "\"\\}" { print $NF; found = 1 }
    END { if (!found) print "" }
  ' "$OUT_DIR/$dir/metrics.txt"
}

report_side() {
  local label="$1" dir="$OUT_DIR/$1"
  echo "--- $label ---"
  printf '  fused_v2=%s gather=%s cascade=%s\n' \
    "$(counter "$label" fused_v2)" "$(counter "$label" gather)" "$(counter "$label" cascade)"
  echo "  backend: $(grep -m1 'custom kernel backend' "$dir/server.log" 2>/dev/null || echo '<probe absent: pre-#1803 binary>')"
  local outcomes
  outcomes="$(grep -o 'paged decode v2: .*' "$dir/server.log" 2>/dev/null | sort -u)"
  echo "  outcomes (corroboration only, one line per kind):"
  if [[ -n "$outcomes" ]]; then printf '    %s\n' "$outcomes"; else echo "    <none logged>"; fi
}

run_side baseline "$BASELINE_BIN"; rc=$?; [[ $rc -eq 0 ]] || exit $rc
run_side arm      "$ARM_BIN";      rc=$?; [[ $rc -eq 0 ]] || exit $rc

report_side baseline
report_side arm
echo

base_fused="$(counter baseline fused_v2)"
arm_fused="$(counter arm fused_v2)"
base_gather="$(counter baseline gather)"
arm_gather="$(counter arm gather)"

fail=0

if [[ -z "$base_fused" || -z "$arm_fused" ]]; then
  echo "INCONCLUSIVE: the fused_v2 counter is missing from at least one side's /metrics."
  exit 2
fi
if [[ "$base_fused" -eq 0 || "$arm_fused" -eq 0 ]]; then
  # Both sides on gather is the vacuous run this script exists to refuse: it
  # compares the fallback against itself and reports agreement.
  echo "INCONCLUSIVE: fused_v2 is zero on at least one side, so the fused path never ran."
  echo "              Check the outcome lines above for a dispatch-floor or servability decline."
  exit 2
fi
if [[ "$base_fused" -ne "$arm_fused" ]]; then
  echo "DIFFERS: fused_v2 $base_fused (baseline) vs $arm_fused (arm)."
  fail=1
fi
if [[ "$base_gather" -ne "$arm_gather" ]]; then
  echo "DIFFERS: gather $base_gather (baseline) vs $arm_gather (arm)."
  echo "         A gather count that rises only on the arm is a fused path the arm disabled;"
  echo "         the text can still be identical, which is why this is the discriminator."
  fail=1
fi

if cmp -s "$OUT_DIR/baseline/texts.txt" "$OUT_DIR/arm/texts.txt"; then
  echo "text: EQUAL"
else
  echo "text: DIFFERS"
  diff "$OUT_DIR/baseline/texts.txt" "$OUT_DIR/arm/texts.txt" | head -20
  fail=1
fi

echo
if [[ $fail -ne 0 ]]; then
  echo "result: the two arms do not agree. Saved under $OUT_DIR."
  exit 1
fi
echo "result: both arms launched the fused path $arm_fused times with $arm_gather fallbacks, same text."
