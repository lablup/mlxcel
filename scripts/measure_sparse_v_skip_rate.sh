#!/usr/bin/env bash
# Measure the sparse-V post-softmax attention skip rate at a set of decode
# contexts and print a per-layer table (#377).
#
# For each context, the script seeds a prompt to about that many tokens, runs a
# short Turbo4Asym decode with MLXCEL_SPARSE_V_COUNT enabled, and aggregates the
# per-call CSV into overall and per-layer (min / max / median) skip rates. The
# skip rate is the fraction of post-softmax attention weights below
# MLXCEL_SPARSE_V_THRESHOLD (default 1e-6) across all query heads and key
# positions, the same quantity the upstream TurboQuant+ sparse-V paper reports.
#
# The asym decode hook that records the skip rate lives in the dense Qwen3 and
# Llama3 attention paths, so pick a model served by one of those (the paper used
# a dense Qwen3-1.7B for its attention inspection; skip rate is a property of the
# attention distribution, not of MoE routing).
#
# Usage:
#   ./scripts/measure_sparse_v_skip_rate.sh [model_dir] [--contexts 8192,16384,32768] [--decode-tokens 8]

set -uo pipefail

MLXCEL="./target/release/mlxcel"
MODEL="models/qwen3-4b-4bit"
CONTEXTS=(8192 16384 32768)
DECODE_TOKENS=8
OUTDIR="benchmarks/sparse_v_skip"

PARAGRAPH="The TurboQuant KV cache speed gate matrix benchmarks decode and prefill throughput across cache quantization modes on Apple Silicon to validate the per-config compression ratio targets from epic 458 against a fixed-prompt fixed-decode workload that mirrors production-scale inference."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --contexts) IFS=',' read -ra CONTEXTS <<< "$2"; shift 2 ;;
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) MODEL="$1"; shift ;;
  esac
done

[[ -x "$MLXCEL" ]] || { echo "Error: $MLXCEL not found; run 'cargo build --release --features metal,accelerate'." >&2; exit 1; }
[[ -d "$MODEL" ]]   || { echo "Error: model dir $MODEL not present." >&2; exit 1; }

NUM_LAYERS="$(python3 -c "import json,sys; print(json.load(open('$MODEL/config.json'))['num_hidden_layers'])")"
mkdir -p "$OUTDIR"

build_prompt() {
  local reps=$(( ($1 + 49) / 50 )) out="" i
  for ((i = 0; i < reps; i++)); do out+="${PARAGRAPH} "; done
  printf '%s' "$out"
}

echo "model=$MODEL num_layers=$NUM_LAYERS contexts=${CONTEXTS[*]} decode_tokens=$DECODE_TOKENS"
for ctx in "${CONTEXTS[@]}"; do
  csv="$OUTDIR/skip_$(basename "$MODEL")_${ctx}.csv"
  prompt="$(build_prompt "$ctx")"
  >&2 echo "[run] ctx=$ctx -> $csv"
  MLXCEL_SPARSE_V_COUNT="$csv" "$MLXCEL" generate -m "$MODEL" -p "$prompt" \
    -n "$DECODE_TOKENS" --kv-cache-mode turbo4-asym >/dev/null 2>&1 || true
  python3 - "$csv" <<'PY'
import csv, statistics, sys
from collections import defaultdict
# A decode step shares one kv_tokens value across all of its layer calls and
# advances by one token between steps, so we group by step and use the call's
# position within a step as the layer index. This is robust to how many layers
# actually use the Turbo4Asym cache (boundary-V layers stay fp16 and are absent).
rows = list(csv.DictReader(open(sys.argv[1])))
steps = defaultdict(list)
for r in rows:
    steps[int(r["kv_tokens"])].append((int(r["skipped"]), int(r["total"])))
n_layers = max((len(v) for v in steps.values()), default=0)
per_layer_skipped = [0]*n_layers
per_layer_total   = [0]*n_layers
for calls in steps.values():
    for layer, (sk, tot) in enumerate(calls):
        per_layer_skipped[layer] += sk
        per_layer_total[layer]   += tot
rates = [100.0*s/t for s, t in zip(per_layer_skipped, per_layer_total) if t]
overall = 100.0*sum(per_layer_skipped)/max(1, sum(per_layer_total))
ctx = max(steps) if steps else 0
if rates:
    print(f"  kv_tokens~{ctx:>6}  layers={n_layers}  steps={len(steps)}  "
          f"overall={overall:5.1f}%  per-layer min={min(rates):5.1f}%  "
          f"median={statistics.median(rates):5.1f}%  max={max(rates):5.1f}%")
else:
    print(f"  kv_tokens~{ctx:>6}: no records")
PY
done
