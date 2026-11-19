#!/usr/bin/env bash
# Rerunnable verification for issue #3033 (Metal query-chunked SDPA fallback).
#
# Runs `mlxcel generate` once per cell and per MLXCEL_METAL_ATTENTION_CHUNK
# value (0 = baseline, exactly main's Metal dispatch; 1 = the fix), greedy
# (temperature 0), and records in <out_dir>/summary.csv:
#   - peak memory footprint of the process, from `/usr/bin/time -l`;
#   - prompt tokens, prefill ms (TTFT of the CLI run) and decode tok/s, from
#     `mlxcel generate --profile`;
#   - how many prefill SDPA calls the Metal gate chunked (debug lines from
#     `RUST_LOG=mlxcel_core::layers=debug` containing "query-chunking");
#   - whether the generated text (reasoning included) is byte-identical
#     between the two runs of the cell.
#
# Usage, from the repository root after `cargo build --release --bin mlxcel`:
#   scripts/bench/metal_sdpa_chunk/verify.sh <out_dir>
#
# Environment:
#   MLXCEL_BIN   binary (default target/release/mlxcel)
#   MODELS_DIR   checkpoint directory (default ./models)
#   CELLS        space-separated model:prompt_tokens:prefill_chunk:budget_mb
#                cells; prefill_chunk 0 is a single-pass prefill, budget "-"
#                keeps the backend default (1024 MiB CUDA, 256 MiB Metal) (default: the 2026-11-14 cells)
#   GEN_TOKENS   greedy tokens per run (default 48)
#
# Needs one idle GPU: run nothing else on it while this runs.
set -euo pipefail

OUT_DIR=${1:?usage: verify.sh <out_dir>}
MLXCEL_BIN=${MLXCEL_BIN:-target/release/mlxcel}
MODELS_DIR=${MODELS_DIR:-models}
GEN_TOKENS=${GEN_TOKENS:-48}
HERE=$(cd "$(dirname "$0")" && pwd)
CELLS=${CELLS:-"gemma-4-12b-it-4bit:28000:0:- gemma-4-12b-it-4bit:28000:2048:- qwen3.5-4b-4bit:28000:0:- qwen3.5-4b-4bit:28000:2048:- qwen3.8-27b-4bit:28000:0:- qwen3.8-27b-4bit:28000:2048:- qwen3.5-4b-4bit:4600:2048:16 gemma-4-12b-it-4bit:4600:2048:16"}

mkdir -p "$OUT_DIR"
if pgrep -x mlxcel-server >/dev/null || pgrep -x mlxcel >/dev/null; then
  echo "another mlxcel process is running; refusing to share the GPU" >&2
  exit 1
fi

SUMMARY="$OUT_DIR/summary.csv"
[ -f "$SUMMARY" ] || echo "model,prompt_target,prefill_chunk,budget_mb,metal_chunk,prompt_tokens,prefill_ms,decode_tok_s,peak_footprint_bytes,chunked_calls,identical_to_baseline,exit" > "$SUMMARY"

# The generated text: everything `generate` prints after the echoed prompt and
# before the profile block.
generated_text() {
  awk '/^\[Profile Results\]/ {exit} found {print} /^Generating\.\.\.$/ {found=1}' "$1"
}

for cell in $CELLS; do
  IFS=: read -r model tokens chunk budget <<<"$cell"
  prompt_file="$OUT_DIR/prompt_${tokens}.txt"
  [ -f "$prompt_file" ] || python3 "$HERE/make_prompt.py" "$tokens" > "$prompt_file"
  base="$OUT_DIR/${model}_${tokens}_pc${chunk}_b${budget}"
  for metal_chunk in 0 1; do
    tag="${base}_mc${metal_chunk}"
    env_args=(MLXCEL_METAL_ATTENTION_CHUNK="$metal_chunk" MLXCEL_PREFILL_CHUNK="$chunk" RUST_LOG=mlxcel_core::layers=debug NO_COLOR=1)
    if [ "$budget" != "-" ]; then env_args+=(MLXCEL_ATTENTION_CHUNK_BUDGET_MB="$budget"); fi
    set +e
    env "${env_args[@]}" /usr/bin/time -l "$MLXCEL_BIN" generate \
      -m "$MODELS_DIR/$model" -p "$(cat "$prompt_file")" -n "$GEN_TOKENS" --temp 0 \
      --profile --show-reasoning > "$tag.out" 2> "$tag.err"
    rc=$?
    set -e
    generated_text "$tag.out" > "$tag.gen"
    peak=$(awk '/peak memory footprint/ {print $1}' "$tag.err" | tail -1)
    chunked=$(grep -c 'query-chunking' "$tag.err" || true)
    ptoks=$(awk '/Prompt tokens:/ {print $3}' "$tag.out")
    pms=$(awk '/Prefill:/ {print $2}' "$tag.out")
    dtps=$(awk '/Decode:/ {gsub(/\(/, "", $4); print $4}' "$tag.out")
    same=-
    if [ "$metal_chunk" = 1 ]; then
      if /usr/bin/cmp -s "${base}_mc0.gen" "$tag.gen"; then same=yes; else same=no; fi
    fi
    echo "$model,$tokens,$chunk,$budget,$metal_chunk,$ptoks,$pms,$dtps,$peak,$chunked,$same,$rc" >> "$SUMMARY"
    echo "$(basename "$tag"): prompt=$ptoks prefill_ms=$pms peak=$peak chunked_calls=$chunked identical=$same rc=$rc"
  done
done
