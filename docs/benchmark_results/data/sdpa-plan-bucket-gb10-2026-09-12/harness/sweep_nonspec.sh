#!/usr/bin/env bash
# Issue #1820 non-speculative control: classic decode and a long-prompt prefill
# on a dense head_dim-128 model, which is where the patched dispatch is shared
# with every other CUDA family. Bucketing on and off, same binary.
set -uo pipefail
cd "$(dirname "$0")"
BIN=${BIN:?set BIN}
MODEL=${MODEL:-/home/inureyes/models/mlx/qwen3-1.7b-4bit}
OUT=${OUT:-../nonspec_cli.jsonl}
ROUNDS=${ROUNDS:-3}
PROMPT=${PROMPT:-prompt_code0.txt}
NTOK=${NTOK:-200}
exec python3 bench_cli.py \
  --bin "$BIN" --target "$MODEL" --draft "$MODEL" \
  --prompt-file "$PROMPT" --out "$OUT" --rounds "$ROUNDS" \
  --max-tokens "$NTOK" --configs "off,fb-off" \
  --preset "fb=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0" \
  --tag "${TAG:-nonspec}"
