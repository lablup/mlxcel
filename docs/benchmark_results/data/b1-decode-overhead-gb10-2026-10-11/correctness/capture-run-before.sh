#!/usr/bin/env bash
set -euo pipefail

WT=${WT:-/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227}
OUT=${OUT:-/tmp/mlxcel-auto-plan/2227-run-parity/before}
MODEL=${MODEL:-models/mlx/qwen3-1.7b-4bit}
BIN=${BIN:-$WT/target/release/mlxcel}
PROMPT=${PROMPT:-Reply with exactly one short sentence naming the capital of France.}
MAX_TOKENS=${MAX_TOKENS:-64}

mkdir -p "$OUT"
cd "$WT"
test -x "$BIN"

cp /tmp/mlxcel-auto-plan/2227-profile/instrumentation.patch \
  "$OUT/profile-instrumentation.patch"

{
  date --iso-8601=seconds
  hostname
  uname -a
  printf 'source_head='
  git rev-parse HEAD
  printf 'source_status_begin\n'
  git status --short
  printf 'source_status_end\n'
  sha256sum "$OUT/profile-instrumentation.patch"
  sha256sum "$BIN"
  stat --format='binary_size=%s binary_mtime=%y' "$BIN"
  printf 'mlx_pin='
  scripts/ci/mlx_pinned_commit.sh
  nvidia-smi
} >"$OUT/provenance.txt" 2>&1

printf '%s' "$PROMPT" >"$OUT/prompt.txt"
{
  printf 'MLXCEL_B1_DECODE_PROFILE=<unset> '
  printf 'MLXCEL_DEVICE=gpu MLXCEL_SDPA_DETERMINISTIC=1 '
  printf '%q ' "$BIN" run "$MODEL" -p "$PROMPT" -n "$MAX_TOKENS" \
    --temp 0 --top-k 0 --top-p 1 --min-p 0 --seed 2227 --show-reasoning
  printf '\n'
} >"$OUT/command.txt"

gpu-lock run --tag issue-2227-run-before -- \
  env -u MLXCEL_B1_DECODE_PROFILE \
    MLXCEL_DEVICE=gpu MLXCEL_SDPA_DETERMINISTIC=1 \
    "$BIN" run "$MODEL" -p "$PROMPT" -n "$MAX_TOKENS" \
      --temp 0 --top-k 0 --top-p 1 --min-p 0 --seed 2227 --show-reasoning \
  >"$OUT/stdout.txt" 2>"$OUT/stderr.txt"

python3 /tmp/mlxcel-auto-plan/2227-profile/extract-run-text.py \
  "$OUT/stdout.txt" "$OUT/generated.txt"

sha256sum "$OUT/stdout.txt" "$OUT/stderr.txt" "$OUT/generated.txt" \
  >"$OUT/output-sha256.txt"
