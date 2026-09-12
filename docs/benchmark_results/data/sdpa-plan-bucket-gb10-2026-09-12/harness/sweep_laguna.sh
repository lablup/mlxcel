#!/usr/bin/env bash
# Issue #1820 Laguna DFlash A/B: #1817 state (bucketing off) against bucketing on.
#
# Same binary on every arm. `fb-*` sets MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0,
# which restores exactly the dispatch on main (#1817's fallback claims the
# masked verify block); the unprefixed arms use the shipped default. The
# classic arm is measured on both settings, since it must not move.
set -euo pipefail
cd "$(dirname "$0")"
BIN=${BIN:?set BIN to the mlxcel binary under test}
OUT=${OUT:-../laguna_cli.jsonl}
ROUNDS=${ROUNDS:-3}
CONFIGS=${CONFIGS:-off,fb-off,b2,fb-b2,b4,fb-b4,b6,fb-b6,b8,fb-b8,b16,fb-b16}
exec python3 bench_cli.py \
  --bin "$BIN" \
  --target /home/inureyes/models/mlx/laguna-xs-2.1-nvfp4 \
  --draft /home/inureyes/models/mlx/laguna-xs-2.1-dflash \
  --prompt-file prompt_code0.txt \
  --out "$OUT" \
  --rounds "$ROUNDS" \
  --configs "$CONFIGS" \
  --preset "fb=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0" \
  --preset "up=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0,MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0" \
  --tag "${TAG:-1820}"
