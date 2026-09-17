#!/usr/bin/env bash
# Issue #1820 context ladder: does bucketed cuDNN overtake #1799's ops fallback
# as the key length grows? The fallback materializes a [B, heads, q_len, k_len]
# score matrix and cuDNN's flash kernel does not, so #1799 predicted a crossover
# somewhere past a few hundred keys. The short-prompt table found none, so this
# walks the context up until one appears or the range is exhausted.
#
# Only 10 of the pairing's 45 attention calls actually scale with the prompt:
# the target's 10 full-attention layers. Its 30 sliding layers are capped by a
# 512 window, and so are all 5 drafter layers (the drafter's config.json is
# `sliding_window: 512` with every layer `sliding_attention`), so the effect is
# diluted by design and a crossover has to be real to show end to end.
set -uo pipefail
cd "$(dirname "$0")"
BIN=${BIN:?set BIN}
OUT=${OUT:-../laguna_ladder_cli.jsonl}
ROUNDS=${ROUNDS:-3}
NTOK=${NTOK:-150}
PROMPT=${PROMPT:?set PROMPT to a prompt file}
CONFIGS=${CONFIGS:-b4,fb-b4,b8,fb-b8}
exec python3 bench_cli.py \
  --bin "$BIN" \
  --target /home/inureyes/models/mlx/laguna-xs-2.1-nvfp4 \
  --draft /home/inureyes/models/mlx/laguna-xs-2.1-dflash \
  --prompt-file "$PROMPT" --out "$OUT" --rounds "$ROUNDS" \
  --max-tokens "$NTOK" --warmup "${WARMUP:-1}" --configs "$CONFIGS" \
  --preset "fb=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0" \
  --preset "up=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0,MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0" \
  --tag "${TAG:-ladder}"
