#!/usr/bin/env bash
# Issue #1820, the one rung that can still change the recommendation.
#
# 2634 is a wash and 152 measures behind, so if bucketing has a win anywhere it
# is here, where the ops fallback's `[B, heads, q_len, k_len]` score matrix is
# large and cuDNN's flash kernel is not. 8k and 32k are deliberately skipped:
# 8k sits between two rungs that already agree, and 32k only confirms a trend
# while carrying the ladder's largest allocation footprint, which is what the
# driver cannot currently afford.
#
# One invocation per round rather than one per rung, so a halt costs at most
# one round and the caller commits between them. The configuration order
# rotates per round, which is what the earlier rungs did inside a single
# `bench_cli` call; reproducing it here keeps drift spread across the arms
# instead of pooling on whichever one runs last, and keeps the numbers
# comparable to the committed 2634 rung.
#
# Every run is gated on the driver by `bench_cli` itself (window and cumulative
# both), so a trip wire during the rung stops it with the rows so far intact.
# Exit 3 means the gate halted the round; anything else is that round's own rc.
set -uo pipefail
cd "$(dirname "$0")"
BIN=${BIN:?set BIN}
OUT=${OUT:-../laguna_ladder_rebased.jsonl}
ROUND=${ROUND:?set ROUND to 1, 2 or 3}
NTOK=${NTOK:-150}

# Rotating start, matching bench_cli's own round-robin rotation.
case "$ROUND" in
  1) CFGS="b4,fb-b4,b8,fb-b8"; WARM=1 ;;
  2) CFGS="fb-b4,b8,fb-b8,b4"; WARM=0 ;;
  3) CFGS="b8,fb-b8,b4,fb-b4"; WARM=0 ;;
  *) echo "ROUND must be 1, 2 or 3" >&2; exit 2 ;;
esac

exec python3 bench_cli.py \
  --bin "$BIN" \
  --target /home/inureyes/models/mlx/laguna-xs-2.1-nvfp4 \
  --draft /home/inureyes/models/mlx/laguna-xs-2.1-dflash \
  --prompt-file prompt_code_16k.txt --out "$OUT" --rounds 1 \
  --max-tokens "$NTOK" --warmup "$WARM" --configs "$CFGS" \
  --preset "fb=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0" \
  --preset "up=MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0,MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0" \
  --tag "ladder-16k-r$ROUND"
