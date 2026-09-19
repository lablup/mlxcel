#!/usr/bin/env bash
# Phase A/B: per-call cuDNN SDPA key-field trace on the Laguna DFlash pairing.
# $1 = tag, rest = extra env assignments.
set -uo pipefail
WT=/home/inureyes/Development/wt-1820-sdpa-plan-cache
SP=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
D=$WT/docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12
TAG=$1; shift
BLOCK=${BLOCK:-4}
NTOK=${NTOK:-60}
env MLXCEL_SDPA_PLAN_DEBUG=1 MLXCEL_MTP_ALLOW_INEXACT=1 "$@" \
  "$WT/target/release/mlxcel" generate \
  -m /home/inureyes/models/mlx/laguna-xs-2.1-nvfp4 \
  --draft-model /home/inureyes/models/mlx/laguna-xs-2.1-dflash \
  --draft-kind dflash --draft-block-size "$BLOCK" \
  -p "$(cat "$D/harness/prompt_code0.txt")" -n "$NTOK" --temp 0 \
  > "$SP/trace_$TAG.out" 2> "$SP/trace_$TAG.err"
echo "rc=$? lines=$(grep -c mlxcel-sdpa "$SP/trace_$TAG.err")"
tail -3 "$SP/trace_$TAG.out"
