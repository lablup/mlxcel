#!/usr/bin/env bash
# Issue #1820, the decisive measurement: walk the prompt length up until the
# bucketed cuDNN path overtakes #1799's ops fallback, or until the range is
# exhausted. #1799 predicted a crossover because the fallback materializes a
# `[B, heads, q_len, k_len]` score matrix and cuDNN's flash kernel does not.
#
# Run after a rebase onto current main, on the rebased binary, so every rung is
# the same binary and the same base tree. Rungs are walked short to long, so a
# host problem that stops the sweep leaves the cheap rungs complete rather than
# the expensive ones half done.
#
# Only 15 of the pairing's 45 attention calls scale with the prompt (the
# target's 10 full-attention layers and the drafter's 5); its 30 sliding layers
# are capped by a 512 window. The effect is diluted by design, so a crossover
# has to be real to show in end-to-end throughput.
set -uo pipefail
cd "$(dirname "$0")"
BIN=${BIN:?set BIN to the mlxcel binary under test}
OUT=${OUT:-../laguna_ladder_rebased.jsonl}
ROUNDS=${ROUNDS:-3}
NTOK=${NTOK:-150}
RUNGS=${RUNGS:-prompt_code_long.txt prompt_code_8k.txt prompt_code_16k.txt prompt_code_32k.txt}

for p in $RUNGS; do
  # The `up` arm (upstream dispatch: neither fix active, so cuDNN rebuilds a
  # plan every round) is measured at 16k only. It separates "what the plan
  # rebuild costs" from "which kernel runs", and at 150 tokens it stays under
  # the lifetime-miss abort. Running it at every rung would double the sweep
  # for a term that is already constant in the key length.
  cfgs="b4,fb-b4,b8,fb-b8"
  [ "$p" = "prompt_code_16k.txt" ] && cfgs="b4,fb-b4,b8,fb-b8,up-b4,up-b8"
  echo "=== rung $p (configs $cfgs) ===" >&2
  BIN="$BIN" OUT="$OUT" ROUNDS="$ROUNDS" NTOK="$NTOK" PROMPT="$p" \
    CONFIGS="$cfgs" TAG="ladder-rebased" ./sweep_context_ladder.sh || echo "rung $p rc=$?" >&2
done
echo "=== ladder complete ===" >&2
