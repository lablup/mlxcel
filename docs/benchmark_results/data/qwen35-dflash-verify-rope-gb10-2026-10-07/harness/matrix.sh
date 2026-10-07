#!/usr/bin/env bash
# Attribution matrix for #2191: {row rope off,on} x {LARGE_D 1,0} x {width 2,4} x n repeats,
# each cell running the byte bisect on the recorded 158-token transcript.
set -uo pipefail
WT=${WT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
D=$WT/docs/benchmark_results/data/dflash-width-2-4-residual-gb10-2026-09-21
OUT=${OUT:-$(dirname "$0")/../matrix}
BIN=${BIN:?set BIN to the lib test binary}
N=${N:-3}
TEST=${TEST:-models::qwen3_5::qwen3_5_dflash_probe_tests::block_versus_chain_byte_bisect_on_the_real_transcript}
mkdir -p "$OUT"
export MLXCEL_Q35_PROBE_TARGET=models/mlx/qwen3.5-4b-4bit
export MLXCEL_Q35_PROBE_PROMPT="$(cat $D/prompt_ids.json)"
export MLXCEL_Q35_PROBE_REFERENCE="$(cat $D/classic_ids.json)"
cd "$WT"
for rep in $(seq 1 "$N"); do
  for w in 2 4; do
    for rr in 0 1; do
      for ld in 1 0; do
        tag="w${w}-rr${rr}-ld${ld}-r${rep}"
        MLXCEL_Q35_ROW_ROPE=$rr MLXCEL_SDPA_VECTOR_LARGE_D=$ld MLXCEL_Q35_PROBE_BLOCK=$w \
        MLXCEL_Q35_PROBE_ACCEPTS="$(cat $D/arms/accepts_w${w}.txt)" \
          gpu-lock run --tag issue-2191-matrix -- "$BIN" --ignored --exact --test-threads=1 --nocapture "$TEST" \
          > "$OUT/$tag.log" 2>&1
        echo "$tag exit=$? $(grep -h '\[1935\] FIRST byte\|kept rows differ\|\[2191\] SUMMARY' "$OUT/$tag.log" | tr '\n' ' ')"
      done
    done
  done
done
echo MATRIX DONE
