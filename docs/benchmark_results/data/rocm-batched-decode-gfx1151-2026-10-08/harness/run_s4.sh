#!/usr/bin/env bash
# Session 4 for #2156: bf16 checkpoints (the fold sends bf16 [B,1,K] to the WMMA route).
set -uo pipefail
WT=.
D=<meas>
B=$D/bin_before
OUT="$1"; mkdir -p "$OUT"; cd "$WT"
log() { echo "$1 $2 $(date -Is)" >> "$OUT/s.log"; }
"$(cat $D/test_bin_path)" --test-threads=1 --nocapture > "$OUT/test.txt" 2>&1; log test $?
for b in 2 4 8; do target/release/examples/qmm_batch_rows_probe $b 64 5 bf16 > "$OUT/probe_after_bf16_b$b.txt" 2>&1; log probe_bf16_b$b $?; done
for m in Qwen3-0.6B-4bit gemma-3-4b-it-4bit; do
  for r in 1 2; do
    $B/profile_batched_decode -m models/mlx/$m --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_before_${m}_r$r.txt" 2>&1; log pbd_before_${m}_r$r $?
    target/release/examples/profile_batched_decode -m models/mlx/$m --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_after_${m}_r$r.txt" 2>&1; log pbd_after_${m}_r$r $?
  done
done
