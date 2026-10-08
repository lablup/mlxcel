#!/usr/bin/env bash
set -uo pipefail
WT=.
OUT="$1"; mkdir -p "$OUT"; cd "$WT"
M=models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
target/release/examples/qmm_batch_rows_probe 4 64 5 > "$OUT/probe_b4.txt" 2>&1; echo "probe4 $?" >> "$OUT/s.log"
target/release/examples/qmm_batch_rows_probe 2 64 5 > "$OUT/probe_b2.txt" 2>&1; echo "probe2 $?" >> "$OUT/s.log"
target/release/examples/profile_batched_decode -m $M --batch-sizes 1,4 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd.txt" 2>&1; echo "pbd $?" >> "$OUT/s.log"
rocprofv3 --kernel-trace --stats -f csv -d "$OUT/trace" -o pbd -- target/release/examples/profile_batched_decode -m $M --batch-sizes 4 --decode-steps 20 --warmup 3 --runs 1 --prompt-len 1024 > "$OUT/pbd_prof.txt" 2>&1; echo "pbdprof $?" >> "$OUT/s.log"
