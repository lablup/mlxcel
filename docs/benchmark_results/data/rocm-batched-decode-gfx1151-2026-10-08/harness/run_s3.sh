#!/usr/bin/env bash
# Session 3 for #2156: regression test, after/before probes, server matrix after, before arm, single-stream A/B.
set -uo pipefail
WT=.
D=<meas>
B=$D/bin_before
OUT="$1"; mkdir -p "$OUT"; cd "$WT"
M=models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
log() { echo "$1 $2 $(date -Is)" >> "$OUT/s.log"; }
TEST_BIN=$(cat "$D/test_bin_path")
"$TEST_BIN" --test-threads=1 --nocapture > "$OUT/test.txt" 2>&1; log test $?
for b in 2 4 8; do target/release/examples/qmm_batch_rows_probe $b 64 5 > "$OUT/probe_after_b$b.txt" 2>&1; log probe_after_b$b $?; done
$B/qmm_batch_rows_probe 8 64 5 > "$OUT/probe_before_b8.txt" 2>&1; log probe_before_b8 $?
target/release/examples/profile_batched_decode -m $M --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_after.txt" 2>&1; log pbd_after $?
$B/profile_batched_decode -m $M --batch-sizes 2,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_before.txt" 2>&1; log pbd_before $?
rocprofv3 --kernel-trace --stats -f csv -d "$OUT/trace" -o pbd -- target/release/examples/profile_batched_decode -m $M --batch-sizes 4 --decode-steps 20 --warmup 3 --runs 1 --prompt-len 1024 > "$OUT/pbd_prof.txt" 2>&1; log pbd_prof $?
for r in 1 2 3; do
  BENCH_DECODE=1 BENCH_BIN=$B/mlxcel-bench-decode $D/run_matrix.sh "$OUT/bd_before_$r" bd 1 x > /dev/null 2>&1; log bd_before_$r $?
  BENCH_DECODE=1 $D/run_matrix.sh "$OUT/bd_after_$r" bd 1 x > /dev/null 2>&1; log bd_after_$r $?
done
$D/run_matrix.sh "$OUT" after 3 "1:1024 4:1024 1:16384 4:16384" > "$OUT/after.log" 2>&1; log after $?
SERVER_BIN=$B/mlxcel-server $D/run_matrix.sh "$OUT" before 3 "1:1024 4:1024" > "$OUT/before.log" 2>&1; log before $?
