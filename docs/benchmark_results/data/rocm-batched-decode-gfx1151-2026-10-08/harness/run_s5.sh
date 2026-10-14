#!/usr/bin/env bash
# Final session for #2156 on the final binary (batch fold + small-M bf16 route).
set -uo pipefail
WT=.
D=<meas>
B=$D/bin_before
OUT="$1"; mkdir -p "$OUT"; cd "$WT"
M=models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
P=target/release/examples/profile_batched_decode
log() { echo "$1 $2 $(date -Is)" >> "$OUT/s.log"; }
"$(cat $D/test_bin_path)" --test-threads=1 --nocapture > "$OUT/test.txt" 2>&1; log test $?
target/release/deps/rocm_qmm_dequant_cache-8c202d56014f3649 --test-threads=1 > "$OUT/test_dequant_cache.txt" 2>&1; log test_dequant_cache $?
for b in 2 4 8; do target/release/examples/qmm_batch_rows_probe $b 64 5 bf16 > "$OUT/probe_bf16_b$b.txt" 2>&1; log probe_bf16_b$b $?; done
target/release/examples/qmm_batch_rows_probe 4 64 5 > "$OUT/probe_f16_b4.txt" 2>&1; log probe_f16_b4 $?
for m in Qwen3-0.6B-4bit gemma-3-4b-it-4bit; do
  for r in 1 2; do
    $B/profile_batched_decode -m models/mlx/$m --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_before_${m}_r$r.txt" 2>&1; log pbd_before_${m}_r$r $?
    $P -m models/mlx/$m --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_after_${m}_r$r.txt" 2>&1; log pbd_after_${m}_r$r $?
  done
done
$P -m $M --batch-sizes 1,2,4,8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_llama_1k.txt" 2>&1; log pbd_llama_1k $?
$P -m $M --batch-sizes 1,4 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 16384 > "$OUT/pbd_llama_16k.txt" 2>&1; log pbd_llama_16k $?
for cfg in 1:1024 4:1024 1:16384 4:16384; do b=${cfg%%:*}; p=${cfg##*:}
  rocprofv3 --kernel-trace --stats -f csv -d "$OUT/trace_b${b}_p${p}" -o pbd -- $P -m $M --batch-sizes $b --decode-steps 20 --warmup 3 --runs 1 --prompt-len $p > "$OUT/pbd_prof_b${b}_p${p}.txt" 2>&1; log prof_b${b}_p${p} $?
done
MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 $P -m $M --batch-sizes 8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_b8_default.txt" 2>&1; log b8_default $?
MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=9 $P -m $M --batch-sizes 8 --decode-steps 50 --warmup 5 --runs 3 --prompt-len 1024 > "$OUT/pbd_b8_thr9.txt" 2>&1; log b8_thr9 $?
for r in 1 2 3; do
  BENCH_DECODE=1 BENCH_BIN=$B/mlxcel-bench-decode $D/run_matrix.sh "$OUT/bd_before_$r" bd 1 x > /dev/null 2>&1; log bd_before_$r $?
  BENCH_DECODE=1 $D/run_matrix.sh "$OUT/bd_after_$r" bd 1 x > /dev/null 2>&1; log bd_after_$r $?
done
$D/run_matrix.sh "$OUT" final 3 "1:1024 4:1024" > "$OUT/final.log" 2>&1; log final $?
