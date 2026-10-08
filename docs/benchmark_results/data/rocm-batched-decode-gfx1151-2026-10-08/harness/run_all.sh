#!/usr/bin/env bash
# One guarded session for #2156: baseline matrix, threshold arm, profile, single-stream.
set -uo pipefail
D=<meas>
OUT="$1"
mkdir -p "$OUT"
cd "$D"
echo "start $(date -Is)" > "$OUT/session.log"
./run_matrix.sh "$OUT" base 3 "1:1024 4:1024 1:16384 4:16384" >> "$OUT/session.log" 2>&1; echo "base exit $?" >> "$OUT/session.log"
SERVER_ENV="MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=64" ./run_matrix.sh "$OUT" thr64 3 "4:1024" >> "$OUT/session.log" 2>&1; echo "thr64 exit $?" >> "$OUT/session.log"
PROFILE=1 ./run_matrix.sh "$OUT" prof 1 "1:1024 4:1024 1:16384 4:16384" >> "$OUT/session.log" 2>&1; echo "prof exit $?" >> "$OUT/session.log"
BENCH_DECODE=1 ./run_matrix.sh "$OUT" bd 3 "1:64" >> "$OUT/session.log" 2>&1; echo "bd exit $?" >> "$OUT/session.log"
echo "end $(date -Is)" >> "$OUT/session.log"
