#!/usr/bin/env bash
# Uniform-B batched decode sweep (issue #449 M3 Stage 1).
#
# Emits the batched decode_step graph for each batch size B, compiles it for the
# target device, drives the real Llama-3.2-1B through it, and reports the
# token-exact verdict plus aggregate tok/s. Proves (1) the batched graph stays
# token-exact at every B and (2) aggregate throughput scales with B (the GPU
# batch-1 path was bandwidth/launch-starved).
#
#   ./validate_batch.sh cpu             # local-task sweep (prebuilt IREE dist)
#   ./validate_batch.sh cuda            # cuda sweep (source-built cuda runtime)
#   ./validate_batch.sh cpu "1 2 4 8"   # custom B list
#
# The CPU path uses the prebuilt dist ($IREE_DIST) + its iree-compile. The CUDA
# path uses the source-built cuda runtime ($IREE_CUDA_HOME) + a cuda-capable
# iree-compile ($IREE_CUDA_COMPILE, the pip one). The two link modes are
# mutually exclusive, so the bin is rebuilt for the chosen device. Same prompt /
# HF reference as the scalar llama bin.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMITTER="$HERE/../rust-emitter"
REF="$HERE/../openxla"
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
OUT="${OUT:-$HERE/out_batch}"
mkdir -p "$OUT"

DEVICE="${1:-cpu}"
BLIST="${2:-1 2 4 8 16 24 32 64}"

# Toolchain paths (override via env). Defaults match the spark-101 dev box.
IREE_DIST="${IREE_DIST:-$HERE/iree-dist}"
IREE_CUDA_HOME="${IREE_CUDA_HOME:-/home/inureyes/Development/iree-cuda}"
IREE_CUDA_COMPILE="${IREE_CUDA_COMPILE:-$REF/.venv/bin/iree-compile}"

CPU_FLAGS=(--iree-input-type=stablehlo --iree-hal-target-device=local
           --iree-hal-local-target-device-backends=llvm-cpu)
CUDA_FLAGS=(--iree-input-type=stablehlo --iree-hal-target-device=cuda)

echo "== build emitter =="
( cd "$EMITTER" && "$CARGO" build --release >/dev/null )
EMIT="$EMITTER/target/release/emit"

case "$DEVICE" in
  cpu)
    HALDEV="local-task"; SUF="vmfb"
    IC="$IREE_DIST/bin/iree-compile"; CFLAGS=("${CPU_FLAGS[@]}")
    echo "== build llama_batch (CPU/dist mode) =="
    ( cd "$HERE" && IREE_DIST="$IREE_DIST" "$CARGO" build --release --bin llama_batch >/dev/null 2>&1 )
    RUN_ENV=(IREE_DIST="$IREE_DIST")
    ;;
  cuda)
    HALDEV="cuda"; SUF="cuda.vmfb"
    IC="$IREE_CUDA_COMPILE"; CFLAGS=("${CUDA_FLAGS[@]}")
    echo "== build llama_batch (CUDA mode) =="
    ( cd "$HERE" && IREE_CUDA_HOME="$IREE_CUDA_HOME" "$CARGO" build --release --bin llama_batch >/dev/null 2>&1 )
    RUN_ENV=(IREE_CUDA_HOME="$IREE_CUDA_HOME")
    ;;
  *) echo "usage: validate_batch.sh [cpu|cuda] [\"B list\"]"; exit 2 ;;
esac

BIN="$HERE/target/release/llama_batch"

# Single-seq prefill graph (shared across all B); compiled once per device.
echo "== compile prefill ($HALDEV) =="
"$EMIT" prefill-argmax "$OUT/prefill.mlir" >/dev/null
"$IC" "${CFLAGS[@]}" "$OUT/prefill.mlir" -o "$OUT/prefill.$SUF"

echo "== sweep B in: $BLIST (device $HALDEV) =="
SUMMARY=()
for B in $BLIST; do
  "$EMIT" decode-batch-argmax "$B" "$OUT/db$B.mlir" >/dev/null
  "$IC" "${CFLAGS[@]}" "$OUT/db$B.mlir" -o "$OUT/db$B.$SUF"
  line="$(env "${RUN_ENV[@]}" "$BIN" --batch "$B" --device "$HALDEV" \
            --prefill "$OUT/prefill.$SUF" --decode "$OUT/db$B.$SUF" \
          | grep '^SCALE' || true)"
  echo "$line"
  SUMMARY+=("$line")
done

echo
echo "== summary ($HALDEV) =="
printf '%-4s %-12s %-14s %-14s %s\n' "B" "ms/step" "per-seq tok/s" "agg tok/s" "pass"
for l in "${SUMMARY[@]}"; do
  b=$(sed -n 's/.*\bB=\([0-9]*\).*/\1/p' <<<"$l")
  ms=$(sed -n 's/.*ms_per_step=\([0-9.]*\).*/\1/p' <<<"$l")
  ps=$(sed -n 's/.*per_seq_tok_s=\([0-9.]*\).*/\1/p' <<<"$l")
  ag=$(sed -n 's/.*agg_tok_s=\([0-9.]*\).*/\1/p' <<<"$l")
  pp=$(sed -n 's/.*pass=\([a-z]*\).*/\1/p' <<<"$l")
  printf '%-4s %-12s %-14s %-14s %s\n' "$b" "$ms" "$ps" "$ag" "$pp"
done
