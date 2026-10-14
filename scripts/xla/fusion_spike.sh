#!/usr/bin/env bash
# Reproduce the issue #573 / #574 / #851 spike: does IREE fuse the packed int4/int8
# dequant into the matmul on the GPU target (CUDA or ROCm/HIP), so the reconstructed
# weight is not materialized to DRAM every decode step? Measures the static decode
# dispatch count (the fusion signal: `iree-compile --compile-to=flow` then count
# `flow.dispatch`) for the real packed decode graph under a flag sweep, plus three
# isolating microtests and executable lowering with `--iree-global-opt-data-tiling`.
#
# Prereqs:
#   - CUDA: `make iree-cuda && eval "$(scripts/iree/setup-cuda.sh --env)"`
#   - ROCm: `make iree-rocm && eval "$(scripts/iree/setup-rocm.sh --env)"`
#
# Usage:
#   scripts/xla/fusion_spike.sh [DECODE_MLIR]
# With no arg it uses the newest packed decode graph mlxcel left in its vmfb cache
# ($TMPDIR/mlxcel-xla-vmfb/decode-*.mlir with ui32 weight args).
set -uo pipefail

DEVICE="${MLXCEL_XLA_DEVICE:-}"
if [ -z "$DEVICE" ]; then
  if [ -n "${IREE_ROCM_HOME:-}" ] || { ! command -v nvcc >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; }; then
    DEVICE="hip"
  else
    DEVICE="cuda"
  fi
fi

IREE_COMPILE="${MLXCEL_XLA_IREE_COMPILE:-}"
if [ "$DEVICE" = "hip" ]; then
  if [ -z "$IREE_COMPILE" ] || [ -z "${MLXCEL_XLA_ROCM_TARGET:-}" ]; then
    eval "$("$(dirname "$0")/../iree/setup-rocm.sh" --env 2>/dev/null)" || true
    IREE_COMPILE="${MLXCEL_XLA_IREE_COMPILE:-$IREE_COMPILE}"
  fi
  [ -x "$IREE_COMPILE" ] || { echo "no rocm iree-compile; run: eval \"\$(scripts/iree/setup-rocm.sh --env)\"" >&2; exit 1; }
  [ -n "${MLXCEL_XLA_ROCM_TARGET:-}" ] || { echo "MLXCEL_XLA_ROCM_TARGET is not set" >&2; exit 1; }
  TARGET_FLAGS=(--iree-input-type=stablehlo --iree-hal-target-device=hip "--iree-rocm-target=${MLXCEL_XLA_ROCM_TARGET}")
  echo "target device: hip (rocm target: ${MLXCEL_XLA_ROCM_TARGET})"
else
  [ -n "$IREE_COMPILE" ] || IREE_COMPILE="$(eval "$(dirname "$0")/../iree/setup-cuda.sh --env" 2>/dev/null; echo "${MLXCEL_XLA_IREE_COMPILE:-}")"
  [ -x "$IREE_COMPILE" ] || { echo "no cuda iree-compile; run: eval \"\$(scripts/iree/setup-cuda.sh --env)\"" >&2; exit 1; }
  TARGET_FLAGS=(--iree-input-type=stablehlo --iree-hal-target-device=cuda)
  echo "target device: cuda"
fi

CACHE="${TMPDIR:-/tmp}/mlxcel-xla-vmfb"
DECODE="${1:-}"
if [ -z "$DECODE" ]; then
  for f in $(ls -t "$CACHE"/decode-*.mlir 2>/dev/null); do
    [ "$(grep -c ui32 "$f")" -gt 0 ] && DECODE="$f" && break
  done
fi
[ -n "$DECODE" ] && [ -f "$DECODE" ] || { echo "no packed decode graph found; run a packed decode first (see header)" >&2; exit 1; }
echo "packed decode graph: $DECODE"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

dispatches() { # <extra flags...> < in.mlir path is $DECODE
  local out="$WORK/o.mlir"
  if "$IREE_COMPILE" "${TARGET_FLAGS[@]}" "$@" --compile-to=flow "$DECODE" -o "$out" 2>"$WORK/e"; then
    grep -c 'flow.dispatch' "$out"
  else
    echo "FAIL($(tail -1 "$WORK/e" | cut -c1-60))"
  fi
}

echo ""
echo "== packed decode: flow.dispatch under an iree-compile flag sweep =="
printf "  %-24s %s\n" "baseline" "$(dispatches)"
printf "  %-24s %s\n" "aggressive-fuse" "$(dispatches --iree-dispatch-creation-enable-aggressive-fusion)"
printf "  %-24s %s\n" "generalize-mm" "$(dispatches --iree-opt-generalize-matmul)"
printf "  %-24s %s\n" "early-trunc" "$(dispatches --iree-dispatch-creation-enable-early-trunc-fusion)"
printf "  %-24s %s\n" "horiz-contract" "$(dispatches --iree-dispatch-creation-enable-fuse-horizontal-contractions)"
printf "  %-24s %s\n" "fuse-multi-use" "$(dispatches --iree-dispatch-creation-fuse-multi-use)"
printf "  %-24s %s\n" "data-tiling" "$(dispatches --iree-global-opt-data-tiling)"
if [ "$DEVICE" = "hip" ]; then
  printf "  %-24s %s\n" "data-tiling+rocm-pad" "$(dispatches --iree-global-opt-data-tiling --iree-rocm-encoding-layout-resolver=pad)"
  printf "  %-24s %s\n" "data-tiling+rocm-dt" "$(dispatches --iree-global-opt-data-tiling --iree-rocm-encoding-layout-resolver=data-tiling)"
fi
printf "  %-24s %s\n" "all-fusion" "$(dispatches --iree-dispatch-creation-enable-aggressive-fusion --iree-opt-generalize-matmul --iree-dispatch-creation-enable-early-trunc-fusion --iree-dispatch-creation-enable-fuse-horizontal-contractions --iree-dispatch-creation-fuse-multi-use)"

echo ""
echo "== full VMFB compilation check under data-tiling (#574) =="
if "$IREE_COMPILE" "${TARGET_FLAGS[@]}" --iree-global-opt-data-tiling "$DECODE" -o "$WORK/dt.vmfb" 2>"$WORK/dt.err"; then
  echo "  data-tiling full compile: PASS ($(stat -c%s "$WORK/dt.vmfb" 2>/dev/null || wc -c <"$WORK/dt.vmfb") bytes)"
else
  echo "  data-tiling full compile: FAIL ($(grep -E 'error:' "$WORK/dt.err" | head -n 2 | tr '\n' ' '))"
fi

micro() { # <name> <mlir file>
  local out="$WORK/m.mlir"
  if "$IREE_COMPILE" "${TARGET_FLAGS[@]}" --compile-to=flow "$2" -o "$out" 2>"$WORK/e"; then
    printf "  %-22s %s dispatch(es)\n" "$1" "$(grep -c 'flow.dispatch' "$out")"
  else
    printf "  %-22s FAIL (%s)\n" "$1" "$(tail -1 "$WORK/e" | cut -c1-70)"
  fi
}

cat > "$WORK/dq.mlir" <<'MLIR'
func.func @dq_mm(%act: tensor<8192xf16>, %w: tensor<2048x1024xui32>, %scale: tensor<2048xf16>) -> tensor<2048xf32> {
  %b = stablehlo.broadcast_in_dim %w, dims = [0, 1] : (tensor<2048x1024xui32>) -> tensor<2048x1024x8xui32>
  %sh = stablehlo.constant dense<[0, 4, 8, 12, 16, 20, 24, 28]> : tensor<8xui32>
  %shb = stablehlo.broadcast_in_dim %sh, dims = [2] : (tensor<8xui32>) -> tensor<2048x1024x8xui32>
  %r = stablehlo.shift_right_logical %b, %shb : tensor<2048x1024x8xui32>
  %mask = stablehlo.constant dense<15> : tensor<2048x1024x8xui32>
  %a = stablehlo.and %r, %mask : tensor<2048x1024x8xui32>
  %rs = stablehlo.reshape %a : (tensor<2048x1024x8xui32>) -> tensor<2048x8192xui32>
  %f = stablehlo.convert %rs : (tensor<2048x8192xui32>) -> tensor<2048x8192xf16>
  %sb = stablehlo.broadcast_in_dim %scale, dims = [0] : (tensor<2048xf16>) -> tensor<2048x8192xf16>
  %w16 = stablehlo.multiply %f, %sb : tensor<2048x8192xf16>
  %d = stablehlo.dot_general %act, %w16, contracting_dims = [0] x [1] : (tensor<8192xf16>, tensor<2048x8192xf16>) -> tensor<2048xf32>
  return %d : tensor<2048xf32>
}
MLIR
cat > "$WORK/f16.mlir" <<'MLIR'
func.func @f16_mm(%act: tensor<8192xf16>, %w: tensor<2048x8192xf16>) -> tensor<2048xf32> {
  %d = stablehlo.dot_general %act, %w, contracting_dims = [0] x [1] : (tensor<8192xf16>, tensor<2048x8192xf16>) -> tensor<2048xf32>
  return %d : tensor<2048xf32>
}
MLIR
cat > "$WORK/i8.mlir" <<'MLIR'
func.func @i8_mm(%act: tensor<8192xi8>, %w: tensor<2048x8192xi8>) -> tensor<2048xi32> {
  %d = stablehlo.dot_general %act, %w, contracting_dims = [0] x [1] : (tensor<8192xi8>, tensor<2048x8192xi8>) -> tensor<2048xi32>
  return %d : tensor<2048xi32>
}
MLIR

echo ""
echo "== microtests: one matmul, dispatch count (2 = producer + matmul, not fused) =="
micro "dequant -> matmul" "$WORK/dq.mlir"
micro "bare f16 matmul" "$WORK/f16.mlir"
micro "bare int8 matmul" "$WORK/i8.mlir"

echo ""
echo "int8 kernel arithmetic (upcast vs tensor/matrix cores):"
"$IREE_COMPILE" "${TARGET_FLAGS[@]}" --iree-hal-dump-executable-intermediates-to="$WORK/i8dump" "$WORK/i8.mlir" -o "$WORK/i8.vmfb" 2>/dev/null
LL="$(ls "$WORK/i8dump"/*optimized.ll 2>/dev/null | head -1)"
if [ -n "$LL" ]; then
  echo "  sext i8->i32 (upcast): $(grep -c 'sext i8\|sext <' "$LL")   dp4a/mma/wmma/mfma/sdot: $(grep -cE 'dp4a|mma.*s8|wmma|mfma|sdot' "$LL")"
fi
