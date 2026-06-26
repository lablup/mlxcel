#!/usr/bin/env bash
# End-to-end driver for the Rust StableHLO emitter spike (#451).
#
#   ./validate.sh            # P0 round-trip + full decode_step token check
#   ./validate.sh p0         # P0 toolchain round-trip only
#   ./validate.sh decode     # emit + compile + token-exact decode only
#
# Reuses the existing spike/openxla venv (iree-compile + iree runtime + torch +
# transformers + safetensors). Nothing here builds or touches the mlxcel crates.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="/home/inureyes/Development/mlxcel/spike/openxla"
PY="$REF/.venv/bin/python"
IREE_COMPILE="$REF/.venv/bin/iree-compile"
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
OUT="$HERE/out"
mkdir -p "$OUT"

mode="${1:-all}"

echo "== build emitter =="
( cd "$HERE" && "$CARGO" build --release )
EMIT="$HERE/target/release/emit"

p0() {
  echo "== P0: single dot_general round-trip =="
  "$EMIT" p0 "$OUT/p0.mlir"
  "$IREE_COMPILE" --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu \
    "$OUT/p0.mlir" -o "$OUT/p0.vmfb"
  "$PY" - "$OUT/p0.vmfb" <<'PY'
import sys, numpy as np, iree.runtime as rt
ctx = rt.SystemContext(config=rt.Config("local-task"))
vm = rt.VmModule.from_flatbuffer(ctx.instance, open(sys.argv[1], "rb").read())
ctx.add_vm_module(vm)
fn = getattr(ctx.modules, vm.name).main
rng = np.random.default_rng(0)
a = rng.standard_normal((4, 8), dtype=np.float32); b = rng.standard_normal((8, 3), dtype=np.float32)
ok = np.allclose(np.asarray(fn(a, b)), a @ b, atol=1e-5)
print("P0 ROUND-TRIP:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
}

decode() {
  echo "== P1: full decode_step =="
  "$EMIT" decode "$OUT/decode.mlir"
  "$IREE_COMPILE" --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu \
    "$OUT/decode.mlir" -o "$OUT/decode.vmfb"
  "$PY" "$HERE/python/run_decode.py" --mlir "$OUT/decode.mlir" --vmfb "$OUT/decode.vmfb"
}

case "$mode" in
  p0) p0 ;;
  decode) decode ;;
  all) p0; decode ;;
  *) echo "usage: validate.sh [p0|decode|all]"; exit 2 ;;
esac
