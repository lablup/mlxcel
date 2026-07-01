#!/usr/bin/env bash
# Set up the IREE toolchain for the mlxcel OpenXLA backend (`xla-iree`) on macOS
# (Apple Silicon). IREE publishes no prebuilt macOS `iree-dist`, only linux dists
# and python wheels, so this script:
#
#   1. installs the pinned macOS `iree-compile` (metal-spirv codegen) from the
#      universal2 wheel into a private venv, and
#   2. source-builds the IREE *runtime* (runtime only, no LLVM) with the
#      local-task / local-sync / metal HAL drivers,
#
# then prints the env the cargo `xla-iree` build needs. This mirrors the manual
# CUDA (`IREE_CUDA_HOME`) setup, automated. Idempotent: re-running reuses an
# existing clone/build/venv.
#
# Usage:
#   scripts/iree/setup-macos.sh                 # build into the default dir
#   eval "$(scripts/iree/setup-macos.sh --env)" # just print/apply the exports
#
# After it finishes, build and run the backend:
#   eval "$(scripts/iree/setup-macos.sh --env)"
#   cargo build --release --features metal,accelerate,xla-iree
#   MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=metal \
#     ./target/release/mlxcel generate -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
set -euo pipefail

# --- pinned IREE version (keep in sync with the runtime/compiler the vmfbs are
# authored against; see CLAUDE.md "MLX upstream commit upgrade" sibling note) ---
IREE_TAG="iree-3.12.0rc20260626"
# macOS universal2, cp312-abi3 (works on CPython >= 3.12). sha256 from the
# iree-org GitHub release assets for ${IREE_TAG}.
WHEEL="iree_base_compiler-3.12.0rc20260626-cp312-abi3-macosx_13_0_universal2.whl"
WHEEL_SHA256="21e76f89f206f396c34845ed8d9da93236038398b60e5597dc9ada2b51b70ae6"

IREE_DIR="${MLXCEL_IREE_MACOS_DIR:-$HOME/.cache/mlxcel/iree-macos}"
SRC="$IREE_DIR/src"
BUILD="$IREE_DIR/build"
VENV="$IREE_DIR/venv"
IREE_COMPILE="$VENV/bin/iree-compile"

emit_env() {
  echo "export IREE_MACOS_HOME=$IREE_DIR"
  echo "export IREE_MACOS_COMPILE=$IREE_COMPILE"
  echo "export MLXCEL_XLA_IREE_COMPILE=$IREE_COMPILE"
}

# `--info`: human-readable resolved paths + pinned version + HAL drivers, for
# `make iree-env`. Not eval-safe -- use `--env` to export into a shell.
emit_info() {
  echo "IREE backend:  metal (HAL drivers: local-task, local-sync, metal)"
  echo "IREE home:     $IREE_DIR"
  echo "iree-compile:  $IREE_COMPILE"
  echo "Pinned:        $IREE_TAG  (wheel $WHEEL)"
  echo
  echo "# export the build env with: eval \"\$(scripts/iree/setup-macos.sh --env)\""
  emit_env
}

# `--env`: assume already built; just print the exports.
if [ "${1:-}" = "--env" ]; then
  emit_env
  exit 0
fi
# `--info`: print resolved paths + pinned version (human-readable).
if [ "${1:-}" = "--info" ]; then
  emit_info
  exit 0
fi

log() { printf '\033[1;34m[setup-macos]\033[0m %s\n' "$*" >&2; }

# --- prerequisites ---
[ "$(uname -s)" = "Darwin" ] || { echo "this script is macOS-only" >&2; exit 1; }
for tool in git cmake python3 curl shasum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,12) else 1)' \
  || { echo "need python >= 3.12 for the cp312-abi3 wheel (have $PYV)" >&2; exit 1; }

mkdir -p "$IREE_DIR"

# --- 1. iree-compile from the pinned macOS wheel, into a private venv ---
if [ ! -x "$IREE_COMPILE" ]; then
  log "creating venv and installing $WHEEL"
  python3 -m venv "$VENV"
  curl -sSL -o "$IREE_DIR/$WHEEL" \
    "https://github.com/iree-org/iree/releases/download/${IREE_TAG}/${WHEEL}"
  GOT=$(shasum -a 256 "$IREE_DIR/$WHEEL" | awk '{print $1}')
  [ "$GOT" = "$WHEEL_SHA256" ] || { echo "sha256 mismatch for $WHEEL: got $GOT" >&2; exit 1; }
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q --no-deps "$IREE_DIR/$WHEEL"
  [ -x "$IREE_COMPILE" ] || { echo "iree-compile not found after install" >&2; exit 1; }
else
  log "reusing iree-compile at $IREE_COMPILE"
fi

# --- 2. source-build the IREE runtime (runtime only; local + metal drivers) ---
if [ ! -d "$SRC/.git" ]; then
  log "cloning IREE runtime source @ $IREE_TAG (shallow, no LLVM)"
  git clone --depth 1 --branch "$IREE_TAG" https://github.com/iree-org/iree.git "$SRC"
fi
# The runtime build needs the non-compiler submodules: flatcc (vmfb parsing),
# plus google benchmark and printf, which the top-level CMake pulls in through the
# threading / runtime gates (IREE_ENABLE_THREADING etc.), not IREE_BUILD_*.
# Initialize everything EXCEPT the heavy compiler-only submodules (llvm-project,
# stablehlo, torch-mlir), which a runtime-only build (IREE_BUILD_COMPILER=OFF)
# never configures. The benchmark sentinel also self-heals an older flatcc-only clone.
if [ ! -e "$SRC/third_party/benchmark/.git" ]; then
  log "initializing runtime submodules (all except llvm-project/stablehlo/torch-mlir)"
  git -C "$SRC" \
    -c submodule."third_party/llvm-project".update=none \
    -c submodule."third_party/stablehlo".update=none \
    -c submodule."third_party/torch-mlir".update=none \
    submodule update --init --depth 1
fi

UNIFIED="$BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a"
if [ ! -f "$UNIFIED" ]; then
  log "configuring runtime-only build (local-task/local-sync/metal)"
  cmake -S "$SRC" -B "$BUILD" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_BENCHMARKS=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_METAL=ON
  log "building iree_runtime_unified"
  cmake --build "$BUILD" --target iree_runtime_unified -j "$(sysctl -n hw.ncpu)"
  [ -f "$UNIFIED" ] || { echo "runtime build did not produce $UNIFIED" >&2; exit 1; }
else
  log "reusing runtime build at $BUILD"
fi

log "done. IREE_MACOS_HOME=$IREE_DIR"
echo >&2
echo "# Run this (or: eval \"\$(scripts/iree/setup-macos.sh --env)\") to export the build env:" >&2
emit_env
