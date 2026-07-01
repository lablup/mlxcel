#!/usr/bin/env bash
# Set up the IREE toolchain for the mlxcel OpenXLA backend (`xla-iree`) on Linux
# with an NVIDIA GPU (validated on a GB10, Grace-Blackwell sm_121). The prebuilt
# IREE dist is CPU/Vulkan only (no CUDA driver, and Vulkan cannot allocate against
# the GB10 unified memory), so this script:
#
#   1. installs the CUDA-capable `iree-compile` from the `iree-base-compiler` wheel
#      into a private venv, and
#   2. source-builds the IREE *runtime* (runtime only, no LLVM) with the
#      local-task / local-sync / CUDA HAL drivers, pinned to the exact revision the
#      wheel was built from (version-matched runtime <-> compiler),
#
# then prints the env the cargo `xla-iree` build needs. Mirrors setup-macos.sh.
# Idempotent: re-running reuses an existing clone/build/venv.
#
# Usage:
#   scripts/iree/setup-cuda.sh                 # build into the default dir
#   eval "$(scripts/iree/setup-cuda.sh --env)" # just print/apply the exports
#
# After it finishes, build and run the backend:
#   eval "$(scripts/iree/setup-cuda.sh --env)"
#   cargo build --release --features xla-iree
#   MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda \
#     ./target/release/mlxcel generate -m <Llama-3.2-1B dir> -p "..." -n 48
set -euo pipefail

# --- pinned IREE version. PIP_VER is the wheel; IREE_SHA is the exact iree-org
# commit that wheel reports (iree-compile --version), so the source-built runtime
# is byte-version-matched to the compiler that lowers the vmfbs. ---
PIP_VER="3.11.0"
IREE_SHA="e4a3b0405d7d23554da26403658d0e8c3c5ecf25"

IREE_DIR="${MLXCEL_IREE_CUDA_DIR:-$HOME/.cache/mlxcel/iree-cuda}"
SRC="$IREE_DIR/src"
BUILD="$IREE_DIR/build"
VENV="$IREE_DIR/venv"
IREE_COMPILE="$VENV/bin/iree-compile"
CUDA_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"

emit_env() {
  echo "export IREE_CUDA_HOME=$IREE_DIR"
  echo "export IREE_CUDA_COMPILE=$IREE_COMPILE"
  echo "export MLXCEL_XLA_IREE_COMPILE=$IREE_COMPILE"
}

# `--env`: assume already built; just print the exports.
if [ "${1:-}" = "--env" ]; then
  emit_env
  exit 0
fi

log() { printf '\033[1;34m[setup-cuda]\033[0m %s\n' "$*" >&2; }

# --- prerequisites ---
[ "$(uname -s)" = "Linux" ] || { echo "this script is Linux/CUDA-only (use setup-macos.sh on macOS)" >&2; exit 1; }
for tool in git cmake make python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
[ -x "$CUDA_ROOT/bin/nvcc" ] || { echo "no CUDA toolkit at $CUDA_ROOT (set CUDAToolkit_ROOT)" >&2; exit 1; }

mkdir -p "$IREE_DIR"

# --- 1. CUDA-capable iree-compile from the pinned wheel, into a private venv ---
if [ ! -x "$IREE_COMPILE" ]; then
  log "creating venv and installing iree-base-compiler==$PIP_VER"
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q "iree-base-compiler==$PIP_VER"
  [ -x "$IREE_COMPILE" ] || { echo "iree-compile not found after install" >&2; exit 1; }
else
  log "reusing iree-compile at $IREE_COMPILE"
fi

# --- 2. source-build the IREE runtime (runtime only; local + cuda drivers) ---
# Fetch the exact pinned commit shallowly (the wheel is an rc, not a release tag).
if [ ! -e "$SRC/.git" ]; then
  log "fetching IREE runtime source @ $IREE_SHA (shallow, no LLVM)"
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/iree-org/iree.git 2>/dev/null || true
  git -C "$SRC" fetch --depth 1 origin "$IREE_SHA"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
# The runtime build needs only flatcc (vmfb parsing); skip the compiler-only
# submodules (llvm-project, stablehlo, torch-mlir, ...) entirely.
if [ ! -e "$SRC/third_party/flatcc/.git" ]; then
  log "initializing flatcc submodule only"
  git -C "$SRC" submodule update --init --depth 1 -- third_party/flatcc
fi

UNIFIED="$BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a"
if [ ! -f "$UNIFIED" ]; then
  log "configuring runtime-only build (local-task/local-sync/cuda)"
  cmake -S "$SRC" -B "$BUILD" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_BENCHMARKS=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_CUDA=ON -DCUDAToolkit_ROOT="$CUDA_ROOT"
  log "building iree_runtime_unified"
  make -C "$BUILD" -j"$(nproc)" iree_runtime_unified
  [ -f "$UNIFIED" ] || { echo "runtime build did not produce $UNIFIED" >&2; exit 1; }
else
  log "reusing runtime build at $BUILD"
fi

log "done. IREE_CUDA_HOME=$IREE_DIR"
echo >&2
echo "# Run this (or: eval \"\$(scripts/iree/setup-cuda.sh --env)\") to export the build env:" >&2
emit_env
