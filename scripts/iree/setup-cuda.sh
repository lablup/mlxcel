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

# --- pinned IREE version. The compiler wheel comes from the matching official
# GitHub candidate release; IREE_SHA is that tag's exact iree-org commit, so the
# source-built runtime is byte-version-matched to the compiler that lowers the
# vmfbs. ---
IREE_VERSION="3.12.0rc20260721"
IREE_TAG="iree-$IREE_VERSION"
IREE_SHA="dc9601f88654749456c7cee4ae87e13de2654e1e"

# Version the default cache directory so a pin bump cannot silently reuse an
# older compiler/runtime build. Explicit MLXCEL_IREE_CUDA_DIR overrides remain
# available for callers that manage their own cache lifecycle.
IREE_DIR="${MLXCEL_IREE_CUDA_DIR:-$HOME/.cache/mlxcel/iree-cuda-$IREE_VERSION}"
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

# `--info`: human-readable resolved paths + pinned version + HAL drivers, for
# `make iree-env`. Not eval-safe -- use `--env` to export into a shell.
emit_info() {
  echo "IREE backend:  cuda (HAL drivers: local-task, local-sync, cuda)"
  echo "IREE home:     $IREE_DIR"
  echo "iree-compile:  $IREE_COMPILE"
  echo "CUDA root:     $CUDA_ROOT"
  echo "Pinned:        $IREE_TAG  (iree-org @ $IREE_SHA)"
  echo
  echo "# export the build env with: eval \"\$(scripts/iree/setup-cuda.sh --env)\""
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

log() { printf '\033[1;34m[setup-cuda]\033[0m %s\n' "$*" >&2; }

# --- prerequisites ---
[ "$(uname -s)" = "Linux" ] || { echo "this script is Linux/CUDA-only (use setup-macos.sh on macOS)" >&2; exit 1; }
for tool in git cmake make python3 curl sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
[ -x "$CUDA_ROOT/bin/nvcc" ] || { echo "no CUDA toolkit at $CUDA_ROOT (set CUDAToolkit_ROOT)" >&2; exit 1; }

# Select the official manylinux wheel for this Python ABI and host architecture.
# CPython >= 3.12 can consume IREE's cp312-abi3 wheel.
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PYV" in
  3.10) PY_TAG="cp310-cp310" ;;
  3.11) PY_TAG="cp311-cp311" ;;
  *)
    python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,12) else 1)' \
      || { echo "need CPython >= 3.10 for the IREE compiler wheel (have $PYV)" >&2; exit 1; }
    PY_TAG="cp312-abi3"
    ;;
esac
case "$(uname -m)" in
  aarch64) WHEEL_ARCH="aarch64" ;;
  x86_64) WHEEL_ARCH="x86_64" ;;
  *) echo "unsupported Linux architecture for the IREE compiler wheel: $(uname -m)" >&2; exit 1 ;;
esac
WHEEL="iree_base_compiler-${IREE_VERSION}-${PY_TAG}-manylinux_2_27_${WHEEL_ARCH}.manylinux_2_28_${WHEEL_ARCH}.whl"
case "${PY_TAG}/${WHEEL_ARCH}" in
  cp310-cp310/aarch64) WHEEL_SHA256="292f92ea749da937d9557fae27dafdd227eb79d8d62d2a4dffd25285b8b11aaf" ;;
  cp310-cp310/x86_64) WHEEL_SHA256="8ca7f0610c5c51efc898a4e647687b6f64fef0ccbc6aeb9fd0217597b03768b8" ;;
  cp311-cp311/aarch64) WHEEL_SHA256="fa288fd6824b343fae186d51d6e80a046f1440fe772afc790fa3c6f39deaa6b1" ;;
  cp311-cp311/x86_64) WHEEL_SHA256="91f725e7de1964ad195e9299d922b12e2e416126ee94d550e6452851997d6c80" ;;
  cp312-abi3/aarch64) WHEEL_SHA256="2aa7b17adbeab2d406e54f6c4814896256a9f6687b3e35b44a2b18ca4da42784" ;;
  cp312-abi3/x86_64) WHEEL_SHA256="496e003f14a8eefa9fb5182949fd1ab5604f8b7a5bfb66c897cfd2e3b25d6652" ;;
esac

mkdir -p "$IREE_DIR"

# --- 1. CUDA-capable iree-compile from the pinned wheel, into a private venv ---
if [ ! -x "$IREE_COMPILE" ]; then
  log "creating venv and installing $WHEEL"
  python3 -m venv "$VENV"
  curl -sSL -o "$IREE_DIR/$WHEEL" \
    "https://github.com/iree-org/iree/releases/download/${IREE_TAG}/${WHEEL}"
  GOT=$(sha256sum "$IREE_DIR/$WHEEL" | awk '{print $1}')
  [ "$GOT" = "$WHEEL_SHA256" ] || { echo "sha256 mismatch for $WHEEL: got $GOT" >&2; exit 1; }
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q --no-deps "$IREE_DIR/$WHEEL"
  [ -x "$IREE_COMPILE" ] || { echo "iree-compile not found after install" >&2; exit 1; }
else
  log "reusing iree-compile at $IREE_COMPILE"
fi

# --- 2. source-build the IREE runtime (runtime only; local + cuda drivers) ---
# Fetch the exact pinned release commit shallowly.
if [ ! -e "$SRC/.git" ]; then
  log "fetching IREE runtime source @ $IREE_SHA (shallow, no LLVM)"
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/iree-org/iree.git 2>/dev/null || true
  git -C "$SRC" fetch --depth 1 origin "$IREE_SHA"
  git -C "$SRC" checkout -q FETCH_HEAD
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
