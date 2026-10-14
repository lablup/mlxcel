#!/usr/bin/env bash
# Set up the IREE toolchain for the mlxcel OpenXLA backend (`xla-iree`) on Linux
# with an AMD ROCm/HIP GPU (validated on Strix Halo Radeon 8060S, gfx1151). The
# prebuilt IREE dist is CPU/Vulkan only (no HIP driver), so this script:
#
#   1. installs the ROCm-capable `iree-compile` from the `iree-base-compiler` wheel
#      into a private venv, and
#   2. source-builds the IREE *runtime* (runtime only, no LLVM) with the
#      local-task / local-sync / HIP HAL drivers, pinned to the exact revision the
#      wheel was built from (version-matched runtime <-> compiler),
#
# then prints the env the cargo `xla-iree` build needs. Mirrors setup-cuda.sh.
# Idempotent: re-running reuses an existing clone/build/venv.
#
# Usage:
#   scripts/iree/setup-rocm.sh                 # build into the default dir
#   eval "$(scripts/iree/setup-rocm.sh --env)" # just print/apply the exports
#   scripts/iree/setup-rocm.sh --info          # print resolved paths + version
#
# After it finishes, build and run the backend:
#   eval "$(scripts/iree/setup-rocm.sh --env)"
#   cargo build --release --features xla-iree
#   MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=hip \
#     ./target/release/mlxcel generate -m <model dir> -p "..." -n 48
set -euo pipefail

# --- pinned IREE version. The compiler wheel comes from the matching official
# GitHub candidate release; IREE_SHA is that tag's exact iree-org commit, so the
# source-built runtime is byte-version-matched to the compiler that lowers the
# vmfbs. ---
IREE_VERSION="3.12.0rc20260721"
IREE_TAG="iree-$IREE_VERSION"
IREE_SHA="dc9601f88654749456c7cee4ae87e13de2654e1e"

# Version the default cache directory so a pin bump cannot silently reuse an
# older compiler/runtime build. Explicit MLXCEL_IREE_ROCM_DIR overrides remain
# available for callers that manage their own cache lifecycle.
IREE_DIR="${MLXCEL_IREE_ROCM_DIR:-$HOME/.cache/mlxcel/iree-rocm-$IREE_VERSION}"
SRC="$IREE_DIR/src"
BUILD="$IREE_DIR/build"
VENV="$IREE_DIR/venv"
IREE_COMPILE="$VENV/bin/iree-compile"
ROCM_ROOT="${ROCM_PATH:-/opt/rocm}"

# Detect a single unambiguous gfx* target from the environment or ROCm tools.
# Returns 0 with the target on stdout when unambiguous, or 1 when missing/ambiguous.
detect_rocm_target() {
  if [ -n "${MLXCEL_XLA_ROCM_TARGET:-}" ]; then
    printf '%s\n' "$MLXCEL_XLA_ROCM_TARGET"
    return 0
  fi

  local targets=""
  if command -v amdgpu-arch >/dev/null 2>&1; then
    targets="$(amdgpu-arch 2>/dev/null | grep -E '^gfx[0-9a-z]+$' | sort -u || true)"
  elif [ -x "$ROCM_ROOT/llvm/bin/amdgpu-arch" ]; then
    targets="$("$ROCM_ROOT/llvm/bin/amdgpu-arch" 2>/dev/null | grep -E '^gfx[0-9a-z]+$' | sort -u || true)"
  fi

  if [ -z "$targets" ]; then
    local rocminfo_bin=""
    if command -v rocminfo >/dev/null 2>&1; then
      rocminfo_bin="$(command -v rocminfo)"
    elif [ -x "$ROCM_ROOT/bin/rocminfo" ]; then
      rocminfo_bin="$ROCM_ROOT/bin/rocminfo"
    fi
    if [ -n "$rocminfo_bin" ]; then
      targets="$("$rocminfo_bin" 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]+gfx[0-9a-z]+$/ { print $2 }' | sort -u || true)"
    fi
  fi

  local count
  count="$(printf '%s\n' "$targets" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "$count" -eq 1 ]; then
    printf '%s\n' "$targets"
    return 0
  fi
  return 1
}

emit_env() {
  echo "export IREE_ROCM_HOME=$IREE_DIR"
  echo "export IREE_ROCM_COMPILE=$IREE_COMPILE"
  echo "export MLXCEL_XLA_IREE_COMPILE=$IREE_COMPILE"
  if target="$(detect_rocm_target)"; then
    echo "export MLXCEL_XLA_ROCM_TARGET=$target"
  fi
  if [ -e "$ROCM_ROOT/lib/libamdhip64.so" ]; then
    echo "export LD_LIBRARY_PATH=$ROCM_ROOT/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
  fi
}

# `--info`: human-readable resolved paths + pinned version + HAL drivers, for
# `make iree-env`. Not eval-safe -- use `--env` to export into a shell.
emit_info() {
  local target_desc
  if target_desc="$(detect_rocm_target)"; then
    :
  else
    target_desc="(unset or ambiguous; set MLXCEL_XLA_ROCM_TARGET)"
  fi
  echo "IREE backend:  rocm/hip (HAL drivers: local-task, local-sync, hip)"
  echo "IREE home:     $IREE_DIR"
  echo "iree-compile:  $IREE_COMPILE"
  echo "ROCm root:     $ROCM_ROOT"
  echo "ROCm target:   $target_desc"
  echo "Pinned:        $IREE_TAG  (iree-org @ $IREE_SHA)"
  echo
  echo "# export the build env with: eval \"\$(scripts/iree/setup-rocm.sh --env)\""
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

log() { printf '\033[1;34m[setup-rocm]\033[0m %s\n' "$*" >&2; }

# --- prerequisites ---
[ "$(uname -s)" = "Linux" ] || { echo "this script is Linux/ROCm-only (use setup-macos.sh on macOS)" >&2; exit 1; }
for tool in git cmake make python3 curl sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
if [ ! -d "$ROCM_ROOT" ] && [ ! -e /dev/kfd ]; then
  echo "no ROCm installation at $ROCM_ROOT and no /dev/kfd found (set ROCM_PATH)" >&2
  exit 1
fi

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

# --- 1. ROCm-capable iree-compile from the pinned wheel, into a private venv ---
if [ ! -x "$IREE_COMPILE" ]; then
  log "creating venv and installing $WHEEL"
  rm -rf "$VENV"
  if python3 -m venv "$VENV" >/dev/null 2>&1; then
    VENV_INSTALL_MODE="pip"
  elif command -v uv >/dev/null 2>&1; then
    rm -rf "$VENV"
    uv venv --clear --quiet "$VENV"
    VENV_INSTALL_MODE="uv"
  else
    rm -rf "$VENV"
    python3 -m venv --without-pip "$VENV"
    VENV_INSTALL_MODE="zipfile"
  fi
  curl -fsSL -o "$IREE_DIR/$WHEEL" \
    "https://github.com/iree-org/iree/releases/download/${IREE_TAG}/${WHEEL}"
  GOT=$(sha256sum "$IREE_DIR/$WHEEL" | awk '{print $1}')
  [ "$GOT" = "$WHEEL_SHA256" ] || { echo "sha256 mismatch for $WHEEL: got $GOT" >&2; exit 1; }
  case "$VENV_INSTALL_MODE" in
    pip)
      "$VENV/bin/python" -m pip install -q --upgrade pip
      "$VENV/bin/python" -m pip install -q --no-deps "$IREE_DIR/$WHEEL"
      ;;
    uv)
      uv pip install --quiet --python "$VENV/bin/python" --no-index --no-deps "$IREE_DIR/$WHEEL"
      ;;
    zipfile)
      "$VENV/bin/python" - "$IREE_DIR/$WHEEL" "$VENV" <<'PY'
import configparser, os, stat, sys, sysconfig, zipfile

wheel_path, venv_dir = sys.argv[1], sys.argv[2]
purelib = sysconfig.get_path("purelib", vars={"base": venv_dir, "platbase": venv_dir})
bin_dir = os.path.join(venv_dir, "bin")
os.makedirs(purelib, exist_ok=True)
os.makedirs(bin_dir, exist_ok=True)

with zipfile.ZipFile(wheel_path) as zf:
    zf.extractall(purelib)
    for name in zf.namelist():
        if name.endswith(".dist-info/entry_points.txt"):
            cp = configparser.ConfigParser()
            cp.read_string(zf.read(name).decode("utf-8"))
            if cp.has_section("console_scripts"):
                for script_name, spec in cp.items("console_scripts"):
                    mod, func = [s.strip() for s in spec.split(":", 1)]
                    script_path = os.path.join(bin_dir, script_name)
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(f"#!{os.path.join(bin_dir, 'python')}\nimport sys\nfrom {mod} import {func}\nsys.exit({func}())\n")
                    os.chmod(script_path, 0o755)
for root, _, files in os.walk(purelib):
    for fn in files:
        if fn.startswith("iree-") or fn.endswith(".so"):
            p = os.path.join(root, fn)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
      ;;
  esac
  rm -f "$IREE_DIR/$WHEEL"
  [ -x "$IREE_COMPILE" ] || { echo "iree-compile not found after install" >&2; exit 1; }
else
  log "reusing iree-compile at $IREE_COMPILE"
fi

if ! "$IREE_COMPILE" --iree-hal-list-target-backends 2>/dev/null | grep -Eq '^[[:space:]]*rocm$'; then
  echo "iree-compile at $IREE_COMPILE does not expose 'rocm' target backend" >&2
  exit 1
fi

# --- 2. source-build the IREE runtime (runtime only; local + hip drivers) ---
# Fetch the exact pinned release commit shallowly.
if [ ! -e "$SRC/.git" ]; then
  log "fetching IREE runtime source @ $IREE_SHA (shallow, no LLVM)"
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/iree-org/iree.git 2>/dev/null || true
  git -C "$SRC" fetch --depth 1 origin "$IREE_SHA"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
# The runtime build needs the non-compiler submodules: flatcc (vmfb parsing),
# hip-build-deps, google benchmark, and printf, which the top-level CMake pulls
# in through the threading / runtime gates. Initialize everything EXCEPT the
# heavy compiler-only submodules (llvm-project, stablehlo, torch-mlir), which a
# runtime-only build (IREE_BUILD_COMPILER=OFF) never configures.
if [ ! -e "$SRC/third_party/benchmark/.git" ]; then
  log "initializing runtime submodules (all except llvm-project/stablehlo/torch-mlir)"
  git -C "$SRC" \
    -c submodule."third_party/llvm-project".update=none \
    -c submodule."third_party/stablehlo".update=none \
    -c submodule."third_party/torch-mlir".update=none \
    submodule update --init --depth 1
fi

UNIFIED="$BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a"
HIP_REG="$BUILD/runtime/src/iree/hal/drivers/hip/registration/libiree_hal_drivers_hip_registration_registration.a"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-${CARGO_BUILD_JOBS:-6}}"

if [ ! -f "$UNIFIED" ] || [ ! -f "$HIP_REG" ]; then
  log "configuring runtime-only build (local-task/local-sync/hip)"
  cmake -S "$SRC" -B "$BUILD" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_BENCHMARKS=OFF \
    -DIREE_ROCM_TEST_TARGET_CHIP="" \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_HIP=ON
  log "building iree_runtime_unified + iree-run-module (-j$JOBS)"
  make -C "$BUILD" -j"$JOBS" iree_runtime_unified iree-run-module
  [ -f "$UNIFIED" ] || { echo "runtime build did not produce $UNIFIED" >&2; exit 1; }
  [ -f "$HIP_REG" ] || { echo "runtime build did not produce $HIP_REG" >&2; exit 1; }
else
  log "reusing runtime build at $BUILD"
fi

log "done. IREE_ROCM_HOME=$IREE_DIR"
echo >&2
echo "# Run this (or: eval \"\$(scripts/iree/setup-rocm.sh --env)\") to export the build env:" >&2
emit_env
