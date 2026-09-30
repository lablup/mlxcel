#!/usr/bin/env bash
# Build the ROCm tree at a candidate MLX commit and list what breaks
# (issue #1813).
#
# Usage:
#   scripts/mlxcelverse/check_api_drift.sh [options]
#
# Options:
#   --mlx-commit SHA     MLX commit to build against (default: the current pin)
#   --rocm-from SPEC     `overlay` (default): the committed ROCm overlay;
#                        `fork:<sha>`: that fork commit's mlx/backend/rocm/ with
#                        the overlay's core files, which shows what the fork's
#                        own code needs to compile against the candidate
#   --work-dir DIR       where to put the source tree, build and report
#                        (default: a new temporary directory)
#   --jobs N             parallel build jobs (default: 8)
#   --arch LIST          HIP targets (default: $MLX_ROCM_ARCHITECTURES, else
#                        the first gfx target rocminfo reports)
#   --git-dir DIR, --no-fetch
#                        passed to rocm_overlay.py (git cache, offline mode)
#
# Run it on an AMD host with ROCm installed, typically after
# `rocm_overlay.py retarget` in a pin bump, or before a fork sync with
# `--rocm-from fork:<new commit>`. It configures MLX standalone with the same
# ROCm options src/lib/mlxcel-core/build.rs passes, builds the `mlx` target
# with `make -k` so every failing translation unit is reported rather than the
# first, and then compares the defined and undefined symbols of every object:
# a primitive without a ROCm `eval_gpu` only shows up as an undefined
# mlx::core symbol, because libmlx is a static archive and nothing links it
# here. The report (compile errors, undefined symbols) is printed and written
# to <work-dir>/api-drift-report.txt; the exit status is 1 when anything is
# listed. It is a manual step, not part of `make verify-rocm`: it needs the
# network for MLX's own FetchContent dependencies and takes as long as a cold
# MLX build.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mlx_commit=""
rocm_from="overlay"
work=""
jobs=8
arch="${MLX_ROCM_ARCHITECTURES:-}"
git_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --mlx-commit) mlx_commit="$2"; shift 2 ;;
    --rocm-from) rocm_from="$2"; shift 2 ;;
    --work-dir) work="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --arch) arch="$2"; shift 2 ;;
    --git-dir) git_args+=(--git-dir "$2"); shift 2 ;;
    --no-fetch) git_args+=(--no-fetch); shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "check_api_drift: unknown option $1" >&2; exit 2 ;;
  esac
done

rocm="${ROCM_PATH:-/opt/rocm}"
if [ ! -x "$rocm/bin/hipcc" ]; then
  echo "check_api_drift: $rocm/bin/hipcc not found; set ROCM_PATH" >&2
  exit 2
fi
if [ -z "$arch" ]; then
  arch="$({ "$rocm/bin/rocminfo" 2>/dev/null || rocminfo 2>/dev/null || true; } | grep -o 'gfx[0-9a-f]*' | head -1)"
fi
if [ -z "$arch" ]; then
  echo "check_api_drift: no HIP target; pass --arch or set MLX_ROCM_ARCHITECTURES" >&2
  exit 2
fi

if [ -z "$work" ]; then
  work="$(mktemp -d -t mlxcelverse-api-drift.XXXXXX)"
fi
mkdir -p "$work"
src="$work/src"
build="$work/build"

export_args=(export-tree --dest "$src" --rocm-from "$rocm_from")
if [ -n "$mlx_commit" ]; then
  export_args+=(--mlx-commit "$mlx_commit")
fi
python3 "$here/rocm_overlay.py" "${export_args[@]}" ${git_args[@]+"${git_args[@]}"}

echo "[api-drift] configuring (log: $work/configure.log)"
if ! cmake -S "$src" -B "$build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_ROCM=ON \
  -DCMAKE_HIP_ARCHITECTURES="$arch" \
  -DMLX_ROCM_ARCHITECTURES="$arch" \
  -DCMAKE_PREFIX_PATH="$rocm" \
  -DROCM_PATH="$rocm" \
  -DCMAKE_HIP_COMPILER="$rocm/bin/hipcc" \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_ACCELERATE=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF >"$work/configure.log" 2>&1; then
  tail -30 "$work/configure.log" >&2
  echo "check_api_drift: CMake configure failed; see $work/configure.log" >&2
  exit 2
fi

echo "[api-drift] building target mlx with -k -j$jobs (log: $work/build.log)"
cmake --build "$build" --target mlx -j "$jobs" -- -k >"$work/build.log" 2>&1 || true

find "$build" -name '*.o' -print0 | xargs -0 -r nm -C --defined-only 2>/dev/null >"$work/defined.txt" || true
find "$build" -name '*.o' -print0 | xargs -0 -r nm -C --undefined-only 2>/dev/null >"$work/undefined.txt" || true

python3 "$here/rocm_overlay.py" api-report \
  --build-log "$work/build.log" \
  --defined "$work/defined.txt" \
  --undefined "$work/undefined.txt" \
  --source-dir "$src" \
  --output "$work/api-drift-report.txt"
