#!/usr/bin/env bash
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Standalone launch benchmark of the committed cuda-oxide module (issue #2834).
# Needs the CUDA toolkit only, not the cuda-oxide compiler: it builds
# standalone/oxide_standalone_bench.cpp against the driver API, checks the
# committed PTX against the digest every manifest pins, and runs the program
# under gpu-lock.
#
#   scripts/standalone-bench.sh [--csv FILE] [--label TEXT] [--iters N]
#       [--shapes model|bench2832|all] [--wrap "PROFILER COMMAND ..."]
#
# --wrap puts a profiler in front of the program inside the lock, for example
#   --wrap "nsys profile --trace=cuda --sample=none --cpuctxsw=none -o out"
# The digest check proves that the PTX matches the in-tree manifests, not that
# either is authentic, and the gap between the check and the program's load of
# the file is local-only: whoever can rewrite the PTX in that window already
# runs code as this user.
# Environment: CUDA_HOME (default /usr/local/cuda), CXX (default g++),
# GPU_LOCK_TAG (default issue-2834-standalone).
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_home=${CUDA_HOME:-/usr/local/cuda}
cxx=${CXX:-g++}
tag=${GPU_LOCK_TAG:-issue-2834-standalone}
ptx="$pkg_dir/artifacts/fused_add_rms_norm.sm_121.ptx"

csv=""
label=""
iters=200
shape_set=all
wrap=""
while [ $# -gt 0 ]; do
    case "$1" in
        --csv) csv=$2; shift 2 ;;
        --label) label=$2; shift 2 ;;
        --iters) iters=$2; shift 2 ;;
        --shapes) shape_set=$2; shift 2 ;;
        --wrap) wrap=$2; shift 2 ;;
        *) echo "standalone-bench: unknown argument $1" >&2; exit 2 ;;
    esac
done

if ! command -v gpu-lock >/dev/null 2>&1; then
    echo "standalone-bench: gpu-lock not found; refusing to run an unguarded GPU benchmark" >&2
    exit 2
fi
[ -f "$cuda_home/include/cuda.h" ] || {
    echo "standalone-bench: $cuda_home/include/cuda.h not found; set CUDA_HOME" >&2
    exit 2
}

# The digest every manifest pins must be the digest of the PTX about to load.
python3 - "$pkg_dir" "$ptx" <<'PY'
import glob, hashlib, json, sys
pkg, ptx = sys.argv[1], sys.argv[2]
digest = hashlib.sha256(open(ptx, "rb").read()).hexdigest()
manifests = sorted(glob.glob(f"{pkg}/artifacts/*.sm_121.manifest.json"))
if len(manifests) != 5:
    sys.exit(f"standalone-bench: expected 5 manifests, found {len(manifests)}")
for path in manifests:
    pinned = json.load(open(path))["container"]["payload_sha256"]
    if pinned != digest:
        sys.exit(f"standalone-bench: {path} pins {pinned}, the PTX is {digest}")
print(f"standalone-bench: PTX sha256 {digest} matches {len(manifests)} manifests")
PY

mkdir -p "$pkg_dir/build"
bin="$pkg_dir/build/oxide_standalone_bench"
src="$pkg_dir/standalone/oxide_standalone_bench.cpp"
if [ ! -x "$bin" ] || [ "$src" -nt "$bin" ]; then
    "$cxx" -O2 -std=c++17 -Wall -Wextra -I"$cuda_home/include" "$src" -o "$bin" -lcuda
fi

# The residual joins of the #2834 checkpoint (hidden 2048, bf16/bf16): decode
# at B=1 and B=4, prefill of 128 and 2048 tokens at B=1 and of 128 at B=4.
model_shapes=(bf16:bf16:1x2048 bf16:bf16:4x2048 bf16:bf16:128x2048 bf16:bf16:512x2048 bf16:bf16:2048x2048)
# The shape list of results/bench-sm_121.json (#2832).
bench2832_shapes=()
for dt in f32 bf16 f16; do
    for dims in 1x1024 1x4096 1x8192 5x4096 33x2048 64x8192; do
        bench2832_shapes+=("$dt:$dt:$dims")
    done
done
case "$shape_set" in
    model) shapes=("${model_shapes[@]}") ;;
    bench2832) shapes=("${bench2832_shapes[@]}") ;;
    all) shapes=("${model_shapes[@]}" "${bench2832_shapes[@]}") ;;
    *) echo "standalone-bench: unknown shape set $shape_set" >&2; exit 2 ;;
esac

args=(--ptx "$ptx" --iters "$iters" --label "$label")
[ -n "$csv" ] && args+=(--csv "$csv")
for s in "${shapes[@]}"; do args+=(--shape "$s"); done

# shellcheck disable=SC2086  # --wrap is a command line, split on purpose
gpu-lock run --tag "$tag" -- $wrap "$bin" "${args[@]}"
