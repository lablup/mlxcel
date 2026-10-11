#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BIN=${MLXCEL_BIN:-"$WT/target/release/mlxcel"}
BENCH_BIN=${MLXCEL_BENCH_BIN:-"$WT/target/release/mlxcel-bench-engine"}
MODELS=${MLXCEL_MODELS:-"$WT/models/mlx"}
BUILD_COMMIT=${MLXCEL_BUILD_COMMIT:-88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752}
OUT=${1:-"/tmp/mlxcel-issue-2228-rerun-$(date -u +%Y%m%dT%H%M%SZ)"}

# The policy arm must exercise the adaptive defaults, independent of the
# operator's shell environment.
unset MLXCEL_PREFILL_CHUNK LLAMA_ARG_BATCH

if [[ ${MLXCEL_ISSUE_2228_LOCKED:-0} != 1 ]]; then
    exec gpu-lock run --tag issue-2228-acceptance -- env MLXCEL_ISSUE_2228_LOCKED=1 "$0" "$OUT"
fi

mkdir -p "$OUT"
[[ -x "$BIN" ]] || { echo "missing fresh release binary: $BIN" >&2; exit 2; }
[[ -x "$BENCH_BIN" ]] || { echo "missing fresh release benchmark: $BENCH_BIN" >&2; exit 2; }
[[ -d "$MODELS/llama-3.2-1b-instruct-4bit" ]] || { echo "missing llama model link" >&2; exit 2; }
[[ -d "$MODELS/qwen3-1.7b-4bit" ]] || { echo "missing qwen model link" >&2; exit 2; }
# Subsequent commits only repair/add tests; validate this explicitly rather
# than rebuilding unchanged production sources to refresh a timestamp.
while IFS= read -r changed; do
    case "$changed" in
        src/lib/mlxcel-core/src/prefill_plan_tests.rs|tests/vlm_wrapper_capability_delegation.rs|docs/*|TECHNICAL_REPORTS/*) ;;
        *) echo "production changed since release build: $changed" >&2; exit 2 ;;
    esac
done < <(git -C "$WT" diff --name-only "$BUILD_COMMIT" HEAD)
git -C "$WT" merge-base --is-ancestor "$BUILD_COMMIT" HEAD
[[ -z $(git -C "$WT" status --porcelain --untracked-files=all) ]] || { echo "measurement requires a clean worktree" >&2; exit 2; }
head_time=$(git -C "$WT" show -s --format=%ct "$BUILD_COMMIT")
(( $(stat -c %Y "$BIN") >= head_time )) || { echo "mlxcel predates release production commit" >&2; exit 2; }
(( $(stat -c %Y "$BENCH_BIN") >= head_time )) || { echo "mlxcel-bench-engine predates release production commit" >&2; exit 2; }

core_build=
while IFS= read -r candidate; do
    if [[ -d "$candidate/out/build/include/cccl" && -d "$candidate/out/build/_deps/mlx-src/.git" ]]; then
        core_build=$candidate
        break
    fi
done < <(find "$WT/target/release/build" -maxdepth 1 -type d -name 'mlxcel-core-*' -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
[[ -n $core_build ]] || { echo "missing release MLX/CCCL build tree" >&2; exit 2; }
mlx_src="$core_build/out/build/_deps/mlx-src"
export MLXCEL_CCCL_DIR="$core_build/out/build/include/cccl"
export MLXCEL_CUTLASS_DIR="$core_build/out/build/include"
mlx_tag=$(awk '$1 == "GIT_TAG" { gsub(/[\")]/, "", $2); print $2 }' "$WT/src/lib/mlx-cpp/CMakeLists.txt")
mlx_head=$(git -C "$mlx_src" rev-parse HEAD)
[[ $mlx_head == "$mlx_tag" ]] || { echo "fetched MLX HEAD $mlx_head != declared pin $mlx_tag" >&2; exit 2; }

{
    date -u +%FT%TZ
    hostname
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    git -C "$WT" rev-parse HEAD
    echo "release_production_commit=$BUILD_COMMIT"
    git -C "$WT" diff --stat "$BUILD_COMMIT" HEAD
    echo "mlx_git_tag=$mlx_tag"
    echo "mlx_build_dir=$core_build"
    echo "mlx_fetched_head=$mlx_head"
    git -C "$mlx_src" status --porcelain --untracked-files=all
    git -C "$WT" status --porcelain --untracked-files=all
    git -C "$WT" diff --binary | sha256sum
    stat -c '%y %n' "$BIN" "$BENCH_BIN"
} > "$OUT/environment.txt"

server_pid=
cleanup() {
    if [[ -n $server_pid ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
trap cleanup EXIT INT TERM

wait_ready() {
    local port=$1
    for _ in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

for round in 1 2 3; do
    PYTHONPATH="$WT/docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness" \
        python3 -c 'import hostgate; hostgate.wait_quiet()' \
        >"$OUT/hostgate-admit-r$round.txt" 2>&1
    if ss -H -ltn 'sport = :18080' | grep -q .; then
        echo "port 18080 is already in use; refusing to probe an unrelated server" >&2
        exit 3
    fi
    log="$OUT/server-admit-policy-r$round.log"
    "$BIN" serve -m "$MODELS/llama-3.2-1b-instruct-4bit" --port 18080 --metrics --parallel 8 --ignore-eos >"$log" 2>&1 &
    server_pid=$!
    wait_ready 18080 || { echo "admission server failed to start in round $round" >&2; exit 3; }
    timeout 3600 python3 "$WT/scripts/bench_mixed_step_admission.py" --port 18080 --streams 4 --stream-max-tokens 4096 --admit-prompt-tokens 8192 --expect any >"$OUT/admit-policy-r$round.txt" 2>&1
    cleanup
    sleep 5
done

python3 - "$OUT" <<'PY'
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
values = []
for path in sorted(out.glob("admit-policy-r*.txt")):
    text = path.read_text()
    match = re.search(r"admission window .*?p95 ([0-9.]+) ms", text)
    if match is None:
        raise SystemExit(f"missing admission p95 in {path}")
    value = float(match.group(1))
    values.append(value)
    print(f"{path.name}: admission ITL p95 {value:.1f} ms")
if len(values) != 3:
    raise SystemExit(f"expected three admission rounds, found {len(values)}")
if any(value > 62.0 for value in values):
    raise SystemExit(f"admission acceptance failed: {values} exceeds 62.0 ms")
PY

for model in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
    python3 "$WT/scripts/engine_bench_rounds.py" \
        --bin "$BENCH_BIN" \
        --model "$MODELS/$model" \
        --prompt-tokens 8192 \
        --rounds 5 \
        --hostgate \
        --arm 'c2048=--path server --prefill-chunk 2048' \
        --arm 'policy=--path server' \
        --out "$OUT/ttft-$model.jsonl" \
        | tee "$OUT/ttft-$model.txt"
done

python3 "$SCRIPT_DIR/analyze.py" "$OUT"

echo "acceptance data: $OUT"
