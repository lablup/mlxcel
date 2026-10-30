#!/bin/bash
# Full campaign for docs/benchmark_results/cublaslt-gemv-ab-gb10-2026-10-31.md.
# Run from anywhere; results land in $MLXCEL_AB_OUT (default: ./ab-results).
#
#   MLXCEL_AB_CUBLAS_DIR=/path/to/nvidia/cu13/lib  (required)
#   MLXCEL_AB_BIN_DIR=<repo>/target/release        (optional)
#   MLXCEL_AB_STAGES="version gemv ladder"         (optional subset)
#
# Every GPU run goes through gpu-lock when it is installed (GB10 hosts).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
out=${MLXCEL_AB_OUT:-$PWD/ab-results}
stages=${MLXCEL_AB_STAGES:-"version gemv ladder"}
: "${MLXCEL_AB_CUBLAS_DIR:?set MLXCEL_AB_CUBLAS_DIR to the alternate cuBLAS lib dir}"
mkdir -p "$out"
lock=()
command -v gpu-lock >/dev/null && lock=(gpu-lock run --tag cublaslt-ab --)
cd "$repo" || exit 1

{
  echo "start $(date -Is) host $(hostname) commit $(git rev-parse --short HEAD) kernel $(uname -r)"
  echo "driver $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
  echo "linked cuBLASLt: $(ldd "${MLXCEL_AB_BIN_DIR:-$repo/target/release}/mlxcel-bench-engine" | grep -o '/\S*libcublasLt\S*')"
  echo "alt cuBLASLt: $(ls "$MLXCEL_AB_CUBLAS_DIR"/libcublasLt.so.13* 2>&1 | tr '\n' ' ')"
} | tee -a "$out/campaign.log"

engine() {  # engine <tag> <model> <max-tokens> <prompt-tokens...> -- <arm specs...>
  local tag=$1 m=$2 mt=$3; shift 3
  local pts=(); while [ "$1" != -- ]; do pts+=("$1"); shift; done; shift
  local arms=(); for s in "$@"; do arms+=(--arm "$s"); done
  echo "== $tag $m $(date +%T)" | tee -a "$out/campaign.log"
  "${lock[@]}" python3 scripts/engine_bench_rounds.py --bin "$here/bench_wrap.sh" --model "models/$m" \
    --prompt-tokens "${pts[@]}" --max-tokens "$mt" --rounds 5 --null-every-arm --hostgate \
    "${arms[@]}" --out "$out/${tag}_$m.jsonl" > "$out/${tag}_$m.txt" 2>&1
  echo "rc=$?" | tee -a "$out/campaign.log"
}

for stage in $stages; do
  case $stage in
    version)  # linked vs alternate cuBLASLt, gemv on (default decode path)
      for m in llama-3.2-1b-instruct-bf16 qwen3.5-9b-bf16 qwen3-1.7b-4bit; do
        engine version "$m" 128 256 4096 -- sys="--path cli" alt="--path cli --cublas-alt"
      done ;;
    gemv)  # MLX gemv vs cuBLASLt (linked and alternate) for B=1 decode
      for m in llama-3.2-1b-instruct-bf16 qwen3.5-9b-bf16 llama-3.1-8b-instruct-bf16; do
        engine gemv "$m" 256 256 -- gemv="--path cli" lt-sys="--path cli --no-gemv" \
          lt-alt="--path cli --no-gemv --cublas-alt"
      done ;;
    ladder)  # batched small-M decode through the server
      for m in llama-3.2-1b-instruct-bf16 llama-3.1-8b-instruct-bf16; do
        echo "== ladder $m $(date +%T)" | tee -a "$out/campaign.log"
        "${lock[@]}" python3 "$here/ladder_rounds.py" --model "models/$m" --rounds 3 --hostgate \
          --out "$out/ladder_$m.jsonl" > "$out/ladder_$m.txt" 2>&1
        echo "rc=$?" | tee -a "$out/campaign.log"
      done ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
echo "end $(date -Is)" | tee -a "$out/campaign.log"
