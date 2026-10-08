set -u
F=$F
R=~/Development/backend.ai/mlxcel/scripts
M=~/models/mlx
BASE=~/Development/backend.ai/wt-epic-2166-base/target/release/mlxcel
NEW=~/Development/backend.ai/wt-epic-2166-final/target/release/mlxcel
cd ~/Development/backend.ai/mlxcel
echo "start $(date -u +%FT%TZ)"
{ hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; git -C ~/Development/backend.ai/wt-epic-2166-base log --oneline -1; git -C ~/Development/backend.ai/wt-epic-2166-final log --oneline -1; } > $F/environment.txt
# A. single-stream: pre-epic CxxGenerator (base-cli) vs final CLI (new-cli), plus server arms
for m in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
  python3 $R/engine_bench_rounds.py --bin $F/pick.sh --model $M/$m --prompt-tokens 256 8192 --rounds 5 --hostgate \
    --arm base-cli="--which base --path cli" \
    --arm new-cli="--which new --path cli" \
    --arm new-srv-dense="--which new --path server --decode-storage dense" \
    --arm base-srv="--which base --path server" \
    --arm new-srv="--which new --path server" \
    --out $F/single-$m.jsonl > $F/single-$m.txt 2>&1
  echo "rc=$? :: single $m $(date -u +%T)"
done
serve_wait() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/health >/dev/null 2>&1 && return 0; curl -sf http://127.0.0.1:$1/v1/models >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
serve_run() { # bin label port model extra... ; then runs "$CMD"
  local bin=$1 label=$2 port=$3 model=$4; shift 4
  if [ "$bin" = "$NEW" ]; then export MLXCEL_CCCL_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include/cccl MLXCEL_CUTLASS_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include; else unset MLXCEL_CCCL_DIR MLXCEL_CUTLASS_DIR; fi
  $bin serve -m $model --port $port --metrics "$@" > $F/server-$label.log 2>&1 &
  local pid=$!
  if serve_wait $port; then eval "$CMD" > $F/$label.txt 2>&1; echo "rc=$? :: $label $(date -u +%T)"; else echo "rc=9 :: $label server did not start"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}
# B. batched serving throughput, base vs new, server defaults (paged at --parallel 8), 3 alternating rounds
for r in 1 2 3; do
  for side in base new; do
    BIN=$BASE; [ $side = new ] && BIN=$NEW
    CMD="python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 1,4,8 --prompt-tokens 512 --max-tokens 128 --metrics"
    serve_run $BIN batched-qwen3-$side-r$r 18080 $M/qwen3-1.7b-4bit --parallel 8
    serve_run $BIN batched-llama-turbo4-$side-r$r 18080 $M/llama-3.2-1b-instruct-4bit --parallel 8 --kv-cache-mode turbo4
  done
done
# C. admission ITL: prefill chunk 2048 vs 512 on the final server, 4 streams + one 8192-token admission, 3 rounds
for r in 1 2 3; do
  for c in 2048 512; do
    CMD="python3 $R/bench_mixed_step_admission.py --port 18080 --streams 4 --stream-max-tokens 4096 --admit-prompt-tokens 8192 --expect any"
    serve_run $NEW admit-qwen3-c$c-r$r 18080 $M/qwen3-1.7b-4bit --parallel 8 --prefill-chunk-size $c
  done
done
# D. Gemma 3 storage under concurrency on the final server: paged (auto today) vs dense, 3 rounds
for r in 1 2 3; do
  for st in paged dense; do
    CMD="python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 1,4,8 --prompt-tokens 512 --max-tokens 128 --metrics"
    serve_run $NEW gemma3-$st-r$r 18080 $M/gemma-3-1b-it-4bit --parallel 8 --decode-storage-backend $st
  done
done
echo "end $(date -u +%FT%TZ)"
echo ALLDONE
