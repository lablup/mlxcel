set -u
F=$F
R=~/Development/backend.ai/mlxcel/scripts
M=~/models/mlx
BASE=~/Development/backend.ai/wt-epic-2166-base/target/release/mlxcel
NEW=~/Development/backend.ai/wt-epic-2166-final/target/release/mlxcel
cd ~/Development/backend.ai/mlxcel
echo "start $(date -u +%FT%TZ)"
{ hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; git -C ~/Development/backend.ai/wt-epic-2166-base log --oneline -1; git -C ~/Development/backend.ai/wt-epic-2166-final log --oneline -1; } > $F/environment.txt
serve_wait() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/health >/dev/null 2>&1 && return 0; curl -sf http://127.0.0.1:$1/v1/models >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
serve_run() { # bin label port model extra... ; then runs "$CMD"
  local bin=$1 label=$2 port=$3 model=$4; shift 4
  if [ "$bin" = "$NEW" ]; then export MLXCEL_CCCL_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include/cccl MLXCEL_CUTLASS_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include; else unset MLXCEL_CCCL_DIR MLXCEL_CUTLASS_DIR; fi
  $bin serve -m $model --port $port --metrics "$@" > $F/server-$label.log 2>&1 &
  local pid=$!
  if serve_wait $port; then eval "$CMD" > $F/$label.txt 2>&1; echo "rc=$? :: $label $(date -u +%T)"; else echo "rc=9 :: $label server did not start"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}
# C. admission ITL: prefill chunk 2048 vs 512 on the final server, 4 streams + one 8192-token admission, 3 rounds
for r in 1 2 3; do
  for c in 2048 512; do
    CMD="python3 $R/bench_mixed_step_admission.py --port 18080 --streams 4 --stream-max-tokens 4096 --admit-prompt-tokens 8192 --expect any"
    serve_run $NEW admit-llama-c$c-r$r 18080 $M/llama-3.2-1b-instruct-4bit --parallel 8 --prefill-chunk-size $c --ignore-eos
  done
done
echo ALLDONE
