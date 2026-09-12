#!/usr/bin/env bash
# Post-reboot #1798 chain. Phases: ab | startup | qbatched | qnsys | q35 | qprefill
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
BD=$WT/target/release/mlxcel-bench-decode
CLI=$WT/target/release/mlxcel
SRV=$WT/target/release/mlxcel-server
M=$WT/models/mlx
PRESETS=(--preset nograph=MLX_USE_CUDA_GRAPHS=0 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 \
         --preset restore=MLX_MAX_OPS_PER_BUFFER=20,MLX_MAX_MB_PER_BUFFER=25)
nvrm() { journalctl -k -b 0 --no-pager -q 2>/dev/null | grep -c NV_ERR_NO_MEMORY; }
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
for phase in "$@"; do
  $R/idle_gate.sh
  echo "PHASE_START $phase $(date -Is) load=$(cat /proc/loadavg) nvrm=$(nvrm)"
  case $phase in
    ab)
      python3 $R/bench_bd.py --bin $BD --model $M/laguna-xs-2.1-nvfp4 --prompt-file $R/prompt_code0.txt --no-chat-template \
        --out $R/results_ab.jsonl --tag ab --configs default,restore,both --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}"
      python3 $R/bench_bd.py --bin $BD --model $M/qwen3.5-4b-4bit --prompt-file $R/prompt_code0.txt --no-chat-template \
        --out $R/results_ab.jsonl --tag ab-dense --configs default,restore --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}" ;;
    startup)
      for m in laguna-xs-2.1-nvfp4 qwen3.5-4b-4bit; do
        echo "== generate $m (no env)"; MLX_ENABLE_TF32=0 $CLI generate -m $M/$m -p "def f():" -n 4 --no-chat-template 2>&1 | grep -E "CUDA graph budget|CUDA compute capability|tok/s" | head -3
        echo "== generate $m (operator MLX_MAX_OPS_PER_BUFFER=20 only)"; MLX_MAX_OPS_PER_BUFFER=20 $CLI generate -m $M/$m -p "def f():" -n 4 --no-chat-template 2>&1 | grep -E "CUDA graph budget" | head -1
        echo "== generate $m (operator both restored)"; MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=25 $CLI generate -m $M/$m -p "def f():" -n 4 --no-chat-template 2>&1 | grep -cE "CUDA graph budget"
      done
      echo "== server laguna (no env)"; ( $SRV -m $M/laguna-xs-2.1-nvfp4 --port 18799 > $R/server_startup_laguna.log 2>&1 & echo $! > $R/srv.pid ); for i in $(seq 1 120); do curl -sf http://127.0.0.1:18799/health >/dev/null && break; sleep 1; done; kill $(cat $R/srv.pid); sleep 3; grep -E "CUDA graph budget|compute_capability" $R/server_startup_laguna.log | head -3
      echo "== server qwen3.5-4b (no env)"; ( $SRV -m $M/qwen3.5-4b-4bit --port 18799 > $R/server_startup_qwen.log 2>&1 & echo $! > $R/srv.pid ); for i in $(seq 1 120); do curl -sf http://127.0.0.1:18799/health >/dev/null && break; sleep 1; done; kill $(cat $R/srv.pid); sleep 3; grep -cE "CUDA graph budget" $R/server_startup_qwen.log ;;
    qbatched)
      python3 $R/bench_srv.py --server $SRV --client $WT/scripts/bench_serving_concurrency.py --model $M/qwen3-30b-a3b-4bit \
        --out $R/results_batched2.jsonl --tag batched --configs default,nograph,both --rounds 2 --concurrency 1,4,8 --max-tokens 200 --prompt-tokens 128 --max-batch 8 "${PRESETS[@]}" ;;
    qnsys)
      for cfg in default both; do for n in 400 200; do
        python3 $R/bench_srv.py --server $SRV --client $WT/scripts/bench_serving_concurrency.py --model $M/qwen3-30b-a3b-4bit \
          --out $R/results_nsys_srv.jsonl --tag "nsys-srv-$cfg-$n" --configs $cfg --rounds 1 --concurrency 4 --max-tokens $n --prompt-tokens 128 --max-batch 8 \
          --wrap "nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $R/nsys_srv_${cfg}_${n}" "${PRESETS[@]}"
      done; done ;;
    q35)
      python3 $R/bench_srv.py --server $SRV --client $WT/scripts/bench_serving_concurrency.py --model $M/qwen3.5-35b-a3b-4bit \
        --out $R/results_batched2.jsonl --tag batched --configs default,both,nograph --rounds 3 --concurrency 1,4,8 --max-tokens 200 --prompt-tokens 128 --max-batch 8 "${PRESETS[@]}" ;;
    qprefill)
      for m in qwen3-30b-a3b-4bit qwen3.5-35b-a3b-4bit; do
        python3 $R/bench_bd.py --bin $BD --model $M/$m --prompt-file $R/prompt_code0.txt --no-chat-template \
          --out $R/results_prefill2.jsonl --tag prefill2048 --configs default,both --rounds 2 --warmup 1 --max-tokens 16 --prompt-tokens 2048 "${PRESETS[@]}"
      done ;;
  esac
  echo "PHASE_END $phase $(date -Is) load=$(cat /proc/loadavg) nvrm=$(nvrm)"
done
echo SWEEP7_DONE
