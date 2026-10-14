#!/usr/bin/env bash
# Isolating measurement for the MoE prefill rows that sit far below their
# 2026-10-05 pages on both builds (issue #2192): granite-4.0-h-tiny-4bit and
# Mixtral-8x7B-Instruct-v0.1-4bit at pp512 on main, default vs
# MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0 vs MLXCEL_FUSED_MOE=0, 2 rounds, plus
# one rocprofv3 kernel-stats run of granite so the prefill kernel is named.
# Usage: moe_prefill_session.sh REPO_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; OUT="$2"; MODELS_DIR="$3"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT/guard" "$OUT/raw"
export ROOT OUT MODELS_DIR GUARD
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  echo "lock held ($(date -u +%FT%TZ))"
  g() { "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/guard/$1.log" -- "${@:2}"; }
  CSV="$OUT/moe_prefill.csv"; echo "model,round,arm,prefill_tok_s,decode_tok_s,status" > "$CSV"
  for r in 1 2; do
    for model in granite-4.0-h-tiny-4bit Mixtral-8x7B-Instruct-v0.1-4bit; do
      for arm in default no-expert-batched no-fused-moe; do
        envs=()
        [[ "$arm" == no-expert-batched ]] && envs=(MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0)
        [[ "$arm" == no-fused-moe ]] && envs=(MLXCEL_FUSED_MOE=0)
        log="$OUT/raw/${model}_r${r}_${arm}.log"
        g "${model}_r${r}_${arm}" env "${envs[@]}" "$ROOT/target/release/mlxcel-bench-decode" -m "$MODELS_DIR/$model" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 512 > "$log" 2>&1
        rc=$?
        p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
        d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
        echo "$model,$r,$arm,$p,$d,$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$CSV"
        sleep 5
      done
    done
  done
  cat "$CSV"
  cd "$ROOT" && scripts/rocm_decode_profile.sh --out "$OUT/profile-granite" --idle-secs 5 --no-plain "$MODELS_DIR/granite-4.0-h-tiny-4bit" > "$OUT/profile_granite.log" 2>&1
  echo "granite profile rc=$?"
  ls "$OUT/profile-granite"
'
echo "lock released ($(date -u +%FT%TZ))"
