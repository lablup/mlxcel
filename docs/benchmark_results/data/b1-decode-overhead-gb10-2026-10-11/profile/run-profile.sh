#!/usr/bin/env bash
set -euo pipefail

WT=${WT:-/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227}
OUT=${OUT:-/tmp/mlxcel-auto-plan/2227-profile}
MODEL=${MODEL:-models/mlx/qwen3-1.7b-4bit}
BIN=${BIN:-$WT/target/release/mlxcel-bench-engine}
PROMPT_TOKENS=${PROMPT_TOKENS:-256}
MAX_TOKENS=${MAX_TOKENS:-128}
WARMUP_TOKENS=${WARMUP_TOKENS:-16}

mkdir -p "$OUT"
cd "$WT"
test -x "$BIN"

{
  date --iso-8601=seconds
  hostname
  uname -a
  git rev-parse HEAD
  git status --short
  sha256sum "$BIN"
  stat --format='binary_size=%s binary_mtime=%y' "$BIN"
  rg -n '81ba1c6a0e50a9268b931579c2d4f1158b9aab5a' CMakeLists.txt cmake src 2>/dev/null || true
  nsys --version
  nsys status --environment
  nvidia-smi
} >"$OUT/environment.txt" 2>&1

COMMON=(
  --model "$MODEL"
  --prompt-tokens "$PROMPT_TOKENS"
  --max-tokens "$MAX_TOKENS"
  --warmup-tokens "$WARMUP_TOKENS"
  --decode-storage dense
)

gpu-lock run --tag issue-2227-profile-cli -- \
  env MLXCEL_SDPA_DETERMINISTIC=1 "$BIN" \
    --path cli "${COMMON[@]}" --label profile-cli \
  >"$OUT/cli-plain.log" 2>&1

gpu-lock run --tag issue-2227-profile-server -- \
  env MLXCEL_SDPA_DETERMINISTIC=1 MLXCEL_B1_DECODE_PROFILE=1 "$BIN" \
    --path server "${COMMON[@]}" --label profile-server \
  >"$OUT/server-plain.log" 2>&1

NSYS=(
  nsys profile
  --trace=cuda,osrt
  --sample=none
  --event-sample=none
  --backtrace=none
  --cpuctxsw=none
  --gpuctxsw=false
  --force-overwrite=true
)

gpu-lock run --tag issue-2227-nsys-cli -- \
  "${NSYS[@]}" -o "$OUT/cli-nsys" \
  env MLXCEL_SDPA_DETERMINISTIC=1 "$BIN" \
    --path cli "${COMMON[@]}" --label profile-cli \
  >"$OUT/cli-nsys.log" 2>&1

gpu-lock run --tag issue-2227-nsys-server -- \
  "${NSYS[@]}" -o "$OUT/server-nsys" \
  env MLXCEL_SDPA_DETERMINISTIC=1 MLXCEL_B1_DECODE_PROFILE=1 "$BIN" \
    --path server "${COMMON[@]}" --label profile-server \
  >"$OUT/server-nsys.log" 2>&1

for arm in cli server; do
  nsys stats --force-export=true \
    --report cuda_api_sum,cuda_gpu_kern_sum,osrt_sum \
    "$OUT/$arm-nsys.nsys-rep" >"$OUT/$arm-nsys-stats.txt"
done

python3 "$OUT/parse-profile.py" "$OUT/server-plain.log" \
  >"$OUT/server-host-breakdown.txt"
