#!/usr/bin/env bash
set -u
S=${S:-$(cd "$(dirname "$0")/../branch" && pwd)}
BIN=${BIN:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/target/release/mlxcel-server}
M=${M:-models/mlx}
for arm in classic w4; do
  extra=()
  [ $arm = w4 ] && extra=(--model-draft $M/qwen3.5-4b-dflash --draft-kind dflash --draft-block-size 4)
  MLXCEL_Q35_ROPE_BRANCH_LOG=1 MLXCEL_MTP_ALLOW_INEXACT=1 $BIN -m $M/qwen3.5-4b-4bit --port 18991 --max-batch-size 1 --ignore-eos "${extra[@]}" > $S/branch-$arm.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -sf localhost:18991/health >/dev/null && break; sleep 1; done
  curl -s localhost:18991/v1/completions -H 'content-type: application/json' -d '{"model":"qwen3.5-4b-4bit","prompt":"def fibonacci(n):","max_tokens":12,"temperature":0}' > $S/branch-$arm.json
  kill $pid; wait $pid
done
