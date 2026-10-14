#!/usr/bin/env bash
# Run this whole script under gpu-lock on an otherwise quiet host. The driver
# performs its own --hostgate check before every arm.
set -euo pipefail

BASE_ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-base-2227
NEW_ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227
BASE_HEAD=4c44e317dcbc5a66548e6aed1cd01cdf36f97499
NEW_BINARY_SOURCE_HEAD=${NEW_BINARY_SOURCE_HEAD:-7c8830cd5aa01198e6ff93b661f90c048ccaeb1d}
OUT=${OUT:-/tmp/mlxcel-auto-plan/2227-acceptance}
DRIVER="$NEW_ROOT/scripts/engine_bench_rounds.py"
PICK="$OUT/pick.sh"
MODEL_ROOT=/home/inureyes/models/mlx
BASE_BIN="$BASE_ROOT/target/release/mlxcel-bench-engine"
NEW_BIN="$NEW_ROOT/target/release/mlxcel-bench-engine"

[[ $(git -C "$BASE_ROOT" rev-parse HEAD) == "$BASE_HEAD" ]]
new_checkout_head=$(git -C "$NEW_ROOT" rev-parse HEAD)
git -C "$NEW_ROOT" cat-file -e "$NEW_BINARY_SOURCE_HEAD^{commit}"
while IFS= read -r changed; do
  case $changed in
    docs/*|TECHNICAL_REPORTS/*|.gitattributes|src/server/batch/scheduler_muse_glimmer_parallel_tests.rs|src/server/batch/scheduler_prompt_lookup_tests.rs) ;;
    *) echo "post-build path requires a fresh release build: $changed" >&2; exit 2 ;;
  esac
done < <(git -C "$NEW_ROOT" diff --name-only "$NEW_BINARY_SOURCE_HEAD..$new_checkout_head")
test -x "$BASE_BIN"
test -x "$NEW_BIN"
test -x "$PICK"
test -f "$DRIVER"
unset MLXCEL_B1_DECODE_PROFILE MLXCEL_DEVICE MLXCEL_SDPA_DETERMINISTIC

for model in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
  test ! -e "$OUT/single-$model.jsonl"
  test ! -e "$OUT/single-$model.log"
done

{
  date --iso-8601=seconds
  hostname
  uname -a
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  printf 'base_source_head=%s\n' "$(git -C "$BASE_ROOT" rev-parse HEAD)"
  printf 'base_source_status_begin\n'
  git -C "$BASE_ROOT" status --short
  printf 'base_source_status_end\n'
  sha256sum "$BASE_BIN"
  stat --format='base_binary_size=%s base_binary_mtime=%y' "$BASE_BIN"
  printf 'new_binary_source_head=%s\n' "$NEW_BINARY_SOURCE_HEAD"
  printf 'new_checkout_head=%s\n' "$new_checkout_head"
  printf 'new_source_status_begin\n'
  git -C "$NEW_ROOT" status --short
  printf 'new_source_status_end\n'
  printf 'post_build_diff_begin\n'
  git -C "$NEW_ROOT" diff --name-status "$NEW_BINARY_SOURCE_HEAD" "$new_checkout_head"
  printf 'post_build_diff_end\n'
  sha256sum "$NEW_BIN"
  stat --format='new_binary_size=%s new_binary_mtime=%y' "$NEW_BIN"
  printf 'mlx_pin=%s\n' "$("$NEW_ROOT/scripts/ci/mlx_pinned_commit.sh")"
  printf 'MLXCEL_DEVICE=%s\n' "${MLXCEL_DEVICE-<unset>}"
  printf 'MLXCEL_SDPA_DETERMINISTIC=%s\n' "${MLXCEL_SDPA_DETERMINISTIC-<unset>}"
  printf 'base_is_ancestor_of_binary_source='; git -C "$NEW_ROOT" merge-base --is-ancestor "$BASE_HEAD" "$NEW_BINARY_SOURCE_HEAD" && echo yes
  printf 'source_diff_stat_begin\n'
  git -C "$NEW_ROOT" diff --stat "$BASE_HEAD..$NEW_BINARY_SOURCE_HEAD"
  printf 'source_diff_stat_end\n'
  sha256sum "$0" "$PICK" "$DRIVER" \
    "$OUT/binary-source-7c8830cd.patch" \
    "$OUT/post-build-test-only.diff" \
    "$OUT/binary-source-provenance.txt"
} > "$OUT/environment.txt" 2>&1

for model in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
  python3 "$DRIVER" \
    --bin "$PICK" \
    --model "$MODEL_ROOT/$model" \
    --prompt-tokens 256 8192 \
    --rounds 5 \
    --hostgate \
    --arm base-cli='--which base --path cli' \
    --arm new-srv-dense='--which new --path server --decode-storage dense' \
    --out "$OUT/single-$model.jsonl" \
    > "$OUT/single-$model.log" 2>&1
  printf 'rc=0 model=%s completed=%s\n' "$model" "$(date --iso-8601=seconds)" | tee -a "$OUT/run-status.txt"
done
