#!/usr/bin/env bash
set -euo pipefail

WT=${WT:-/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227}
OUT=${OUT:-/tmp/mlxcel-auto-plan/2227-run-parity/after-valid}
MODEL=${MODEL:-models/mlx/qwen3-1.7b-4bit}
BIN=${BIN:-$WT/target/release/mlxcel}
PROMPT=${PROMPT:-Reply with exactly one short sentence naming the capital of France.}
MAX_TOKENS=${MAX_TOKENS:-64}
BINARY_SOURCE_HEAD=${BINARY_SOURCE_HEAD:-7c8830cd5aa01198e6ff93b661f90c048ccaeb1d}
BEFORE=${BEFORE:-/tmp/mlxcel-auto-plan/2227-run-parity/before-valid/generated.txt}
EXTRACTOR=/tmp/mlxcel-auto-plan/2227-profile/extract-run-text.py

mkdir -p "$OUT"
cd "$WT"
test -x "$BIN"
test -s "$BEFORE"
test -f "$EXTRACTOR"
checkout_head=$(git rev-parse HEAD)
git cat-file -e "$BINARY_SOURCE_HEAD^{commit}"
while IFS= read -r changed; do
  case $changed in
    docs/*|TECHNICAL_REPORTS/*|.gitattributes|src/server/batch/scheduler_muse_glimmer_parallel_tests.rs|src/server/batch/scheduler_prompt_lookup_tests.rs) ;;
    *) echo "post-build path requires a fresh mlxcel release build: $changed" >&2; exit 2 ;;
  esac
done < <(git diff --name-only "$BINARY_SOURCE_HEAD" "$checkout_head")

{
  date --iso-8601=seconds
  hostname
  uname -a
  printf 'binary_source_head=%s\n' "$BINARY_SOURCE_HEAD"
  printf 'checkout_head=%s\n' "$checkout_head"
  printf 'checkout_status_begin\n'
  git status --short
  printf 'checkout_status_end\n'
  printf 'post_build_diff_begin\n'
  git diff --name-status "$BINARY_SOURCE_HEAD" "$checkout_head"
  printf 'post_build_diff_end\n'
  sha256sum "$BIN"
  stat --format='binary_size=%s binary_mtime=%y' "$BIN"
  printf 'mlx_pin='; scripts/ci/mlx_pinned_commit.sh
  sha256sum "$0" "$EXTRACTOR" "$BEFORE"
  nvidia-smi
} > "$OUT/provenance.txt" 2>&1

printf '%s' "$PROMPT" > "$OUT/prompt.txt"
{
  printf 'MLXCEL_B1_DECODE_PROFILE=<unset> '
  printf 'MLXCEL_DEVICE=gpu MLXCEL_SDPA_DETERMINISTIC=1 '
  printf '%q ' "$BIN" run "$MODEL" -p "$PROMPT" -n "$MAX_TOKENS" \
    --temp 0 --top-k 0 --top-p 1 --min-p 0 --seed 2227 --show-reasoning
  printf '\n'
} > "$OUT/command.txt"

gpu-lock run --tag issue-2227-run-after -- \
  env -u MLXCEL_B1_DECODE_PROFILE \
    MLXCEL_DEVICE=gpu MLXCEL_SDPA_DETERMINISTIC=1 \
    "$BIN" run "$MODEL" -p "$PROMPT" -n "$MAX_TOKENS" \
      --temp 0 --top-k 0 --top-p 1 --min-p 0 --seed 2227 --show-reasoning \
  > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"

python3 "$EXTRACTOR" "$OUT/stdout.txt" "$OUT/generated.txt"
sha256sum "$OUT/stdout.txt" "$OUT/stderr.txt" "$OUT/generated.txt" \
  > "$OUT/output-sha256.txt"
cmp "$BEFORE" "$OUT/generated.txt"
printf 'byte_identical_to_before=yes\n' > "$OUT/comparison.txt"
