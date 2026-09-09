#!/usr/bin/env bash
# Output-equality gate for an A/B arm.
#
# The throughput half of an A/B says whether a change is faster. This is the
# other half: whether it still produces the same text. It exists because doing
# that comparison by hand keeps reintroducing the same two errors.
#
#   1. Sampling. `generation_config.json` can enable sampling on its own, and
#      then the comparison is between two samples, not two implementations, so
#      an unchanged arm "fails" and a broken one can pass. Both arms run at
#      `--temp 0` here, always.
#   2. The reasoning channel. `mlxcel generate` suppresses `<think>` content by
#      default, so a reasoning model whose generation is cut off inside the
#      channel prints an empty content channel. Comparing empty against empty
#      passes while comparing nothing at all, and reading it as breakage has
#      twice cost an afternoon (glm-4.1v, then nvidia-nemotron-3-nano-30b-a3b
#      during the RMS-norm A/B). Both arms run with `--show-reasoning` here, so
#      the comparison covers every generated token.
#
# A third error is structural rather than per-run: an output difference is only
# attributable to the arm if the baseline agrees with ITSELF. Some families are
# not bitwise stable run to run (the f16 reduction-order jitter class), and on
# those a difference between arms means nothing. The script therefore runs the
# baseline twice as a control and reports INCONCLUSIVE, not FAIL, when the
# control disagrees.
#
# Usage:
#   ./scripts/ab_output_equality.sh --baseline target/release/mlxcel.before \
#                                   --arm target/release/mlxcel \
#                                   --model models/mlx/granite-4.0-h-tiny-4bit
#
#   --model / --prompt repeat, and every (model, prompt) pair is checked.
#
# Producing the baseline binary: build at the unpatched commit, copy the binary
# aside, apply the patch, rebuild.
#
#   git stash && cargo build --release --features metal,accelerate --bin mlxcel
#   /bin/cp target/release/mlxcel target/release/mlxcel.before
#   git stash pop && cargo build --release --features metal,accelerate --bin mlxcel
#
# Exit status: 0 all pairs equal, 1 at least one pair differs, 2 at least one
# pair inconclusive (unstable baseline) and none differ.

set -uo pipefail

BASELINE_BIN=""
ARM_BIN="./target/release/mlxcel"
MAX_TOKENS=128
OUT_DIR=""
MODELS=()
PROMPTS=()

usage() {
  sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE_BIN="$2"; shift 2 ;;
    --arm) ARM_BIN="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --prompt) PROMPTS+=("$2"); shift 2 ;;
    -n|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$BASELINE_BIN" ]]; then
  echo "error: --baseline is required (the binary built without the change)" >&2
  exit 1
fi
for bin in "$BASELINE_BIN" "$ARM_BIN"; do
  [[ -x "$bin" ]] || { echo "error: not an executable: $bin" >&2; exit 1; }
done
if [[ "$(cd "$(dirname "$BASELINE_BIN")" && pwd)/$(basename "$BASELINE_BIN")" \
      == "$(cd "$(dirname "$ARM_BIN")" && pwd)/$(basename "$ARM_BIN")" ]]; then
  echo "error: --baseline and --arm are the same file; the comparison would be vacuous" >&2
  exit 1
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "error: at least one --model is required" >&2
  exit 1
fi
if [[ ${#PROMPTS[@]} -eq 0 ]]; then
  # Two shapes: a short factual completion, and a longer instruction that keeps
  # a reasoning model inside its channel for a while.
  PROMPTS=(
    "The capital of France is"
    "List three prime numbers greater than 100 and explain briefly how you checked each one."
  )
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d -t ab_output_equality)"
fi
mkdir -p "$OUT_DIR"

echo "baseline: $BASELINE_BIN"
echo "arm:      $ARM_BIN"
echo "tokens:   $MAX_TOKENS   (temp 0, --show-reasoning forced)"
echo "outputs:  $OUT_DIR"
echo

# Run one generation and reduce it to just the generated text.
#
# Everything up to and including the last `Generating...` line is loader banner,
# and the trailing bracketed lines are the timing block. The reasoning notice is
# stripped too, though `--show-reasoning` means it cannot fire here. Redirecting
# stdout to a file also makes `is_terminal()` false, so `--show-reasoning` emits
# no dim SGR codes and the captured text is plain.
run_one() {
  local bin="$1" model="$2" prompt="$3" out="$4"
  "$bin" generate -m "$model" -p "$prompt" -n "$MAX_TOKENS" --temp 0 --show-reasoning \
    > "$out.raw" 2> "$out.err"
  local status=$?
  awk 'f { print } /^Generating\.\.\.$/ { f = 1 }' "$out.raw" \
    | sed -E '/^\[Generated [0-9]+ tokens in /d; /^\[Profile Results\]$/d; /^\[All [0-9]+ generated tokens went to the reasoning channel;/d' \
    | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' \
    > "$out"
  return $status
}

fail=0
inconclusive=0
pair=0
printf '%-52s %-44s %s\n' "MODEL" "PROMPT" "RESULT"
for model in "${MODELS[@]}"; do
  model_tag="$(basename "$model")"
  for prompt in "${PROMPTS[@]}"; do
    pair=$((pair + 1))
    stem="$OUT_DIR/$(printf '%02d' "$pair")_${model_tag}"
    prompt_tag="$(printf '%s' "$prompt" | cut -c1-40)"

    if ! run_one "$BASELINE_BIN" "$model" "$prompt" "${stem}.baseline"; then
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "ERROR (baseline run failed, see ${stem}.baseline.err)"
      fail=1
      continue
    fi
    if ! run_one "$BASELINE_BIN" "$model" "$prompt" "${stem}.control"; then
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "ERROR (control run failed, see ${stem}.control.err)"
      fail=1
      continue
    fi
    if ! run_one "$ARM_BIN" "$model" "$prompt" "${stem}.arm"; then
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "ERROR (arm run failed, see ${stem}.arm.err)"
      fail=1
      continue
    fi

    if ! cmp -s "${stem}.baseline" "${stem}.control"; then
      # The baseline does not reproduce itself, so nothing can be attributed to
      # the arm on this pair. Say so rather than reporting a difference.
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "INCONCLUSIVE (baseline is not run-to-run stable)"
      inconclusive=1
      continue
    fi
    if cmp -s "${stem}.baseline" "${stem}.arm"; then
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "EQUAL"
    else
      printf '%-52s %-44s %s\n' "$model_tag" "$prompt_tag" "DIFFERS"
      fail=1
    fi
  done
done

echo
if [[ $fail -ne 0 ]]; then
  echo "result: at least one pair differs or errored. Diff the saved files under $OUT_DIR."
  exit 1
fi
if [[ $inconclusive -ne 0 ]]; then
  echo "result: no differences, but at least one baseline was not stable against itself."
  echo "        Those pairs prove nothing about the arm; pick a stable checkpoint or"
  echo "        use the teacher-forced logit trace instead (docs/benchmarks.md)."
  exit 2
fi
echo "result: every pair is byte-identical between baseline and arm."
