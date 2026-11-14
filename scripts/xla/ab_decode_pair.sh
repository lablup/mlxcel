#!/usr/bin/env bash
# Paired A/B for two builds of the OpenXLA / IREE backend, on any HAL device.
#
# A single before/after run of `mlxcel generate --profile` is not evidence: the
# same binary moves 5 to 10 percent between runs on a thermally loaded host, so
# an apparent win can be drift. This runs the two arms interleaved, reports
# medians, and says how many paired rounds the arm won. It is the harness the
# decode changes of #3028, #3042 and #3043 were measured with.
#
# Three guards exist because each of them has produced a wrong number before:
#
#   1. Identical arms. A `cargo build` that no-ops leaves both arms the same
#      binary and every round "passes". Refused unless --allow-identical.
#   2. Whole-run accounting. `generate --profile` on the OpenXLA backend used to
#      report `Prefill: 0.00 ms` and fold prefill into the decode rate (#3042),
#      which made decode depend on the token budget. A zero prefill window is
#      refused here unless --allow-whole-run-accounting, because it means the
#      "decode" figure is not a decode figure.
#   3. Cold compilation. The first run of a fresh context compiles the graphs
#      and creates the vmfb cache, so each arm is warmed up before measuring.
#
# Equality is not reimplemented: the text half is delegated to
# scripts/ab_output_equality.sh, which already handles sampling and the
# reasoning channel.
#
# Usage:
#   eval "$(scripts/iree/setup-cuda.sh --env)"      # or setup-rocm.sh / setup-macos.sh
#   scripts/xla/ab_decode_pair.sh --model models/llama-3.2-1b-instruct-4bit \
#                                 --device hip --old main --new HEAD
#
# The eval line is optional: when MLXCEL_XLA_IREE_COMPILE is unset the script
# applies the setup script matching --device itself, so a fresh shell can run
# the command above without it. Builds and both arms inherit that environment.
#
# Options:
#   --model <dir>        checkpoint directory (required)
#   --device <name>      IREE HAL device: cuda, hip, metal, local-task
#                        (default: $MLXCEL_XLA_DEVICE or local-task)
#   --old <ref|binary>   baseline arm: a git ref built in a worktree, or a path
#                        to an already-built binary (default: origin/main)
#   --new <ref|binary>   arm under test, same forms (default: the working tree)
#   --tokens <n>         decode budget (default: 128)
#   --rounds <n>         interleaved rounds (default: 6)
#   --capacity <n>       MLXCEL_XLA_CONTEXT_CAPACITY (default: 512)
#   --quant <mode>       MLXCEL_XLA_QUANT, e.g. native. Note that `native` is
#                        HIP-only: on CUDA it is ignored and the f16 path runs,
#                        so a CUDA figure is not comparable with a HIP one.
#   --features <list>    cargo features for the builds (default: xla-iree)
#   --prompt <text>      prompt (default: "Hello, how are you today?")
#   --lock <prefix>      command prefix that serializes GPU use, including the
#                        `--` that ends its own options. Defaults to
#                        `gpu-lock run --tag xla-ab --` when gpu-lock exists and
#                        the device is cuda, to `scripts/rocm_gpu_guard.sh --hold --`
#                        when the device is hip, and to nothing otherwise.
#   --skip-equality      skip the greedy-text comparison
#   --allow-identical    permit two arms holding the same bytes (for harness work)
#   --allow-whole-run-accounting
#                        permit a zero prefill window in the profile output
#   --keep-worktrees     leave the built worktrees in place
#   -h, --help           this text
#
# Exit status: 0 the A/B ran and the equality half passed (or was skipped),
# 1 a guard refused, the A/B failed, or the outputs differ.

set -uo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "ab_decode_pair: not inside a git work tree" >&2
  exit 1
}
cd "$REPO_ROOT" || exit 1

MODEL=""
DEVICE="${MLXCEL_XLA_DEVICE:-local-task}"
OLD_ARG="origin/main"
NEW_ARG=""           # "" means the working tree
TOKENS=128
ROUNDS=6
CAPACITY=512
QUANT=""
FEATURES="xla-iree"
PROMPT="Hello, how are you today?"
LOCK=""
SKIP_EQUALITY=0
ALLOW_IDENTICAL=0
ALLOW_WHOLE_RUN=0
KEEP_WORKTREES=0

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --old) OLD_ARG="${2:-}"; shift 2 ;;
    --new) NEW_ARG="${2:-}"; shift 2 ;;
    --tokens) TOKENS="${2:-}"; shift 2 ;;
    --rounds) ROUNDS="${2:-}"; shift 2 ;;
    --capacity) CAPACITY="${2:-}"; shift 2 ;;
    --quant) QUANT="${2:-}"; shift 2 ;;
    --features) FEATURES="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --lock) LOCK="${2:-}"; shift 2 ;;
    --skip-equality) SKIP_EQUALITY=1; shift ;;
    --allow-identical) ALLOW_IDENTICAL=1; shift ;;
    --allow-whole-run-accounting) ALLOW_WHOLE_RUN=1; shift ;;
    --keep-worktrees) KEEP_WORKTREES=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "ab_decode_pair: unknown argument: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ab_decode_pair: --model is required" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "ab_decode_pair: model directory not found: $MODEL" >&2
  exit 1
fi
if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]] || (( ROUNDS < 1 )); then
  echo "ab_decode_pair: --rounds must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TOKENS" =~ ^[0-9]+$ ]] || (( TOKENS < 2 )); then
  echo "ab_decode_pair: --tokens must be an integer of at least 2" >&2
  exit 1
fi

# --- Toolchain env ----------------------------------------------------------
# The builds and both arms need the compiler and HAL runtime paths the per-device
# setup scripts print. Apply them when the caller has not, so the documented
# one-liner works on a fresh shell; an already-exported environment is left alone.
if [[ -z "${MLXCEL_XLA_IREE_COMPILE:-}" ]]; then
  setup=""
  case "$DEVICE" in
    hip) setup=scripts/iree/setup-rocm.sh ;;
    cuda) setup=scripts/iree/setup-cuda.sh ;;
    metal) setup=scripts/iree/setup-macos.sh ;;
  esac
  if [[ -n "$setup" && -x "$setup" ]]; then
    echo "ab_decode_pair: applying $setup --env" >&2
    eval "$("$setup" --env)"
  elif [[ -n "$setup" ]]; then
    echo "ab_decode_pair: $setup is missing; export the toolchain env yourself" >&2
  fi
fi

# --- GPU serialization ------------------------------------------------------
if [[ -z "$LOCK" ]]; then
  if [[ "$DEVICE" == "cuda" ]] && command -v gpu-lock >/dev/null 2>&1; then
    LOCK="gpu-lock run --tag xla-ab --"
  elif [[ "$DEVICE" == "hip" && -x scripts/rocm_gpu_guard.sh ]]; then
    LOCK="scripts/rocm_gpu_guard.sh --hold --"
  fi
fi
read -r -a LOCK_CMD <<< "$LOCK"

# --- Arm resolution ---------------------------------------------------------
# A slash means "already-built binary", anything else means "git ref to build".
is_binary_arg() { [[ "$1" == */* ]]; }

WORKTREES=()
cleanup() {
  local wt
  if (( KEEP_WORKTREES )); then
    for wt in "${WORKTREES[@]}"; do
      echo "kept: $wt" >&2
    done
    return
  fi
  for wt in "${WORKTREES[@]}"; do
    git worktree remove --force "$wt" >/dev/null 2>&1 || true
  done
  git worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

build_arm() {
  # $1 = ref, prints the binary path
  local ref=$1 wt
  wt=$(mktemp -d "${TMPDIR:-/tmp}/mlxcel-ab-XXXXXX")
  rm -rf "$wt"
  git worktree add --detach "$wt" "$ref" >&2 || return 1
  WORKTREES+=("$wt")
  ( cd "$wt" && cargo build --release --features "$FEATURES" --bin mlxcel >&2 ) || return 1
  printf '%s/target/release/mlxcel\n' "$wt"
}

resolve_arm() {
  # $1 = argument, $2 = label, prints the binary path
  local arg=$1 label=$2
  if is_binary_arg "$arg"; then
    [[ -x "$arg" ]] || { echo "ab_decode_pair: $label binary is not executable: $arg" >&2; return 1; }
    printf '%s\n' "$arg"
    return 0
  fi
  build_arm "$arg"
}

if [[ -z "$NEW_ARG" ]]; then
  echo "building the working tree" >&2
  cargo build --release --features "$FEATURES" --bin mlxcel >&2 || exit 1
  NEW_BIN="$REPO_ROOT/target/release/mlxcel"
else
  NEW_BIN=$(resolve_arm "$NEW_ARG" new) || exit 1
fi
OLD_BIN=$(resolve_arm "$OLD_ARG" old) || exit 1

if /usr/bin/cmp -s "$OLD_BIN" "$NEW_BIN" && (( ! ALLOW_IDENTICAL )); then
  echo "ab_decode_pair: the two arms hold the same bytes, so the A/B would measure noise." >&2
  echo "  old: $OLD_BIN" >&2
  echo "  new: $NEW_BIN" >&2
  echo "  A cargo build that no-ops does this; rebuild the working tree, or pass --allow-identical." >&2
  exit 1
fi

export MLXCEL_BACKEND=xla
export MLXCEL_XLA_DEVICE="$DEVICE"
export MLXCEL_XLA_CONTEXT_CAPACITY="$CAPACITY"
if [[ -n "$QUANT" ]]; then
  export MLXCEL_XLA_QUANT="$QUANT"
else
  unset MLXCEL_XLA_QUANT
fi

echo "ab_decode_pair: device=$DEVICE tokens=$TOKENS rounds=$ROUNDS capacity=$CAPACITY quant=${QUANT:-none}" >&2
echo "  old: $OLD_BIN" >&2
echo "  new: $NEW_BIN" >&2
[[ -n "$LOCK" ]] && echo "  lock: $LOCK" >&2

# --- Measurement ------------------------------------------------------------
# One run, reduced to "tok/s ms prefill_ms". Nothing here reads the line's ms
# field as a rate: that mix-up once turned a 10 percent win into a reported
# regression.
measure() {
  local bin=$1 out dec dec_ms pre
  out=$("${LOCK_CMD[@]+"${LOCK_CMD[@]}"}" env \
      MLXCEL_BACKEND="$MLXCEL_BACKEND" \
      MLXCEL_XLA_DEVICE="$MLXCEL_XLA_DEVICE" \
      MLXCEL_XLA_CONTEXT_CAPACITY="$MLXCEL_XLA_CONTEXT_CAPACITY" \
      ${MLXCEL_XLA_QUANT:+MLXCEL_XLA_QUANT="$MLXCEL_XLA_QUANT"} \
      "$bin" generate -m "$MODEL" -p "$PROMPT" -n "$TOKENS" --profile 2>&1)
  dec=$(printf '%s' "$out" | grep -oE 'Decode:.*' | tail -1)
  dec_ms=$(printf '%s' "$dec" | grep -oE '[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
  dec_tps=$(printf '%s' "$dec" | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+')
  pre=$(printf '%s' "$out" | grep -oE 'Prefill:.*' | tail -1 | grep -oE '[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
  printf '%s %s %s\n' "${dec_tps:-NA}" "${dec_ms:-NA}" "${pre:-NA}"
}

# Warm-up: creates the vmfb cache and the persistent buffers for each arm.
for arm in old new; do
  bin=$([[ $arm == old ]] && printf '%s' "$OLD_BIN" || printf '%s' "$NEW_BIN")
  warm=$(measure "$bin")
  pre=$(printf '%s' "$warm" | awk '{print $3}')
  if [[ "$pre" == "NA" ]] || [[ "$pre" == "0.00" ]] || [[ "$pre" == "0.0" ]]; then
    if (( ! ALLOW_WHOLE_RUN )); then
      echo "ab_decode_pair: the $arm arm reports Prefill: ${pre:-NA} ms." >&2
      echo "  A zero prefill window means the decode rate covers the whole run (#3042)," >&2
      echo "  so this A/B would compare two different quantities. Use a build with the" >&2
      echo "  phase accounting fix, or pass --allow-whole-run-accounting to compare anyway." >&2
      exit 1
    fi
    echo "ab_decode_pair: WARNING: the $arm arm reports no prefill window; the rate is whole-run" >&2
  fi
done

rm -f /tmp/ab_decode_pair_old.txt /tmp/ab_decode_pair_new.txt
printf 'round old_tok_s new_tok_s old_ms new_ms\n'
fails=0
for r in $(seq 1 "$ROUNDS"); do
  o=$(measure "$OLD_BIN"); n=$(measure "$NEW_BIN")
  o_tps=$(printf '%s' "$o" | awk '{print $1}'); o_ms=$(printf '%s' "$o" | awk '{print $2}')
  n_tps=$(printf '%s' "$n" | awk '{print $1}'); n_ms=$(printf '%s' "$n" | awk '{print $2}')
  printf '%s %s %s %s %s\n' "$r" "$o_tps" "$n_tps" "$o_ms" "$n_ms"
  [[ "$o_tps" == NA || "$n_tps" == NA ]] && fails=$((fails + 1))
  printf '%s\n' "$o_tps" >> /tmp/ab_decode_pair_old.txt
  printf '%s\n' "$n_tps" >> /tmp/ab_decode_pair_new.txt
done

python3 - "$ROUNDS" <<'PY'
import statistics, sys
from pathlib import Path

def load(path):
    vals = []
    for line in Path(path).read_text().split():
        if line in ("", "NA"):
            continue
        try:
            vals.append(float(line))
        except ValueError:
            pass
    return vals

old = load('/tmp/ab_decode_pair_old.txt')
new = load('/tmp/ab_decode_pair_new.txt')
if not old or not new:
    print("ab_decode_pair: no usable measurements")
    sys.exit(1)
mo, mn = statistics.median(old), statistics.median(new)
wins = sum(1 for o, n in zip(old, new) if n > o)
n = min(len(old), len(new))
print(f"old median {mo:.2f} tok/s (n={len(old)})  new median {mn:.2f} tok/s (n={len(new)})  "
      f"delta {100 * (mn - mo) / mo:+.2f}%  rounds won by new {wins}/{n}")
PY

# --- Output equality --------------------------------------------------------
if (( SKIP_EQUALITY )); then
  echo "ab_decode_pair: text equality skipped (--skip-equality)"
else
  extra=()
  (( ALLOW_IDENTICAL )) && extra+=(--allow-identical-binaries)
  "${LOCK_CMD[@]+"${LOCK_CMD[@]}"}" env \
      MLXCEL_BACKEND="$MLXCEL_BACKEND" \
      MLXCEL_XLA_DEVICE="$MLXCEL_XLA_DEVICE" \
      MLXCEL_XLA_CONTEXT_CAPACITY="$MLXCEL_XLA_CONTEXT_CAPACITY" \
      ${MLXCEL_XLA_QUANT:+MLXCEL_XLA_QUANT="$MLXCEL_XLA_QUANT"} \
      scripts/ab_output_equality.sh --baseline "$OLD_BIN" --arm "$NEW_BIN" \
      --model "$MODEL" --prompt "$PROMPT" "${extra[@]+"${extra[@]}"}"
  eq_rc=$?
  case "$eq_rc" in
    0) echo "ab_decode_pair: text EQUAL" ;;
    2) echo "ab_decode_pair: text INCONCLUSIVE (the baseline disagrees with itself)" >&2 ;;
    *) echo "ab_decode_pair: text DIFFERS" >&2 ;;
  esac
  [[ "$eq_rc" == 0 ]] || exit 1
fi

(( fails == 0 )) || { echo "ab_decode_pair: $fails round(s) produced no measurement" >&2; exit 1; }
exit 0
