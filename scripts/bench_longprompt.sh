#!/usr/bin/env bash
# Long-prompt prefill benchmark sweep (epic #623 #624).
#
# Runs a representative subset of models across a prompt-length ladder so that
# prefill throughput is measured in the matmul-bound regime rather than the
# launch-overhead-bound regime that short (8-66 token) prompts produce. Each
# cell reuses scripts/bench_decode.sh --prompt-tokens N, so warmup, OOM
# classification, and the CSV schema are identical to the standard harness; the
# only added column is prompt_target_len.
#
# Usage:
#   ./scripts/bench_longprompt.sh
#   ./scripts/bench_longprompt.sh --lengths "512 2048 8192"
#   ./scripts/bench_longprompt.sh --models "llama-3.1-8b-4bit qwen2.5-7b-4bit"
#   ./scripts/bench_longprompt.sh --output benchmarks/custom.csv
#
# Default output: benchmarks/{backend}_{hardware}_longprompt_{YYYY-MM-DD}.csv
#   e.g. benchmarks/cuda_gb10_longprompt_2026-07-03.csv
#
# A model+length cell that OOMs (common at 32768 for large MoE models) is
# recorded with the usual SKIP:oom / SKIP:oom_estimate classification and the
# sweep continues.
#
# --- OOM avoidance (issue #807) ---------------------------------------------
#
# Near-OOM CUDA runs have repeatedly left the GB10 driver in a degraded state
# (NVRM NV_ERR_NO_MEMORY accumulating for hours, eventually wedging the
# kernel), so cells that are certain to OOM should never be launched at all.
# Both layers below are purely empirical: they react to an observed SKIP:oom
# cell, they never predict from a memory model. Only the SKIP:oom
# classification (a true OOM at load/run time, see is_oom_failure() in
# bench_decode.sh) feeds either layer; FAIL:bench, timeout (exit 124),
# SKIP:oom_estimate, and cells for a missing ./models symlink or missing model
# directory never do.
#
# 1. Monotonic backstop (within one sweep run). Once a (model, len) cell
#    classifies as SKIP:oom, every remaining cell for that same model with
#    length >= the OOM'd length is skipped before launch and logged as an
#    explicit SKIP:oom_backstop CSV row (same 16-field layout as other rows)
#    plus a stderr message.
#
# 2. Persistent OOM record (across runs). Each true SKIP:oom cell is appended
#    to a gitignored record file at benchmarks/.oom-record, one line per
#    event:
#      model,prompt_target_len,date,commit,peak_bytes,source_csv
#    `peak_bytes` is always blank today (the harness does not capture peak
#    memory); `commit` is `git rev-parse --short HEAD` (or "unknown" outside a
#    git checkout). On later runs, any cell whose model has a recorded OOM at
#    a length <= the cell's length is pre-skipped and logged as a
#    SKIP:oom_record row (the stderr message includes the matching record
#    line). Set MLXCEL_BENCH_IGNORE_OOM_RECORD=1 to disable the pre-skip for
#    one run; new SKIP:oom cells are still appended even when the pre-skip is
#    disabled. A missing or empty record file is treated as "no records"; a
#    malformed line (wrong field count, non-numeric length) is skipped with a
#    stderr warning rather than aborting the sweep.
#
#    Staleness / pruning: records go stale whenever memory behavior improves.
#    For example, the gemma-4-31b 32768-token OOM that originally motivated
#    this design was fixed by the chunked-prefill change in PR #676, so a
#    record line dated before that merge is no longer valid evidence. The
#    per-line date and commit exist precisely so stale entries can be found
#    and removed by hand, e.g.:
#      grep -v '^gemma-4-31b-it-4bit,' benchmarks/.oom-record > /tmp/pruned \
#        && mv /tmp/pruned benchmarks/.oom-record
#    or by deleting lines whose date/commit predate a known fix. There is no
#    automatic expiry; MLXCEL_BENCH_IGNORE_OOM_RECORD=1 is the low-effort way
#    to re-test a single run without editing the file.
#
# 3. MLXCEL_BENCH_FAKE_OOM="model:len[,model:len...]" (testing hook). When the
#    current cell's (model, len) pair is listed, the sweep skips invoking
#    bench_decode.sh entirely and synthesizes a cell result containing a
#    SKIP:oom row, so the backstop and persistent-record logic can be
#    exercised deterministically without a real OOM.

set -euo pipefail

# This script uses associative arrays (declare -A), a bash 4+ feature. Stock
# macOS ships bash 3.2 as /bin/bash, and this harness explicitly supports
# Darwin/Metal (see detect_hardware_short below), so re-exec under a newer
# bash when one is on PATH (Homebrew installs it at /opt/homebrew/bin or
# /usr/local/bin). If none is found, fail with an actionable message instead
# of the cryptic "declare: -A: invalid option" abort bash 3.2 would emit. The
# guard itself is bash-3-compatible and only re-execs into a bash whose major
# version is >= 4, so it cannot loop.
if [[ -z "${BASH_VERSINFO:-}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  for _alt_bash in /opt/homebrew/bin/bash /usr/local/bin/bash "$(command -v bash 2>/dev/null || true)"; do
    if [[ -n "$_alt_bash" && -x "$_alt_bash" ]]; then
      # shellcheck disable=SC2016  # intentional: the candidate bash must expand this, not us
      _alt_major=$("$_alt_bash" -c 'echo "${BASH_VERSINFO[0]}"' 2>/dev/null || echo 0)
      if [[ "$_alt_major" -ge 4 ]]; then
        exec "$_alt_bash" "$0" "$@"
      fi
    fi
  done
  echo "Error: bench_longprompt.sh requires bash >= 4 (associative arrays); running under ${BASH_VERSION:-unknown}." >&2
  echo "       On macOS install a newer bash, e.g. 'brew install bash', or invoke it explicitly: /opt/homebrew/bin/bash scripts/bench_longprompt.sh" >&2
  exit 1
fi

trap 'echo "Interrupted (signal received)" >&2; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DECODE="${SCRIPT_DIR}/bench_decode.sh"
MODELS_DIR="./models"
BENCHMARKS_DIR="./benchmarks"
DATE=$(date '+%Y-%m-%d')

# Same 15-column header bench_decode.sh writes (its CSV_HEADER constant), so
# aggregate rows -- whether they come from a real cell, a synthesized fake-OOM
# cell, or a backstop/record skip row emitted directly by this script -- share
# one schema. Trailing SKIP:*/FAIL:* tokens are an unnamed 16th field.
CSV_HEADER="model,model_path,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,date,hardware,mlx_version,build_type,max_tokens,prompt,prompt_target_len"

# Representative subset: mix of dense (llama, qwen2.5), MoE (qwen3-a3b,
# mixtral), and a large multimodal-capable text model (gemma-4).
MODELS_DEFAULT="llama-3.1-8b-4bit qwen2.5-7b-4bit qwen3-30b-a3b-4bit mixtral-8x7b-4bit gemma-4-31b-it-4bit"
LADDER_DEFAULT="512 2048 8192 32768"

MODELS="$MODELS_DEFAULT"
LADDER="$LADDER_DEFAULT"
OUTPUT=""
MAX_TOKENS=32
WARMUP_TOKENS=4
COOLDOWN=0
BIG_COOLDOWN=0
EXTRA_ARGS=()

# OOM avoidance state (issue #807). See the header comment above for the
# design; IGNORE_OOM_RECORD and FAKE_OOM_SET are read from the environment
# only (no CLI flag) so a stray flag typo can never silently disable them.
OOM_RECORD_FILE="${BENCHMARKS_DIR}/.oom-record"
IGNORE_OOM_RECORD="${MLXCEL_BENCH_IGNORE_OOM_RECORD:-0}"
declare -A OOM_MIN_LEN=()       # model_name -> smallest length observed SKIP:oom at
declare -A FAKE_OOM_SET=()      # "model:len" -> 1, from MLXCEL_BENCH_FAKE_OOM
OOM_RECORD_MODEL=()             # parallel arrays loaded from OOM_RECORD_FILE
OOM_RECORD_LEN=()
OOM_RECORD_LINE=()

usage() {
  cat <<'EOF'
Usage: bench_longprompt.sh [options]

Options:
  --models "A B C"    Space-separated model basenames under ./models
                      (default: llama-3.1-8b-4bit qwen2.5-7b-4bit
                      qwen3-30b-a3b-4bit mixtral-8x7b-4bit gemma-4-31b-it-4bit)
  --lengths "N N N"   Space-separated prompt-token ladder
                      (default: 512 2048 8192 32768)
  --max-tokens N      Decode tokens per cell (default: 32; kept small so the
                      run measures prefill, not decode)
  --warmup-tokens N   Warmup decode tokens per cell (default: 4)
  --cooldown N        Seconds to sleep after each cell (default: 0)
  --big-cooldown N    Extra seconds after a >10GB model (default: 0)
  --output PATH       Aggregate CSV path (default: auto-named under benchmarks/)
  --help              Show this help

All cells share one aggregate CSV with the 15-column bench_decode.sh schema
(prompt_target_len appended). Cell order is model-major, length-minor.

OOM avoidance (issue #807):
  A cell that classifies as SKIP:oom (a true OOM at load/run time; see
  is_oom_failure() in bench_decode.sh) triggers two purely empirical,
  reactive-only skip layers -- neither one predicts from a memory model:

  1. Monotonic backstop: every remaining cell for that model at length >= the
     OOM'd length is skipped for the rest of this run, logged as
     SKIP:oom_backstop.
  2. Persistent record: the OOM is appended to the gitignored
     benchmarks/.oom-record file (model,prompt_target_len,date,commit,
     peak_bytes,source_csv). On later runs, any cell whose model has a
     recorded OOM at a length <= the cell's length is pre-skipped and logged
     as SKIP:oom_record.

  Only SKIP:oom feeds either layer. FAIL:bench, a timeout (exit 124),
  SKIP:oom_estimate, a missing ./models symlink, and a missing model
  directory never do.

Environment variables:
  MLXCEL_BENCH_IGNORE_OOM_RECORD=1
      Disable the persistent-record pre-skip for this run (new SKIP:oom cells
      are still appended to benchmarks/.oom-record). Use this to re-test
      cells after a memory fix lands; see the staleness/pruning guidance in
      this script's header comment for removing the corresponding record
      lines once the fix is confirmed.
  MLXCEL_BENCH_FAKE_OOM="model:len[,model:len...]"
      Testing hook: for each listed (model, len) pair, skip invoking
      bench_decode.sh and synthesize a cell result containing a SKIP:oom row,
      so the backstop and persistent-record logic can be validated without a
      real OOM.
EOF
}

# ---------------------------------------------------------------------------
# Minimal hardware/backend detection for the default filename (mirrors
# bench_decode.sh so the two share a naming convention).
# ---------------------------------------------------------------------------
detect_backend() {
  if [[ "$(uname)" == "Linux" ]] && nvidia-smi &>/dev/null; then
    echo "cuda"
  else
    echo "metal"
  fi
}

detect_hardware_short() {
  local chip=""
  if [[ "$(uname)" == "Darwin" ]]; then
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
  else
    chip=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  fi
  case "$chip" in
    *M1\ Ultra*) echo "m1ultra" ;;
    *M5\ Max*)   echo "m5max" ;;
    *GB10*)      echo "gb10" ;;
    *)           echo "$chip" | tr '[:upper:] ' '[:lower:]_' | tr ',' '_' | cut -c1-20 ;;
  esac
}

# ---------------------------------------------------------------------------
# OOM avoidance helpers (issue #807)
# ---------------------------------------------------------------------------

# Extract the trailing SKIP:*/FAIL:* classification token from the data row
# in $TMP_CELL, or the empty string for a successful row (no trailing token)
# or an empty/missing cell file. The row's quoted prompt field can itself
# contain a comma (e.g. the default "Hello, how are you today?"), so this
# parses from the end of the line (last comma-separated field) rather than by
# a fixed field index.
cell_classification() {
  local row token
  row=$(tail -n +2 "$TMP_CELL" 2>/dev/null | head -n 1)
  if [[ -z "$row" ]]; then
    echo ""
    return 0
  fi
  token=$(awk -F, '{print $NF}' <<< "$row")
  if [[ "$token" =~ ^(SKIP|FAIL): ]]; then
    echo "$token"
  else
    echo ""
  fi
}

# Append a schema-compatible 16-field skip row for a cell that was never
# launched (monotonic backstop or persistent-record pre-skip). Writes the
# shared CSV header first if this is the first row of the aggregate output.
# This script does not compute `hardware`/`mlx_version`/`build_type` (only
# bench_decode.sh does), so those three fields are left blank; downstream
# consumers filter on the SKIP: prefix and treat these rows as capacity
# exclusions regardless.
emit_skip_row() {
  local model_name="$1" model_path="$2" len="$3" classification="$4"
  if [[ "$header_written" -eq 0 ]]; then
    echo "$CSV_HEADER" >> "$OUTPUT"
    header_written=1
  fi
  echo "${model_name},${model_path},,,,,,,${DATE},,,,${MAX_TOKENS},\"\",${len},${classification}" >> "$OUTPUT"
}

# Append a true-OOM cell to the persistent record file. peak_bytes is always
# blank today; the harness does not capture peak memory.
append_oom_record() {
  local model_name="$1" len="$2"
  local commit
  commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
  mkdir -p "$(dirname "$OOM_RECORD_FILE")"
  echo "${model_name},${len},${DATE},${commit},,${OUTPUT}" >> "$OOM_RECORD_FILE"
}

# Load benchmarks/.oom-record into parallel arrays once, up front, so the
# per-cell lookup is a cheap in-memory scan and malformed lines are only
# warned about once (not once per cell). Tolerates a missing/empty file.
load_oom_record() {
  [[ -f "$OOM_RECORD_FILE" ]] || return 0
  local line_num=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_num=$((line_num + 1))
    [[ -z "$line" ]] && continue
    local rec_model rec_len field_count
    rec_model=$(cut -d, -f1 <<< "$line")
    rec_len=$(cut -d, -f2 <<< "$line")
    # The documented record format is six fields
    # (model,prompt_target_len,date,commit,peak_bytes,source_csv); reject any
    # line with fewer, so a truncated/hand-edited entry like "model,32768" is
    # skipped rather than treated as an active OOM record. A source path that
    # itself contains a comma only inflates the count, so >= 6 stays lenient.
    field_count=$(awk -F, '{print NF}' <<< "$line")
    if [[ "$field_count" -lt 6 || -z "$rec_model" || ! "$rec_len" =~ ^[0-9]+$ ]]; then
      >&2 echo "Warning: malformed line $line_num in $OOM_RECORD_FILE, skipping: $line"
      continue
    fi
    OOM_RECORD_MODEL+=("$rec_model")
    OOM_RECORD_LEN+=("$rec_len")
    OOM_RECORD_LINE+=("$line")
  done < "$OOM_RECORD_FILE"
}

# Print the first record line whose model matches $1 and whose recorded
# length is <= $2 (the current cell's length), and return success. Returns
# failure with no output when there is no match.
check_oom_record() {
  local model_name="$1" len="$2" i
  for i in "${!OOM_RECORD_MODEL[@]}"; do
    if [[ "${OOM_RECORD_MODEL[$i]}" == "$model_name" && "$len" -ge "${OOM_RECORD_LEN[$i]}" ]]; then
      echo "${OOM_RECORD_LINE[$i]}"
      return 0
    fi
  done
  return 1
}

# True when MLXCEL_BENCH_FAKE_OOM lists this exact (model, len) pair.
is_fake_oom_cell() {
  local model_name="$1" len="$2"
  [[ -n "${FAKE_OOM_SET[${model_name}:${len}]:-}" ]]
}

# Write a synthetic SKIP:oom cell result to $TMP_CELL in place of invoking
# bench_decode.sh, for MLXCEL_BENCH_FAKE_OOM.
synthesize_fake_oom_cell() {
  local model_name="$1" model_path="$2" len="$3"
  {
    echo "$CSV_HEADER"
    echo "${model_name},${model_path},,,,,,,${DATE},,,,${MAX_TOKENS},\"(faked via MLXCEL_BENCH_FAKE_OOM)\",${len},SKIP:oom"
  } > "$TMP_CELL"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)        MODELS="$2"; shift 2 ;;
    --lengths)       LADDER="$2"; shift 2 ;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
    --warmup-tokens) WARMUP_TOKENS="$2"; shift 2 ;;
    --cooldown)      COOLDOWN="$2"; shift 2 ;;
    --big-cooldown)  BIG_COOLDOWN="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --help)          usage; exit 0 ;;
    *)               echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -x "$BENCH_DECODE" ]]; then
  echo "Error: $BENCH_DECODE not found or not executable" >&2
  exit 1
fi

# Every ladder length must be a plain non-negative integer. The OOM backstop
# and the persistent-record pre-skip both compare the cell length with `[[
# "$len" -ge N ]]`, and a non-numeric entry (a typo such as "2o48", or "32k")
# makes that arithmetic error out and silently evaluate false -- which would
# launch a cell the backstop was supposed to skip. Since launching a
# near-OOM cell is exactly what wedges the GB10 driver (issue #807), a
# malformed length is a fatal config error, caught before the sweep starts
# rather than after hours of running.
for _len in $LADDER; do
  if [[ ! "$_len" =~ ^[0-9]+$ ]]; then
    echo "Error: --lengths must be space-separated non-negative integers; got \"$_len\"." >&2
    exit 1
  fi
done

if [[ -z "$OUTPUT" ]]; then
  BACKEND=$(detect_backend)
  HW=$(detect_hardware_short)
  OUTPUT="${BENCHMARKS_DIR}/${BACKEND}_${HW}_longprompt_${DATE}.csv"
fi

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"   # truncate; header is copied from the first cell (or written by emit_skip_row)

if [[ -n "${MLXCEL_BENCH_FAKE_OOM:-}" ]]; then
  IFS=',' read -ra _fake_entries <<< "$MLXCEL_BENCH_FAKE_OOM"
  for _entry in "${_fake_entries[@]}"; do
    [[ -z "$_entry" ]] && continue
    FAKE_OOM_SET["$_entry"]=1
  done
fi

if [[ "$IGNORE_OOM_RECORD" != "1" ]]; then
  load_oom_record
fi

TMP_CELL=$(mktemp -t bench_longprompt_cell.XXXXXX.csv)
cleanup() { rm -f "$TMP_CELL"; }
trap 'cleanup; echo "Interrupted (signal received)" >&2; exit 130' INT TERM
trap cleanup EXIT

>&2 echo "Long-prompt sweep"
>&2 echo "  output:  $OUTPUT"
>&2 echo "  models:  $MODELS"
>&2 echo "  lengths: $LADDER"
>&2 echo "  max-tokens=$MAX_TOKENS warmup-tokens=$WARMUP_TOKENS"
if [[ "$IGNORE_OOM_RECORD" == "1" ]]; then
  >&2 echo "  oom-record: ${#OOM_RECORD_MODEL[@]} entries loaded but pre-skip DISABLED (MLXCEL_BENCH_IGNORE_OOM_RECORD=1)"
else
  >&2 echo "  oom-record: ${#OOM_RECORD_MODEL[@]} entries loaded from $OOM_RECORD_FILE"
fi
>&2 echo ""

header_written=0
for model_name in $MODELS; do
  model_path="${MODELS_DIR}/${model_name}"
  if [[ ! -d "$model_path" ]]; then
    >&2 echo ">>> [miss]   $model_name (not found under $MODELS_DIR, skipping)"
    continue
  fi
  for len in $LADDER; do
    >&2 echo "=== $model_name @ ${len} tokens ==="

    # 1. Monotonic backstop: this model already OOM'd at <= len earlier in
    #    this same sweep.
    if [[ -n "${OOM_MIN_LEN[$model_name]:-}" ]] && [[ "$len" -ge "${OOM_MIN_LEN[$model_name]}" ]]; then
      >&2 echo "    backstop: $model_name already SKIP:oom at ${OOM_MIN_LEN[$model_name]} tokens <= ${len}, skipping before launch (SKIP:oom_backstop)"
      emit_skip_row "$model_name" "$model_path" "$len" "SKIP:oom_backstop"
      continue
    fi

    # 2. Persistent record: a prior run recorded this model OOM'ing at <= len.
    if [[ "$IGNORE_OOM_RECORD" != "1" ]]; then
      if record_match=$(check_oom_record "$model_name" "$len"); then
        >&2 echo "    oom-record: matched \"$record_match\", skipping before launch (SKIP:oom_record)"
        emit_skip_row "$model_name" "$model_path" "$len" "SKIP:oom_record"
        continue
      fi
    fi

    # 3. Testing hook: synthesize a SKIP:oom cell instead of launching.
    if is_fake_oom_cell "$model_name" "$len"; then
      >&2 echo "    fake-oom: synthesizing SKIP:oom for $model_name @ ${len} (MLXCEL_BENCH_FAKE_OOM)"
      synthesize_fake_oom_cell "$model_name" "$model_path" "$len"
    else
      # Reuse the standard runner for one (model, length) cell. --output
      # isolates the cell so the aggregate CSV is never truncated mid-sweep.
      "$BENCH_DECODE" "$model_path" \
        --prompt-tokens "$len" \
        --max-tokens "$MAX_TOKENS" \
        --warmup-tokens "$WARMUP_TOKENS" \
        --cooldown "$COOLDOWN" \
        --big-cooldown "$BIG_COOLDOWN" \
        --output "$TMP_CELL" \
        "${EXTRA_ARGS[@]}" >/dev/null || {
          >&2 echo "    cell runner exited non-zero (continuing)"
        }
    fi

    if [[ ! -s "$TMP_CELL" ]]; then
      >&2 echo "    no output produced for this cell (continuing)"
      continue
    fi
    if [[ "$header_written" -eq 0 ]]; then
      cat "$TMP_CELL" >> "$OUTPUT"
      header_written=1
    else
      tail -n +2 "$TMP_CELL" >> "$OUTPUT"
    fi

    # Only a true SKIP:oom feeds the backstop and the persistent record.
    # FAIL:bench, timeouts, SKIP:oom_estimate, etc. never do.
    classification=$(cell_classification)
    if [[ "$classification" == "SKIP:oom" ]]; then
      if [[ -z "${OOM_MIN_LEN[$model_name]:-}" ]] || [[ "$len" -lt "${OOM_MIN_LEN[$model_name]}" ]]; then
        OOM_MIN_LEN[$model_name]="$len"
      fi
      append_oom_record "$model_name" "$len"
    fi
  done
done

>&2 echo ""
>&2 echo "Long-prompt results saved to: $OUTPUT"
