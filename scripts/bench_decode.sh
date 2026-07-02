#!/usr/bin/env bash
# Decode benchmark with warmup preheating for accurate measurements.
#
# Usage:
#   ./scripts/bench_decode.sh models/llama3-1b-4bit
#   ./scripts/bench_decode.sh all
#   ./scripts/bench_decode.sh all --vlm --image tests/fixtures/test_image.png
#   ./scripts/bench_decode.sh all --output benchmarks/custom_name.csv
#
# Default output: benchmarks/{backend}_{hardware}_{YYYY-MM-DD}.csv
#   e.g. benchmarks/metal_m1ultra_2026-03-31.csv
#   VLM: benchmarks/metal_m1ultra_vlm_2026-03-31.csv
#
# The script uses `mlxcel-bench-decode` so each model is loaded once and the
# warmup pass (discarded) and measured pass run in the same process. This keeps
# Metal/MLX warm state alive for the measured prefill, matching the Python
# `bench_mlxlm.py` harness and avoiding cold-prefill skew from two separate
# `mlxcel generate` invocations.
#
# CSV columns (15):
#   model, model_path, prompt_tokens, generated_tokens,
#   prefill_ms, prefill_tok_s, decode_ms, decode_tok_s,
#   date, hardware, mlx_version, build_type, max_tokens, prompt,
#   prompt_target_len
#
# The first 14 columns are unchanged from the historical schema so older CSVs
# stay comparable. Column 15 (prompt_target_len) records the --prompt-tokens
# target for long-prompt prefill runs (epic #623 #624) and is empty for the
# default short-prompt path. The actual prompt length used (after capping at the
# model context) is still reported in the prompt_tokens column. Any trailing
# result-classification token (SKIP:*/FAIL:*) follows prompt_target_len.
#
# Result classifications (trailing CSV token):
#   (none)               successful decode with profiling numbers
#   SKIP:oom_estimate    model weight size exceeds the up-front memory budget;
#                        skipped before launch (see model_fits_in_memory)
#   SKIP:oom             process exited with an OOM signal/exception at load or
#                        run time; a capacity exclusion, not a code failure
#   FAIL:bench           benchmark process exited with a non-OOM failure
#   FAIL:no_output       benchmark succeeded but produced no decode numbers
#   SKIP:vlm_image_not_found  VLM run skipped because the test image is absent
#
# Both SKIP:oom_estimate and SKIP:oom share the SKIP: prefix, so downstream
# consumers that filter on SKIP: treat them identically as capacity exclusions.
#
# Filename convention:
#   {backend}_{hardware}_{YYYY-MM-DD}.csv                (text suite, 'all')
#   {backend}_{hardware}_vlm_{YYYY-MM-DD}.csv            (VLM suite, 'all --vlm')
#   {backend}_{hardware}_{YYYY-MM-DD}_{suffix}.csv       (with --suffix)
#   {backend}_{hardware}_{YYYY-MM-DD}_single_{model}.csv (single-model run)
#
# The `_single_{model}` form exists so that ad-hoc sanity runs against a
# specific model cannot silently truncate the day's full-suite CSV. Pass
# --output PATH to override the default for either mode.

set -euo pipefail

# Forward SIGINT/SIGTERM through cooldown sleeps so Ctrl-C aborts immediately
# instead of waiting out the current sleep.
trap 'echo "Interrupted (signal received)" >&2; exit 130' INT TERM

MLXCEL="./target/release/mlxcel"
MLXCEL_BENCH="./target/release/mlxcel-bench-decode"
MODELS_DIR="./models"
BENCHMARKS_DIR="./benchmarks"
TEXT_PROMPT="Hello, how are you today?"
VLM_PROMPT="What is in this image?"
MAX_TOKENS=100
WARMUP_TOKENS=20
# Optional deterministic long-prompt length (--prompt-tokens N). Empty keeps the
# short-prompt default so historical CSV rows are byte-compatible.
PROMPT_TOKENS=""
TIMEOUT=300
JIT_PREHEAT_TIMEOUT=600
VLM_IMAGE="tests/fixtures/test_image.png"
VLM_MODE=0
NO_CHAT_TEMPLATE=0
OUTPUT=""
SUFFIX=""
DATE=$(date '+%Y-%m-%d')
# Version recorded in the CSV `mlx_version` column. This is the MLXCEL
# version from Cargo.toml (the /update-benchmarks staleness check compares
# this column against Cargo.toml); it was a stale hardcoded "0.31.2" until
# 2026-06-12.
MLX_VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
[[ -n "$MLX_VERSION" ]] || MLX_VERSION="unknown"
BUILD_TYPE="release"
# Optional overhead multiplier applied to the weight-size estimate in
# model_fits_in_memory(). Default 1.0 preserves the existing pass/skip
# decisions exactly. Set to e.g. 1.2 to require weights to consume at most
# 85%/1.2 ≈ 71% of system memory, giving headroom for activations and KV
# cache on constrained hosts. Overridden via BENCH_MEM_OVERHEAD_FACTOR env.
BENCH_MEM_OVERHEAD_FACTOR="${BENCH_MEM_OVERHEAD_FACTOR:-1.0}"

# Thermal cooldown between models. Defaults to 0 to preserve existing
# behaviour for CI and desktop hardware. Laptops (e.g. MacBook Pro M5 Max)
# should pass --cooldown / --big-cooldown explicitly to avoid throttling
# during long all-model runs.
COOLDOWN_SECS=0
BIG_MODEL_COOLDOWN_SECS=0
# Pre-warm before `all` sweeps (default ON). The first sweep right after a
# fresh binary build measures early models up to 2x slow: GPU pipeline
# caches are cold and the machine is still hot from compiling. The pre-warm
# runs one small throwaway generation (compiles shared kernels) and then
# settles for PRE_WARM_SETTLE_SECS so thermals recover before the first
# measured model. Disable with --no-pre-warm for intentionally cold runs.
PRE_WARM=1
PRE_WARM_SETTLE_SECS=30
# Models whose total weight bytes exceed this threshold get an additional
# BIG_MODEL_COOLDOWN_SECS pause after running.
BIG_MODEL_THRESHOLD_BYTES=$((10 * 1024 * 1024 * 1024))

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

# Full hardware string for CSV content (e.g. Apple_M1_Ultra_128GB)
detect_hardware_full() {
  local chip mem
  if [[ "$(uname)" == "Darwin" ]]; then
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    mem=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB", $1/1073741824}')
  else
    # Linux: detect NVIDIA GPU or fall back to CPU
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    local cuda_ver
    cuda_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || echo "")
    if [[ -n "$gpu_name" ]]; then
      # Avoid double "NVIDIA" prefix if gpu_name already starts with it
      if [[ "$gpu_name" == NVIDIA* ]]; then
        chip="${gpu_name}"
      else
        chip="NVIDIA_${gpu_name}"
      fi
      [[ -n "$cuda_ver" ]] && chip="${chip}_CUDA${cuda_ver}"
    else
      chip=$(cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | sed 's/.*: //' || echo "unknown")
    fi
    mem=$(free -b 2>/dev/null | awk '/^Mem:/{printf "%.0fGB", $2/1073741824}' || echo "")
  fi
  echo "${chip}_${mem}" | tr ' ' '_'
}

# Short hardware name for filenames (e.g. m1ultra, m5max, gb10)
detect_hardware_short() {
  local full
  full=$(detect_hardware_full)

  # Map known hardware to short names
  case "$full" in
    *M1_Ultra*)  echo "m1ultra" ;;
    *M1_Max*)    echo "m1max" ;;
    *M1_Pro*)    echo "m1pro" ;;
    *M2_Ultra*)  echo "m2ultra" ;;
    *M2_Max*)    echo "m2max" ;;
    *M3_Ultra*)  echo "m3ultra" ;;
    *M3_Max*)    echo "m3max" ;;
    *M4_Ultra*)  echo "m4ultra" ;;
    *M4_Max*)    echo "m4max" ;;
    *M5_Ultra*)  echo "m5ultra" ;;
    *M5_Max*)    echo "m5max" ;;
    *GB10*)      echo "gb10" ;;
    *)           echo "${full}" | tr '[:upper:]' '[:lower:]' | tr ',' '_' | cut -c1-20 ;;
  esac
}

# Detect backend from binary / platform
detect_backend() {
  if [[ "$(uname)" == "Linux" ]] && nvidia-smi &>/dev/null; then
    echo "cuda"
  elif "$MLXCEL" generate --help 2>&1 | grep -q "cuda"; then
    echo "cuda"
  else
    echo "metal"
  fi
}

HARDWARE_FULL=$(detect_hardware_full)
HARDWARE_SHORT=$(detect_hardware_short)
BACKEND=$(detect_backend)

# ---------------------------------------------------------------------------
# Memory-based model size check
# ---------------------------------------------------------------------------
# Detect available system memory in bytes.
detect_memory_bytes() {
  if [[ "$(uname)" == "Darwin" ]]; then
    sysctl -n hw.memsize 2>/dev/null || echo 0
  else
    free -b 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0
  fi
}

SYSTEM_MEMORY_BYTES=$(detect_memory_bytes)
# Reserve 15% for OS/runtime overhead; use 85% as the usable limit.
MEMORY_LIMIT_BYTES=$(( SYSTEM_MEMORY_BYTES * 85 / 100 ))

# Estimate model weight size from safetensors files (bytes).
# Returns 0 if no safetensors files found.
estimate_model_size() {
  local model_path="$1"
  local total=0
  local size
  for f in "$model_path"/*.safetensors; do
    [[ -f "$f" ]] || continue
    size=$(stat --format='%s' "$f" 2>/dev/null || stat -f'%z' "$f" 2>/dev/null || echo 0)
    total=$((total + size))
  done
  echo "$total"
}

# Check if a model likely fits in memory.  Returns 0 (fits) or 1 (too large).
# BENCH_MEM_OVERHEAD_FACTOR (default 1.0) scales the weight-byte estimate to
# add headroom for activations, KV cache, and framework overhead. The default
# leaves existing pass/skip decisions unchanged.
model_fits_in_memory() {
  local model_path="$1"
  local model_bytes
  model_bytes=$(estimate_model_size "$model_path")
  [[ "$model_bytes" -eq 0 ]] && return 0  # can't determine size, try anyway
  local effective_bytes
  # Pass values as awk data (-v), not interpolated program text, so a
  # non-numeric BENCH_MEM_OVERHEAD_FACTOR degrades to 0 instead of an awk
  # syntax error (which could abort the sweep under set -e).
  effective_bytes=$(awk -v b="$model_bytes" -v f="$BENCH_MEM_OVERHEAD_FACTOR" 'BEGIN{printf "%.0f", b * f}')
  [[ "$effective_bytes" -le "$MEMORY_LIMIT_BYTES" ]]
}

# Returns 0 (true) when a failed run looks like an out-of-memory condition.
# OOM is matched by EITHER the exit signal OR a recognisable allocator message,
# independently. Requiring both would miss the two real OOM paths: the OS
# OOM-killer SIGKILLs the process before it can print anything (signal, no
# message), while an MLX/Metal allocator exception propagated through cxx exits
# with code 1 and carries the allocator text (message, non-signal exit code).
#
# Exit codes:
#   137 = SIGKILL          -> OS OOM-killer; treated as OOM on its own
#   124 = timeout(1) expiry -> explicitly NOT OOM (a slow model is not OOM)
#   134 = SIGABRT          -> usually std::bad_alloc; classified by message below
#   1   = cxx-propagated MLX/Metal exception -> classified by message below
#
# OOM error text patterns (case-insensitive). Kept specific and grounded in
# strings the runtime actually emits, so a non-OOM failure is never hidden as a
# capacity skip (a too-broad pattern like a bare "exceeds .*limit" would match
# unrelated errors such as the KV-cache "exceeds limit" frame checks):
#   Generic:      out of memory, out-of-memory, insufficient memory,
#                 failed to allocate, cannot allocate memory, unable to allocate,
#                 bad_alloc, "memory allocation of N bytes failed" (Rust abort)
#   MLX Metal:    "greater than the maximum allowed buffer size", metal::malloc
#
# Bare "oom" is intentionally excluded to avoid false positives (e.g. "zoom").
is_oom_failure() {
  local rc="$1"
  local err_text="$2"

  # timeout(1) expiry is never OOM (a slow model is not out of memory).
  [[ "$rc" -eq 124 ]] && return 1

  # SIGKILL is the OS OOM-killer; the process is usually killed before it can
  # print anything, so the exit code alone is sufficient.
  [[ "$rc" -eq 137 ]] && return 0

  # Otherwise (exit 1 cxx exception, 134 bad_alloc abort, ...) it is OOM only
  # if the captured output names an allocator failure.
  printf '%s\n' "$err_text" | grep -qiE \
    'out of memory|out-of-memory|insufficient memory|failed to allocate|cannot allocate memory|unable to allocate|bad_alloc|memory allocation of [0-9]|greater than the maximum allowed buffer size|metal::malloc'
}

# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: bench_decode.sh <model_path|all> [options]

Options:
  --vlm               VLM mode (use image prompt)
  --image PATH        Image for VLM benchmark (default: tests/fixtures/test_image.png)
  --prompt TEXT        Override text prompt
  --no-chat-template   Disable automatic chat template application in the
                       mlxcel benchmark runner. Use this when comparing
                       against raw-prompt mlx-lm stream_generate baselines.
  --max-tokens N      Max tokens to generate (default: 100)
  --warmup-tokens N   Tokens for warmup pass (default: 20)
  --prompt-tokens N   Synthesize a deterministic prompt of exactly N tokens
                      (repeated corpus, tokenized, truncated; capped at the
                      model context) for long-prompt prefill benchmarking.
                      Recorded in the prompt_target_len CSV column. When unset
                      the short-prompt --prompt path runs unchanged.
  --timeout N         Timeout per run in seconds (default: 300)
  --jit-preheat-timeout N  Timeout for CUDA JIT warmup in seconds (default: 600)
  --output PATH       Write CSV to specific file (overrides auto-naming)
  --suffix TAG        Append suffix to auto-generated filename (e.g. --suffix baseline)
  --cooldown N        Sleep N seconds after every model to let the GPU cool
                      down (default: 0). Use on thermally constrained
                      hardware such as the MacBook Pro M5 Max where back-to-
                      back large models cause Metal to throttle.
  --big-cooldown N    Additional sleep after a model whose weight bytes
                      exceed --big-threshold-gb (default: 0). The total pause
                      after a "big" model is COOLDOWN + BIG_COOLDOWN.
  --big-threshold-gb N  Size in GB that triggers --big-cooldown
                        (default: 10).
  --no-pre-warm       Skip the pre-warm pass before an `all` sweep. The
                      pre-warm runs one small throwaway generation and then
                      settles for --pre-warm-settle seconds, so that cold GPU
                      pipeline caches and post-build thermals do not depress
                      the first measured models (observed up to 2x slow on
                      the first sweep after a fresh build, 2026-06-12).
  --pre-warm-settle N Settle sleep after the pre-warm pass (default: 30)
  --no-cooldown       Force --cooldown=0 and --big-cooldown=0, overriding
                      any earlier values on the same command line.
  --help              Show this help

Environment variables:
  BENCH_MEM_OVERHEAD_FACTOR  Multiply the safetensors weight-size estimate by
                             this factor before comparing against the 85% memory
                             budget. Default 1.0 (no change). Set to e.g. 1.2
                             to add headroom for activations and KV cache on
                             memory-constrained hosts; models that exceed the
                             adjusted limit are classified SKIP:oom_estimate.

Result classifications (trailing CSV token):
  (none)               successful decode with profiling numbers
  SKIP:oom_estimate    model weight bytes exceed the up-front memory budget
  SKIP:oom             process OOM'd at load or run time (SIGKILL/SIGABRT +
                       OOM error text); a capacity exclusion, not a code failure
  FAIL:bench           non-OOM process failure
  FAIL:no_output       process succeeded but produced no decode numbers
  SKIP:vlm_image_not_found  VLM run skipped because the test image is absent

Filename convention:
  {backend}_{hardware}_{YYYY-MM-DD}.csv                text suite ('all')
  {backend}_{hardware}_vlm_{YYYY-MM-DD}.csv            VLM suite ('all --vlm')
  {backend}_{hardware}_{YYYY-MM-DD}_{suffix}.csv       with --suffix
  {backend}_{hardware}_{YYYY-MM-DD}_single_{model}.csv single-model run
                                                       (separate file so sanity
                                                       runs do not truncate the
                                                       full-suite CSV)
EOF
}

# ---------------------------------------------------------------------------
# Generate default output filename
# ---------------------------------------------------------------------------
# Composes orthogonal segments in this order:
#   {backend}_{hardware}[_vlm]_{date}[_{suffix}][_single_{model}].csv
#
# Rationale for `_single_{model}` after `_{suffix}`: this groups all output
# from one --suffix run together in lexical directory listings, so e.g.
# `ls metal_m5max_2026-04-13_probe_*` shows the full-suite probe CSV right
# next to its per-model probe CSVs. Reversing the order would group runs
# of the same model across unrelated suffixes instead, which is rarely
# what the human comparing two benchmark runs wants.
#
# Single-model invocations get their own auto-named CSV so that ad-hoc
# sanity runs cannot truncate the day's full-suite CSV. The full-
# suite path is reserved exclusively for `all` runs and for any caller
# that explicitly passes --output.
default_output_path() {
  local name="${BACKEND}_${HARDWARE_SHORT}"
  if [[ "$VLM_MODE" -eq 1 ]]; then
    name="${name}_vlm"
  fi
  name="${name}_${DATE}"
  if [[ -n "$SUFFIX" ]]; then
    name="${name}_${SUFFIX}"
  fi
  if [[ "$MODEL_ARG" != "all" && -n "$MODEL_ARG" ]]; then
    local model_basename
    model_basename=$(basename "${MODEL_ARG%/}")
    name="${name}_single_${model_basename}"
  fi
  echo "${BENCHMARKS_DIR}/${name}.csv"
}

# ---------------------------------------------------------------------------
# Parse --profile output into CSV fields
# ---------------------------------------------------------------------------
parse_profile() {
  local output="$1"
  local prompt_tok gen_tok prefill_ms prefill_tps decode_ms decode_tps

  prompt_tok=$(echo "$output" | sed -n 's/.*Prompt tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  gen_tok=$(echo "$output"    | sed -n 's/.*Generated tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  prefill_ms=$(echo "$output" | sed -n 's/.*Prefill:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  prefill_tps=$(echo "$output" | sed -n 's/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
  decode_ms=$(echo "$output"  | sed -n 's/.*Decode:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  decode_tps=$(echo "$output" | sed -n 's/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)

  echo "${prompt_tok:-},${gen_tok:-},${prefill_ms:-},${prefill_tps:-},${decode_ms:-},${decode_tps:-}"
}

# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------
bench_one() {
  local model_path="$1"
  local model_name
  model_name=$(basename "$model_path")

  # Long-prompt target recorded in the prompt_target_len CSV column; empty for
  # the short-prompt default so historical rows stay byte-compatible.
  local ptl="$PROMPT_TOKENS"

  # Skip models that won't fit in memory
  if ! model_fits_in_memory "$model_path"; then
    local est_bytes effective_mb limit_mb
    est_bytes=$(estimate_model_size "$model_path")
    # Report the effective (overhead-scaled) size actually used for the decision
    # so the message stays consistent under a non-default BENCH_MEM_OVERHEAD_FACTOR.
    effective_mb=$(awk -v b="$est_bytes" -v f="$BENCH_MEM_OVERHEAD_FACTOR" 'BEGIN{printf "%.0f", b * f / 1048576}')
    limit_mb=$(( MEMORY_LIMIT_BYTES / 1048576 ))
    >&2 printf '>>> [skip]   %s (%d MB > %d MB limit)\n' "$model_name" "$effective_mb" "$limit_mb"
    echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$TEXT_PROMPT\",${ptl},SKIP:oom_estimate"
    return
  fi

  local prompt="$TEXT_PROMPT"
  local extra_args=()
  if [[ "$VLM_MODE" -eq 1 ]]; then
    prompt="$VLM_PROMPT"
    if [[ ! -f "$VLM_IMAGE" ]]; then
      echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",${ptl},SKIP:vlm_image_not_found"
      return
    fi
    extra_args+=(--image "$VLM_IMAGE")
  fi

  # Long-prompt prefill mode: synthesize an exactly-N-token prompt in the runner.
  if [[ -n "$PROMPT_TOKENS" ]]; then
    extra_args+=(--prompt-tokens "$PROMPT_TOKENS")
  fi

  # --- Same-process warmup + measured pass ---
  # On CUDA, the first run of a model triggers JIT kernel compilation which can
  # take several minutes. The bench runner performs warmup and measurement in a
  # single process, so allow the CUDA run to consume both the JIT preheat budget
  # and the normal measured-run budget.
  local run_timeout="$TIMEOUT"
  if [[ "$BACKEND" == "cuda" ]]; then
    run_timeout=$((JIT_PREHEAT_TIMEOUT + TIMEOUT))
  fi
  if [[ "$NO_CHAT_TEMPLATE" -eq 1 ]]; then
    extra_args+=(--no-chat-template)
  fi

  >&2 printf '>>> [bench]  %s (same-process warmup=%s) ...\n' "$model_name" "$WARMUP_TOKENS"
  local raw rc=0
  # Capture combined stdout+stderr and exit code. The || suppresses set -e so
  # a non-zero exit code is captured in rc rather than aborting the script.
  raw=$(timeout "$run_timeout" "$MLXCEL_BENCH" \
      -m "$model_path" -p "$prompt" -n "$MAX_TOKENS" \
      --warmup-tokens "$WARMUP_TOKENS" \
      ${extra_args[@]+"${extra_args[@]}"} 2>&1) || rc=$?

  if [[ "$rc" -ne 0 ]]; then
    if is_oom_failure "$rc" "$raw"; then
      >&2 printf '    OOM at load/run (exit %d) — SKIP:oom\n' "$rc"
      echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",${ptl},SKIP:oom"
    else
      >&2 printf '    benchmark failed (exit %d)\n' "$rc"
      echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",${ptl},FAIL:bench"
    fi
    return
  fi

  local fields
  fields=$(parse_profile "$raw")
  local decode_tps
  decode_tps=$(echo "$fields" | cut -d, -f6)

  if [[ -z "$decode_tps" ]]; then
    >&2 echo "    no decode output"
    echo "${model_name},${model_path},${fields},$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",${ptl},FAIL:no_output"
  else
    >&2 printf '    decode: %s tok/s\n' "$decode_tps"
    echo "${model_name},${model_path},${fields},$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",${ptl}"
  fi
}

# ---------------------------------------------------------------------------
# Thermal cooldown
# ---------------------------------------------------------------------------
# Pause after a model finishes so the GPU can cool down before the next run.
# The pause is COOLDOWN_SECS, plus BIG_MODEL_COOLDOWN_SECS if the model's
# total safetensors size exceeds BIG_MODEL_THRESHOLD_BYTES. With the default
# values (both 0) this function is a no-op, so existing callers see no change.
cooldown_after() {
  local model_path="$1"
  local sleep_secs="$COOLDOWN_SECS"
  if [[ "$BIG_MODEL_COOLDOWN_SECS" -gt 0 ]]; then
    local model_bytes
    model_bytes=$(estimate_model_size "$model_path")
    if [[ "$model_bytes" -gt "$BIG_MODEL_THRESHOLD_BYTES" ]]; then
      sleep_secs=$(( sleep_secs + BIG_MODEL_COOLDOWN_SECS ))
    fi
  fi
  if [[ "$sleep_secs" -gt 0 ]]; then
    >&2 printf '    cooldown: %ds\n' "$sleep_secs"
    sleep "$sleep_secs"
  fi
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODEL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vlm)            VLM_MODE=1; shift ;;
    --image)          VLM_IMAGE="$2"; VLM_MODE=1; shift 2 ;;
    --prompt)         TEXT_PROMPT="$2"; shift 2 ;;
    --no-chat-template) NO_CHAT_TEMPLATE=1; shift ;;
    --max-tokens)     MAX_TOKENS="$2"; shift 2 ;;
    --warmup-tokens)  WARMUP_TOKENS="$2"; shift 2 ;;
    --prompt-tokens)  PROMPT_TOKENS="$2"; shift 2 ;;
    --timeout)        TIMEOUT="$2"; shift 2 ;;
    --jit-preheat-timeout) JIT_PREHEAT_TIMEOUT="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --suffix)         SUFFIX="$2"; shift 2 ;;
    --cooldown)       COOLDOWN_SECS="$2"; shift 2 ;;
    --big-cooldown)   BIG_MODEL_COOLDOWN_SECS="$2"; shift 2 ;;
    --big-threshold-gb)
                      BIG_MODEL_THRESHOLD_BYTES=$(( $2 * 1024 * 1024 * 1024 ))
                      shift 2 ;;
    --no-cooldown)    COOLDOWN_SECS=0; BIG_MODEL_COOLDOWN_SECS=0; shift ;;
    --no-pre-warm)    PRE_WARM=0; shift ;;
    --pre-warm-settle) PRE_WARM_SETTLE_SECS="$2"; shift 2 ;;
    --help)           usage; exit 0 ;;
    -*)               echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)                MODEL_ARG="$1"; shift ;;
  esac
done

if [[ -z "$MODEL_ARG" ]]; then
  echo "Error: model path or 'all' required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$MLXCEL" ]]; then
  echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
  exit 1
fi
if [[ ! -x "$MLXCEL_BENCH" ]]; then
  echo "Error: $MLXCEL_BENCH not found. Run 'cargo build --release --bin mlxcel-bench-decode' first." >&2
  exit 1
fi

# Validate single-model paths before generating the output filename so a
# typo in `models/foo` exits cleanly without leaving a header-only orphan
# CSV behind in benchmarks/. The `all` mode is allowed to proceed without
# this check because it iterates the existing entries of $MODELS_DIR.
if [[ "$MODEL_ARG" != "all" && ! -d "$MODEL_ARG" ]]; then
  echo "Error: model directory '$MODEL_ARG' not found" >&2
  exit 1
fi

# Auto-generate output path if not specified
if [[ -z "$OUTPUT" ]]; then
  OUTPUT=$(default_output_path)
fi

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------
CSV_HEADER="model,model_path,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,date,hardware,mlx_version,build_type,max_tokens,prompt,prompt_target_len"

mkdir -p "$(dirname "$OUTPUT")"
echo "$CSV_HEADER" > "$OUTPUT"
>&2 echo "Output: $OUTPUT"
>&2 echo "Hardware: $HARDWARE_FULL ($HARDWARE_SHORT)"
>&2 echo "Backend: $BACKEND"
>&2 echo ""

emit() {
  echo "$1" | tee -a "$OUTPUT"
}

# ---------------------------------------------------------------------------
# Models known to crash the GPU (Metal timeout / address fault).
# Run these LAST to prevent GPU state corruption from affecting other models.
# ---------------------------------------------------------------------------
GPU_CRASH_MODELS="gemma3n-e4b-bf16"

is_gpu_crash_model() {
  local name
  name=$(basename "$1")
  for m in $GPU_CRASH_MODELS; do
    [[ "$name" == "$m" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pre-warm before `all` sweeps: on the first run after a build the shared
# kernels are not cached yet (CUDA JIT-compiles for 3-8 minutes per model;
# Metal compiles pipeline variants on first dispatch) and the machine is
# still hot from compiling, which measured up to 2x slow on the first
# models of a fresh sweep (2026-06-12 M1 Ultra refresh). Run one small
# throwaway generation so shared kernels are cached, then settle so
# thermals recover before the first measured model. --no-pre-warm skips.
# ---------------------------------------------------------------------------
if [[ "$PRE_WARM" == "1" && "$MODEL_ARG" == "all" ]]; then
  # Find a small model to preheat with
  preheat_model=""
  for candidate in smollm-135m-4bit ernie-4.5-0.3b-4bit qwen2.5-0.5b-bf16; do
    if [[ -d "$MODELS_DIR/$candidate" ]]; then
      preheat_model="$MODELS_DIR/$candidate"
      break
    fi
  done
  if [[ -n "$preheat_model" ]]; then
    >&2 echo "=== Pre-warm: $(basename "$preheat_model") ==="
    if [[ "$BACKEND" == "cuda" ]]; then
      >&2 echo "    First run compiles CUDA JIT kernels (may take several minutes)..."
    else
      >&2 echo "    Warming shared GPU pipeline caches..."
    fi
    if timeout "$JIT_PREHEAT_TIMEOUT" "$MLXCEL" generate \
        -m "$preheat_model" -p "Hello" -n 5 --profile >/dev/null 2>&1; then
      >&2 echo "    Pre-warm complete."
    else
      >&2 echo "    Pre-warm failed (non-fatal, continuing)."
    fi
    if [[ "$PRE_WARM_SETTLE_SECS" -gt 0 ]]; then
      >&2 echo "    Settling for ${PRE_WARM_SETTLE_SECS}s (thermal recovery)..."
      sleep "$PRE_WARM_SETTLE_SECS"
    fi
    >&2 echo ""
  fi
fi

if [[ "$MODEL_ARG" == "all" ]]; then
  # First pass: run all models except known GPU-crash models
  for dir in "$MODELS_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    is_gpu_crash_model "$dir" && continue
    result=$(bench_one "$dir")
    emit "$result"
    cooldown_after "$dir"
  done
  # Second pass: run known GPU-crash models last
  for dir in "$MODELS_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    is_gpu_crash_model "$dir" || continue
    result=$(bench_one "$dir")
    emit "$result"
    cooldown_after "$dir"
  done
else
  # Path validation already happened before $OUTPUT was generated.
  result=$(bench_one "$MODEL_ARG")
  emit "$result"
  cooldown_after "$MODEL_ARG"
fi

>&2 echo ""
>&2 echo "Results saved to: $OUTPUT"
