#!/usr/bin/env bash
# Issue #2241 measurement session (run under scripts/rocm_gpu_guard.sh --hold).
# Before/after MoE prefill at pp512 (3 rounds, arms alternated), a rocprofv3
# kernel trace of granite on both arms, and logit traces (w8, w256) on both
# arms for the parity comparison.
# Usage: session.sh REPO_ROOT OUT_DIR MODELS_DIR BEFORE_DIR AFTER_DIR
set -uo pipefail
ROOT="$1"; OUT="$2"; MODELS_DIR="$3"; BEFORE="$4"; AFTER="$5"
mkdir -p "$OUT/raw" "$OUT/prof" "$OUT/traces"
echo "session start $(date -u +%FT%TZ)"
CSV="$OUT/moe_prefill.csv"
echo "model,round,arm,prefill_tok_s,decode_tok_s,status" > "$CSV"
MODELS=(granite-4.0-h-tiny-4bit Mixtral-8x7B-Instruct-v0.1-4bit Qwen3-30B-A3B-4bit gpt-oss-20b-MXFP4-Q4)
for r in 1 2 3; do
  for model in "${MODELS[@]}"; do
    # Alternate which arm goes first per round.
    if (( r % 2 == 1 )); then arms=(before after); else arms=(after before); fi
    for arm in "${arms[@]}"; do
      bin="$BEFORE/mlxcel-bench-decode"; [[ "$arm" == after ]] && bin="$AFTER/mlxcel-bench-decode"
      log="$OUT/raw/${model}_r${r}_${arm}.log"
      "$bin" -m "$MODELS_DIR/$model" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 512 > "$log" 2>&1
      rc=$?
      p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      echo "$model,$r,$arm,$p,$d,$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$CSV"
      echo "$(date -u +%T) $model r$r $arm prefill=$p decode=$d rc=$rc"
      sleep 3
    done
  done
done
cat "$CSV"

# Kernel trace: which gather kernel the granite prefill runs on each arm.
ROCPROF="${ROCPROFV3:-$(command -v rocprofv3 || echo /opt/rocm/bin/rocprofv3)}"
for arm in before after; do
  bin="$BEFORE/mlxcel-bench-decode"; [[ "$arm" == after ]] && bin="$AFTER/mlxcel-bench-decode"
  d="$OUT/prof/$arm"; mkdir -p "$d"
  MLXCEL_BENCH_PHASE_MARKS=1 "$ROCPROF" --kernel-trace --stats -f csv -d "$d" -o granite -- \
    "$bin" -m "$MODELS_DIR/granite-4.0-h-tiny-4bit" -p x -n 8 --warmup-tokens 1 --ignore-eos --prompt-tokens 512 > "$d/bench.log" 2>&1
  echo "profile $arm rc=$? $(date -u +%T)"
  grep -i "gather_qmv" "$d/granite_kernel_stats.csv" | cut -d, -f1-4 | head -8
done

# Logit traces for parity (teacher-forced; w8 and w256 as the 10-05 page).
C="$ROOT/tests/fixtures/wikitext2_excerpt.txt"
for pair in granite-4.0-h-tiny:granite-4.0-h-tiny-4bit mixtral-8x7b-instruct:Mixtral-8x7B-Instruct-v0.1-4bit qwen3-30b-a3b:Qwen3-30B-A3B-4bit; do
  tag="${pair%%:*}"; ckpt="${pair##*:}"
  for width in w8 w256; do
    case "$width" in w8) args=(8 80 8 512);; w256) args=(256 2 8 0);; esac
    for arm in before after; do
      lt="$BEFORE/logit_trace"; [[ "$arm" == after ]] && lt="$AFTER/logit_trace"
      "$lt" "$MODELS_DIR/$ckpt" "$C" "${args[@]}" > "$OUT/traces/${tag}_${arm}_${width}.tsv" 2> "$OUT/traces/${tag}_${arm}_${width}.err"
      echo "trace $tag $width $arm rc=$? rows=$(wc -l < "$OUT/traces/${tag}_${arm}_${width}.tsv") $(date -u +%T)"
    done
  done
done
echo "session end $(date -u +%FT%TZ)"
