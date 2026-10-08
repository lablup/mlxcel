#!/usr/bin/env bash
# Measurement session for issue #2192 on gfx1151:
#   measure_session.sh REPO_ROOT PRE_EPIC_ROOT OUT_DIR MODELS_DIR
#
# Every GPU process below runs as its own scripts/rocm_gpu_guard.sh command
# (GUARD, default the repo's script), so a contended sample rejects one run
# rather than a whole session on a host four units share. Steps:
#   1. bench_rounds.sh: 3 interleaved rounds of pre-epic (4c44e317, CxxGenerator)
#      vs post-epic (main) `mlxcel-bench-decode`, plus a null arm, pp512/tg128
#      and pp2048 where the pre-epic pages have a pp2048 row;
#   2. `mlxcel-bench-engine` on both builds, `--path both --decode-storage dense`,
#      3 rounds, for the server-engine path `mlxcel run` and chat take now;
#   3. the `#[ignore]` 20 GiB test `paged_pool_past_u32_elements_matches_gather`
#      from the prebuilt mlxcel-core test binary;
#   4. one Gemma 3 server decode with the lookahead counter read from /metrics;
#   5. scripts/rocm_decode_profile.sh once, end to end, on one model.
set -uo pipefail
ROOT="$1"; PRE="$2"; OUT="$3"; MODELS_DIR="$4"
mkdir -p "$OUT"
H="$ROOT/docs/benchmark_results/data/rocm-unified-engine-gfx1151-2026-10-08/harness"
BASE_BENCH="$PRE/target/release/mlxcel-bench-decode"
NEW_BENCH="$ROOT/target/release/mlxcel-bench-decode"
BASE_ENGINE="$PRE/target/release/mlxcel-bench-engine"
NEW_ENGINE="$ROOT/target/release/mlxcel-bench-engine"
ROUNDS="${ROUNDS:-3}"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
export GUARD GUARD_IDLE_SECS="${GUARD_IDLE_SECS:-60}" GUARD_ATTEMPTS="${GUARD_ATTEMPTS:-10}"
mkdir -p "$OUT/guard"
g() { "$GUARD" --idle-secs "$GUARD_IDLE_SECS" --max-attempts "$GUARD_ATTEMPTS" --log "$OUT/guard/$1.log" -- "${@:2}"; }
echo "session start $(date -u +%FT%TZ)"

# 1. bench_decode rounds (bench_rounds.sh guards each run through GUARD).
"$H/bench_rounds.sh" "$OUT/rounds" "$ROUNDS" "$BASE_BENCH" "$NEW_BENCH" "$MODELS_DIR" \
  Qwen3-0.6B-4bit:512 Qwen3-0.6B-4bit:2048 \
  Meta-Llama-3.1-8B-Instruct-4bit:512 \
  gemma-3-4b-it-4bit:512 gemma-3-4b-it-4bit:2048 \
  Qwen3-30B-A3B-4bit:512 Qwen3-30B-A3B-4bit:2048 \
  granite-4.0-h-tiny-4bit:512 \
  NVIDIA-Nemotron-3-Nano-30B-A3B-4bit:512 \
  gpt-oss-20b-MXFP4-Q4:512 \
  Mixtral-8x7B-Instruct-v0.1-4bit:512 \
  > "$OUT/bench_rounds.log" 2>&1
echo "bench_rounds rc=$? $(date -u +%H:%M:%S)"

# 2. bench_engine rounds (server-dense is the `mlxcel run` / chat path).
mkdir -p "$OUT/engine"
for ((r = 1; r <= ROUNDS; r++)); do
  for model in Qwen3-0.6B-4bit Meta-Llama-3.1-8B-Instruct-4bit; do
    for arm in base new; do
      bin="$NEW_ENGINE"; [[ "$arm" == base ]] && bin="$BASE_ENGINE"
      g "engine_${model}_r${r}_${arm}" "$bin" -m "$MODELS_DIR/$model" --path both --decode-storage dense \
        --prompt-tokens 256,2048 -n 128 --warmup-tokens 20 \
        --csv "$OUT/engine/rounds.csv" --label "$arm-r$r" \
        > "$OUT/engine/${model}_r${r}_${arm}.log" 2>&1 || echo "bench_engine $model $arm r$r rc=$?"
      sleep 5
    done
  done
done
echo "bench_engine done $(date -u +%H:%M:%S)"

# 3. The 20 GiB ignored test, from the prebuilt test binary (no cargo here:
#    the guard counts cargo as a compiler).
CORE_TEST=$(ls -t "$ROOT"/target/test-fast/deps/mlxcel_core-* 2>/dev/null | grep -v '\.d$' | head -1)
echo "mlxcel-core test binary: $CORE_TEST"
g paged_pool_past_u32 "$CORE_TEST" paged_pool_past_u32_elements_matches_gather --ignored --test-threads=1 --nocapture \
  > "$OUT/paged_pool_past_u32.log" 2>&1
echo "paged_pool_past_u32 rc=$? $(date -u +%H:%M:%S)"
grep -h "test result\|skipping" "$OUT/paged_pool_past_u32.log"

# 4. Gemma 3 server decode: does the lookahead prime on ROCm?
#    This is a correctness check (does the counter move), not a timing, so a
#    contended window is accepted after a few attempts.
PORT=18611
cat > "$OUT/gemma3_probe.sh" <<EOS
#!/usr/bin/env bash
RUST_LOG=info "$ROOT/target/release/mlxcel" serve -m "$MODELS_DIR/gemma-3-4b-it-4bit" --parallel 1 \
  --host 127.0.0.1 --port $PORT > "$OUT/gemma3_server.log" 2>&1 &
GPID=\$!
for _ in \$(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain in three sentences why the sky is blue."}],"max_tokens":96,"temperature":0}' \
  > "$OUT/gemma3_response.json"
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/gemma3_metrics.txt"
kill \$GPID; for _ in \$(seq 1 30); do kill -0 \$GPID 2>/dev/null || break; sleep 1; done
EOS
"$GUARD" --idle-secs 20 --max-attempts 3 --log "$OUT/guard/gemma3_probe.log" -- bash "$OUT/gemma3_probe.sh"
echo "gemma3 probe rc=$?"
grep -h "lookahead" "$OUT/gemma3_metrics.txt" | grep -v "^#"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("gemma3 tokens:", d.get("usage")); print(d["choices"][0]["message"]["content"][:300])' "$OUT/gemma3_response.json"
echo "gemma3 done $(date -u +%H:%M:%S)"

# 5. Decode profile end to end (nested guard skips the lock).
cd "$ROOT" && scripts/rocm_decode_profile.sh --out "$OUT/profile" --idle-secs 5 \
  "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-4bit" > "$OUT/decode_profile.log" 2>&1
echo "decode_profile rc=$? $(date -u +%H:%M:%S)"
ls "$OUT/profile" 2>/dev/null | head
echo "session end $(date -u +%FT%TZ)"
