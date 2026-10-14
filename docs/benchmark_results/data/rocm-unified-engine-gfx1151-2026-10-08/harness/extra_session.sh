#!/usr/bin/env bash
# Follow-up GPU items for issue #2192, one lock hold:
#   1. the Gemma 3 lookahead probe (a chat request on a one-slot server, then
#      `mlxcel_batch_decode_lookahead_steps_total` from /metrics);
#   2. an isolating measurement for the gemma-3-4b-it-4bit pp2048 decode cell:
#      base (pre-epic CxxGenerator), new (DirectEngine, pipelined) and
#      new-sync (DirectEngine with MLXCEL_FORCE_SYNC=1), 3 interleaved rounds.
# Usage: extra_session.sh REPO_ROOT PRE_EPIC_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; PRE="$2"; OUT="$3"; MODELS_DIR="$4"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT/guard" "$OUT/raw"
export ROOT PRE OUT MODELS_DIR GUARD
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  echo "lock held ($(date -u +%FT%TZ))"
  g() { "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/guard/$1.log" -- "${@:2}"; }

  # 1. Gemma 3 lookahead probe.
  PORT=18612
  cat > "$OUT/gemma3_probe.sh" <<EOS
#!/usr/bin/env bash
RUST_LOG=info "$ROOT/target/release/mlxcel" serve -m "$MODELS_DIR/gemma-3-4b-it-4bit" --parallel 1 --metrics --host 127.0.0.1 --port $PORT > "$OUT/gemma3_server.log" 2>&1 &
GPID=\$!
for _ in \$(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma-3-4b-it-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in three sentences why the sky is blue.\"}],\"max_tokens\":96,\"temperature\":0}" \
  > "$OUT/gemma3_response.json"
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/gemma3_metrics.txt"
kill \$GPID; for _ in \$(seq 1 30); do kill -0 \$GPID 2>/dev/null || break; sleep 1; done
EOS
  "$GUARD" --idle-secs 10 --max-attempts 3 --log "$OUT/guard/gemma3_probe.log" -- bash "$OUT/gemma3_probe.sh"
  echo "gemma3 probe rc=$?"
  grep -h "lookahead" "$OUT/gemma3_metrics.txt" | grep -v "^#"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(\"gemma3 usage:\", d.get(\"usage\")); print(d[\"choices\"][0][\"message\"][\"content\"][:200])" "$OUT/gemma3_response.json"

  # 2. gemma-3-4b pp2048 isolation.
  CSV="$OUT/gemma3_pp2048.csv"
  echo "round,arm,prefill_tok_s,decode_tok_s,status" > "$CSV"
  ARMS=(base new new-sync)
  for r in 1 2 3; do
    shift_by=$(( (r - 1) % 3 ))
    order=("${ARMS[@]:$shift_by}" "${ARMS[@]:0:$shift_by}")
    for arm in "${order[@]}"; do
      bin="$ROOT/target/release/mlxcel-bench-decode"; envs=()
      [[ "$arm" == base ]] && bin="$PRE/target/release/mlxcel-bench-decode"
      [[ "$arm" == new-sync ]] && envs=(MLXCEL_FORCE_SYNC=1)
      log="$OUT/raw/gemma3_pp2048_r${r}_${arm}.log"
      g "gemma3_pp2048_r${r}_${arm}" env "${envs[@]}" "$bin" -m "$MODELS_DIR/gemma-3-4b-it-4bit" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 2048 > "$log" 2>&1
      rc=$?
      p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      echo "$r,$arm,$p,$d,$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$CSV"
      sleep 5
    done
  done
  cat "$CSV"
'
echo "lock released ($(date -u +%FT%TZ))"
