#!/usr/bin/env bash
# Gemma 3 lookahead probe alone (issue #2192): a one-slot server started with
# --metrics, one greedy chat request, then mlxcel_batch_decode_lookahead_steps_total.
# Usage: gemma3_probe_session.sh REPO_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; OUT="$2"; MODELS_DIR="$3"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT/guard"
export ROOT OUT MODELS_DIR GUARD
PORT=18613
cat > "$OUT/gemma3_probe.sh" <<EOS
#!/usr/bin/env bash
RUST_LOG=info "$ROOT/target/release/mlxcel" serve -m "$MODELS_DIR/gemma-3-4b-it-4bit" --parallel 1 --metrics --host 127.0.0.1 --port $PORT > "$OUT/gemma3_server.log" 2>&1 &
GPID=\$!
for _ in \$(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/gemma3_metrics_before.txt"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma-3-4b-it-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in three sentences why the sky is blue.\"}],\"max_tokens\":96,\"temperature\":0}" \
  > "$OUT/gemma3_response.json"
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/gemma3_metrics.txt"
kill \$GPID; for _ in \$(seq 1 30); do kill -0 \$GPID 2>/dev/null || break; sleep 1; done
EOS
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  "$GUARD" --idle-secs 10 --max-attempts 3 --log "$OUT/guard/gemma3_probe.log" -- bash "$OUT/gemma3_probe.sh"
  echo "gemma3 probe rc=$?"
'
grep -h "lookahead" "$OUT/gemma3_metrics.txt" | grep -v "^#"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('gemma3 usage:', d.get('usage'))" "$OUT/gemma3_response.json"
