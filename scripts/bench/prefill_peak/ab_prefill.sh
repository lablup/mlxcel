#!/usr/bin/env bash
# Interleaved A/B of two mlxcel binaries (issue #3045): prefill peak footprint
# and timing on the CLI (`generate --profile`) and the server (`serve
# --parallel 1`, one greedy streaming chat request, TTFT from server_ttft.py).
#
# Each cell runs REPS repetitions and alternates the arm order (base first on
# odd repetitions, new first on even ones), so load from anything else on the
# host falls on both arms alike. Every row records the 1-minute load average.
# The peak is the process footprint from `/usr/bin/time -l`. After the runs it
# compares the first repetition's generated text between the arms.
#
# Usage, from the repository root:
#   BASE_BIN=<main build> NEW_BIN=<branch build> scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
# Environment:
#   MODELS_DIR     checkpoint directory (default ./models)
#   CLI_CELLS      space-separated model:prefill_chunk cells (0 = single pass)
#   SERVER_MODELS  space-separated models for the server cells ("" skips them)
#   TOKENS         prompt length (default 28000), GEN_TOKENS (default 32)
#   REPS           repetitions per arm (default 2), PORT (default 8237)
#   EXTRA_ENV      space-separated VAR=VAL pairs applied to both arms
# Needs one GPU to itself while it runs.
set -uo pipefail
OUT=${1:?usage: ab_prefill.sh <out_dir>}
HERE=$(cd "$(dirname "$0")/../metal_sdpa_chunk" && pwd)
MODELS_DIR=${MODELS_DIR:-models}
: "${BASE_BIN:?set BASE_BIN}" "${NEW_BIN:?set NEW_BIN}"
TOKENS=${TOKENS:-28000}
GEN_TOKENS=${GEN_TOKENS:-32}
REPS=${REPS:-2}
PORT=${PORT:-8237}
CLI_CELLS=${CLI_CELLS:-"qwen3.5-4b-4bit:0 qwen3.5-4b-4bit:2048 qwen3.8-27b-4bit:0 qwen3.8-27b-4bit:2048"}
SERVER_MODELS=${SERVER_MODELS:-"qwen3.5-4b-4bit qwen3.8-27b-4bit"}
EXTRA_ENV=${EXTRA_ENV:-}
mkdir -p "$OUT"
prompt="$OUT/prompt_${TOKENS}.txt"
[ -f "$prompt" ] || python3 "$HERE/make_prompt.py" "$TOKENS" > "$prompt"
SUM="$OUT/summary.csv"
[ -f "$SUM" ] || echo "kind,model,chunk,arm,rep,prompt_tokens,prefill_or_ttft,decode_tok_s,peak_bytes,load1,rc" > "$SUM"

bin_of() { if [ "$1" = base ]; then echo "$BASE_BIN"; else echo "$NEW_BIN"; fi; }
load1() { uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/'; }
gen_text() { awk '/^\[Profile Results\]/ {exit} found {print} /^Generating\.\.\.$/ {found=1}' "$1"; }
order() { if [ $(( $1 % 2 )) = 1 ]; then echo "base new"; else echo "new base"; fi; }

for cell in $CLI_CELLS; do
  IFS=: read -r model chunk <<<"$cell"
  for rep in $(seq 1 "$REPS"); do
    for arm in $(order "$rep"); do
      tag="$OUT/cli_${model}_pc${chunk}_${arm}_r${rep}"
      ld=$(load1)
      env $EXTRA_ENV MLXCEL_PREFILL_CHUNK="$chunk" NO_COLOR=1 /usr/bin/time -l "$(bin_of $arm)" generate \
        -m "$MODELS_DIR/$model" -p "$(cat "$prompt")" -n "$GEN_TOKENS" --temp 0 --profile --show-reasoning \
        > "$tag.out" 2> "$tag.err"
      rc=$?
      gen_text "$tag.out" > "$tag.gen"
      peak=$(awk '/peak memory footprint/ {print $1}' "$tag.err" | tail -1)
      pt=$(awk '/Prompt tokens:/ {print $3}' "$tag.out")
      pms=$(awk '/Prefill:/ {print $2}' "$tag.out")
      dt=$(awk '/Decode:/ {gsub(/\(/, "", $4); print $4}' "$tag.out")
      echo "cli,$model,$chunk,$arm,$rep,$pt,$pms,$dt,$peak,$ld,$rc" >> "$SUM"
      echo "cli $model pc=$chunk $arm r$rep prefill_ms=$pms peak=$peak load=$ld rc=$rc"
    done
  done
done

for model in $SERVER_MODELS; do
  for rep in $(seq 1 "$REPS"); do
    for arm in $(order "$rep"); do
      tag="$OUT/server_${model}_${arm}_r${rep}"
      ld=$(load1)
      env $EXTRA_ENV NO_COLOR=1 /usr/bin/time -l "$(bin_of $arm)" serve -m "$MODELS_DIR/$model" \
        --host 127.0.0.1 --port "$PORT" --parallel 1 > "$tag.server.out" 2> "$tag.server.err" &
      tpid=$!
      ok=0
      for _ in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
        kill -0 "$tpid" 2>/dev/null || break
        sleep 1
      done
      if [ $ok = 1 ]; then python3 "$HERE/server_ttft.py" "$PORT" "$prompt" "$GEN_TOKENS" "$model" > "$tag.json"; fi
      spid=$(pgrep -P "$tpid" | head -1)
      [ -n "$spid" ] && kill -TERM "$spid"
      wait "$tpid"; rc=$?
      peak=$(awk '/peak memory footprint/ {print $1}' "$tag.server.err" | tail -1)
      ttft=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%.3f" % d["ttft_s"])' "$tag.json" 2>/dev/null)
      pt=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["prompt_tokens"])' "$tag.json" 2>/dev/null)
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["text"], end="")' "$tag.json" > "$tag.gen" 2>/dev/null
      echo "server,$model,-,$arm,$rep,$pt,$ttft,,$peak,$ld,$rc" >> "$SUM"
      echo "server $model $arm r$rep ttft_s=$ttft peak=$peak load=$ld rc=$rc"
    done
  done
done

echo "--- identity (base vs new, rep 1) ---"
for f in "$OUT"/*_base_r1.gen; do
  g=${f/_base_r1.gen/_new_r1.gen}
  if /usr/bin/cmp -s "$f" "$g"; then echo "identical $(basename "${f%_base_r1.gen}")"; else echo "DIFFERS $(basename "${f%_base_r1.gen}")"; fi
done
