#!/bin/bash
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Real-checkpoint validation of Hy4 (`hy_v4`, issue #2354).
#
# usage: validate_hy_v4.sh MODEL_DIR OUT_DIR [STEP...]
#
# The port was written against synthetic-checkpoint tests only: the 4-bit
# checkpoint is 433 GB on disk and peaks at 435 to 455 GB resident, above the
# default macOS GPU wired limit of a 512 GB host. Raise the limit first, for
# example `sudo sysctl iogpu.wired_limit_mb=490000`, and run nothing else on
# the GPU. Each step loads the checkpoint once, so the steps run one after
# another, never in parallel.
#
# Steps (default: all, in this order):
#   preflight  wired limit, checkpoint completeness, release binaries
#   generate   `mlxcel generate -p "Hello" -n 50` through the normal CLI path
#   server     `mlxcel-server` smoke: one chat completion
#   trace      teacher-forced logit trace of mlxcel against mlx-vlm
#   arms       the same trace with the sinks or the gate dropped (must fail),
#              and with half-split RoPE pairs (settles the pairing by NLL)
#   latent     `--mla-latent-quant int8` and `int4` against the fp16 latent
#
# Environment:
#   CORPUS       text to teacher-force (default docs/architecture.md)
#   GPU_LOCK     lock file shared with other GPU jobs on this host; when set,
#                every model run is wrapped in `lockf -k`
#   MIN_WIRED_MB wired limit the preflight requires (default 470000)
#   MLX_VLM      mlx-vlm checkout the reference arm runs (default
#                references/mlx-vlm, the vendored upstream)
#   PORT         server smoke port (default 18354)
#
# Gates, read from the `compare_logit_traces.py` reports in OUT_DIR:
#   * ref vs mlxcel-no-swiglu: 0 disagreements at decided positions. This is
#     the arm that matches mlx-vlm term for term (mlx-vlm applies no
#     `swiglu_limit`).
#   * ref vs mlxcel (shipped, clamp on): report it. A difference from the
#     previous line is the effect of the routed-expert clamp.
#   * ref vs no_sinks and ref vs no_gate: decided disagreements above zero.
#   * mean NLL of mlxcel vs rope_half_split: the lower one is the pairing the
#     checkpoint was trained with. The shipped default is interleaved (what
#     mlx-vlm does); the transformers reference rotates half-split pairs. If
#     half-split wins, mlx-vlm is wrong too and the default must change.
#   * fp16 vs int8 and fp16 vs int4: record top-1 agreement in the PR.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: validate_hy_v4.sh MODEL_DIR OUT_DIR [STEP...] (steps and gates: see this file's header)" >&2
    exit 2
fi

MODEL=$1
OUT=$2
shift 2
STEPS=("$@")
[ "${#STEPS[@]}" -gt 0 ] || STEPS=(preflight generate server trace arms latent)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORPUS=${CORPUS:-$ROOT/docs/architecture.md}
MLX_VLM=${MLX_VLM:-$ROOT/references/mlx-vlm}
MIN_WIRED_MB=${MIN_WIRED_MB:-470000}
PORT=${PORT:-18354}
BIN=$ROOT/target/release
TRACE=$BIN/examples/logit_trace
# CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL: 4 chunks of 64 traced rows behind a
# 512-token context. Past 2048 tokens of context the DSA indexer starts to
# select; rerun `trace` with LONG=1 to cover that.
SHAPE=(64 4 8 512)
[ -z "${LONG:-}" ] || SHAPE=(64 2 8 4096)

mkdir -p "$OUT"

run() {
    if [ -n "${GPU_LOCK:-}" ]; then
        /usr/bin/lockf -k "$GPU_LOCK" "$@"
    else
        "$@"
    fi
}

mlxcel_trace() {
    # mlxcel_trace NAME [ENV=VALUE...]
    local name=$1
    shift
    echo "== trace $name" >&2
    run env MLXCEL_TRACE_MLA_STATS=1 "$@" "$TRACE" "$MODEL" "$CORPUS" "${SHAPE[@]}" \
        > "$OUT/$name.tsv" 2> "$OUT/$name.log"
}

compare() {
    # compare REFERENCE CANDIDATE
    echo "== compare $1 vs $2" >&2
    python3 "$ROOT/scripts/compare_logit_traces.py" "$OUT/$1.tsv" "$OUT/$2.tsv" --decided 2 \
        | tee "$OUT/$1-vs-$2.txt"
}

mean_nll() {
    awk -F'\t' '!/^#/ { sum += $4; n += 1 } END { printf "%.4f over %d positions\n", sum / n, n }' \
        "$OUT/$1.tsv"
}

step_preflight() {
    local wired
    wired=$(sysctl -n iogpu.wired_limit_mb)
    if [ "$wired" -lt "$MIN_WIRED_MB" ]; then
        echo "iogpu.wired_limit_mb is $wired, below $MIN_WIRED_MB: raise it before loading Hy4" >&2
        exit 1
    fi
    [ -f "$MODEL/config.json" ] || { echo "$MODEL has no config.json" >&2; exit 1; }
    grep -q '"model_type": *"hy_v4"' "$MODEL/config.json" \
        || { echo "$MODEL is not a hy_v4 checkpoint" >&2; exit 1; }
    (cd "$ROOT" && cargo build --release --features metal,accelerate \
        --bin mlxcel --bin mlxcel-server --example logit_trace)
    "$BIN/mlxcel" arch | grep -i 'hy_v4' || { echo "mlxcel arch does not list hy_v4" >&2; exit 1; }
}

step_generate() {
    run "$BIN/mlxcel" generate -m "$MODEL" -p "Hello" -n 50 2> "$OUT/generate.log" \
        | tee "$OUT/generate.txt"
}

step_server() {
    "$BIN/mlxcel-server" -m "$MODEL" --port "$PORT" > "$OUT/server.log" 2>&1 &
    local pid=$!
    trap 'kill "$pid" 2>/dev/null || true' RETURN
    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 "$pid" 2>/dev/null || { echo "server exited; see $OUT/server.log" >&2; return 1; }
        [ "$waited" -lt 3600 ] || { echo "server not healthy after 1 h" >&2; return 1; }
        sleep 10
        waited=$((waited + 10))
    done
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":40,"temperature":0}' \
        | tee "$OUT/server-response.json"
    echo
}

step_trace() {
    echo "== reference trace (mlx-vlm at $MLX_VLM)" >&2
    run uv run --python 3.12 --with-editable "$MLX_VLM" --with tokenizers --with numpy \
        "$ROOT/scripts/mlx_lm_logit_trace.py" "$MODEL" "$CORPUS" "${SHAPE[@]}" --mlx-vlm \
        > "$OUT/ref.tsv" 2> "$OUT/ref.log"
    mlxcel_trace mlxcel
    mlxcel_trace mlxcel-no-swiglu MLXCEL_HY4_ARM=no_swiglu_limit
    compare ref mlxcel-no-swiglu
    compare ref mlxcel
}

step_arms() {
    [ -f "$OUT/ref.tsv" ] || { echo "run the trace step first" >&2; exit 1; }
    mlxcel_trace no-sinks MLXCEL_HY4_ARM=no_sinks,no_swiglu_limit
    mlxcel_trace no-gate MLXCEL_HY4_ARM=no_gate,no_swiglu_limit
    mlxcel_trace rope-half-split MLXCEL_HY4_ARM=rope_half_split
    compare ref no-sinks
    compare ref no-gate
    compare mlxcel rope-half-split
    echo "mean NLL, interleaved RoPE (shipped): $(mean_nll mlxcel)" | tee "$OUT/rope-nll.txt"
    echo "mean NLL, half-split RoPE:            $(mean_nll rope-half-split)" \
        | tee -a "$OUT/rope-nll.txt"
}

step_latent() {
    # The fp16 baseline at the shape the quantized arms run at.
    mlxcel_trace latent-fp16 MLXCEL_TRACE_DECODE_STEP=1
    for quant in int8 int4; do
        run "$BIN/mlxcel" generate -m "$MODEL" -p "Hello" -n 50 --mla-latent-quant "$quant" \
            2> "$OUT/generate-$quant.log" | tee "$OUT/generate-$quant.txt"
        # The trace harness sets the storage mode itself; the option picks the width.
        mlxcel_trace "latent-$quant" MLXCEL_TRACE_KV_CACHE_MODE=int8 \
            MLXCEL_MLA_LATENT_QUANT="$quant" MLXCEL_TRACE_DECODE_STEP=1
        compare latent-fp16 "latent-$quant"
    done
}

for step in "${STEPS[@]}"; do
    case "$step" in
        preflight | generate | server | trace | arms | latent) "step_$step" ;;
        *) echo "unknown step: $step" >&2; exit 2 ;;
    esac
done
