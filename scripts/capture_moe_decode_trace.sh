#!/usr/bin/env bash
# Capture a Metal GPU trace of one warm MoE decode token, for the fused
# decode-MoE kernel work (#268). See
# docs/benchmark_results/fused-moe-decode-kernel-design.md.
#
# Two modes:
#   gputrace  (default) the bench's measured pass after a warmup, captured
#             through MLXCEL_METAL_CAPTURE_PATH: the short prompt's prefill
#             plus one decode token (`-n 2`; the first token comes from the
#             prefill). Open in Xcode: `open <out>.gputrace`. The Summary
#             (Command Buffers / Compute Encoders / Dispatch Calls) is readable
#             without the slow Profile pass; the decode token is the last
#             command buffers, so compare expert-path idle/dispatch counts
#             there before vs after the kernel lands.
#   xctrace   Metal System Trace over N decode tokens (timeline view).
#
# Usage:
#   scripts/capture_moe_decode_trace.sh [model] [mode] [out]
#   scripts/capture_moe_decode_trace.sh models/qwen3-30b-a3b-4bit
#   scripts/capture_moe_decode_trace.sh models/dots.llm1.inst-mixed-4-6bit xctrace
#
# Traces dump the streamed weights (multi-GB) into $TMPDIR; they are ephemeral.
set -euo pipefail

MODEL="${1:-models/qwen3-30b-a3b-4bit}"
MODE="${2:-gputrace}"
BIN="./target/release/mlxcel-bench-decode"
PROMPT="Once upon a time, there was a young inventor named Ada who loved to build machines. One day, she decided to"

[[ -x "$BIN" ]] || { echo "build first: cargo build --release --features metal,accelerate --bin mlxcel-bench-decode" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

case "$MODE" in
  gputrace)
    OUT="${3:-/tmp/mlxcel_moe_$(basename "$MODEL").gputrace}"
    rm -rf "$OUT"
    echo "Capturing the measured prefill and one warm decode token -> $OUT"
    MTL_CAPTURE_ENABLED=1 MLXCEL_METAL_CAPTURE_PATH="$OUT" \
      "$BIN" -m "$MODEL" -p "$PROMPT" -n 2 --warmup-tokens 6 --no-chat-template
    echo "Done. A finalized bundle has hex-named archive files at top level."
    echo "Open: open '$OUT'"
    ;;
  xctrace)
    OUT="${3:-/tmp/mlxcel_moe_$(basename "$MODEL").trace}"
    rm -rf "$OUT"
    echo "Recording Metal System Trace -> $OUT"
    xcrun xctrace record --template "Metal System Trace" --output "$OUT" \
      --launch -- "$BIN" -m "$MODEL" -p "$PROMPT" -n 40 --warmup-tokens 10 --no-chat-template
    echo "Inspect tables: xcrun xctrace export --input '$OUT' --toc"
    ;;
  *)
    echo "unknown mode: $MODE (use gputrace or xctrace)" >&2; exit 1 ;;
esac
