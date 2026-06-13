#!/usr/bin/env bash
# Capture a Metal GPU trace of one warm MoE decode token, for the fused
# decode-MoE kernel work (#268). See
# docs/benchmark_results/fused-moe-decode-kernel-design.md.
#
# Two modes:
#   gputrace  (default) one warm decode token via the in-process
#             MLXCEL_CAPTURE_DECODE hook, then the process exits so the bundle
#             finalizes. Open in Xcode: `open <out>.gputrace`. The Summary
#             (Command Buffers / Compute Encoders / Dispatch Calls) is readable
#             without the slow Profile pass; compare expert-path idle/dispatch
#             counts before vs after the kernel lands.
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
    echo "Capturing one warm decode token -> $OUT"
    MTL_CAPTURE_ENABLED=1 MLXCEL_CAPTURE_DECODE="$OUT" \
      "$BIN" -m "$MODEL" -p "$PROMPT" -n 1 --warmup-tokens 6 --no-chat-template
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
