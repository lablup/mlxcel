#!/usr/bin/env bash
# Run the page-gather decode microbench (epic #116 Phase 0 / #117).
#
# Times gather-then-SDPA vs contiguous SDPA across a context/batch/block-size
# sweep to inform the paged-attention strategy and KV pool layout in
# docs/adr/0001-paged-attention-gather-vs-fused-kernel.md.
#
# Usage:
#   scripts/run_page_gather_microbench.sh                 # default sweep
#   scripts/run_page_gather_microbench.sh --context-lengths 1024,4096
#
# Any extra arguments are forwarded to the example. Runs under `caffeinate -i`
# so the host does not idle-throttle the GPU mid-run; let the machine cool
# between sweeps for stable numbers.

set -euo pipefail
cd "$(dirname "$0")/.."

caffeinate -i cargo run --release --features metal,accelerate \
  --example page_gather_microbench -- "$@"
