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
# Trace a verify block against the decode step, row by row, at equal context.
#
# usage: matched_width_trace.sh MODEL_DIR CORPUS_TXT OUT_DIR
#
# Width W (a verify block) is traced once, starting at token PREFILL so that
# every block sees a PREFILL-token context window. Row j of a width-W chunk
# follows PREFILL + j tokens of history, so its width-1 reference is traced with
# PREFILL' = PREFILL + j and start token PREFILL + j: that run's chunk W*c then
# sees exactly the same token history as block c, row j. The width-1 history is
# one prefill forward, not W sequential decode steps. Comparing the rows of
# one width-W run with the rows of a single width-1 run would mix context
# lengths, which moves the logits by itself.
#
# Environment:
#   WIDTH    block width, the verify block size under test (default 8)
#   PREFILL  tokens of history before the first row (default 128)
#   BLOCKS   width-W chunks to trace (default 48)
#   LOGIT_TRACE  path to the logit_trace binary (default
#            ./target/release/examples/logit_trace)
#   MLXCEL_TRACE_KV_CACHE_MODE  stores KV like `--kv-cache-mode` (model-owned
#            caches only: Gemma 3 and Gemma 4 among the shipped families)
#
# Compare with: scripts/compare_matched_width.py OUT_DIR
set -euo pipefail
usage="usage: matched_width_trace.sh MODEL_DIR CORPUS_TXT OUT_DIR"
if [ "$#" -ne 3 ]; then
  echo "$usage" >&2
  exit 2
fi
M=$1
C=$2
O=$3
W=${WIDTH:-8}
P=${PREFILL:-128}
N=${BLOCKS:-48}
T=${LOGIT_TRACE:-./target/release/examples/logit_trace}
for pair in "WIDTH=$W" "BLOCKS=$N"; do
  case "${pair#*=}" in
    '' | *[!0-9]* | 0) echo "matched_width_trace: ${pair%%=*} must be a positive integer, got '${pair#*=}'" >&2; exit 2 ;;
  esac
done
case "$P" in
  '' | *[!0-9]*) echo "matched_width_trace: PREFILL must be a non-negative integer, got '$P'" >&2; exit 2 ;;
esac
[ -d "$M" ] || { echo "matched_width_trace: model directory not found: $M" >&2; exit 2; }
[ -r "$C" ] || { echo "matched_width_trace: corpus not readable: $C" >&2; exit 2; }
[ -x "$T" ] || { echo "matched_width_trace: trace binary not executable: $T (set LOGIT_TRACE or build --example logit_trace)" >&2; exit 2; }
mkdir -p -- "$O"
MLXCEL_TRACE_START_TOKEN=$P "$T" "$M" "$C" "$W" "$N" 8 "$P" > "$O/w${W}.tsv"
for ((j = 0; j < W; j++)); do
  MLXCEL_TRACE_START_TOKEN=$((P + j)) \
    "$T" "$M" "$C" 1 $(((N - 1) * W + 1)) 8 $((P + j)) > "$O/w1_p$((P + j)).tsv"
done
