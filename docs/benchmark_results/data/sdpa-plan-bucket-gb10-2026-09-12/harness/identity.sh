#!/usr/bin/env bash
# Greedy token-id identity, compared as ids, between two binaries or two env
# settings of one binary. Prints the id line for each arm and whether they match.
#
#   identity.sh <label> <binary> <model> [extra mlxcel args...]
# Env: NTOK (default 200), plus any MLXCEL_* the caller exports.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
LABEL=$1; BIN=$2; MODEL=$3; shift 3
NTOK=${NTOK:-200}
env MLXCEL_PRINT_TOKEN_IDS=1 MLXCEL_MTP_ALLOW_INEXACT=1 \
  "$BIN" generate -m "$MODEL" -p "$(cat "$D/prompt_code0.txt")" \
  -n "$NTOK" --temp 0 "$@" 2>&1 \
  | sed -n 's/.*\[token ids ([0-9]*): \([0-9 ]*\)\].*/\1/p' > "/tmp/ids_$LABEL.txt"
echo "$LABEL: $(wc -w < "/tmp/ids_$LABEL.txt") ids, sha=$(sha256sum < "/tmp/ids_$LABEL.txt" | cut -c1-16)"
