#!/usr/bin/env bash
# One #1797 measurement session: both families, both arms of the acceptance
# criteria, one binary, one host identity.
#
#   run_session.sh <mlxcel-server binary> [outdir]
#
# Two things here are deliberate and easy to get wrong if this is rewritten.
#
# 1. The binary is COPIED to `mlxcel1797-server` before anything runs. The
#    #1820 host gate blocks on a foreign inference process by `/proc/<pid>/comm`
#    (`mlxcel`, `mlxcel-server`), which is what stops a peer session's model
#    from sharing the GPU with a timed run. A server started under the stock
#    name would match that list and gate itself forever. `/proc/comm` truncates
#    at 15 characters, so `mlxcel1797-serv` is what the gate sees and it is
#    unambiguously not the name a peer's server carries.
#
# 2. `MLX_CUDA_ARCHITECTURES=121` is exported for the build AND for every
#    server process. build.rs auto-detects `121a` on this host, which is not
#    what `.github/workflows/release.yml` ships and not what any earlier GB10
#    record used. A draft block width is a kernel-dispatch result, so the arch
#    is part of the measured configuration rather than build trivia.
set -uo pipefail
BIN=${1:?usage: run_session.sh <mlxcel-server binary> [outdir]}
D="$(cd "$(dirname "$0")" && pwd)"
OUT=${2:-$(cd "$D/.." && pwd)}
export MLX_CUDA_ARCHITECTURES=${MLX_CUDA_ARCHITECTURES:-121}

# The renamed copy lives outside the repo: it is a 100 MB binary and the data
# directory is committed.
RUN_DIR=${RUN_BIN_DIR:-$(mktemp -d -t mlxcel1797.XXXXXX)}
RUN_BIN="$RUN_DIR/mlxcel1797-server"
/bin/cp -f "$BIN" "$RUN_BIN"
chmod +x "$RUN_BIN"

REPO=$(git -C "$D" rev-parse --show-toplevel)
"$D/identity.sh" "$RUN_BIN" \
  "$REPO/models/mlx/qwen3.5-4b-4bit" "$REPO/models/mlx/qwen3.5-4b-dflash" \
  "$REPO/models/mlx/laguna-xs-2.1-nvfp4" "$REPO/models/mlx/laguna-xs-2.1-dflash" \
  | tee "$OUT/identity.txt"

# The affine family first: it is the one with a prior record to agree with, so
# a disagreement there is a signal about the session rather than about NVFP4.
# `default` is the arm the issue exists for (no flag at all); `16` is the
# explicit-override control and `env16` the environment-override control.
python3 "$D/sweep_server_widths.py" \
  --server "$RUN_BIN" \
  --target "$REPO/models/mlx/qwen3.5-4b-4bit" \
  --drafter "$REPO/models/mlx/qwen3.5-4b-dflash" \
  --widths default,2,3,4,5,6,7,8,16,env16 \
  --n 3 --out "$OUT/affine.jsonl" --logdir "$OUT/logs-affine" --port 18797

python3 "$D/sweep_server_widths.py" \
  --server "$RUN_BIN" \
  --target "$REPO/models/mlx/laguna-xs-2.1-nvfp4" \
  --drafter "$REPO/models/mlx/laguna-xs-2.1-dflash" \
  --widths default,2,3,4,5,6,7,8,16 \
  --n 3 --out "$OUT/nvfp4.jsonl" --logdir "$OUT/logs-nvfp4" --port 18798

python3 "$D/summarize.py" "$OUT/affine.jsonl" --label "affine (qwen3.5-4b-4bit + qwen3.5-4b-dflash)"
python3 "$D/summarize.py" "$OUT/nvfp4.jsonl" --label "NVFP4 (laguna-xs-2.1-nvfp4 + laguna-xs-2.1-dflash)"
