#!/usr/bin/env bash
# Profile one Laguna DFlash width at two token budgets so the pair can be
# differenced. Usage: nsys_laguna.sh TAG BLOCK [env assignments...]
set -uo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)"
BIN=${BIN:?set BIN}
tag=$1; block=$2; shift 2
L=/home/inureyes/models/mlx/laguna-xs-2.1-nvfp4
DR=/home/inureyes/models/mlx/laguna-xs-2.1-dflash
for n in 200 100; do
  spec=(--draft-model "$DR" --draft-kind dflash --draft-block-size "$block")
  [ "$block" = "0" ] && spec=()
  env MLXCEL_MTP_ALLOW_INEXACT=1 "$@" \
    nsys profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true \
      -o "$D/nsys_${tag}_${n}" \
      "$BIN" generate -m "$L" -p "$(cat "$D/harness/prompt_code0.txt")" \
      -n "$n" --temp 0 "${spec[@]}" > "$D/nsys_${tag}_${n}.out" 2>&1
  echo "$tag n=$n rc=$? $(grep -o 'rounds=[0-9]*' "$D/nsys_${tag}_${n}.out" | head -1) $(grep -o '= [0-9.]* tok/s' "$D/nsys_${tag}_${n}.out" | head -1)"
done
