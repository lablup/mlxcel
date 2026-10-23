#!/bin/bash
set -u
cd /home/devops/Development/wt-epic-2382-issue-2412 || exit 90
binary=/home/devops/Development/mlxcel/.cache/epic-2382/sam31-artifacts/proof/integrated-tested/mlxcel-1fa3d785
expected=4878babc21ab7d631b1371c96087d3d619b83b3245ada6900f64d04520919243
actual=$(sha256sum "$binary")
[[ ${actual%% *} == "$expected" ]] || exit 91
out="$PWD/target/epic-2382-ppdoc-reference/rt-heron-baseline.json"
[[ ! -e "$out" ]] || exit 92
MLX_ENABLE_TF32=0 "$binary" detect -m "$PWD/target/epic-2382-ppdoc-reference/regression-checkpoints/heron/8f39ad3c0b4c58e9c2d2c84a38465abf757272d8" -i "$PWD/target/epic-2382-ppdoc-reference/bert-page/two_column_page.png" --format json > "$out" 2> /tmp/epic-2382-ppdoc-rt-baseline-v1.log
rc=$?
printf '%s\n' "$rc" > /tmp/epic-2382-ppdoc-rt-baseline-v1.child-status
exit "$rc"
