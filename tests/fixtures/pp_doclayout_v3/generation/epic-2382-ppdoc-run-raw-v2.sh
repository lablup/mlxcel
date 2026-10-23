#!/bin/bash
set -u
kind=$1
cd /home/devops/Development/wt-epic-2382-issue-2412 || exit 90
log="/tmp/epic-2382-ppdoc-raw-v2-${kind}.log"
child="/tmp/epic-2382-ppdoc-raw-v2-${kind}.child-status"
PYTHONDONTWRITEBYTECODE=1 MLX_ENABLE_TF32=0 PYTHONPATH="$PWD/target/epic-2382-ppdoc-reference/python-deps:/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core/python" LD_LIBRARY_PATH=/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core:/opt/rocm/lib /home/devops/Development/mlx-rocm-phase1/.venv-xcel/bin/python -O /tmp/epic-2382-ppdoc-raw-reference-v2.py "$kind" --device gpu > "$log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$child"
tail -n 12 "$log"
exit "$rc"
