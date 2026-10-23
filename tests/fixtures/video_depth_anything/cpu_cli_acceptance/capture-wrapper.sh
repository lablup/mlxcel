#!/bin/bash
set -u
kind=$1
device=$2
frames=$3
cd /home/devops/Development/wt-epic-2382-issue-2352 || exit 2
guard=()
if [[ "$device" == gpu ]]; then guard=(scripts/rocm_gpu_guard.sh --hold --max-wait 300 --); fi
"${guard[@]}" env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/epic-2382-vda-reference/python-deps" /home/devops/Development/mlx-rocm-phase1/.venv-xcel/bin/python /tmp/epic-2382-vda-cli-capture.py "$kind" --device "$device" --frames "$frames" > "/tmp/epic-2382-vda-cli-$kind-$device-$frames.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "/tmp/epic-2382-vda-cli-$kind-$device-$frames.status"
tail -c 1600 "/tmp/epic-2382-vda-cli-$kind-$device-$frames.log"
exit "$rc"
