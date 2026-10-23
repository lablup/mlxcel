#!/bin/bash
set -u
kind=$1
shift
cd /home/devops/Development/wt-epic-2382-issue-2431 || exit 90
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MLX_ENABLE_TF32=0
export PYTHONPATH=/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core-layernorm-v1/python:/home/devops/Development/mlxcel/.cache/epic-2382/video-depth-anything-validated-target/epic-2382-vda-reference/python-deps:/home/devops/Development/mlxcel/.cache/epic-2382/pp-doclayout-v3-validated-target/epic-2382-ppdoc-reference/python-deps
export LD_LIBRARY_PATH=/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core-layernorm-v1:/opt/rocm/lib
/home/devops/Development/mlx-rocm-phase1/.venv-xcel/bin/python -B -O /tmp/epic-2382-rfdetr-reference-v2.py "$kind" "$@"
