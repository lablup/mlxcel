#!/usr/bin/env bash
set -euo pipefail
cd /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227
python3 - <<'PY' > /tmp/mlxcel-auto-plan/2227-profile/hostgate.log 2>&1
import sys,datetime
sys.path.insert(0,'docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness')
import hostgate
print('start',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
hostgate.wait_quiet(log=sys.stdout)
print('quiet',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
PY
bash /tmp/mlxcel-auto-plan/2227-profile/run-profile.sh
OUT=/tmp/mlxcel-auto-plan/2227-run-parity/before bash /tmp/mlxcel-auto-plan/2227-profile/capture-run-before.sh
