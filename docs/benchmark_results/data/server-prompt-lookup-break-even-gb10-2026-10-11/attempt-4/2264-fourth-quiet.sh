#!/usr/bin/env bash
set -euo pipefail
cd /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2264
export REPO_ROOT=$PWD
export SERVER_BIN=$PWD/target/release/mlxcel-server
export MODELS=$PWD/models/mlx
export BINARY_SOURCE_COMMIT=$(cat /tmp/mlxcel-auto-plan/2264-fourth-source.txt)
[[ $(git rev-parse HEAD) == "$BINARY_SOURCE_COMMIT" ]]
sha256sum --check /tmp/mlxcel-auto-plan/2264-fourth-binary.sha256
D=docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-4
[[ ! -e "$D/hostgate.log" ]]
python3 -B - <<'GATE' > "$D/hostgate.log" 2>&1
import sys
sys.path.insert(0, 'docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness')
import hostgate
print('gate_wait_s=', hostgate.wait_quiet(log=sys.stderr), flush=True)
print('ci_job_running=', hostgate.ci_job_running(), flush=True)
GATE
bash "$D/run_all.sh"
