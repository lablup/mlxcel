#!/usr/bin/env bash
set -euo pipefail
cd /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2259
OUT=/tmp/mlxcel-auto-plan/2259-timing
mkdir -p "$OUT"
test ! -e "$OUT/dry-step-cost.log"
BIN=target/test-fast/deps/mlxcel_core-5feb86ac05c84f53
{
 date -u --iso-8601=seconds
 hostname
 uname -a
 git rev-parse HEAD
 git status --short
 printf "configured_mlx_pin="
 scripts/ci/mlx_pinned_commit.sh
 for mlx_source_dir in target/test-fast/build/mlxcel-core-*/out/build/_deps/mlx-src; do
  if [[ -d "$mlx_source_dir/.git" ]]; then
   printf "fetched_mlx_source=%s commit=" "$mlx_source_dir"
   git -C "$mlx_source_dir" rev-parse HEAD
  fi
 done
 sha256sum "$BIN" "$0"
 nvidia-smi
 gh run list --status in_progress --limit 30 --json databaseId,name,status,headBranch
} > "$OUT/environment.txt" 2>&1
python3 -B - <<'GATE' > "$OUT/hostgate.log" 2>&1
import sys
sys.path.insert(0, 'docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness')
import hostgate
print('gate_wait_s=', hostgate.wait_quiet(log=sys.stderr), flush=True)
print('ci_job_running=', hostgate.ci_job_running(), flush=True)
GATE
"$BIN" sampling::tests::dry_step_cost_report --ignored --nocapture --test-threads=1 > "$OUT/dry-step-cost.log" 2>&1
date -u --iso-8601=seconds >> "$OUT/environment.txt"
