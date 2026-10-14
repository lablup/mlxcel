#!/usr/bin/env bash
# Run only under the coordinator's exclusive gpu-lock, after all builds and CI stop.
set -euo pipefail
D=$(cd -- "$(dirname -- "$0")" && pwd)
export REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
export SERVER_BIN=${SERVER_BIN:-$REPO_ROOT/target/release/mlxcel-server}
export MODELS=${MODELS:-$REPO_ROOT/models/mlx}
[[ ! -e "$D/environment.txt" ]] || { echo "Use a fresh attempt directory; evidence already exists" >&2; exit 1; }
[[ -x "$SERVER_BIN" ]]
if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then
  echo "Port 18080 already has a server" >&2; exit 1
fi
{
  date -u +%FT%TZ
  hostname
  git -C "$REPO_ROOT" rev-parse HEAD
  git -C "$REPO_ROOT" status --porcelain
  sha256sum "$SERVER_BIN"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$D/environment.txt"
python3 -I -B "$D/provenance.py" final-acceptance > "$D/environment.json"
for script in run3.sh run.sh run2.sh; do
  bash "$D/$script" > "$D/${script%.sh}-driver.txt" 2>&1
done
sha256sum "$SERVER_BIN" > "$D/binary-after.sha256"
python3 -I -B "$D/analyze_break_even.py" "$D" > "$D/summary.txt"
