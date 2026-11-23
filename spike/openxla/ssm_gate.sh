#!/usr/bin/env bash
# Single-layer Mamba2 mixer gate (issue #3337): every acceptance case of the SSM
# probe against the HF fp32 oracle, on each device given (default local-task
# and hip). Build the example first:
#   cargo build --release --features xla-iree --example xla_ssm_probe
# Run on ROCm under the guard:
#   scripts/rocm_gpu_guard.sh --hold -- spike/openxla/ssm_gate.sh OUT_DIR [devices...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: ssm_gate.sh OUT_DIR [device...]}"
shift || true
DEVICES=("$@")
[ ${#DEVICES[@]} -gt 0 ] || DEVICES=(local-task hip)
PY="${MLXCEL_ORACLE_PYTHON:-$ROOT/spike/openxla/.venv/bin/python}"
PROBE="$ROOT/target/release/examples/xla_ssm_probe"
ORACLE="$ROOT/spike/openxla/ssm_oracle.py"
CAP="${MLXCEL_SSM_CAPACITY:-256}"

# name | oracle arguments
CASES=(
  "decode|--mode decode"
  "decode-b3-bias|--mode decode --batch 3 --use-bias"
  "prefill-partial|--mode prefill --capacity $CAP --real-len 100"
  "prefill-full|--mode prefill --capacity $CAP --real-len $CAP"
  "prefill-short|--mode prefill --capacity $CAP --real-len 2"
  "prefill-empty|--mode prefill --capacity $CAP --real-len 0"
  "prefill-empty-carried|--mode prefill --capacity $CAP --real-len 0 --init-state"
  "prefill-carried|--mode prefill --capacity $CAP --real-len 77 --init-state --use-bias"
)

failed=0
for case in "${CASES[@]}"; do
  name="${case%%|*}"
  read -r -a oracle_args <<<"${case#*|}"
  inputs="$OUT/$name/inputs"
  "$PY" "$ORACLE" --emit-inputs "$inputs" "${oracle_args[@]}" 2>/dev/null
  for device in "${DEVICES[@]}"; do
    result="$OUT/$name/$device"
    echo "== $name on $device"
    if "$PROBE" --inputs "$inputs" --device "$device" --out "$result" &&
      "$PY" "$ORACLE" --check "$result" --json "$result/check.json"; then
      :
    else
      failed=1
    fi
  done
done
[ "$failed" -eq 0 ] && echo "SSM GATE: PASS" || echo "SSM GATE: FAIL"
exit "$failed"
