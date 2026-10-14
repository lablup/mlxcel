#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230
BIN="$ROOT/target/release/mlxcel"
MODEL="$ROOT/models/mlx/granite-4.0-h-tiny-4bit"
PROMPT='Explain the Mamba state update.'
OUT_ROOT=/tmp/mlxcel-auto-plan

if [[ ${ISSUE2230_GRANITE_LOCKED:-0} != 1 ]]; then
    command -v gpu-lock >/dev/null
    exec gpu-lock run --tag issue2230-granite-parity -- \
        env ISSUE2230_GRANITE_LOCKED=1 "$0" "$@"
fi

[[ -x "$BIN" ]] || {
    echo "missing release binary: $BIN" >&2
    exit 1
}
[[ -f "$MODEL/config.json" ]] || {
    echo "missing Granite checkpoint: $MODEL" >&2
    exit 1
}

mkdir -p "$OUT_ROOT"
RUN_DIR=$(mktemp -d "$OUT_ROOT/2230-granite-parity.XXXXXXXX")
COMMON=(
    generate
    -m "$MODEL"
    -p "$PROMPT"
    -n 64
    --temp 0
)

{
    echo "run_dir=$RUN_DIR"
    echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
    echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
    printf 'kernel_command='
    printf '%q ' env -u MLXCEL_SSM_KERNEL -u MLXCEL_SSM_CUDA_KERNEL \
        MLXCEL_SDPA_DETERMINISTIC=1 MLXCEL_PRINT_TOKEN_IDS=1 \
        "$BIN" "${COMMON[@]}"
    printf '\n'
    printf 'graph_command='
    printf '%q ' env -u MLXCEL_SSM_KERNEL \
        MLXCEL_SDPA_DETERMINISTIC=1 MLXCEL_PRINT_TOKEN_IDS=1 \
        MLXCEL_SSM_CUDA_KERNEL=0 "$BIN" "${COMMON[@]}"
    printf '\n'
} | tee "$RUN_DIR/provenance.txt"

env -u MLXCEL_SSM_KERNEL -u MLXCEL_SSM_CUDA_KERNEL \
    MLXCEL_SDPA_DETERMINISTIC=1 \
    MLXCEL_PRINT_TOKEN_IDS=1 \
    "$BIN" "${COMMON[@]}" \
    >"$RUN_DIR/kernel.stdout" \
    2>"$RUN_DIR/kernel.stderr"

env -u MLXCEL_SSM_KERNEL \
    MLXCEL_SDPA_DETERMINISTIC=1 \
    MLXCEL_PRINT_TOKEN_IDS=1 \
    MLXCEL_SSM_CUDA_KERNEL=0 \
    "$BIN" "${COMMON[@]}" \
    >"$RUN_DIR/graph.stdout" \
    2>"$RUN_DIR/graph.stderr"

python3 - "$RUN_DIR" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
line_pattern = re.compile(r"\[(prompt|token) ids \((\d+)\): ([0-9 ]*)\]")


def read_ids(arm: str) -> dict[str, list[int]]:
    text = (run_dir / f"{arm}.stderr").read_text()
    found: dict[str, list[int]] = {}
    for kind, declared_text, payload in line_pattern.findall(text):
        ids = [int(value) for value in payload.split()] if payload.strip() else []
        declared = int(declared_text)
        if declared != len(ids):
            raise SystemExit(
                f"{arm}: {kind} id line declares {declared}, contains {len(ids)}"
            )
        if kind in found:
            raise SystemExit(f"{arm}: duplicate {kind} id line")
        found[kind] = ids
    if set(found) != {"prompt", "token"}:
        raise SystemExit(
            f"{arm}: expected prompt and token id lines, found {sorted(found)}"
        )
    if len(found["token"]) != 64:
        raise SystemExit(
            f"{arm}: expected 64 generated token ids, found {len(found['token'])}"
        )
    return found


kernel = read_ids("kernel")
graph = read_ids("graph")
if kernel["prompt"] != graph["prompt"]:
    raise SystemExit("kernel and graph arms tokenized the prompt differently")

first_divergence = next(
    (
        index
        for index, (kernel_id, graph_id) in enumerate(
            zip(kernel["token"], graph["token"], strict=True)
        )
        if kernel_id != graph_id
    ),
    None,
)
payload = " ".join(map(str, kernel["token"])).encode()
comparison = {
    "prompt_ids": len(kernel["prompt"]),
    "generated_ids": len(kernel["token"]),
    "token_ids_identical": first_divergence is None,
    "first_divergence_zero_based": first_divergence,
    "kernel_token_ids": kernel["token"],
    "graph_token_ids": graph["token"],
    "kernel_token_ids_sha256": hashlib.sha256(payload).hexdigest(),
}
(run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
if first_divergence is not None:
    raise SystemExit(
        "Granite greedy kernel/graph outputs diverged at generated token "
        f"{first_divergence + 1}"
    )
PY

echo "Granite kernel and graph paths produced the same 64 greedy token ids."
echo "Artifacts: $RUN_DIR"
