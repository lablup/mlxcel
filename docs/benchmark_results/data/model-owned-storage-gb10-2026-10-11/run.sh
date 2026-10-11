#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230
BIN=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/target/release/mlxcel
CLIENT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/scripts/bench_serving_concurrency.py
MODEL_ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/models/mlx
DATA=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11
GATE_DRIVER=${DATA}/gate.py
HOSTGATE_DIR=${WORKTREE}/docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness
PORT=18080

export MLXCEL_CCCL_DIR=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include/cccl
export MLXCEL_CUTLASS_DIR=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include
MLX_SOURCE_DIR=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/_deps/mlx-src

SERVER_PID=
cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=
}
trap cleanup_server EXIT INT TERM

mkdir -p "${DATA}"
exec > >(tee -a "${DATA}/run.log") 2>&1

for required in "${BIN}" "${CLIENT}" "${GATE_DRIVER}" "${HOSTGATE_DIR}/hostgate.py" "${MODEL_ROOT}/gemma-3-1b-it-4bit" "${MODEL_ROOT}/llama-4-scout-17b-16e-instruct-4bit" "${MLXCEL_CCCL_DIR}" "${MLXCEL_CUTLASS_DIR}" "${MLX_SOURCE_DIR}/.git"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required benchmark path: ${required}" >&2
        exit 2
    fi
done

if [[ -e "${DATA}/complete" ]]; then
    echo "benchmark already completed; remove ${DATA}/complete to rerun" >&2
    exit 2
fi

: > "${DATA}/hostgate.jsonl"
python3 "${GATE_DRIVER}" --hostgate-dir "${HOSTGATE_DIR}" --evidence "${DATA}/hostgate.jsonl" --label run-preflight --port "${PORT}" --port-only

mlx_cmake_tag=$(awk '$1 == "GIT_TAG" { gsub(/[")]/, "", $2); print $2; exit }' "${WORKTREE}/src/lib/mlx-cpp/CMakeLists.txt")
mlx_checkout_commit=$(git -C "${MLX_SOURCE_DIR}" rev-parse HEAD)
if [[ -z "${mlx_cmake_tag}" || "${mlx_cmake_tag}" != "${mlx_checkout_commit}" ]]; then
    echo "MLX pin mismatch: CMake GIT_TAG=${mlx_cmake_tag:-missing}, release checkout=${mlx_checkout_commit}" >&2
    exit 2
fi

{
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "worktree=${WORKTREE}"
    echo "binary=${BIN}"
    echo "binary_sha256=$(sha256sum "${BIN}" | awk '{print $1}')"
    echo "commit=$(git -C "${WORKTREE}" rev-parse HEAD)"
    echo "mlx_cmake_git_tag=${mlx_cmake_tag}"
    echo "mlx_checkout=${MLX_SOURCE_DIR}"
    echo "mlx_checkout_commit=${mlx_checkout_commit}"
    echo "hostgate_source=${HOSTGATE_DIR}/hostgate.py"
    echo "hostgate_sha256=$(sha256sum "${HOSTGATE_DIR}/hostgate.py" | awk '{print $1}')"
    echo "host=$(hostname)"
    uname -a
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
} > "${DATA}/environment.txt"

: > "${DATA}/schedule.tsv"
printf 'model\tprompt_tokens\tround\tposition\tarm\tbackend\tstarted_utc\n' >> "${DATA}/schedule.tsv"

wait_for_server() {
    local attempt
    for attempt in $(seq 1 600); do
        if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done
    return 1
}

run_arm() {
    local model_name=$1
    local model_path=$2
    local prompt_tokens=$3
    local round=$4
    local position=$5
    local arm=$6
    local backend=paged
    if [[ "${arm}" == dense ]]; then
        backend=dense
    fi

    local stem="${model_name}-p${prompt_tokens}-r${round}-${arm}"
    local server_log="${DATA}/server-${stem}.log"
    local client_log="${DATA}/client-${stem}.txt"

    echo "GATE ${stem} $(date -u +%FT%TZ)"
    python3 "${GATE_DRIVER}" --hostgate-dir "${HOSTGATE_DIR}" --evidence "${DATA}/hostgate.jsonl" --label "${stem}" --port "${PORT}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${model_name}" "${prompt_tokens}" "${round}" "${position}" "${arm}" "${backend}" "$(date -u +%FT%TZ)" >> "${DATA}/schedule.tsv"
    echo "START ${stem} $(date -u +%FT%TZ)"
    "${BIN}" serve -m "${model_path}" --port "${PORT}" --metrics --parallel 8 --decode-storage-backend "${backend}" > "${server_log}" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "server failed to become ready for ${stem}" >&2
        tail -n 100 "${server_log}" >&2 || true
        exit 3
    fi

    python3 "${CLIENT}" --port "${PORT}" --concurrency 1,4,8 --prompt-tokens "${prompt_tokens}" --max-tokens 128 --metrics > "${client_log}" 2>&1
    if ! awk '/^[[:space:]]*[148][[:space:]]/ { rows += 1; if ($3 != 0) failed = 1 } END { exit !(rows == 3 && failed == 0) }' "${client_log}"; then
        echo "client run reported missing rows or request failures for ${stem}" >&2
        cat "${client_log}" >&2
        exit 4
    fi

    cleanup_server
    sleep 5
    echo "DONE ${stem} $(date -u +%FT%TZ)"
}

for model_name in gemma3 llama4; do
    case "${model_name}" in
        gemma3) model_path="${MODEL_ROOT}/gemma-3-1b-it-4bit" ;;
        llama4) model_path="${MODEL_ROOT}/llama-4-scout-17b-16e-instruct-4bit" ;;
    esac
    for prompt_tokens in 512 4096; do
        for round in 1 2 3 4 5; do
            case $(( (round - 1) % 3 )) in
                0) order=(paged-a dense paged-b) ;;
                1) order=(dense paged-b paged-a) ;;
                2) order=(paged-b paged-a dense) ;;
            esac
            position=0
            for arm in "${order[@]}"; do
                position=$((position + 1))
                run_arm "${model_name}" "${model_path}" "${prompt_tokens}" "${round}" "${position}" "${arm}"
            done
        done
    done
done

echo "completed_utc=$(date -u +%FT%TZ)" >> "${DATA}/environment.txt"
touch "${DATA}/complete"
echo "ALLDONE $(date -u +%FT%TZ)"
