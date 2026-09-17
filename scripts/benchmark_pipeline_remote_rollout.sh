#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_WORKTREE="$(git -C "$ROOT_DIR" worktree list --porcelain | sed -n 's/^worktree //p' | head -n1)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
if [[ ! -d "$MODEL_ROOT" && -n "$PRIMARY_WORKTREE" && -d "$PRIMARY_WORKTREE/models" ]]; then
  MODEL_ROOT="$PRIMARY_WORKTREE/models"
fi

DATE_TAG="${DATE_TAG:-$(date +%F)}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/target/release/mlxcel-server}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/llama-3.2-1b-instruct-4bit}"
MODEL_ALIAS="${MODEL_ALIAS:-$(basename "$MODEL_PATH")}"
PROMPT="${PROMPT:-Hello from multi-machine pipeline parallel validation.}"
MAX_TOKENS="${MAX_TOKENS:-16}"
REQUESTS="${REQUESTS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-30}"
COORDINATOR_URL="${COORDINATOR_URL:-http://127.0.0.1:18080}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/benchmarks/pipeline_parallel_remote_rollout_${DATE_TAG}.csv}"

CLUSTER_NAME="${CLUSTER_NAME:-remote-pp}"
TRANSPORT_BACKEND="${TRANSPORT_BACKEND:-tcp}"
COORDINATOR_CONTROL_ADDR="${COORDINATOR_CONTROL_ADDR:-127.0.0.1:19000}"
STAGE0_ADDR="${STAGE0_ADDR:-127.0.0.1:19001}"
STAGE1_ADDR="${STAGE1_ADDR:-127.0.0.1:19002}"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmark_pipeline_remote_rollout.sh write-config [output.toml]
  scripts/benchmark_pipeline_remote_rollout.sh smoke
  scripts/benchmark_pipeline_remote_rollout.sh benchmark

Commands:
  write-config  Emit a 2-stage remote pipeline cluster TOML using env-configured addresses.
  smoke         Probe a running coordinator once and print the response body.
  benchmark     Send repeated requests to a running coordinator and append CSV rows.

Key environment variables:
  MODEL_PATH                Model directory. Default: models/llama-3.2-1b-instruct-4bit
  COORDINATOR_URL           Coordinator HTTP base URL. Default: http://127.0.0.1:18080
  OUTPUT_CSV                CSV path for benchmark rows.
  CLUSTER_NAME              Cluster name for write-config.
  TRANSPORT_BACKEND         tcp or thunderbolt.
  COORDINATOR_CONTROL_ADDR  Coordinator control socket address.
  STAGE0_ADDR               Stage 0 transport address.
  STAGE1_ADDR               Stage 1 transport address.
EOF
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required tool: $1" >&2
    exit 1
  fi
}

extract_json_number() {
  local key="$1"
  sed -n "s/.*\"${key}\":\\([0-9][0-9.]*\\).*/\\1/p"
}

write_header_if_missing() {
  mkdir -p "$(dirname "$OUTPUT_CSV")"
  if [[ ! -f "$OUTPUT_CSV" ]]; then
    printf '%s\n' \
      'date,scenario,transport,topology,model_alias,request_index,status_or_http_code,latency_sec,completion_tokens,prompt,max_tokens,notes' \
      >"$OUTPUT_CSV"
  fi
}

sanitize_csv_field() {
  local value="${1//$'\n'/ }"
  value="${value//,/;}"
  printf '%s' "$value"
}

wait_for_health() {
  local url="$1/health"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  exit 1
}

write_config() {
  local output="${1:-$ROOT_DIR/examples/distributed/generated_pipeline_remote_2node_${TRANSPORT_BACKEND}.toml}"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<EOF
[cluster]
name = "${CLUSTER_NAME}"
pipeline_parallel_size = 2
transport_backend = "${TRANSPORT_BACKEND}"

[[nodes]]
id = "coordinator"
address = "${COORDINATOR_CONTROL_ADDR}"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "${STAGE0_ADDR}"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "${STAGE1_ADDR}"
role = "pipeline_stage"
stage = 1
EOF
  echo "wrote $output"
}

smoke() {
  require_tool curl
  wait_for_health "$COORDINATOR_URL"
  curl -fsS "${COORDINATOR_URL}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_ALIAS}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
  printf '\n'
}

benchmark() {
  require_tool curl
  require_tool awk
  wait_for_health "$COORDINATOR_URL"
  write_header_if_missing

  local request_url="${COORDINATOR_URL}/v1/completions"
  local body http_code latency completion_tokens response_file curl_meta
  local prompt_field notes_field
  prompt_field="$(sanitize_csv_field "$PROMPT")"
  notes_field="$(sanitize_csv_field "coordinator_url=${COORDINATOR_URL}")"

  for request_index in $(seq 1 "$REQUESTS"); do
    response_file="$(mktemp)"
    curl_meta="$(
      curl -sS -o "$response_file" -w '%{http_code} %{time_total}' \
        --max-time "$REQUEST_TIMEOUT" \
        "$request_url" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_ALIAS}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
    )"
    http_code="${curl_meta%% *}"
    latency="${curl_meta##* }"
    body="$(cat "$response_file")"
    rm -f "$response_file"
    completion_tokens="$(printf '%s\n' "$body" | extract_json_number completion_tokens)"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$DATE_TAG" \
      "remote_pipeline_http" \
      "$TRANSPORT_BACKEND" \
      "2node" \
      "$MODEL_ALIAS" \
      "$request_index" \
      "$http_code" \
      "$latency" \
      "${completion_tokens:-}" \
      "$prompt_field" \
      "$MAX_TOKENS" \
      "$notes_field" \
      >>"$OUTPUT_CSV"
  done

  local avg_latency
  avg_latency="$(
    awk -F, 'NR > 1 { sum += $8; count += 1 } END { if (count == 0) { print "0" } else { printf "%.6f", sum / count } }' "$OUTPUT_CSV"
  )"
  echo "wrote $OUTPUT_CSV (avg latency ${avg_latency}s)"
}

command="${1:-}"
case "$command" in
  write-config)
    write_config "${2:-}"
    ;;
  smoke)
    smoke
    ;;
  benchmark)
    benchmark
    ;;
  *)
    usage
    exit 1
    ;;
esac
