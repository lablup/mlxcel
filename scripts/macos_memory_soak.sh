#!/usr/bin/env bash
#
# Copyright 2025-2026 Lablup Inc.
# Licensed under the Apache License, Version 2.0.
#
# Measure how an idle mlxcel-server holding a large model reacts to large file
# writes on macOS (issue #2883).
#
# For each wired-limit setting the script starts the server, sends one request,
# leaves it idle, writes a 20 GiB and then a SIZE file to the internal SSD, and
# sends a second request. A sampler records memory pressure, vm_stat, the file
# cache floor, the server footprint and MLX active/cache memory once a second.
# Results go to one CSV per setting plus a summary.md comparing time to first
# token before and after the write.
#
# This loads a model of tens of GiB and writes tens of GiB to disk. Run it on a
# quiet machine. --dry-run prints the plan and does neither.

set -u

usage() {
    cat <<'EOF'
Usage: scripts/macos_memory_soak.sh [options] MODEL_DIR SIZE

  MODEL_DIR   Model directory, e.g. models/Qwen3.8-Flash-Next-OptiQ-2bit
  SIZE        Size of the large write: NG, e.g. 40G. A 20G write runs first
              when SIZE is larger than 20G.

Options:
  --wired-limits LIST  Comma-separated MLXCEL_WIRED_LIMIT settings
                       (default: default,none,60G; "default" leaves it unset)
  --wire-weights       Also set MLXCEL_WIRE_WEIGHTS=1 for every run
  --allocate SIZE      Hold a SIZE allocation beside the write, e.g. 16G
  --server PATH        Server binary (default: target/release/mlxcel-server)
  --port N             Server port (default: 18083)
  --write-dir DIR      Directory for the files written (default: $TMPDIR)
  --out DIR            Results directory
                       (default: bench-results/macos-memory-soak-<host>-<date>,
                       with a -wire-weights suffix under --wire-weights)
  --idle-seconds N     Idle time before and after the writes (default: 30)
  --dry-run            Print the plan and exit: no server, no write
  -h, --help           Show this help
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

WIRED_LIMITS="default,none,60G"
WIRE_WEIGHTS=0
ALLOCATE=""
SERVER="target/release/mlxcel-server"
PORT=18083
WRITE_DIR="${TMPDIR:-/tmp}"
OUT=""
IDLE_SECONDS=30
DRY_RUN=0
POSITIONAL=()

while [ $# -gt 0 ]; do
    case "$1" in
        --wired-limits) WIRED_LIMITS="${2:?}"; shift 2 ;;
        --wire-weights) WIRE_WEIGHTS=1; shift ;;
        --allocate) ALLOCATE="${2:?}"; shift 2 ;;
        --server) SERVER="${2:?}"; shift 2 ;;
        --port) PORT="${2:?}"; shift 2 ;;
        --write-dir) WRITE_DIR="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        --idle-seconds) IDLE_SECONDS="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) usage >&2; die "unknown option $1" ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

[ "${#POSITIONAL[@]}" -eq 2 ] || { usage >&2; exit 2; }
MODEL="${POSITIONAL[0]}"
SIZE="${POSITIONAL[1]}"

gib_of() {
    case "$1" in
        *[Gg]) echo "${1%?}" ;;
        *) die "size must look like 40G, got $1" ;;
    esac
}

SIZE_GIB=$(gib_of "$SIZE")
case "$SIZE_GIB" in ''|*[!0-9]*) die "size must look like 40G, got $SIZE" ;; esac

WRITES=""
[ "$SIZE_GIB" -gt 20 ] && WRITES="20 "
WRITES="${WRITES}${SIZE_GIB}"

if [ -z "$OUT" ]; then
    # A wired-weights run gets its own directory so it does not overwrite the
    # summary and samples of a plain run from the same day.
    SUFFIX=""
    [ "$WIRE_WEIGHTS" = 1 ] && SUFFIX="-wire-weights"
    OUT="bench-results/macos-memory-soak-$(hostname -s)-$(date +%Y-%m-%d)${SUFFIX}"
fi

IFS=',' read -r -a LIMITS <<<"$WIRED_LIMITS"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Model:          $MODEL"
    echo "Server:         $SERVER (port $PORT)"
    echo "Wired limits:   ${LIMITS[*]}"
    echo "Wire weights:   $WIRE_WEIGHTS"
    echo "Writes (GiB):   $WRITES, into $WRITE_DIR"
    echo "Allocation:     ${ALLOCATE:-none}"
    echo "Idle seconds:   $IDLE_SECONDS"
    echo "Results:        $OUT"
    echo "Sampled once per second: memory_pressure -Q, vm_stat (compressions,"
    echo "decompressions, swapins, swapouts), vm.vm_page_filecache_min, footprint,"
    echo "compressed bytes, MLX active and cache memory."
    exit 0
fi

[ "$(uname -s)" = "Darwin" ] || die "this procedure is macOS only"
[ -x "$SERVER" ] || die "server binary not found: $SERVER (cargo build --release --features metal,accelerate)"
[ -d "$MODEL" ] || die "model directory not found: $MODEL"
[ -d "$WRITE_DIR" ] || die "write directory not found: $WRITE_DIR"
command -v python3 >/dev/null || die "python3 is required"

need_gib=$((SIZE_GIB + 25))
free_gib=$(df -g "$WRITE_DIR" | awk 'NR==2 {print $4}')
[ "$free_gib" -ge "$need_gib" ] || die "$WRITE_DIR has ${free_gib} GiB free, needs ${need_gib} GiB"

mkdir -p "$OUT" || die "cannot create $OUT"
BASE="http://127.0.0.1:$PORT"
SERVER_PID=""
SAMPLER_PID=""
ALLOC_PID=""
WRITE_FILE=""
DD_PID=""

# Remove the write file first, so a server slow to stop cannot delay it. Only
# children this script started are signalled.
cleanup() {
    local tries=0
    if [ -n "$DD_PID" ]; then
        kill "$DD_PID" 2>/dev/null
        wait "$DD_PID" 2>/dev/null
    fi
    [ -n "$WRITE_FILE" ] && rm -f "$WRITE_FILE"
    for pid in "$SAMPLER_PID" "$ALLOC_PID" "$SERVER_PID"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null
    done
    while [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge 60 ]; then
            kill -9 "$SERVER_PID" 2>/dev/null
            break
        fi
        sleep 1
    done
    wait 2>/dev/null
    SERVER_PID=""; SAMPLER_PID=""; ALLOC_PID=""; WRITE_FILE=""; DD_PID=""
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM HUP

# Stream one short completion and print the seconds to the first content delta.
measure_ttft() {
    python3 - "$BASE" <<'PY'
import json
import sys
import time
import urllib.request

body = json.dumps(
    {
        "model": "default",
        "messages": [{"role": "user", "content": "Count from one to five."}],
        "max_tokens": 16,
        "stream": True,
    }
).encode()
request = urllib.request.Request(
    sys.argv[1] + "/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
start = time.monotonic()
with urllib.request.urlopen(request, timeout=1800) as response:
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choices = json.loads(line[6:]).get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        if delta.get("content") or delta.get("reasoning_content"):
            print(f"{time.monotonic() - start:.3f}")
            break
PY
}

metric() {
    curl -s --max-time 2 "$BASE/metrics" | awk -v name="$1" '$1 == name {print $2; exit}'
}

vmstat_field() {
    echo "$1" | awk -v key="$2" 'index($0, key) == 1 {gsub(/[^0-9]/, "", $NF); print $NF; exit}'
}

sample_loop() {
    local csv="$1" phase_file="$2"
    echo "unix_time,phase,memory_free_pct,compressions,decompressions,swapins,swapouts,pages_in_compressor,filecache_min_bytes,phys_footprint_bytes,compressed_bytes,mlx_active_bytes,mlx_cache_bytes" >"$csv"
    while :; do
        local vm free_pct
        vm=$(vm_stat)
        free_pct=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub(/%/, "", $2); print $2}')
        echo "$(date +%s),$(cat "$phase_file"),$free_pct,$(vmstat_field "$vm" 'Compressions:'),$(vmstat_field "$vm" 'Decompressions:'),$(vmstat_field "$vm" 'Swapins:'),$(vmstat_field "$vm" 'Swapouts:'),$(vmstat_field "$vm" 'Pages occupied by compressor:'),$(sysctl -n vm.vm_page_filecache_min 2>/dev/null),$(metric mlxcel_process_phys_footprint_bytes),$(metric mlxcel_process_compressed_bytes),$(metric mlxcel_mlx_active_bytes),$(metric mlxcel_mlx_cache_bytes)" >>"$csv"
        sleep 1
    done
}

allocator() {
    python3 -c '
import sys, time
size = int(sys.argv[1]) << 30
block = bytearray(size)
for offset in range(0, size, 4096):
    block[offset] = 1
time.sleep(100000)
' "$(gib_of "$ALLOCATE")" &
    ALLOC_PID=$!
}

SUMMARY="$OUT/summary.md"
{
    echo "# macOS memory soak: $MODEL"
    echo
    echo "Host: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) >> 30)) GiB, macOS $(sw_vers -productVersion). Writes: ${WRITES} GiB. Allocation: ${ALLOCATE:-none}. MLXCEL_WIRE_WEIGHTS: $WIRE_WEIGHTS."
    echo
    echo "| MLXCEL_WIRED_LIMIT | TTFT before (s) | TTFT after (s) | Compressions | Swapouts |"
    echo "|---|---|---|---|---|"
} >"$SUMMARY"

run_one() {
    local limit="$1" log csv phase_file ttft_before ttft_after tries=0
    log="$OUT/server-$limit.log"
    csv="$OUT/samples-$limit.csv"
    phase_file="$OUT/.phase"
    echo loading >"$phase_file"

    # Without this check a foreign server would answer /health and be measured.
    if curl -s --max-time 2 -o /dev/null "$BASE/health"; then
        echo "port $PORT already answers; choose another with --port" >&2
        return 1
    fi

    (
        unset MLXCEL_WIRED_LIMIT MLXCEL_WIRE_WEIGHTS
        [ "$limit" = "default" ] || export MLXCEL_WIRED_LIMIT="$limit"
        [ "$WIRE_WEIGHTS" -eq 1 ] && export MLXCEL_WIRE_WEIGHTS=1
        exec "$SERVER" -m "$MODEL" --port "$PORT" --metrics
    ) >"$log" 2>&1 &
    SERVER_PID=$!

    until curl -sf --max-time 2 "$BASE/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited, see $log" >&2; cleanup; return 1; }
        tries=$((tries + 1))
        [ "$tries" -lt 1800 ] || { echo "server not ready after 30 min" >&2; cleanup; return 1; }
        sleep 1
    done

    sample_loop "$csv" "$phase_file" &
    SAMPLER_PID=$!

    echo first_request >"$phase_file"
    ttft_before=$(measure_ttft)
    if [ -z "$ttft_before" ]; then
        echo "[$limit] no first token, skipping the write; see $log" >&2
        cleanup
        return 1
    fi
    echo "[$limit] TTFT before write: ${ttft_before}s"
    echo idle_before >"$phase_file"
    sleep "$IDLE_SECONDS"

    [ -n "$ALLOCATE" ] && allocator

    local before_c before_s after_c after_s gib vm
    vm=$(vm_stat); before_c=$(vmstat_field "$vm" 'Compressions:'); before_s=$(vmstat_field "$vm" 'Swapouts:')
    for gib in $WRITES; do
        echo "write_${gib}G" >"$phase_file"
        WRITE_FILE="$WRITE_DIR/mlxcel-soak-$$-$gib.bin"
        # In the background so that a signal interrupts the wait at once.
        dd if=/dev/urandom of="$WRITE_FILE" bs=1048576 count=$((gib * 1024)) 2>>"$log" &
        DD_PID=$!
        if ! wait "$DD_PID"; then
            DD_PID=""
            echo "[$limit] write of ${gib}G failed, see $log" >&2
            cleanup
            return 1
        fi
        DD_PID=""
        rm -f "$WRITE_FILE"
        WRITE_FILE=""
    done
    vm=$(vm_stat); after_c=$(vmstat_field "$vm" 'Compressions:'); after_s=$(vmstat_field "$vm" 'Swapouts:')

    echo idle_after >"$phase_file"
    sleep "$IDLE_SECONDS"
    echo second_request >"$phase_file"
    ttft_after=$(measure_ttft)
    echo "[$limit] TTFT after write: ${ttft_after}s"

    echo "| $limit | $ttft_before | $ttft_after | $((after_c - before_c)) | $((after_s - before_s)) |" >>"$SUMMARY"
    cleanup
}

status=0
for limit in "${LIMITS[@]}"; do
    run_one "$limit" || status=1
done
rm -f "$OUT/.phase"
echo "Results: $OUT"
exit "$status"
