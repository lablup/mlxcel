#!/usr/bin/env bash
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The commands behind every CSV in this directory (issue #2834). The protocol
# they implement is docs/mlxcelverse/cuda-oxide-benchmark-evidence.md, frozen
# before the first of them ran.
#
#   run.sh <stage> [<stage> ...]
#   stages: env m3 m4 m5 m6 m6b m1 m2 m2c m7 m8 m9
#   SET=<suffix> writes m4/m5 to files with that suffix (a repeat set).
#
# Build first, outside gpu-lock:
#   MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 \
#     cargo build --release --features cuda \
#       --example oxide_fused_norm_checkpoint_bench --example logit_trace
#   MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 \
#     cargo build --release -p mlxcel-core --features cuda --example oxide_fused_norm_bench
#
# WORK holds everything that is not a result row (profiler captures, logit
# traces, the PTX caches). It is outside the checkout and must be a directory
# only the current user can write: the PTX disk cache refuses anything else.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
WORK=${WORK:?set WORK to a private scratch directory outside the checkout}
MODEL=${MODEL:-$repo/models/llama-3.2-1b-instruct-bf16}
spike=$repo/spike/cuda-oxide-fused-norm
opbench=$repo/target/release/examples/oxide_fused_norm_bench
ckptbench=$repo/target/release/examples/oxide_fused_norm_checkpoint_bench
trace=$repo/target/release/examples/logit_trace
cuda=${CUDA_HOME:-/usr/local/cuda}
corpus=$repo/tests/fixtures/bench/fused-moe-dff-prompt-439-words.txt

# --wrap in standalone-bench.sh splits its argument on whitespace, so a WORK
# with whitespace would split the profiler's -o path.
case "$WORK" in
    *[[:space:]]*) echo "run.sh: WORK must not contain whitespace: $WORK" >&2; exit 2 ;;
esac
mkdir -p "$WORK"
# chmod 700 below must never reach into the checkout.
work_real=$(cd "$WORK" && pwd -P)
case "$work_real/" in
    "$repo"/*) echo "run.sh: WORK ($work_real) is inside the checkout $repo; refusing" >&2; exit 2 ;;
esac
WORK=$work_real
mkdir -p "$WORK/ptx-warm"
chmod 700 "$WORK" "$WORK/ptx-warm"
export MLX_PTX_CACHE_DIR=$WORK/ptx-warm

lock() { # lock <purpose> <command...>
    local purpose=$1
    shift
    gpu-lock run --tag "issue-2834-$purpose" -- "$@"
}

# `pgrep -x`, not `-f`: a shell that merely mentions the runner's process name
# in its command line (a monitoring loop, for instance) must not count as a job.
wait_runner() { # never start measuring while a CI job runs on this host
    local waited=0 limit=${RUNNER_WAIT:-1800}
    while pgrep -x Runner.Worker >/dev/null && [ "$waited" -lt "$limit" ]; do
        sleep 2
        waited=$((waited + 2))
    done
    if pgrep -x Runner.Worker >/dev/null; then
        echo "run.sh: a CI job is still running after ${waited}s; refusing to measure" >&2
        exit 3
    fi
}

stage_env() {
    # Written beside the target and renamed only when the Python step succeeds,
    # so a failure leaves the committed provenance.json alone.
    python3 - "$repo" "$MODEL" "$spike" "$opbench" "$ckptbench" "$cuda" "$trace" >"$here/provenance.json.new" <<'PY'
import hashlib, json, os, platform, subprocess, sys
repo, model, spike, opbench, ckptbench, cuda, trace = sys.argv[1:8]
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=repo).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None
out = {
    "issue": 2834,
    "host": platform.node(),
    "kernel": platform.release(),
    "machine": platform.machine(),
    "gpu": sh("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
    "gpu_clocks_sm_mhz": sh("nvidia-smi", "--query-gpu=clocks.current.sm,clocks.max.sm",
                            "--format=csv,noheader"),
    "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    "cuda_toolkit": sh(os.path.join(cuda, "bin", "nvcc"), "--version").splitlines()[-2:],
    "nsys": sh("nsys", "--version"),
    "rustc": sh("rustc", "--version"),
    "cxx": sh("g++", "--version").splitlines()[:1],
    "mlxcel_commit": sh("git", "rev-parse", "HEAD"),
    "mlxcel_dirty_files": sh("git", "status", "--porcelain", "--untracked-files=no").splitlines(),
    "mlx_pin": sh("bash", "scripts/ci/mlx_pinned_commit.sh"),
    "build": {
        "profile": "release",
        "features": "cuda",
        "env": {"MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM": "1", "MLXCEL_CUDA_ARCHITECTURES": "121"},
        "binaries_sha256": {os.path.basename(p): sha(p) for p in (opbench, ckptbench, trace)},
    },
    "artifact": {
        "ptx_sha256": sha(os.path.join(spike, "artifacts", "fused_add_rms_norm.sm_121.ptx")),
        "cubin_sha256": sha(os.path.join(spike, "artifacts", "fused_add_rms_norm.sm_121.cubin")),
    },
    "checkpoint": {
        "path": model,
        "config_sha256": sha(os.path.join(model, "config.json")),
        "model_safetensors_sha256": sha(os.path.join(model, "model.safetensors")),
        "tokenizer_sha256": sha(os.path.join(model, "tokenizer.json")),
    },
    "gpu_lock": sh("which", "gpu-lock"),
    "ci_job_running_at_capture": bool(sh("pgrep", "-x", "Runner.Worker")),
}
json.dump(out, sys.stdout, indent=2)
print()
PY
    mv "$here/provenance.json.new" "$here/provenance.json"
    cat "$here/provenance.json"
}

# M3: standalone, CUDA events, 5 processes.
stage_m3() {
    wait_runner
    for rep in 1 2 3 4 5; do
        GPU_LOCK_TAG=issue-2834-standalone "$spike/scripts/standalone-bench.sh" \
            --shapes all --iters 200 --label "rep$rep" --csv "$here/m3-standalone-events.csv"
    done
}

# M4: kernel durations from one profiler, standalone and through MLX
# (graphs off so kernels are attributable), 3 captures each.
stage_m4() {
    wait_runner
    local nsys_args=(profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true)
    mkdir -p "$WORK/nsys"
    for rep in 1 2 3; do
        GPU_LOCK_TAG=issue-2834-nsys "$spike/scripts/standalone-bench.sh" --shapes model \
            --iters 200 --label "nsys-rep$rep" --csv "$WORK/nsys/standalone-under-nsys.csv" \
            --wrap "nsys ${nsys_args[*]} -o $WORK/nsys/standalone-$rep"
        nsys export --type sqlite --force-overwrite=true -o "$WORK/nsys/standalone-$rep.sqlite" \
            "$WORK/nsys/standalone-$rep.nsys-rep" >/dev/null
        python3 "$repo/scripts/nsys_kernel_table.py" "$WORK/nsys/standalone-$rep.sqlite" \
            --label "standalone-rep$rep" --match fused_add_rms_norm \
            --kernels "$WORK/nsys/k-standalone-$rep.csv" --api "$WORK/nsys/a-standalone-$rep.csv"

        MLX_USE_CUDA_GRAPHS=0 lock nsys nsys "${nsys_args[@]}" -o "$WORK/nsys/mlx-$rep" \
            "$opbench" --mode iso --shapes model --rounds 5 --iters 200 --warmup 20 \
            --label "nsys-rep$rep" --csv "$WORK/nsys/mlx-iso-under-nsys.csv"
        nsys export --type sqlite --force-overwrite=true -o "$WORK/nsys/mlx-$rep.sqlite" \
            "$WORK/nsys/mlx-$rep.nsys-rep" >/dev/null
        python3 "$repo/scripts/nsys_kernel_table.py" "$WORK/nsys/mlx-$rep.sqlite" \
            --label "mlx-graphs-off-rep$rep" --match fused_add_rms_norm --match rms_norm \
            --match binary_vv \
            --kernels "$WORK/nsys/k-mlx-$rep.csv" --api "$WORK/nsys/a-mlx-$rep.csv"
    done
    # One table per kind: header once, then every capture's rows.
    for kind in k a; do
        local out=$here/m4-nsys-kernels${SET:-}.csv
        [ "$kind" = a ] && out=$here/m4-nsys-cuda-api${SET:-}.csv
        head -1 "$WORK/nsys/$kind-standalone-1.csv" >"$out"
        for f in "$WORK"/nsys/"$kind"-standalone-?.csv "$WORK"/nsys/"$kind"-mlx-?.csv; do
            tail -n +2 "$f" >>"$out"
        done
    done
}

# M5: op level through MLX in production mode (graphs on), 3 processes.
stage_m5() {
    wait_runner
    for rep in 1 2 3; do
        for mode in iso pipe; do
            lock op "$opbench" --mode "$mode" --shapes model,bench2832 --rounds 5 --iters 200 \
                --warmup 20 --label "rep$rep" --csv "$here/m5-op-level-mlx${SET:-}.csv"
        done
    done
}

# M6: cold start, a fresh process per sample, 5 samples per state.
stage_m6() {
    wait_runner
    local shapes=bf16:bf16:1x2048,bf16:bf16:1x4096
    # Rows go to a staging directory and replace the committed CSVs only after
    # every sample ran; a failed stage leaves nothing that a rerun would append to.
    local stage=$WORK/m6-stage
    [ -e "$stage" ] && { echo "run.sh: $stage exists from an earlier run; remove it or use a fresh WORK" >&2; exit 2; }
    mkdir -p "$stage"
    local out=$stage/m6-cold-start
    for rep in 1 2 3 4 5; do
        # fused, empty disk cache: NVRTC compiles the kernel.
        local empty=$WORK/ptx-empty-$rep
        [ -e "$empty" ] && { echo "run.sh: $empty exists; use a fresh WORK" >&2; exit 2; }
        mkdir -p "$empty" && chmod 700 "$empty"
        MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold --arms fused \
            --shapes "$shapes" --iters 50 --label "fused-empty-cache-rep$rep" --csv "$out"
        # fused, the same directory again: served from the disk cache.
        MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold --arms fused \
            --shapes "$shapes" --iters 50 --label "fused-warm-cache-rep$rep" --csv "$out"
        MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold --arms oxide \
            --shapes "$shapes" --iters 50 --label "oxide-rep$rep" --csv "$out"
        MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold --arms graph \
            --shapes "$shapes" --iters 50 --label "graph-rep$rep" --csv "$out"
    done
    mv "$out.cold.csv" "$here/m6-cold-start.csv"
    mv "$out" "$here/m6-cold-start-steady.csv"
    rmdir "$stage"
}

# M6, driver compute cache off (protocol adjustment 1). The CUDA driver keeps
# its own cache of PTX it has compiled (~/.nv/ComputeCache), so a PTX module
# loads in well under a millisecond on a machine that has loaded it before.
# CUDA_CACHE_DISABLE=1 shows the load a machine pays the first time, and after
# a driver update. The existing kernel's disk-cache entries are cubins, which
# the driver does not compile, so its two states are repeated as controls.
stage_m6b() {
    wait_runner
    local shapes=bf16:bf16:1x2048,bf16:bf16:1x4096
    local stage=$WORK/m6b-stage
    [ -e "$stage" ] && { echo "run.sh: $stage exists from an earlier run; remove it or use a fresh WORK" >&2; exit 2; }
    mkdir -p "$stage"
    local out=$stage/m6b-cold-start-driver-cache-off
    for rep in 1 2 3 4 5; do
        local empty=$WORK/ptx-empty-nodrv-$rep
        [ -e "$empty" ] && { echo "run.sh: $empty exists; use a fresh WORK" >&2; exit 2; }
        mkdir -p "$empty" && chmod 700 "$empty"
        CUDA_CACHE_DISABLE=1 MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold \
            --arms fused --shapes "$shapes" --iters 50 --label "fused-empty-cache-rep$rep" --csv "$out"
        CUDA_CACHE_DISABLE=1 MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold \
            --arms fused --shapes "$shapes" --iters 50 --label "fused-warm-cache-rep$rep" --csv "$out"
        CUDA_CACHE_DISABLE=1 MLX_PTX_CACHE_DIR=$empty lock cold "$opbench" --mode cold \
            --arms oxide --shapes "$shapes" --iters 50 --label "oxide-rep$rep" --csv "$out"
        CUDA_CACHE_DISABLE=1 GPU_LOCK_TAG=issue-2834-standalone "$spike/scripts/standalone-bench.sh" \
            --shapes model --iters 200 --label "driver-cache-off-rep$rep" \
            --csv "$stage/m6b-standalone-driver-cache-off.csv"
    done
    mv "$out.cold.csv" "$here/m6b-cold-start-driver-cache-off.csv"
    mv "$out" "$here/m6b-cold-start-driver-cache-off-steady.csv"
    mv "$stage/m6b-standalone-driver-cache-off.csv" "$here/m6b-standalone-driver-cache-off.csv"
    rmdir "$stage"
}

# M8, exploratory (protocol adjustment 3): CUDA API calls per op, one arm per
# process, so the calls can be attributed to an arm. Graphs off and graphs on.
stage_m8() {
    wait_runner
    mkdir -p "$WORK/nsys-arm"
    local out=$here/m8-cuda-api-per-arm.csv
    local first=1
    for graphs in 0 1; do
        for arm in graph fused oxide; do
            local base=$WORK/nsys-arm/$arm-graphs$graphs
            MLX_USE_CUDA_GRAPHS=$graphs lock nsys nsys profile --trace=cuda --sample=none \
                --cpuctxsw=none --force-overwrite=true -o "$base" \
                "$opbench" --mode pipe --arms "$arm" --shapes bf16:bf16:1x2048 --rounds 1 \
                --iters 2000 --warmup 20 --label "api-$arm-graphs$graphs" \
                --csv "$WORK/nsys-arm/pipe-under-nsys.csv"
            nsys export --type sqlite --force-overwrite=true -o "$base.sqlite" "$base.nsys-rep" >/dev/null
            python3 "$repo/scripts/nsys_kernel_table.py" "$base.sqlite" \
                --label "$arm-graphs$graphs-2020-ops" --api "$base-api.csv" >/dev/null
            if [ "$first" = 1 ]; then head -1 "$base-api.csv" >"$out"; first=0; fi
            tail -n +2 "$base-api.csv" >>"$out"
        done
    done
    cat "$out"
}

# M9, exploratory (protocol adjustment 4): host cost of building the lazy node
# alone, nothing evaluated, so no launch and no kernel. 3 processes.
stage_m9() {
    wait_runner
    for rep in 1 2 3; do
        lock op "$opbench" --mode build --shapes model --rounds 5 --iters 2000 --warmup 20 \
            --label "rep$rep" --csv "$here/m9-node-build-cost.csv"
    done
}

# M1: end to end, production mode, 7 interleaved rounds with a null arm.
stage_m1() {
    wait_runner
    python3 "$repo/scripts/oxide_fused_norm_rounds.py" run --bin "$ckptbench" --model "$MODEL" \
        --rounds "${ROUNDS:-7}" --first-round "${FIRST_ROUND:-0}" --out "$here/m1-end-to-end.csv"
    python3 "$repo/scripts/oxide_fused_norm_rounds.py" summarize "$here/m1-end-to-end.csv" \
        --out "$here/m1-end-to-end-summary.csv"
}

# M2: teacher-forced logit traces per arm, three shapes, with route counters.
stage_m2() {
    wait_runner
    mkdir -p "$WORK/traces"
    local summary=$here/m2-logit-trace-summary.csv
    echo "shape,arm,route_line,trace_sha256,positions" >"$summary"
    run_trace() { # run_trace <shape> <chunk tokens> <arm> [env...]
        local shape=$1 chunk=$2 arm=$3
        shift 3
        local out=$WORK/traces/$shape-$arm.tsv
        env -u MLXCEL_FUSED_ADD_RMSNORM -u MLXCEL_CUDA_OXIDE_FUSED_NORM \
            -u MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR "$@" \
            gpu-lock run --tag issue-2834-trace -- "$trace" "$MODEL" "$corpus" "$chunk" 1 8 \
            >"$out" 2>"$out.err"
        local route
        route=$(grep '^# fused_norm_route' "$out.err" | tr '\t' ' ') || true
        if [ -z "$route" ]; then
            echo "run.sh: no '# fused_norm_route' line on stderr for $shape/$arm (see $out.err); without it the arm's route is unproven" >&2
            exit 4
        fi
        echo "$shape,$arm,$route,$(sha256sum "$out" | cut -d' ' -f1),$(grep -vc '^#' "$out")" >>"$summary"
    }
    for shape in prefill256 decode-b1 decode-b4; do
        # The corpus is 480 tokens: one 256-token forward for the prefill
        # shape, and 112-token chunks for the decode shapes so that four rows
        # with their own prefixes fit.
        local chunk=256 extra=()
        [ "$shape" = decode-b1 ] && chunk=112 && extra=(MLXCEL_TRACE_BATCH_ROWS=1)
        [ "$shape" = decode-b4 ] && chunk=112 && extra=(MLXCEL_TRACE_BATCH_ROWS=4)
        run_trace "$shape" "$chunk" graph "${extra[@]}"
        run_trace "$shape" "$chunk" fused MLXCEL_FUSED_ADD_RMSNORM=1 "${extra[@]}"
        run_trace "$shape" "$chunk" fused-null MLXCEL_FUSED_ADD_RMSNORM=1 "${extra[@]}"
        run_trace "$shape" "$chunk" oxide MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_CUDA_OXIDE_FUSED_NORM=1 "${extra[@]}"
    done
    {
        for shape in prefill256 decode-b1 decode-b4; do
            for cand in oxide fused-null graph; do
                echo "== $shape: $cand against fused"
                if cmp -s "$WORK/traces/$shape-fused.tsv" "$WORK/traces/$shape-$cand.tsv"; then
                    echo "byte-identical traces"
                else
                    echo "traces differ"
                fi
                python3 "$repo/scripts/compare_logit_traces.py" "$WORK/traces/$shape-fused.tsv" \
                    "$WORK/traces/$shape-$cand.tsv" --decided 2
            done
        done
    } >"$here/m2-logit-trace-comparison.txt"
    cat "$summary"
}

# M2(c): the #2832 130-case matrix through the adapter, as the 31 tests of the
# opted-in mlxcel-core lib test binary whose name contains cuda_oxide. A
# correctness rerun, not a measurement. LIB is that binary, built outside
# gpu-lock with
#   MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 \
#     cargo test --release -p mlxcel-core --features cuda --lib --no-run
stage_m2c() {
    # No wait_runner: this checks outputs, it does not time anything, so a CI
    # job on the host does not affect the result.
    local lib=${LIB:?set LIB to the opted-in mlxcel_core lib test binary}
    # The raw capture stays in WORK (raw logs are not committed under
    # docs/benchmark_results/); the compact CSV derived from it is committed.
    local out=$WORK/m2c-adapter-matrix-test.log
    local status=0
    {
        echo "# command: gpu-lock run --tag issue-2834-finalize -- $lib cuda_oxide --test-threads=1 --nocapture"
        echo "# mlxcel_commit: $(git -C "$repo" rev-parse HEAD)"
        echo "# binary_sha256: $(sha256sum "$lib" | cut -d' ' -f1)"
        echo "# started_utc: $(date -u +%FT%TZ)"
        gpu-lock run --tag issue-2834-finalize -- "$lib" cuda_oxide --test-threads=1 --nocapture 2>&1 || status=$?
        echo "# exit_status: $status"
    } >"$out.part"
    mv "$out.part" "$out"
    grep -E 'RESULT |^test result' "$out"
    python3 "$repo/scripts/oxide_fused_norm_m2c_summarize.py" "$out" "$here/m2c-adapter-matrix.csv"
    return "$status"
}

# M7: resource usage and instruction counts, one tool on both cubins.
stage_m7() {
    local out=$here/m7-kernel-resources.csv
    echo "implementation,module,kernel,registers,stack_bytes,shared_bytes,constant_bank_bytes,sass_instructions" >"$out"
    resources() { # resources <implementation> <module file>
        python3 - "$1" "$2" "$cuda/bin/cuobjdump" >>"$out" <<'PY'
import os, re, subprocess, sys
impl, module, tool = sys.argv[1:4]
res = subprocess.run([tool, "-res-usage", module], capture_output=True, text=True).stdout
sass = subprocess.run([tool, "-sass", module], capture_output=True, text=True).stdout
counts, current = {}, None
for line in sass.splitlines():
    m = re.search(r"Function : (\S+)", line)
    if m:
        current = m.group(1)
        counts[current] = 0
    elif current and re.search(r"/\*[0-9a-f]{4}\*/", line):
        counts[current] += 1
for m in re.finditer(r"Function (\S+?):\s*\n\s*(.*)", res):
    name, usage = m.group(1), m.group(2)
    field = lambda key: (re.search(key + r":(\d+)", usage) or [None, ""])[1]
    cbank = sum(int(v) for v in re.findall(r"CONSTANT\[\d+\]:(\d+)", usage))
    print(",".join([impl, os.path.basename(module), name, field("REG"), field("STACK"),
                    field("SHARED"), str(cbank), str(counts.get(name, ""))]))
PY
    }
    resources cuda-oxide "$spike/artifacts/fused_add_rms_norm.sm_121.cubin"
    # nullglob: an empty cache must end the stage with a message, not feed the
    # literal pattern to cuobjdump.
    local ptx found=0
    shopt -s nullglob
    for ptx in "$MLX_PTX_CACHE_DIR"/sm121-nvrtc13.0/custom_kernel_mlxcel_fused_add_rms_norm_*.ptx; do
        resources existing-nvrtc "$ptx"
        found=1
    done
    shopt -u nullglob
    if [ "$found" = 0 ]; then
        echo "run.sh: no NVRTC fused_add_rms_norm PTX under $MLX_PTX_CACHE_DIR; run m5 or m6 first" >&2
        exit 4
    fi
    cat "$out"
}

# One row per stage run: when it ran, the load average around it, and whether
# a CI job was running on the host when it ended (one can start mid-stage).
host_state=$here/host-state.csv
[ -f "$host_state" ] || echo "stage,started_utc,ended_utc,load1_start,load1_end,ci_job_running_at_end" >"$host_state"
for stage in "$@"; do
    case "$stage" in
        env | m1 | m2 | m2c | m3 | m4 | m5 | m6 | m6b | m7 | m8 | m9)
            started=$(date -u +%FT%TZ)
            load_start=$(cut -d' ' -f1 /proc/loadavg)
            "stage_$stage"
            ci=false
            pgrep -x Runner.Worker >/dev/null && ci=true
            echo "$stage,$started,$(date -u +%FT%TZ),$load_start,$(cut -d' ' -f1 /proc/loadavg),$ci" >>"$host_state"
            ;;
        *) echo "run.sh: unknown stage $stage" >&2; exit 2 ;;
    esac
done
