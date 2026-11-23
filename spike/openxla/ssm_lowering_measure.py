"""Prefill-lowering measurements for the SSM scan spike (issue #3337).

For each prefill form and context capacity this records what
`FINDINGS_ssm.md` reports: iree-compile wall time and peak RSS for `llvm-cpu`
and `rocm` (gfx1151), the StableHLO op count, the dispatch count (IREE stream
statistics), and the max-abs error against the HF fp32 oracle on `local-task`
and `hip`. It drives the existing pieces only: `ssm_oracle.py --emit-inputs`,
`xla_ssm_probe --emit-only` (the emitted graph), a direct `iree-compile` of that
graph, `xla_ssm_probe` (the shim run), and `ssm_oracle.py --check`.

Run under the GPU guard, after building the example, with the ROCm env loaded:

    eval "$(scripts/iree/setup-rocm.sh --env)"
    cargo build --release --features xla-iree --example xla_ssm_probe
    scripts/rocm_gpu_guard.sh --hold -- spike/openxla/.venv/bin/python \
        spike/openxla/ssm_lowering_measure.py --out /tmp/ssm-measure \
        --forms chunked:64,chunked:128,chunked:256,associative,sequential \
        --capacities 256,4096

Compiles run one at a time (31 GiB unified memory); `--compile-timeout` bounds
each one, and a timed-out or failed step is recorded rather than fatal.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(ROOT, "target", "release", "examples", "xla_ssm_probe")
TARGETS = {
    "llvm-cpu": [
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
    ],
    "rocm": ["--iree-hal-target-device=hip"],
}


def compile_measured(mlir, target, rocm_arch, out_dir, tag, timeout):
    compiler = os.environ["MLXCEL_XLA_IREE_COMPILE"]
    flags = list(TARGETS[target])
    if target == "rocm":
        flags.append(f"--iree-rocm-target={rocm_arch}")
    stats = os.path.join(out_dir, f"{tag}.{target}.stats.json")
    cmd = [
        compiler,
        "--iree-input-type=stablehlo",
        *flags,
        "--iree-scheduling-dump-statistics-format=json",
        f"--iree-scheduling-dump-statistics-file={stats}",
        mlir,
        "-o",
        os.path.join(out_dir, f"{tag}.{target}.vmfb"),
    ]
    log = os.path.join(out_dir, f"{tag}.{target}.log")
    started = time.monotonic()
    with open(log, "wb") as err:
        # Own session: iree-compile is a wrapper around the native compiler, and a
        # timeout must kill both.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True
        )
    status, peak = None, None
    while status is None:
        pid, raw, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid == proc.pid:
            status = os.waitstatus_to_exitcode(raw)
            peak = round(usage.ru_maxrss / 1024, 1)
            proc.returncode = status
        elif time.monotonic() - started > timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "timeout"
        else:
            time.sleep(0.2)
    seconds = time.monotonic() - started
    result = dict(status=status, seconds=round(seconds, 2), peak_rss_mib=peak)
    if status == 0 and os.path.exists(stats):
        with open(stats) as fh:
            execution = json.load(fh)["stream-aggregate"]["execution"]
        result["dispatches"] = execution["dispatch-count"]
        result["transient_mib"] = round(execution["transient-memory-size"] / 2**20, 1)
    elif status != 0:
        with open(log, errors="replace") as fh:
            result["error"] = fh.read()[-600:]
    return result


def op_count(mlir):
    with open(mlir) as fh:
        text = fh.read()
    return len(re.findall(r"(?:= |\")stablehlo\.[a-z_]+", text))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="chunked:64,chunked:128,chunked:256,associative,sequential")
    ap.add_argument("--capacities", default="256,4096")
    ap.add_argument("--dims", default="mamba2-130m")
    ap.add_argument("--devices", default="local-task,hip")
    ap.add_argument("--targets", default="llvm-cpu,rocm")
    ap.add_argument("--compile-timeout", type=int, default=1800)
    ap.add_argument("--skip", default="", help="form@capacity pairs to skip, comma separated")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    py = sys.executable
    rocm_arch = os.environ.get("MLXCEL_XLA_ROCM_TARGET", "gfx1151")
    skip = set(filter(None, args.skip.split(",")))
    results_path = os.path.join(args.out, "results.json")
    results = []
    if os.path.exists(results_path):
        with open(results_path) as fh:
            results = json.load(fh)
    done = {(r["form"], r["capacity"]) for r in results}

    for capacity in map(int, args.capacities.split(",")):
        real_len = capacity - capacity // 8  # padding rows are exercised
        inputs = os.path.join(args.out, f"inputs-{capacity}")
        if not os.path.exists(os.path.join(inputs, "meta.json")):
            subprocess.run(
                [py, os.path.join(HERE, "ssm_oracle.py"), "--emit-inputs", inputs,
                 "--mode", "prefill", "--capacity", str(capacity),
                 "--real-len", str(real_len), "--dims", args.dims],
                check=True,
            )
        for spec in args.forms.split(","):
            form, _, chunk = spec.partition(":")
            label = spec
            if (label, capacity) in done or f"{label}@{capacity}" in skip:
                continue
            tag = f"{label.replace(':', '-')}-{capacity}"
            row = dict(form=label, capacity=capacity, real_len=real_len, dims=args.dims)
            mlir = os.path.join(args.out, f"{tag}.mlir")
            extra = ["--form", form] + (["--chunk-size", chunk] if chunk else [])
            subprocess.run(
                [PROBE, "--inputs", inputs, "--emit-only", mlir, *extra], check=True
            )
            row["stablehlo_ops"] = op_count(mlir)
            row["mlir_kib"] = round(os.path.getsize(mlir) / 1024, 1)
            print(f"[{tag}] ops={row['stablehlo_ops']}", flush=True)
            for target in args.targets.split(","):
                r = compile_measured(mlir, target, rocm_arch, args.out, tag, args.compile_timeout)
                row[f"compile_{target}"] = r
                print(f"[{tag}] compile {target}: {r}", flush=True)
            for device in args.devices.split(","):
                target = "llvm-cpu" if device.startswith("local") else "rocm"
                if row.get(f"compile_{target}", {}).get("status") != 0:
                    row[f"error_{device}"] = "not compiled"
                    continue
                result_dir = os.path.join(args.out, f"{tag}-{device}")
                run = subprocess.run(
                    [PROBE, "--inputs", inputs, "--device", device, "--out", result_dir, *extra],
                    capture_output=True, text=True,
                )
                if run.returncode != 0:
                    row[f"error_{device}"] = (run.stderr or run.stdout)[-600:]
                    print(f"[{tag}] run {device} failed: {row[f'error_{device}']}", flush=True)
                    continue
                report = os.path.join(result_dir, "check.json")
                subprocess.run(
                    [py, os.path.join(HERE, "ssm_oracle.py"), "--check", result_dir,
                     "--json", report],
                    capture_output=True,
                )
                with open(report) as fh:
                    checked = json.load(fh)
                with open(os.path.join(result_dir, "run.json")) as fh:
                    timing = json.load(fh)
                row[f"error_{device}"] = {
                    k: checked[k]["max_abs"] for k in ("y", "h", "conv")
                }
                row[f"run_seconds_{device}"] = round(timing["run_seconds"], 4)
                print(f"[{tag}] {device}: {row[f'error_{device}']}", flush=True)
            results.append(row)
            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=1)
    print(f"results -> {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
