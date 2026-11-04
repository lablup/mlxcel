#!/usr/bin/env python3
"""Image-generation server throughput ladder (GB10 / CUDA and Apple Silicon).

Starts one `mlxcel-server --image-model <dir>` per checkpoint (image-only
mode), waits for readiness, warms up once, then times
`POST /v1/images/generations` over a fixed prompt/size/step shape and appends
one CSV row per model. Only stdlib.

Usage:
  scripts/bench_imagegen.py --bin target/release/mlxcel-server \
      --out benchmarks/<file>.csv [--only NAME ...] [--repeats 3] [--port 18092]

The CSV `backend` column is cuda-<nvcc release>, rocm-<release> or metal;
MLXCEL_BENCH_BACKEND overrides it for a build the probe cannot name.

The measurement is end-to-end serving latency: HTTP, tokenization, the full
denoising loop, VAE decode and PNG/base64 encoding. It is not a kernel number.
Every model runs the same 512x512, 8-step, batch-1 shape so the rows compare;
a family whose quality needs more steps will look faster here than its default
settings would, which is why `steps` is a CSV column.
"""
import argparse
import base64
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same store probe as scripts/bench_embeddings.py: `models/` holds checkpoints
# directly on this host, `models/mlx/` on another, and `MLXCEL_MODEL_STORE`
# overrides when a host has neither shape.
def _default_store():
    override = os.environ.get("MLXCEL_MODEL_STORE")
    if override:
        return Path(override)
    for candidate in (Path("models"), Path("models/mlx")):
        if candidate.is_dir():
            return candidate
    return Path("models")


STORE = _default_store()
PROMPT = "a red fox sitting in snow, soft morning light, detailed fur"
EDIT_PROMPT = "make it a night scene with moonlight and falling snow"
IMAGE = Path("tests/fixtures/test_image.png")
NEGATIVE = ""
SIZE = "512x512"
STEPS = 8
SEED = 0

CSV_HEADER = [
    "model", "model_path", "prompt", "size", "steps", "seed", "images",
    "repeats", "p50_ms", "mean_ms", "min_ms", "s_per_image", "load_ms",
    "date", "hardware", "mlxcel_version", "build_type", "mlxcel_commit",
    "mlx_commit", "backend", "notes",
]


def _host_memory_gb():
    """Linux host memory as `NNGB`: unified-memory parts answer [N/A] for
    memory.total, and this is the figure bench_decode.sh records."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return f"{round(int(line.split()[1]) / 1024 ** 2)}GB"
    except (OSError, ValueError, IndexError):
        pass
    return ""


def _rocm_hardware():
    """(gfx target, marketing name, ROCm version, VRAM bytes) or None.

    Same detection as `scripts/bench_embeddings.py` and `bench_decode.sh`:
    `rocminfo` names the first GPU agent, `/opt/rocm/.info/version` carries the
    release, and sysfs carries the device VRAM (the host figure is wrong on a
    UMA carve-out). `ROCM_PATH` overrides the install root.
    """
    import re
    import shutil

    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    tool = shutil.which("rocminfo") or str(root / "bin/rocminfo")
    if not Path(tool).is_file():
        return None
    try:
        out = subprocess.run([tool], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    gfx = ""
    name = ""
    for line in out.splitlines():
        match = re.match(r"\s*Name:\s+(gfx[0-9a-f]+)\s*$", line)
        if match and not gfx:
            gfx = match.group(1)
            continue
        match = re.match(r"\s*Marketing Name:\s+(.+?)\s*$", line)
        if match and gfx and not name:
            name = match.group(1)
    if not gfx:
        return None
    version = ""
    for candidate in (root / ".info/version", root / "share/rocm/.info/version"):
        try:
            version = candidate.read_text().strip().splitlines()[0]
            break
        except OSError:
            pass
    vram = 0
    for node in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        try:
            vram = max(vram, int(node.read_text().strip()))
        except (OSError, ValueError):
            pass
    return gfx, name or "AMD_GPU", version, vram


def detect_hardware():
    """Host string for the CSV `hardware` column, matching bench_decode.sh."""
    import platform

    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        ).stdout.strip()
        return chip or "unknown"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
    except OSError:
        gpu = []
    if gpu:
        name, memory = [part.strip() for part in gpu[0].split(",")]
        mib = "".join(ch for ch in memory if ch.isdigit())
        gb = f"{round(int(mib) / 1024)}GB" if mib else _host_memory_gb()
        return f"{name.replace(' ', '_')}_{gb}".strip("_")
    rocm = _rocm_hardware()
    if rocm:
        gfx, name, version, vram = rocm
        parts = [name.replace(" ", "_"), gfx]
        if version:
            parts.append(f"ROCm{version}")
        if vram:
            parts.append(f"{round(vram / 1024**3)}GB")
        return "_".join(parts).replace(",", "_")
    return "unknown"


def _nvcc_release():
    """The toolkit release behind the binary, e.g. `13.0`, or the empty string.

    `nvcc` is not on PATH on every CUDA host (the GB10 box needs
    /usr/local/cuda/bin added), so the default install root is the fallback.
    """
    import re
    import shutil

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    match = re.search(r"release\s+([0-9][0-9.]*)", out)
    return match.group(1) if match else ""


def detect_backend():
    """Versioned label for the CSV `backend` column.

    The hardware column names the box, not the stack: the same GB10 can run
    CUDA 13.0 or 13.4, a ROCm host changes release, and the cuda-rust bridge,
    Vulkan or OpenXLA paths are distinct stacks on that same box.
    `MLXCEL_BENCH_BACKEND` overrides the probe for a build it cannot name.
    """
    override = os.environ.get("MLXCEL_BENCH_BACKEND")
    if override:
        return override
    import platform
    import shutil

    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvidia-smi"):
        release = _nvcc_release()
        return f"cuda-{release}" if release else "cuda"
    rocm = _rocm_hardware()
    if rocm:
        version = rocm[2]
        return f"rocm-{version}" if version else "rocm"
    return "unknown"


def build_info(cwd):
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ""

    commit = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if commit and dirty:
        commit += "-dirty"
    version = ""
    try:
        for line in (Path(cwd) / "Cargo.toml").read_text().splitlines():
            if line.startswith("version"):
                version = line.split('"')[1]
                break
    except OSError:
        pass
    mlx_commit = ""
    # Single source of truth for the MLX pin (#1047); the short hash is what
    # the other benchmark CSVs record.
    import re
    pin = Path(cwd) / "src/lib/mlxcelverse/CMakeLists.txt"
    try:
        match = re.search(r'GIT_TAG\s+"?([0-9a-f]{40})"?', pin.read_text())
        if match:
            mlx_commit = match.group(1)[:8]
    except OSError:
        pass
    return version, commit, mlx_commit


def is_image_checkpoint(entry):
    """A Diffusers-style image checkpoint, in either layout this repo ships.

    `model_index.json` marks the classic layout, but several checkpoints here
    are shipped without it (the `flux2-klein-*-4bit` and `ming-image` sets):
    they carry `transformer/` (or `transformer-packed-*`) plus `vae/` instead.
    Both markers are accepted; the server still decides whether this build can
    run the family, and an unsupported one becomes a FAIL row.
    """
    if not entry.is_dir():
        return False
    if (entry / "model_index.json").is_file():
        return True
    children = [child for child in entry.iterdir() if child.is_dir()]
    transformer = any(
        child.name == "transformer" or child.name.startswith("transformer-")
        for child in children
    )
    return transformer and any(child.name == "vae" for child in children)


def discover(store):
    if not store.is_dir():
        return []
    return [entry for entry in sorted(store.iterdir()) if is_image_checkpoint(entry)]


def start_server(binary, model_dir, port, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [binary, "--image-model", str(model_dir), "--host", "127.0.0.1",
         "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    return proc, log


def wait_ready(port, proc, timeout_s=900):
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as resp:
                if resp.status == 200:
                    return (time.perf_counter() - start) * 1000.0
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return None


def request_image(port, task="generate", image_uri=None, timeout_s=900):
    prompt = EDIT_PROMPT if task == "edit" else PROMPT
    body = {
        "prompt": prompt,
        "size": SIZE,
        "steps": STEPS,
        "seed": SEED,
        "n": 1,
        "response_format": "b64_json",
    }
    if NEGATIVE:
        body["negative_prompt"] = NEGATIVE
    route = "/v1/images/generations"
    if task == "edit":
        if not image_uri:
            raise RuntimeError("edit task needs a reference image")
        body["image"] = image_uri
        route = "/v1/images/edits"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    parsed = json.loads(payload)
    images = parsed.get("data") or []
    if not images:
        raise RuntimeError("response carried no image data")
    first = images[0]
    if first.get("b64_json"):
        base64.b64decode(first["b64_json"], validate=False)
    elif not first.get("url"):
        raise RuntimeError("response carried neither b64_json nor url")
    return elapsed_ms


def stop_server(proc, log):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--task", choices=["generate", "edit"], default="generate",
                    help="generate: POST /v1/images/generations; edit: POST "
                         "/v1/images/edits with --image as the reference")
    ap.add_argument("--image", default=str(IMAGE),
                    help="reference image for --task edit")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=18092)
    ap.add_argument("--build-type", default="release")
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    cwd = Path(args.cwd)
    logdir = Path(args.logdir) if args.logdir else Path("/tmp/mlxcel-imagegen-logs")
    logdir.mkdir(parents=True, exist_ok=True)
    version, commit, mlx_commit = build_info(cwd)
    image_uri = None
    if args.task == "edit":
        reference = Path(args.image)
        if not reference.is_file():
            print(f"--task edit needs a reference image, missing {reference}", file=sys.stderr)
            return 1
        image_uri = "data:image/png;base64," + base64.b64encode(
            reference.read_bytes()
        ).decode()
    prompt = EDIT_PROMPT if args.task == "edit" else PROMPT
    hardware = detect_hardware()
    backend = detect_backend()
    date = time.strftime("%Y-%m-%d")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    models = discover(STORE)
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m.name in wanted]
    if not models:
        print(f"no image checkpoints under {STORE}", file=sys.stderr)
        return 1

    csv_file = open(out, "w", newline="")
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    csv_file.flush()

    for model_dir in models:
        name = model_dir.name
        row = [name, str(model_dir), prompt, SIZE, STEPS, SEED, 1, args.repeats,
               "", "", "", "", "", date, hardware, version, args.build_type,
               commit, mlx_commit, backend, ""]
        print(f">>> {name}", file=sys.stderr, flush=True)
        proc, log = start_server(args.bin, model_dir, args.port, logdir / f"{name}.log")
        load_ms = wait_ready(args.port, proc)
        if load_ms is None:
            stop_server(proc, log)
            row[-1] = "FAIL:start"
            writer.writerow(row)
            csv_file.flush()
            print("    FAIL:start", file=sys.stderr, flush=True)
            continue
        try:
            request_image(args.port, args.task, image_uri)  # warmup, untimed
            samples = [
                request_image(args.port, args.task, image_uri)
                for _ in range(args.repeats)
            ]
        except Exception as err:  # noqa: BLE001 - the row records the failure
            stop_server(proc, log)
            row[-1] = f"FAIL:request ({type(err).__name__}: {err})"[:200]
            writer.writerow(row)
            csv_file.flush()
            print(f"    {row[-1]}", file=sys.stderr, flush=True)
            continue
        stop_server(proc, log)
        p50 = statistics.median(samples)
        row[8] = f"{p50:.2f}"
        row[9] = f"{statistics.mean(samples):.2f}"
        row[10] = f"{min(samples):.2f}"
        row[11] = f"{p50 / 1000.0:.3f}"
        row[12] = f"{load_ms:.1f}"
        writer.writerow(row)
        csv_file.flush()
        print(f"    p50 {p50:.0f} ms ({p50 / 1000.0:.2f} s/image)", file=sys.stderr, flush=True)

    csv_file.close()
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
