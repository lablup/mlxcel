#!/usr/bin/env python3
"""Embedding and rerank server throughput ladder (epic #1348).

Starts one mlxcel-server per checkpoint, waits for readiness, warms up once,
then times POST /v1/embeddings or POST /v1/rerank over fixed workloads and
appends one CSV row per (model, input_kind, batch) cell. Only stdlib.

Usage:
  scripts/bench_embeddings.py --bin target/release/mlxcel-server --out benchmarks/<file>.csv \
      [--only NAME ...] [--repeats 5] [--port 18091]
"""
import argparse, base64, csv, json, os, signal, statistics, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

HOME = Path.home()
STORE = HOME / ".cache/mlxcel/models"
IMAGE = Path("tests/fixtures/test_image.png").resolve()

SHORT = "The quick brown fox jumps over the lazy dog near the river bank at dawn."
LONG = " ".join([
    "Retrieval augmented generation pairs a dense retriever with a language model so that answers are grounded in documents.",
    "The retriever encodes queries and passages into vectors, and the nearest passages are placed in the prompt.",
    "Embedding quality decides which passages are found, and pooling, normalization and prompt prefixes all move the result.",
    "A reranker then rescores the shortlist with a cross encoder or a generative model that reads the pair jointly.",
] * 6)

# (name, path, kind) kind in {"text", "vl", "multivector", "rerank", "rerank_vl"}
MODELS = [
    ("all-MiniLM-L6-v2", STORE / "sentence-transformers/all-MiniLM-L6-v2", "text"),
    ("multilingual-e5-small", STORE / "intfloat/multilingual-e5-small", "text"),
    ("bge-m3-safetensors", STORE / "seansitter/bge-m3-safetensors", "text"),
    ("modernbert-embed-base", STORE / "nomic-ai/modernbert-embed-base", "text"),
    ("siglip-base-patch16-224", STORE / "google/siglip-base-patch16-224", "text"),
    ("embeddinggemma-300m-4bit", STORE / "mlx-community/embeddinggemma-300m-4bit", "text"),
    ("Qwen3-Embedding-0.6B", STORE / "Qwen/Qwen3-Embedding-0.6B", "text"),
    ("llama-nemotron-embed-1b-v2", STORE / "nvidia/llama-nemotron-embed-1b-v2", "text"),
    ("Nemotron-3-Embed-1B-BF16", STORE / "nvidia/Nemotron-3-Embed-1B-BF16", "text"),
    ("Nemotron-3-Embed-1B-BF16-8bit", STORE / "mlx-community/Nemotron-3-Embed-1B-BF16-8bit", "text"),
    ("LFM2.5-Embedding-350M", STORE / "LiquidAI/LFM2.5-Embedding-350M", "text"),
    ("Qwen3-VL-Embedding-2B", STORE / "Qwen/Qwen3-VL-Embedding-2B", "vl"),
    ("llama-nemotron-embed-vl-1b-v2", STORE / "nvidia/llama-nemotron-embed-vl-1b-v2", "vl"),
    ("colSmol-256M-merged", STORE / "local/colSmol-256M-merged", "multivector"),
    ("colqwen2.5-v0.2-merged", STORE / "local/colqwen2.5-v0.2-merged", "multivector"),
    ("ms-marco-MiniLM-L6-v2", STORE / "cross-encoder/ms-marco-MiniLM-L6-v2", "rerank"),
    ("bge-reranker-v2-m3", STORE / "BAAI/bge-reranker-v2-m3", "rerank"),
    ("gte-reranker-modernbert-base", STORE / "Alibaba-NLP/gte-reranker-modernbert-base", "rerank"),
    ("Qwen3-Reranker-0.6B-4bit", STORE / "mlx-community/Qwen3-Reranker-0.6B-4bit", "rerank"),
    ("Qwen3-VL-Reranker-2B", STORE / "Qwen/Qwen3-VL-Reranker-2B", "rerank_vl"),
]

CSV_HEADER = ["model", "model_path", "kind", "input_kind", "batch", "inputs", "prompt_tokens", "repeats",
              "p50_ms", "mean_ms", "min_ms", "inputs_per_s", "tokens_per_s", "load_ms", "date", "hardware",
              "mlxcel_version", "build_type", "mlxcel_commit", "mlx_commit", "notes"]


def detect_hardware():
    """Host string for the CSV `hardware` column, matching bench_decode.sh.

    Was hardcoded to the GB10 box, which silently mislabels every CSV produced
    on another host and breaks cross-hardware comparison after the fact.
    """
    import platform
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout.strip()
        gb = f"{round(int(mem) / 1024**3)}GB" if mem.isdigit() else ""
        return f"{chip.replace(' ', '_')}_{gb}".strip("_") or "unknown"
    name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if name:
        first = name.splitlines()[0]
        gpu, _, total = first.partition(",")
        mib = "".join(ch for ch in total if ch.isdigit())
        gb = f"{round(int(mib) / 1024)}GB" if mib else ""
        return f"{gpu.strip().replace(' ', '_')}_{gb}".strip("_")
    return "unknown"


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000.0, data


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return (time.time() - t0) * 1000.0, json.loads(r.read())
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("server did not become ready")


def _image_data_uri():
    """The fixture as a base64 `data:` URI.

    A `file://` URL also works, but only when it is relative to the server's
    --media-path root: an absolute path is concatenated onto the root rather
    than replacing it (llama-server b10621 compatibility, pinned by
    `an_absolute_looking_path_is_concatenated_not_joined`), so the absolute
    form this harness used to send was probed under the root and reported as
    "file does not exist or cannot be opened". A data URI is used instead
    because it needs no server flag, which keeps the ladder reproducible on a
    host where --media-path was never set. The fixture is 679 bytes, so
    inlining it costs nothing.
    """
    import base64
    return "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode()


def image_item():
    return {"type": "image_url", "image_url": {"url": _image_data_uri()}}


def rerank_doc_image():
    return {"image_url": _image_data_uri()}


def embed_body(kind, input_kind, batch):
    if input_kind == "short":
        return {"input": [SHORT] * batch}
    if input_kind == "long":
        return {"input": [LONG] * batch}
    if input_kind == "image":
        return {"input": [image_item()] * batch}
    raise ValueError(input_kind)


def rerank_body(input_kind, batch):
    if input_kind == "docs_short":
        docs = [SHORT + f" Variant {i}." for i in range(batch)]
    elif input_kind == "docs_long":
        docs = [LONG + f" Variant {i}." for i in range(batch)]
    elif input_kind == "docs_image":
        docs = [rerank_doc_image()] * batch
    else:
        raise ValueError(input_kind)
    return {"query": "What does a reranker do in a retrieval pipeline?", "documents": docs}


def cells_for(kind):
    if kind in ("text", "multivector"):
        return [("short", 1), ("short", 8), ("short", 32), ("long", 1), ("long", 8), ("long", 32)]
    if kind == "vl":
        return [("short", 1), ("short", 8), ("long", 8), ("image", 1), ("image", 4)]
    if kind == "rerank":
        return [("docs_short", 8), ("docs_short", 32), ("docs_long", 8)]
    if kind == "rerank_vl":
        return [("docs_short", 8), ("docs_image", 4)]
    raise ValueError(kind)


def run_model(name, path, kind, args, meta, writer, fh):
    if not path.exists():
        print(f"[skip] {name}: {path} missing", flush=True)
        return
    port = args.port
    log = open(args.logdir / f"server-{name}.log", "w")
    if kind.startswith("rerank"):
        # Rerank-only shape: `-m` is mandatory, and passing the same directory to both
        # flags loads the weights once (the chat worker stays unloaded).
        cmd = [args.bin, "-m", str(path), "--reranker-model", str(path), "--host", "127.0.0.1", "--port", str(port)]
    else:
        cmd = [args.bin, "-m", str(path), "--host", "127.0.0.1", "--port", str(port)]
    # Kept so a relative `file://` URL would resolve if the image request shape
    # is ever switched back from the data URI in `_image_data_uri`. Since #1481
    # the server refuses `file://` media unless a root is allow-listed.
    cmd += ["--media-path", str(IMAGE.parent)]
    print(f"[start] {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=args.cwd)
    try:
        load_ms, models = wait_ready(port, proc)
        served = models.get("data", [{}])[0].get("id", name)
        endpoint = "rerank" if kind.startswith("rerank") else "embeddings"
        url = f"http://127.0.0.1:{port}/v1/{endpoint}"
        # warmup
        warm = rerank_body("docs_short", 2) if endpoint == "rerank" else embed_body(kind, "short", 1)
        post(url, warm)
        for input_kind, batch in cells_for(kind):
            body = rerank_body(input_kind, batch) if endpoint == "rerank" else embed_body(kind, input_kind, batch)
            times, toks = [], []
            note = ""
            for _ in range(args.repeats):
                try:
                    ms, data = post(url, body)
                except urllib.error.HTTPError as e:
                    note = f"HTTP {e.code}: {e.read()[:120].decode(errors='replace')}"
                    break
                times.append(ms)
                toks.append(int(data.get("usage", {}).get("prompt_tokens", 0)))
            if not times:
                writer.writerow([name, str(path), kind, input_kind, batch, batch, "", 0, "", "", "", "", "", f"{load_ms:.1f}",
                                 meta["date"], meta["hardware"], meta["mlxcel_version"], meta["build_type"], meta["commit"], meta["mlx_commit"], note])
                fh.flush()
                continue
            p50 = statistics.median(times)
            mean = statistics.fmean(times)
            mn = min(times)
            ptoks = toks[0]
            writer.writerow([name, str(path), kind, input_kind, batch, batch, ptoks, len(times), f"{p50:.2f}", f"{mean:.2f}",
                             f"{mn:.2f}", f"{batch / (p50 / 1000.0):.2f}", f"{ptoks / (p50 / 1000.0):.1f}" if ptoks else "",
                             f"{load_ms:.1f}", meta["date"], meta["hardware"], meta["mlxcel_version"], meta["build_type"],
                             meta["commit"], note])
            fh.flush()
            print(f"  {name} {input_kind} b={batch}: p50={p50:.1f}ms tokens={ptoks}", flush=True)
    except Exception as e:
        writer.writerow([name, str(path), kind, "", "", "", "", 0, "", "", "", "", "", "", meta["date"], meta["hardware"],
                         meta["mlxcel_version"], meta["build_type"], meta["commit"], f"ERROR: {e}"])
        fh.flush()
        print(f"[error] {name}: {e}", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--build-type", default="release")
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()
    args.logdir = Path(args.logdir or Path(args.out).parent / "perf_logs").resolve()
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.bin = str(Path(args.bin).resolve())
    commit = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=args.cwd).stdout.strip()
    # An MLX pin bump changes kernels without moving the mlxcel version or commit.
    mlx_commit = subprocess.run(["scripts/ci/mlx_pinned_commit.sh"], capture_output=True, text=True, cwd=args.cwd).stdout.strip()[:8] or "unknown"
    version = subprocess.run([args.bin.replace("mlxcel-server", "mlxcel"), "--version"], capture_output=True, text=True).stdout.strip().split()[-1]
    meta = {"date": time.strftime("%Y-%m-%d"), "hardware": detect_hardware(), "mlxcel_version": version,
            "build_type": args.build_type, "commit": commit,
            "mlx_commit": mlx_commit}
    out = Path(args.out)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CSV_HEADER)
        for name, path, kind in MODELS:
            if args.only and name not in args.only:
                continue
            run_model(name, path, kind, args, meta, w, fh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
