#!/usr/bin/env python3
"""A/B memory-footprint bench for mlxcel server versions.

Starts one fresh server process per (config, scenario), drives a fixed
HTTP workload, and samples the server's OS-level memory (psutil RSS at
100 ms + /usr/bin/footprint phys-footprint at 2.5 s). Reports per-phase
peak and settled values so versions can be compared on identical load.

On Apple Silicon, MLX Metal buffers do not show up in RSS at all (a
server holding gigabytes of KV cache reports a flat ~0.5 GiB RSS), so
phys-footprint is the primary metric and RSS is kept only as a
CPU-heap-side cross-check.

Scenarios:
  idle                weights-only anchor, no requests for 20 s
  shared_prefix_burst one warmup prefill of a ~4k-token shared system
                      prompt, then 8 concurrent requests reusing it
  seqshare            8 sequential requests on the same shared prefix
                      (each completes before the next, so prompt-cache
                      donate/adopt chains can engage)
  multiturn           one conversation, 8 accumulating turns
  churn               32 distinct sequential short requests

Prefer a non-thinking model (e.g. llama-3.2-1b): reasoning models split
<think> content out of the API reply, so the echoed conversation no
longer re-renders byte-identically and strict-containment prompt-cache
matching can never hit on multiturn.

Usage:
  python3 bench_memory_footprint.py \
      --config new:/path/to/new/mlxcel \
      --config new-dense:/path/to/new/mlxcel:--decode-storage-backend=dense \
      --config old:/path/to/old/mlxcel \
      --model /path/to/models/qwen3-0.6b-4bit \
      --out results.json
"""

import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import psutil

FOOTPRINT_RE = re.compile(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)")
FOOTPRINT_UNIT = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def read_footprint(pid: int):
    try:
        out = subprocess.run(
            ["/usr/bin/footprint", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    m = FOOTPRINT_RE.search(out)
    if not m:
        return None
    return int(float(m.group(1)) * FOOTPRINT_UNIT[m.group(2)])


class MemSampler:
    """Samples RSS every 100 ms and phys footprint every 2.5 s."""

    def __init__(self, pid: int):
        self.proc = psutil.Process(pid)
        self.pid = pid
        self.samples = []          # (t, rss_bytes)
        self.fp_samples = []       # (t, footprint_bytes)
        self.marks = []            # (t, name)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rss_thread = threading.Thread(target=self._rss_loop, daemon=True)
        self._fp_thread = threading.Thread(target=self._fp_loop, daemon=True)

    def start(self):
        self._rss_thread.start()
        self._fp_thread.start()

    def _rss_loop(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
            except psutil.Error:
                break
            with self._lock:
                self.samples.append((time.monotonic(), rss))
            self._stop.wait(0.1)

    def _fp_loop(self):
        while not self._stop.is_set():
            fp = read_footprint(self.pid)
            if fp is not None:
                with self._lock:
                    self.fp_samples.append((time.monotonic(), fp))
            self._stop.wait(2.5)

    def mark(self, name: str):
        with self._lock:
            self.marks.append((time.monotonic(), name))

    def stop(self):
        self._stop.set()
        self._rss_thread.join(timeout=2)
        self._fp_thread.join(timeout=12)

    def phase_stats(self):
        """Per-phase peak/last for both metrics."""
        out = []
        bounds = self.marks + [(time.monotonic(), "_end")]
        for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
            rss = [v for t, v in self.samples if t0 <= t < t1]
            fp = [v for t, v in self.fp_samples if t0 <= t < t1]
            out.append({
                "phase": name,
                "duration_s": round(t1 - t0, 2),
                "rss_peak": max(rss) if rss else None,
                "rss_last": rss[-1] if rss else None,
                "footprint_peak": max(fp) if fp else None,
                "footprint_last": fp[-1] if fp else None,
            })
        return out


class ServerClient:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.model_id = None
        self.lock = threading.Lock()
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_cached_tokens = 0
        self.cached_tokens_sum = 0

    def _get(self, path, timeout=10):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def wait_ready(self, proc, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={proc.returncode}")
            try:
                models = self._get("/v1/models", timeout=3)
                self.model_id = models["data"][0]["id"]
                return
            except Exception:
                time.sleep(0.5)
        raise TimeoutError("server did not become ready")

    def chat(self, messages, max_tokens, timeout=600):
        body = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        usage = resp.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        with self.lock:
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            self.last_prompt_tokens = usage.get("prompt_tokens", 0)
            self.last_cached_tokens = details.get("cached_tokens", 0)
            self.cached_tokens_sum += self.last_cached_tokens
        return resp["choices"][0]["message"]["content"]

    def cache_stats(self):
        try:
            return self._get("/v1/cache/stats", timeout=5)
        except Exception:
            return None


# --- deterministic prompt builders -----------------------------------------

def long_text(n_chars: int, salt: str = "") -> str:
    """Deterministic filler prose, ~4 chars/token on Qwen tokenizers."""
    parts = []
    i = 0
    while sum(len(p) + 1 for p in parts) < n_chars:
        parts.append(
            f"Fact {salt}{i}: the archive shelf number {i * 7 % 991} holds a "
            f"ledger describing harvest season {1900 + i % 120} in district "
            f"{i % 47}, including rainfall, grain prices, and road repairs."
        )
        i += 1
    return " ".join(parts)[:n_chars]


SHARED_PREFIX = (
    "You are an archival research assistant. Use only the facts below. "
    + long_text(16000, salt="P")
)


# --- scenarios ---------------------------------------------------------------

def scen_idle(client, sampler):
    sampler.mark("idle")
    time.sleep(20)


def scen_shared_prefix_burst(client, sampler, n_parallel=8):
    sampler.mark("warmup_prefill")
    client.chat(
        [{"role": "system", "content": SHARED_PREFIX},
         {"role": "user", "content": "Reply with the single word OK."}],
        max_tokens=8,
    )
    prompt_tokens = client.last_prompt_tokens
    time.sleep(3)
    sampler.mark("burst")

    def one(i):
        return client.chat(
            [{"role": "system", "content": SHARED_PREFIX},
             {"role": "user",
              "content": f"In one sentence, summarize what fact P{i} says."}],
            max_tokens=128,
        )

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        list(ex.map(one, range(n_parallel)))
    sampler.mark("settle")
    time.sleep(8)
    return {"shared_prefix_tokens": prompt_tokens, "parallel": n_parallel}


def scen_distinct_burst(client, sampler, n_parallel=8):
    """8 concurrent requests with per-request UNIQUE long system prompts.

    Nothing is shareable, so every prefill extends the paged pool: this is
    the slab-growth stress the shared scenarios no longer exercise once
    prefix sharing engages (issue #235)."""
    sampler.mark("burst")

    def one(i):
        sysmsg = ("You are an archival research assistant. "
                  + long_text(16000, salt=f"D{i}"))
        return client.chat(
            [{"role": "system", "content": sysmsg},
             {"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=32,
        )

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        list(ex.map(one, range(n_parallel)))
    sampler.mark("settle")
    time.sleep(8)
    return {"parallel": n_parallel,
            "prompt_tokens_each": client.last_prompt_tokens}


def scen_seqshare(client, sampler, n=8):
    sampler.mark("seqshare")
    cached = []
    for i in range(n):
        client.chat(
            [{"role": "system", "content": SHARED_PREFIX},
             {"role": "user",
              "content": f"In one sentence, summarize what fact P{i} says."}],
            max_tokens=128,
        )
        cached.append(client.last_cached_tokens)
    sampler.mark("settle")
    time.sleep(8)
    return {"requests": n, "cached_tokens_per_request": cached,
            "shared_prefix_tokens": client.last_prompt_tokens}


def scen_multiturn(client, sampler, turns=8):
    sampler.mark("multiturn")
    messages = [{"role": "system",
                 "content": "You are a meticulous archival assistant."}]
    for t in range(turns):
        messages.append({"role": "user", "content": long_text(1000, salt=f"T{t}")
                         + f" Question {t}: acknowledge in one sentence."})
        reply = client.chat(messages, max_tokens=128)
        messages.append({"role": "assistant", "content": reply})
    sampler.mark("settle")
    time.sleep(8)
    return {"turns": turns, "final_prompt_tokens": client.last_prompt_tokens}


def scen_churn(client, sampler, n=32):
    sampler.mark("churn")
    quarter_marks = {}
    for i in range(n):
        client.chat(
            [{"role": "user", "content": long_text(2000, salt=f"C{i}")
              + " Answer with one short sentence."}],
            max_tokens=32,
        )
        if (i + 1) % (n // 4) == 0:
            try:
                rss = psutil.Process(sampler.pid).memory_info().rss
                quarter_marks[f"after_{i + 1}"] = rss
            except psutil.Error:
                pass
    sampler.mark("settle")
    time.sleep(8)
    return {"requests": n, "rss_checkpoints": quarter_marks}


SCENARIOS = {
    "idle": scen_idle,
    "shared_prefix_burst": scen_shared_prefix_burst,
    "distinct_burst": scen_distinct_burst,
    "seqshare": scen_seqshare,
    "multiturn": scen_multiturn,
    "churn": scen_churn,
}


# --- runner ------------------------------------------------------------------

def run_one(binary, extra_flags, model, scenario_name, log_dir):
    port = free_port()
    cmd = [binary, "serve", "-m", model,
           "--host", "127.0.0.1", "--port", str(port),
           "--max-batch-size", "8",
           "--prompt-cache-enabled=true"] + extra_flags
    log_path = os.path.join(log_dir, f"{scenario_name}.server.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    client = ServerClient(port)
    sampler = None
    try:
        client.wait_ready(proc)
        sampler = MemSampler(proc.pid)
        sampler.start()
        sampler.mark("post_load_settle")
        time.sleep(5)
        t0 = time.monotonic()
        extra = SCENARIOS[scenario_name](client, sampler) or {}
        wall = time.monotonic() - t0
        time.sleep(0.5)
        phases = sampler.phase_stats()
        return {
            "scenario": scenario_name,
            "cmd": " ".join(cmd),
            "phases": phases,
            "wall_s": round(wall, 2),
            "completion_tokens": client.total_completion_tokens,
            "cached_tokens_sum": client.cached_tokens_sum,
            "cache_stats_end": client.cache_stats(),
            **extra,
        }
    finally:
        if sampler:
            sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def parse_config(spec):
    # label:binary[:flag1 flag2 ...]  flags use = form to avoid colon issues
    parts = spec.split(":", 2)
    label, binary = parts[0], parts[1]
    flags = parts[2].split() if len(parts) > 2 else []
    return label, binary, flags


def fmt_mib(v):
    return "-" if v is None else f"{v / 1024**2:8.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="label:binary[:extra flags, = form]")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenarios",
                    default="idle,shared_prefix_burst,seqshare,multiturn,churn")
    ap.add_argument("--out", default="memfp_results.json")
    ap.add_argument("--log-dir", default="memfp_logs")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results = []
    for spec in args.config:
        label, binary, flags = parse_config(spec)
        cfg_log = os.path.join(args.log_dir, label)
        os.makedirs(cfg_log, exist_ok=True)
        for scen in scenarios:
            print(f"=== {label} / {scen} ===", flush=True)
            r = run_one(binary, flags, args.model, scen, cfg_log)
            r["config"] = label
            results.append(r)
            print(f"  wall {r['wall_s']}s, completion {r['completion_tokens']} tok, "
                  f"cached {r['cached_tokens_sum']} tok", flush=True)
            for p in r["phases"]:
                print(f"  {p['phase']:<18} {p['duration_s']:7.1f}s  "
                      f"rss_peak {fmt_mib(p['rss_peak'])} MiB  "
                      f"rss_last {fmt_mib(p['rss_last'])} MiB  "
                      f"fp_peak {fmt_mib(p['footprint_peak'])} MiB",
                      flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
