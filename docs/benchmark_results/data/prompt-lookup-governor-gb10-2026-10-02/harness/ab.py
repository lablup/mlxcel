#!/usr/bin/env python3
"""Interleaved A/B for issue #2091: plain decoding against prompt lookup arms.

Run from the repository root. Each repetition runs every arm back to back
(round-robin), so slow drift in machine load hits all arms alike. The rate is
the CLI's whole-call `[Generated N tokens in Xs = Y tok/s]`, prefill
included, after `mlxcel generate`'s own in-process warmup. Writes one JSON
line per case to stdout, the first repetition's reply text per arm for the
greedy parity check, and `results.json`.

Environment:
  MLXCEL_BIN          binary under test (default target/release/mlxcel)
  MLXCEL_MODELS_ROOT  checkpoint directory (default models/mlx)
  CASES               model:prompt:max_tokens,... (prompts/ holds the texts)
  ARMS                JSON {arm: [extra args] or null for plain}
  REPS                repetitions per arm (default 3)
  BENCH_OUT           output directory
"""
import json, os, re, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = os.environ.get("MLXCEL_BIN", "target/release/mlxcel")
MODELS_ROOT = Path(os.environ.get("MLXCEL_MODELS_ROOT", "models/mlx"))
REPS = int(os.environ.get("REPS", "3"))
OUT = Path(os.environ.get("BENCH_OUT", "out-ab"))
GEN_RE = re.compile(r"\[Generated (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s\]")
PL_RE = re.compile(r"\[Prompt lookup\] (.*)")

# name -> extra args (None = plain decoding)
ARMS = json.loads(os.environ.get("ARMS", json.dumps({
    "plain": None,
    "graded": ["--prompt-lookup", "--prompt-lookup-policy", "graded"],
    "gated": ["--prompt-lookup", "--prompt-lookup-policy", "gated"],
})))
CASES = [c.split(":") for c in os.environ.get("CASES", "qwen3-1.7b-4bit:write:400").split(",")]


def prompt_text(model, name):
    text = (HERE / "prompts" / f"{name}.txt").read_text().strip()
    return text + " /no_think" if model.startswith("qwen3") else text


def run(model, prompt, n, extra):
    cmd = [BIN, "generate", "-m", str(MODELS_ROOT / model), "-p", prompt_text(model, prompt), "-n", n, "--temp", "0"] + (extra or [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd}\n{p.stderr[-1500:]}")
    m = GEN_RE.search(p.stdout)
    st = p.stdout.find("Generating...\n")
    body = p.stdout[st + 14 if st >= 0 else 0 : m.start()]
    pl = PL_RE.search(p.stderr)
    stats = dict(kv.split("=") for kv in pl.group(1).split()) if pl else {}
    return float(m.group(3)), int(m.group(1)), body, stats


def main():
    OUT.mkdir(exist_ok=True)
    results = []
    for model, prompt, n in CASES:
        rates = {a: [] for a in ARMS}
        info = {}
        for rep in range(REPS):
            for arm, extra in ARMS.items():
                r, toks, body, stats = run(model, prompt, n, extra)
                rates[arm].append(r)
                if rep == 0:
                    info[arm] = {"tokens": toks, "stats": stats}
                    (OUT / f"{model}.{prompt}.{arm}.txt").write_text(body)
        row = {"model": model, "prompt": prompt, "n": n}
        base = statistics.median(rates["plain"]) if "plain" in rates else None
        for arm in ARMS:
            med = statistics.median(rates[arm])
            row[arm] = {"median": med, "rates": rates[arm], **info[arm]}
            if base and arm != "plain":
                row[arm]["x"] = med / base
                a = (OUT / f"{model}.{prompt}.plain.txt").read_text()
                b = (OUT / f"{model}.{prompt}.{arm}.txt").read_text()
                row[arm]["parity"] = "identical" if a == b else f"diverges@{next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))}"
        results.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
