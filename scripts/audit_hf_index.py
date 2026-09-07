#!/usr/bin/env python3
"""Audit `model.safetensors.index.json` against the shards a repo actually ships.

Some mlx-community VLM conversions publish the *source* model's index verbatim
next to the quantized weights, so every shard the index names is absent and the
declared `total_size` is the full-precision size. mlxcel tolerates this by
falling back to a directory glob (see `docs/adr/0006-safetensors-shard-discovery-globs-past-a-stale-index.md`);
this script is how that claim is checked and re-checked.

Usage:

    # Audit repositories by id.
    python3 scripts/audit_hf_index.py mlx-community/Qwen3-VL-32B-Instruct-4bit

    # Audit every repo named in the READMEs of local checkpoints.
    python3 scripts/audit_hf_index.py --local models/qwen3-vl-32b-4bit models/gemma-3-4b-it-4bit

    # Audit a slice of the org, most-downloaded first.
    python3 scripts/audit_hf_index.py --search image-text-to-text --limit 60

Exit status is 1 when any audited repository has a stale index, so this can gate
a check.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co"
UA = {"User-Agent": "mlxcel-index-audit"}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def gb(n):
    return f"{n / 1e9:.1f} GB"


def audit(repo):
    """Return (repo, verdict, detail) for one repository."""
    try:
        tree = json.loads(fetch(f"{API}/api/models/{urllib.parse.quote(repo)}/tree/main?recursive=1"))
    except Exception as exc:
        return repo, "ERROR", f"tree: {exc}"
    shards = {
        entry["path"]: (entry.get("lfs") or {}).get("size", entry.get("size", 0))
        for entry in tree
        if entry.get("type") == "file" and entry["path"].endswith(".safetensors")
    }
    try:
        index = json.loads(fetch(f"{API}/{repo}/raw/main/model.safetensors.index.json"))
    except Exception:
        return repo, "NO-INDEX", f"{len(shards)} shards, {gb(sum(shards.values()))}"
    named = sorted(set((index.get("weight_map") or {}).values()))
    absent = [name for name in named if name not in shards]
    declared = (index.get("metadata") or {}).get("total_size") or 0
    detail = (
        f"repo {len(shards)} shards {gb(sum(shards.values()))}, "
        f"index {len(named)} shards {gb(declared)}"
    )
    if not absent:
        return repo, "CONSISTENT", detail
    return repo, "STALE", f"{detail}, {len(absent)} of {len(named)} declared shards absent"


def repo_from_local(model_dir):
    """Recover the source repo id from a downloaded checkpoint's README."""
    readme = Path(model_dir) / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\b([A-Za-z0-9][\w.-]*/[\w.-]+)\b", text.split("---")[-1])
    return match.group(1) if match else None


def search(pipeline_tag, limit):
    url = (
        f"{API}/api/models?author=mlx-community&pipeline_tag={urllib.parse.quote(pipeline_tag)}"
        f"&sort=downloads&direction=-1&limit={limit}"
    )
    return [m["id"] for m in json.loads(fetch(url))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="*", help="HuggingFace repository ids")
    parser.add_argument("--local", nargs="+", default=[], metavar="DIR",
                        help="local checkpoint directories; the repo id is read from each README")
    parser.add_argument("--search", metavar="PIPELINE_TAG",
                        help="audit mlx-community repos with this pipeline tag, most-downloaded first")
    parser.add_argument("--limit", type=int, default=60, help="how many repos --search returns (default 60)")
    args = parser.parse_args()

    repos = list(args.repos)
    for model_dir in args.local:
        repo = repo_from_local(model_dir)
        if repo:
            repos.append(repo)
        else:
            print(f"{model_dir}: no repo id in README, skipped", file=sys.stderr)
    if args.search:
        repos.extend(search(args.search, args.limit))
    repos = list(dict.fromkeys(repos))
    if not repos:
        parser.error("nothing to audit: pass repo ids, --local, or --search")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(audit, repos):
            results.append(result)

    tally = {}
    for _, verdict, _ in results:
        tally[verdict] = tally.get(verdict, 0) + 1
    for repo, verdict, detail in sorted(results):
        if verdict != "CONSISTENT":
            print(f"{verdict:<10} {repo}\n           {detail}")
    print(f"\n{len(results)} audited: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    return 1 if tally.get("STALE") else 0


if __name__ == "__main__":
    sys.exit(main())
