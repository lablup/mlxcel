#!/usr/bin/env python3
"""Checkpoint fingerprint for cross-host identity checks.

A directory name is not an identity: two hosts can hold different checkpoints
under the same basename, or the same checkpoint under different names. This
prints, per model, the sha256 of config.json plus the architecture and the
config dimensions that a re-download would preserve and a different model
would not.
"""
import csv, hashlib, json, os, sys
from pathlib import Path

def rows(csv_path):
    for r in list(csv.reader(open(csv_path)))[1:]:
        if r: yield r

def fp(model_dir):
    p = Path(model_dir)
    cfg = p / "config.json"
    if not cfg.exists(): return None
    raw = open(cfg, "rb").read()
    h = hashlib.sha256(raw).hexdigest()[:12]
    try: d = json.loads(raw)
    except Exception: return {"cfg_sha": h}
    t = d.get("text_config") or d
    q = d.get("quantization") or {}
    # Sum the shards the checkpoint actually loads, not every file in the
    # directory. Two directories on this host disagree with a plain glob:
    # `gemma-4-12b-it-4bit` keeps a superseded 2-shard export beside the
    # 3-shard set its index names, which inflated the total by 6.3 GB, and 12
    # others carry an index left over from a pre-quantization export that
    # names shards no longer present. mlxcel globs, so both load; a bytes
    # figure taken from the glob is a property of the directory rather than of
    # the model, and two hosts holding the same model then disagree.
    on_disk = {f.name: os.stat(f).st_size for f in p.glob("*.safetensors") if f.is_file()}
    named = set()
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        try: named = set(json.load(open(idx)).get("weight_map", {}).values())
        except Exception: named = set()
    if named and named <= set(on_disk):
        shards = sorted((n, on_disk[n]) for n in named)
        source = "index"
    else:
        shards = sorted(on_disk.items())
        source = "glob-stale-index" if named else "glob"
    total = sum(s for _, s in shards)
    return {
        "cfg_sha": h,
        "arch": (d.get("architectures") or [d.get("model_type", "?")])[0],
        "hidden": t.get("hidden_size"),
        "layers": t.get("num_hidden_layers"),
        "vocab": t.get("vocab_size"),
        "bits": q.get("bits"),
        "group": q.get("group_size"),
        "shards": len(shards),
        "bytes": total,
        "shard_source": source,
    }

if __name__ == "__main__":
    out = {}
    for csv_path in sys.argv[1:]:
        for r in rows(csv_path):
            name, path = r[0], r[1]
            if name in out: continue
            f = fp(path)
            if f: out[name] = f
    json.dump(out, sys.stdout, sort_keys=True, indent=0)
