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
    shards = sorted((f.name, os.stat(f).st_size) for f in p.glob("*.safetensors") if f.is_file())
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
