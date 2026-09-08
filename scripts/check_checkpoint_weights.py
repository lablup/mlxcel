import json, struct, sys, pathlib
def check(p: pathlib.Path):
    size = p.stat().st_size
    if size < 8: return "file shorter than the 8-byte header length"
    with p.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n <= 0 or n + 8 > size: return f"header length {n} does not fit in {size} bytes"
        try: hdr = json.loads(f.read(n))
        except Exception as e: return f"header is not valid JSON: {e}"
    end = 0
    for k, v in hdr.items():
        if k == "__metadata__": continue
        try: o = v["data_offsets"]
        except Exception: return f"tensor {k} has no data_offsets"
        end = max(end, o[1])
    want = 8 + n + end
    if want != size: return f"declared {want} bytes, file is {size} ({size-want:+d})"
    return None
roots = [pathlib.Path(x) for x in sys.argv[1:]]
bad = 0; nfile = 0; ndir = 0
for root in roots:
    if not root.exists(): continue
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        shards = sorted(d.glob('*.safetensors'))
        if not shards:
            # A directory with a config and no weights is an interrupted
            # download, and skipping it here is how two of them survived a
            # whole audit: the check only looked at directories that already
            # had shards, so the ones with none were never examined.
            other = list(d.glob('*.npz')) + list(d.glob('*.bin')) + list(d.glob('*.gguf'))
            if not other and (d/'config.json').exists():
                bad += 1
                print(f"  BAD  {d.name}: config.json present, no weight file")
            continue
        ndir += 1
        for s in shards:
            nfile += 1
            err = check(s)
            if err:
                bad += 1
                print(f"  BAD  {d.name}/{s.name}: {err}")
print(f"checked {nfile} shards across {ndir} directories; {bad} bad")
