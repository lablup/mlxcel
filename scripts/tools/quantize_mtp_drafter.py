"""Quantize a qwen3_5_mtp drafter checkpoint to match its target's scheme.

Out-of-band tooling (issue #1185 Phase 3). The drafter borrows embed_tokens and
the LM head from the target, so only its own 2D projections are converted; the
1D norms stay bf16, matching what every shipped mlx-community checkpoint does.

The scheme is taken from the target rather than chosen here, because the
mlxcel loader passes the *target's* group_size/bits to UnifiedLinear when it
builds the drafter. A drafter quantized any other way would be read with the
wrong parameters.
"""
import json, shutil, sys
from pathlib import Path
import mlx.core as mx

src, tgt, dst = (Path(p) for p in sys.argv[1:4])
scheme = json.loads((tgt / "config.json").read_text())["quantization"]
group_size, bits, mode = scheme["group_size"], scheme["bits"], scheme.get("mode", "affine")
print(f"target scheme: group_size={group_size} bits={bits} mode={mode}")

w = mx.load(str(src / "model.safetensors"))
out, converted, kept = {}, 0, 0
before = sum(v.nbytes for v in w.values())
for key in sorted(w):
    v = w[key]
    if v.ndim == 2 and key.endswith(".weight"):
        prefix = key[: -len(".weight")]
        if v.shape[-1] % group_size:
            raise SystemExit(f"{key}: last dim {v.shape[-1]} not divisible by {group_size}")
        packed, scales, biases = mx.quantize(v, group_size=group_size, bits=bits)
        out[key] = packed
        out[f"{prefix}.scales"] = scales
        out[f"{prefix}.biases"] = biases
        converted += 1
        print(f"  quantized {key:48s} {tuple(v.shape)} -> {tuple(packed.shape)}")
    else:
        out[key] = v
        kept += 1

dst.mkdir(parents=True, exist_ok=True)
mx.eval(list(out.values()))
mx.save_safetensors(str(dst / "model.safetensors"), out)
cfg = json.loads((src / "config.json").read_text())
cfg["quantization"] = {"group_size": group_size, "bits": bits, "mode": mode}
(dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md"):
    if (src / extra).exists():
        shutil.copy2(src / extra, dst / extra)

after = sum(v.nbytes for v in out.values())
print(f"\nconverted {converted} projections, kept {kept} tensors as-is")
print(f"{before/2**20:.1f} MiB -> {after/2**20:.1f} MiB ({after/before:.3f}x)")
