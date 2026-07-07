#!/usr/bin/env python3
# Requantize a checkpoint's 8-bit affine module overrides down to 4-bit,
# producing quantization variants for quality/throughput frontier studies
# (issue #683). Validated against gemma-4-12b-it-4bit, whose 144 8-bit MLP
# modules (gate/up/down x 48 layers) account for ~9 GB of the ~10.3 GB
# streamed per decode token on GB10 and cap decode at ~14 tok/s; the uniform
# 4-bit variant measured 1.59 to 1.62x decode with 3.6 GB lower peak.
#
# Pure numpy on raw safetensors (8-byte LE header length + JSON header +
# buffer); bf16 handled as uint16 with manual f32 conversion. MLX affine
# layout: quantized weights are u32-packed LOW-FIRST (8-bit: 4 values/u32 ==
# LE byte order; 4-bit: 8 nibbles/u32, nibble k at bits [4k, 4k+4)); dequant
# is q * scale + bias per group along the input axis. The new 4-bit q is
# computed against the bf16-ROUNDED scale/bias so the stored triple is
# self-consistent regardless of how MLX's own quantizer would have chosen
# them.
#
# Usage:
#   python3 scripts/requantize_mlp.py --src MODEL_DIR --dst OUT_DIR \
#       [--target-group-size 64] [--match REGEX]
#
# --match limits requantization to module names matching REGEX (e.g.
# 'mlp\.(gate|up)_proj$' keeps down_proj at 8-bit). Modules whose resulting
# (bits, group_size) equal the config's global values lose their override;
# others get an explicit override entry.

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

TARGET_BITS = 4


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f: np.ndarray) -> np.ndarray:
    u = f.astype(np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)  # round-to-nearest-even
    return (rounded >> 16).astype(np.uint16)


def read_safetensors(path: Path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        buf = np.fromfile(f, dtype=np.uint8)
    meta = header.pop("__metadata__", None)
    return header, buf, meta


def write_safetensors(path: Path, tensors: dict, meta):
    header = {}
    if meta is not None:
        header["__metadata__"] = meta
    offset = 0
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dt,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)  # align data start
    with open(path, "wb") as f:
        f.write(len(hj).to_bytes(8, "little"))
        f.write(hj)
        for _, (_, _, raw) in tensors.items():
            f.write(raw)
    return offset


def requant_module(w_u32, w_shape, sc_u16, sc_shape, bi_u16, src_gs, tgt_gs):
    out, packed = w_shape
    inp = packed * 4  # source is 8-bit: 4 values per u32
    g_src = inp // src_gs
    assert sc_shape == [out, g_src], (sc_shape, [out, g_src])

    q8 = w_u32.view(np.uint8).reshape(out, g_src, src_gs).astype(np.float32)
    sc = bf16_to_f32(sc_u16).reshape(out, g_src, 1)
    bi = bf16_to_f32(bi_u16).reshape(out, g_src, 1)
    w = (q8 * sc + bi).reshape(out, inp)

    g_tgt = inp // tgt_gs
    w = w.reshape(out, g_tgt, tgt_gs)
    mn = w.min(axis=2)
    mx = w.max(axis=2)
    scale = ((mx - mn) / 15.0).astype(np.float32)
    scale = np.where(scale <= 0, np.float32(1.0), scale)
    sc4_u16 = f32_to_bf16(scale)
    bi4_u16 = f32_to_bf16(mn)
    sc4 = bf16_to_f32(sc4_u16).reshape(out, g_tgt, 1)
    sc4 = np.where(sc4 == 0, np.float32(1e-6), sc4)
    bi4 = bf16_to_f32(bi4_u16).reshape(out, g_tgt, 1)
    q4 = np.clip(np.rint((w - bi4) / sc4), 0, 15).astype(np.uint32)

    err = float(np.abs((q4 * sc4 + bi4) - w).max())

    q4 = q4.reshape(out, inp // 8, 8)
    packed4 = np.zeros((out, inp // 8), dtype=np.uint32)
    for k in range(8):
        packed4 |= q4[:, :, k] << np.uint32(4 * k)

    return packed4, [out, inp // 8], sc4_u16, bi4_u16, [out, g_tgt], err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--target-group-size", type=int, default=64, choices=(32, 64))
    ap.add_argument("--match", default=None, help="regex limiting which 8-bit modules are requantized")
    args = ap.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    q = cfg["quantization"]
    global_bits = q.get("bits")
    global_gs = q.get("group_size")
    pat = re.compile(args.match) if args.match else None
    targets = {
        k: v
        for k, v in q.items()
        if isinstance(v, dict) and v.get("bits") == 8 and (pat is None or pat.search(k))
    }
    if not targets:
        print("no matching 8-bit module overrides found; nothing to do")
        return 1
    src_gs = {v.get("group_size", global_gs) for v in targets.values()}
    assert src_gs == {64}, f"only group_size-64 8-bit sources are supported, got {src_gs}"
    tgs = args.target_group_size
    print(f"{len(targets)} modules -> {TARGET_BITS}-bit group_size {tgs}")

    idx = json.loads((args.src / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))

    args.dst.mkdir(exist_ok=True)
    new_total = 0
    max_err = 0.0
    n_done = 0

    for shard in shards:
        header, buf, meta = read_safetensors(args.src / shard)
        out_tensors = {}
        for name in header:
            info = header[name]
            s, e = info["data_offsets"]
            raw = buf[s:e]
            prefix, _, suffix = name.rpartition(".")
            if prefix in targets and suffix in ("weight", "scales", "biases"):
                out_tensors[name] = ("PENDING", prefix, suffix, info, raw)
            else:
                out_tensors[name] = (info["dtype"], info["shape"], raw.tobytes())

        prefixes = sorted({v[1] for v in out_tensors.values() if v[0] == "PENDING"})
        for p in prefixes:
            wn, sn, bn = f"{p}.weight", f"{p}.scales", f"{p}.biases"
            if any(n not in out_tensors or out_tensors[n][0] != "PENDING" for n in (wn, sn, bn)):
                raise RuntimeError(f"module {p} triple not co-located in {shard}")
            wi, si = out_tensors[wn][3], out_tensors[sn][3]
            assert wi["dtype"] == "U32" and si["dtype"] == "BF16", (wi, si)
            w_u32 = np.frombuffer(out_tensors[wn][4].tobytes(), dtype=np.uint32).reshape(wi["shape"])
            sc_u16 = np.frombuffer(out_tensors[sn][4].tobytes(), dtype=np.uint16)
            b_u16 = np.frombuffer(out_tensors[bn][4].tobytes(), dtype=np.uint16)
            pw, pw_shape, s4, b4, sb_shape, err = requant_module(
                w_u32, wi["shape"], sc_u16, si["shape"], b_u16, 64, tgs
            )
            out_tensors[wn] = ("U32", pw_shape, pw.tobytes())
            out_tensors[sn] = ("BF16", sb_shape, s4.tobytes())
            out_tensors[bn] = ("BF16", sb_shape, b4.tobytes())
            max_err = max(max_err, err)
            n_done += 1

        new_total += write_safetensors(args.dst / shard, out_tensors, meta)
        print(f"wrote {shard} ({n_done}/{len(targets)} modules done)")

    idx["metadata"]["total_size"] = new_total
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))

    for key in ("quantization", "quantization_config"):
        if key in cfg and isinstance(cfg[key], dict):
            for t in targets:
                if t not in cfg[key]:
                    continue
                if TARGET_BITS == cfg[key].get("bits", global_bits) and tgs == cfg[key].get(
                    "group_size", global_gs
                ):
                    del cfg[key][t]
                else:
                    cfg[key][t] = {"bits": TARGET_BITS, "group_size": tgs}
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in args.src.iterdir():
        if f.name.startswith("model-") or f.name in (
            "model.safetensors.index.json",
            "config.json",
        ):
            continue
        if f.is_file():
            shutil.copy2(f, args.dst / f.name)

    print(
        f"done: {n_done} modules, max group reconstruction err {max_err:.5f}, "
        f"total {new_total / 1e9:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
