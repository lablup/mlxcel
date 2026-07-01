#!/usr/bin/env python3
"""Architecture-generic HF fp32 greedy oracle for the OpenXLA token-exactness gate
(issue #496).

Given a checkpoint (bf16 / f16 / f32, or an MLX affine 4-bit / 8-bit quantized
checkpoint), this produces the external reference the `xla_oracle_check` example
diffs against: the pure next-token-argmax trajectory for N steps with NO EOS stop,
loaded and run in fp32 (the exact widening the XLA path applies to its weights).
The first generated token is the argmax after the FULL prompt, matching
`XlaReferenceEngine::prefill_first`.

For a quantized checkpoint it first dequantizes the packed weights to f32 offline
(the "offline dequant to f32 oracle" step), using the same affine formula the Rust
loader applies (`src/lib/mlxcel-xla/src/weights.rs::dequantize_affine`:
`w = q * scale + bias`, `q` unpacked low-order-first, f16 scales/biases), so the
oracle weights match what the engine dequantizes to and token-exactness is
meaningful.

Usage:
    # bf16 / f32 checkpoint (no dequant needed):
    spike/openxla/.venv/bin/python spike/openxla/oracle_continuation.py \\
        --model /models/qwen2.5-0.5b-bf16 --out /tmp/oracle.json \\
        --prompt "The capital of France is" --max-new 40

    # MLX 4-bit / 8-bit checkpoint (dequantized to f32 first):
    ... --model /models/qwen2.5-0.5b-4bit --out /tmp/oracle.json

    # verify the dequant math mirrors the Rust loader (numpy only, no model):
    ... --selftest

Output JSON: {"prompt_text", "prompt_ids": [int...], "ref_token_ids": [int...]}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np


def dequantize_affine(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int,
    group_size: int,
) -> np.ndarray:
    """Dequantize one MLX affine-quantized weight to row-major ``[out, in]`` f32.

    Mirrors ``src/lib/mlxcel-xla/src/weights.rs::dequantize_affine`` exactly so the
    oracle and the XLA engine dequantize to identical f32 weights.

    Args:
        packed: ``[out, in_packed]`` uint32 weight (``in_packed = in * bits / 32``),
            with ``bits``-wide codes packed low-order-first per u32.
        scales: ``[out, in / group_size]`` per-group scale (widened to f32).
        biases: ``[out, in / group_size]`` per-group bias (widened to f32).
        bits: 4 or 8.
        group_size: input columns sharing one scale/bias.

    Returns:
        ``[out, in]`` float32 array, ``w[o, i] = q[o, i] * scale[o, i // group_size]
        + bias[o, i // group_size]``.
    """
    if bits not in (4, 8):
        raise ValueError(f"unsupported quantization bits {bits} (expected 4 or 8)")
    out, in_packed = packed.shape
    per_u32 = 32 // bits
    in_ = in_packed * per_u32
    n_groups = scales.shape[1]
    if group_size <= 0 or in_ != n_groups * group_size:
        raise ValueError(
            f"group_size {group_size} x n_groups {n_groups} != in dimension {in_}"
        )
    mask = np.uint32((1 << bits) - 1)
    shifts = np.arange(per_u32, dtype=np.uint32) * np.uint32(bits)
    # [out, in_packed, per_u32]: code j of packed[o, p] is (u >> (bits*j)) & mask.
    codes = (packed[:, :, None] >> shifts[None, None, :]) & mask
    codes = codes.reshape(out, in_).astype(np.float32)
    scale_full = np.repeat(scales.astype(np.float32), group_size, axis=1)
    bias_full = np.repeat(biases.astype(np.float32), group_size, axis=1)
    return codes * scale_full + bias_full


def _selftest() -> None:
    """Assert the numpy dequant matches the Rust ``weights.rs`` hand-examples."""
    # 4-bit: u32 0x8765_4321 -> nibbles [1..8] low-first; groups (2.0, +10), (0.5, -1).
    packed4 = np.array([[0x87654321]], dtype=np.uint32)
    scales = np.array([[2.0, 0.5]], dtype=np.float32)
    biases = np.array([[10.0, -1.0]], dtype=np.float32)
    w4 = dequantize_affine(packed4, scales, biases, 4, 4)
    want4 = np.array([[12.0, 14.0, 16.0, 18.0, 1.5, 2.0, 2.5, 3.0]], dtype=np.float32)
    assert np.array_equal(w4, want4), f"4-bit mismatch: {w4}"

    # 8-bit: u32 0x281E_140A -> bytes [10, 20, 30, 40] low-first; same two groups.
    packed8 = np.array([[0x281E140A]], dtype=np.uint32)
    w8 = dequantize_affine(packed8, scales, biases, 8, 2)
    want8 = np.array([[30.0, 50.0, 14.0, 19.0]], dtype=np.float32)
    assert np.array_equal(w8, want8), f"8-bit mismatch: {w8}"


def read_quantization(model_dir: str) -> dict | None:
    """Return ``{"bits", "group_size"}`` if ``config.json`` marks an MLX-quantized
    checkpoint, else ``None``."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if not q:
        return None
    return {"bits": int(q["bits"]), "group_size": int(q["group_size"])}


def _shard_files(model_dir: str) -> list[tuple[str, list[str]]]:
    """``[(safetensors_path, [tensor_name...])]`` for a single-file or sharded
    checkpoint."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            by_file.setdefault(filename, []).append(name)
        return [(os.path.join(model_dir, fn), names) for fn, names in by_file.items()]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open

        with safe_open(single, framework="numpy") as f:
            return [(single, list(f.keys()))]
    raise FileNotFoundError(f"no model.safetensors or index.json in {model_dir}")


def dequant_checkpoint(model_dir: str, out_dir: str, bits: int, group_size: int) -> None:
    """Write an HF-loadable f32 checkpoint (dequantized weights, no quantization
    block) into ``out_dir``."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    os.makedirs(out_dir, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for path, names in _shard_files(model_dir):
        nameset = set(names)
        # numpy handle for the uint32 packed weights; torch handle so bf16 / f16
        # copy-through tensors and f16 scales/biases widen to f32 exactly.
        with (
            safe_open(path, framework="numpy") as fnp,
            safe_open(path, framework="pt") as fpt,
        ):
            for name in names:
                if name.endswith(".scales") or name.endswith(".biases"):
                    continue  # consumed with the paired .weight
                base = name[: -len(".weight")] if name.endswith(".weight") else None
                if base is not None and f"{base}.scales" in nameset:
                    packed = fnp.get_tensor(name).astype(np.uint32)
                    scales = fpt.get_tensor(f"{base}.scales").float().numpy()
                    biases = fpt.get_tensor(f"{base}.biases").float().numpy()
                    tensors[name] = dequantize_affine(
                        packed, scales, biases, bits, group_size
                    )
                else:
                    tensors[name] = fpt.get_tensor(name).float().numpy()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Tokenizer + generation config so the dequantized dir loads like the original.
    passthrough = {
        "generation_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    }
    for fn in os.listdir(model_dir):
        if fn.startswith("tokenizer") or fn in passthrough:
            shutil.copy(os.path.join(model_dir, fn), os.path.join(out_dir, fn))


def hf_greedy_oracle(model_dir: str, prompt: str, n_new: int) -> dict:
    """Run HF fp32 greedy (pure argmax, no EOS stop) for ``n_new`` steps."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    prompt_ids = tok(prompt, return_tensors="pt").input_ids  # [1, L]
    ids = prompt_ids.clone()
    ref: list[int] = []
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(ids).logits[:, -1, :]  # [1, V]
            nxt = int(torch.argmax(logits, dim=-1).item())
            ref.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return {
        "prompt_text": prompt,
        "prompt_ids": prompt_ids[0].tolist(),
        "ref_token_ids": ref,
        "decoded": tok.decode(ref),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="checkpoint directory")
    ap.add_argument("--out", help="output oracle JSON path")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument(
        "--dequant-dir",
        default=None,
        help="where to write the dequantized f32 checkpoint (default: a temp dir "
        "removed on exit); implies --keep-dequant if set",
    )
    ap.add_argument(
        "--keep-dequant",
        action="store_true",
        help="do not remove the dequantized checkpoint afterwards",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="verify the dequant math mirrors the Rust loader (numpy only), then exit",
    )
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        print("oracle_continuation self-test: OK (dequant matches weights.rs)")
        return 0

    if not args.model or not args.out:
        ap.error("--model and --out are required (unless --selftest)")

    quant = read_quantization(args.model)
    source = args.model
    tmp_dir = None
    if quant is not None:
        target = args.dequant_dir or tempfile.mkdtemp(prefix="mlxcel-dequant-")
        tmp_dir = None if (args.dequant_dir or args.keep_dequant) else target
        print(
            f"[oracle] {args.model} is MLX-quantized (bits={quant['bits']}, "
            f"group_size={quant['group_size']}); dequantizing to f32 -> {target}",
            flush=True,
        )
        dequant_checkpoint(args.model, target, quant["bits"], quant["group_size"])
        source = target

    print(
        f"[oracle] HF fp32 greedy from {source}: prompt={args.prompt!r} "
        f"max_new={args.max_new}",
        flush=True,
    )
    result = hf_greedy_oracle(source, args.prompt, args.max_new)
    with open(args.out, "w") as f:
        json.dump(
            {
                "prompt_text": result["prompt_text"],
                "prompt_ids": result["prompt_ids"],
                "ref_token_ids": result["ref_token_ids"],
            },
            f,
        )
    print(
        f"[oracle] wrote {args.out}: {len(result['prompt_ids'])} prompt tokens, "
        f"{len(result['ref_token_ids'])} reference tokens",
        flush=True,
    )
    print(f"[oracle] continuation: {result['decoded']!r}", flush=True)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
