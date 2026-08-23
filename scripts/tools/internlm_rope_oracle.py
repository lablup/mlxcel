"""Greedy-decode oracle for InternLM3, on the mlx-lm stack.

Two modes:

  --mode stock   mlx-lm 0.31.3 exactly as installed. Its
                 `Attention.__init__` computes
                 `rope_scale = 1/factor if rope_type == "linear" else 2.0`,
                 so a `dynamic` block gets position scale 2.0, and its
                 `DynamicNTKScalingRoPE.__call__` then reuses that same 2.0 as
                 the NTK factor and reads the sequence length from
                 `x.shape[1]`, which is the head axis of a
                 [B, n_heads, L, head_dim] tensor.

  --mode fixed   the schedule the checkpoint's own remote code implements
                 (`modeling_internlm3.py` -> transformers
                 ROPE_INIT_FUNCTIONS["dynamic"] ->
                 `_compute_dynamic_ntk_parameters`): positions are never
                 scaled, `seq_len` is clamped up to
                 `max_position_embeddings`, and only the base moves.

Everything else (weight loading, dequantisation, attention, sampling) is
mlx-lm's. Only the rope module is replaced.
"""

import argparse
import json
import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


class FixedDynamicNtkRoPE(nn.Module):
    def __init__(self, dims, max_position_embeddings, traditional, base, rope_type, factor):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.original_base = base
        self.rope_type = rope_type
        self.factor = factor

    def __call__(self, x, offset: int = 0):
        # x is [B, n_heads, L, head_dim]; the sequence axis is -2.
        seq = x.shape[-2] + offset
        if self.rope_type == "dynamic":
            seq_len = max(seq, self.max_position_embeddings)
            base = self.original_base * (
                (self.factor * seq_len / self.max_position_embeddings) - (self.factor - 1)
            ) ** (self.dims / (self.dims - 2))
            scale = 1.0
        elif self.rope_type == "linear":
            base = self.original_base
            scale = 1.0 / self.factor
        else:
            base = self.original_base
            scale = 1.0
        return mx.fast.rope(
            x, self.dims, traditional=self.traditional, base=base, scale=scale, offset=offset
        )


def patch(model):
    args = model.args
    scaling = args.rope_scaling or {}
    rope_type = scaling.get("rope_type", scaling.get("type", "default"))
    factor = float(scaling.get("factor", 1.0))
    head_dim = args.hidden_size // args.num_attention_heads
    for layer in model.model.layers:
        # internlm3 names the block `self_attn`; internlm2 names it `attention`.
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention")
        attn.rope = FixedDynamicNtkRoPE(
            head_dim,
            args.max_position_embeddings,
            args.rope_traditional,
            float(args.rope_theta),
            rope_type,
            factor,
        )
    return rope_type, factor, head_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["stock", "fixed"], required=True)
    ap.add_argument("--ids-file")
    ap.add_argument("--prompt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dump-ids")
    args = ap.parse_args()

    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    info = {}
    if args.mode == "fixed":
        rope_type, factor, head_dim = patch(model)
        info = {"rope_type": rope_type, "factor": factor, "head_dim": head_dim}

    if args.ids_file:
        ids = [int(t) for t in open(args.ids_file).read().split()]
    else:
        ids = tokenizer.encode(args.prompt)
    if args.dump_ids:
        open(args.dump_ids, "w").write(" ".join(str(i) for i in ids))

    cache = make_prompt_cache(model)
    x = mx.array([ids])
    logits = model(x, cache=cache)

    out, margins = [], []
    for _ in range(args.n):
        row = logits[0, -1].astype(mx.float32)
        order = mx.argsort(row)
        top1 = int(order[-1].item())
        top2 = int(order[-2].item())
        margins.append(float(row[top1].item() - row[top2].item()))
        out.append(top1)
        logits = model(mx.array([[top1]]), cache=cache)

    print(json.dumps({
        "mode": args.mode,
        "info": info,
        "n_prompt": len(ids),
        "ids": out,
        "margins": [round(m, 5) for m in margins],
        "text": tokenizer.decode(out),
    }))


if __name__ == "__main__":
    main()
