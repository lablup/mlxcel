#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-forced logit trace of a Kolibri-1 checkpoint, computed by mlx-vlm.

The reference arm for `examples/logit_trace.rs` on Aleph Alpha Kolibri-1
(#2357), which mlx-vlm implements (`mlx_vlm/models/kolibri1`) and mlx-lm does
not. It runs the chunk plan `scripts/mlx_lm_logit_trace.py` runs and writes the
same TSV, so `scripts/compare_logit_traces.py` reads the pair unmodified.

`--rope-on-full` is the negative arm: the full-attention layers also apply the
sliding layers' RoPE, which upstream deliberately omits (they are NoPE). A
trace that cannot tell this arm from the real model has not reached a context
long enough to measure the full layers.

Usage:
    kolibri1_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--rope-on-full] [--no-prefix-sharing] > reference.tsv

The mlxcel decode arm (`MLXCEL_TRACE_BATCH_ROWS=1`) traces from token PREFILL,
so with `256 8 8 1024` here its chunks 0 to 3 are this trace's chunks 4 to 7.

Needs `mlx`, `mlx-vlm` (any build with `mlx_vlm.models.kolibri1`, for example
the vendored `references/mlx-vlm` on `PYTHONPATH`), `tokenizers` and `numpy`.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laguna_oracle_trace import (  # noqa: E402
    OracleError,
    load_tokenizer,
    plan_chunks,
    plan_forwards,
    usize_arg,
)
from mlx_lm_logit_trace import trace  # noqa: E402


def rope_every_layer(language) -> int:
    """Give every full-attention layer the sliding layers' rotation.

    Upstream applies RoPE only `if self.is_sliding`, and `is_sliding` also
    picks the layer's mask, so the flag cannot simply be flipped. Each full
    layer's class is swapped for a subclass whose `__call__` is upstream's
    with the rotation made unconditional, sharing a sliding layer's `rope`
    module; that is what an implementation that forgot the NoPE rule computes.
    """
    from mlx_vlm.models.base import scaled_dot_product_attention
    from mlx_vlm.models.kolibri1.language import Attention

    class RopeEverywhere(Attention):
        def __call__(self, x, mask=None, cache=None):
            b, n, _ = x.shape
            q = self.q_proj(x).reshape(b, n, self.n_heads, self.head_dim)
            k = self.k_proj(x).reshape(b, n, self.n_kv_heads, self.head_dim)
            v = self.v_proj(x).reshape(b, n, self.n_kv_heads, self.head_dim)
            q = self.q_norm(q).transpose(0, 2, 1, 3)
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, n, -1))

    rope = next(layer.self_attn.rope for layer in language.layers if layer.self_attn.is_sliding)
    patched = 0
    for layer in language.layers:
        attn = layer.self_attn
        if not attn.is_sliding:
            attn.rope = rope
            attn.__class__ = RopeEverywhere
            patched += 1
    return patched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("text_file")
    ap.add_argument("chunk_tokens", nargs="?")
    ap.add_argument("max_chunks", nargs="?")
    ap.add_argument("topk", nargs="?")
    ap.add_argument("prefill", nargs="?")
    ap.add_argument(
        "--rope-on-full", action="store_true", help="negative arm that rotates the NoPE layers"
    )
    ap.add_argument(
        "--no-prefix-sharing",
        action="store_true",
        help="one forward per chunk, to check how closely a shared forward agrees with it",
    )
    args = ap.parse_args(argv)

    chunk_tokens = usize_arg(args.chunk_tokens, 256)
    max_chunks = usize_arg(args.max_chunks, 4)
    topk = usize_arg(args.topk, 8)
    prefill = usize_arg(args.prefill, 0)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        import mlx.core as mx
        import mlx_vlm
        import tokenizers
        from mlx_vlm.utils import load_model
    except ModuleNotFoundError as err:
        log(f"error: {err.name} is not installed here; run this with a Python that has mlx-vlm")
        return 1

    try:
        if chunk_tokens == 0 or topk == 0:
            raise OracleError("CHUNK_TOKENS and TOPK must be positive")
        model_dir = Path(args.model_dir)
        text = Path(args.text_file).read_bytes().decode("utf-8")
        tokenizer = load_tokenizer(model_dir)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        bos_prefix = tokenizer.encode("", add_special_tokens=True).ids[:1]
        chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, prefill)
        groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)

        started = time.perf_counter()
        model = load_model(model_dir)
        language = model.language_model
        patched = rope_every_layer(language) if args.rope_on_full else 0
        if args.rope_on_full and patched == 0:
            raise OracleError("--rope-on-full found no full-attention layer to rotate")
        log(
            f"loaded {model_dir} with mlx-vlm {mlx_vlm.__version__} in "
            f"{time.perf_counter() - started:.0f}s; rope_on_full={args.rope_on_full} "
            f"({patched} layers patched)"
        )
        results = trace(lambda inputs: language(inputs).logits, groups, topk, log)
        for chunk in chunks:
            if not all(math.isfinite(v) for v in results[chunk.index][0]):
                raise OracleError(f"chunk {chunk.index} produced a non-finite NLL")
    except OracleError as err:
        log(f"error: {err}")
        return 1

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    arm = "mlx-vlm-rope-on-full" if args.rope_on_full else "mlx-vlm"
    out.write(
        f"# oracle\t{arm}\t{mlx_vlm.__version__}\tmlx\t{mx.__version__}"
        f"\ttokenizers\t{tokenizers.__version__}\n"
    )
    out.write(f"# device\t{mx.default_device()}\n")
    out.write(f"# corpus\t{args.text_file}\ttokens\t{len(ids)}\n")
    out.write(
        f"# chunks\t{len(chunks)}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}\n"
    )
    sharing = str(not args.no_prefix_sharing).lower()
    out.write(f"# forwards\t{len(groups)}\tprefix_sharing\t{sharing}\n")
    out.write("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits\n")
    bos_anchor = ",".join(str(t) for t in bos_prefix) or "none"
    out.write(f"# bos_anchor\t{bos_anchor}\tchunks\t{'all' if bos_prefix else 'none'}\n")
    for chunk in chunks:
        nll, top_ids, top_logits = results[chunk.index]
        for j, target in enumerate(chunk.targets):
            ids_s = ",".join(str(i) for i in top_ids[j])
            lg_s = ",".join(f"{v:.6f}" for v in top_logits[j])
            out.write(f"{chunk.index}\t{j}\t{target}\t{nll[j]:.6f}\t{ids_s}\t{lg_s}\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
