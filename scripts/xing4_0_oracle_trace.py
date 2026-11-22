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
"""Teacher-forced logit trace of a Xing4.0 checkpoint, computed by mlx-vlm.

The reference arm for `examples/logit_trace.rs` on Xing4.0 (#2356), which
mlx-vlm implements (`mlx_vlm/models/xing4_0`) and mlx-lm does not. It runs the
chunk plan `scripts/mlx_lm_logit_trace.py` runs and writes the same TSV, so
`scripts/compare_logit_traces.py` reads the pair unmodified.

Xing4.0 ships a SentencePiece `tokenizer.model` and no `tokenizer.json`, so
the text is tokenized with `sentencepiece` (no BOS, then the `<_start>` BOS
anchor that the checkpoint's `add_bos_token` asks for). The comparison checks
that both arms traced the same target ids.

Negative arms, each replacing one detail of the hyper connection with a
plausible wrong one, so a trace that cannot tell them from the real model is
not measuring the connection:

* `--comb-transpose`: the expand mixes with `comb.T @ x` (DeepSeek-V4's form);
* `--v4-sinkhorn`: softmax rows, `+ hc_eps`, an initial column pass and
  `iters - 1` rounds (DeepSeek-V4's Sinkhorn);
* `--no-yarn`: the attention RoPE at the plain `rope_theta` frequencies,
  ignoring the checkpoint's YaRN block.

Usage:
    xing4_0_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--comb-transpose|--v4-sinkhorn|--no-yarn]
        [--no-prefix-sharing] > reference.tsv

Needs `mlx`, `mlx-vlm` (any build with `mlx_vlm.models.xing4_0`, for example
the vendored `references/mlx-vlm` on `PYTHONPATH`), `sentencepiece` and
`numpy`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laguna_oracle_trace import (  # noqa: E402
    OracleError,
    plan_chunks,
    plan_forwards,
    usize_arg,
)
from mlx_lm_logit_trace import trace  # noqa: E402


def patch_comb_transpose(language_mod) -> None:
    def call(self, x, mask=None, cache=None):
        post, comb, collapsed = self.attn_hc(x)
        out = self.self_attn(self.input_layernorm(collapsed), mask, cache)
        x = post[..., None] * out[:, :, None] + comb.swapaxes(-1, -2) @ x
        post, comb, collapsed = self.ffn_hc(x)
        out = self.mlp(self.post_attention_layernorm(collapsed))
        return post[..., None] * out[:, :, None] + comb.swapaxes(-1, -2) @ x

    language_mod.DecoderLayer.__call__ = call


def patch_v4_sinkhorn(language_mod, mx) -> None:
    def call(self, x):
        config = self.config
        hc = config.hc_mult
        flat = x.reshape(*x.shape[:2], -1).astype(mx.float32)
        flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + config.rms_norm_eps)
        mix = (flat.astype(x.dtype) @ self.hc_fn.astype(x.dtype).T).astype(mx.float32)
        pre, post, comb = mx.split(mix, [hc, 2 * hc], axis=-1)
        pre_b, post_b, comb_b = mx.split(self.hc_base, [hc, 2 * hc])
        pre = mx.sigmoid(pre * self.hc_scale[0] + pre_b)
        post = 2 * mx.sigmoid(post * self.hc_scale[1] + post_b)
        comb = comb.reshape(*comb.shape[:-1], hc, hc) * self.hc_scale[2] + comb_b.reshape(hc, hc)
        comb = mx.softmax(comb, axis=-1, precise=True) + config.hc_eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + config.hc_eps)
        for _ in range(config.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + config.hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + config.hc_eps)
        collapsed = (pre[..., None].astype(x.dtype) * x).sum(axis=2)
        return post.astype(x.dtype), comb.astype(x.dtype), collapsed

    language_mod.HyperConnection.__call__ = call


def drop_yarn(language, nn) -> None:
    for layer in language.model.layers:
        attn = layer.self_attn
        attn.rope = nn.RoPE(
            attn.qk_rope_head_dim, traditional=True, base=language.args.rope_theta
        )


def load_sentencepiece(model_dir: Path):
    import sentencepiece as spm

    path = model_dir / "tokenizer.model"
    if not path.is_file():
        raise OracleError(f"{path} not found")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(path))
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    add_bos = config.get("add_bos_token", True)
    bos = config.get("bos_token", "<_start>")
    if isinstance(bos, dict):
        bos = bos.get("content")
    bos_prefix = [sp.piece_to_id(bos)] if add_bos and bos else []
    return sp, bos_prefix


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
    arms = ap.add_mutually_exclusive_group()
    arms.add_argument("--comb-transpose", action="store_true", help="negative arm: comb.T @ x")
    arms.add_argument("--v4-sinkhorn", action="store_true", help="negative arm: V4 Sinkhorn")
    arms.add_argument("--no-yarn", action="store_true", help="negative arm: plain RoPE")
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
        import mlx.nn as nn
        import mlx_vlm
        from mlx_vlm.models.xing4_0 import language as language_mod
        from mlx_vlm.utils import load_model
    except ModuleNotFoundError as err:
        log(f"error: {err.name} is not installed here; run this with a Python that has mlx-vlm")
        return 1

    arm = "mlx-vlm"
    if args.comb_transpose:
        patch_comb_transpose(language_mod)
        arm = "mlx-vlm-comb-transpose"
    elif args.v4_sinkhorn:
        patch_v4_sinkhorn(language_mod, mx)
        arm = "mlx-vlm-v4-sinkhorn"
    elif args.no_yarn:
        arm = "mlx-vlm-no-yarn"

    try:
        if chunk_tokens == 0 or topk == 0:
            raise OracleError("CHUNK_TOKENS and TOPK must be positive")
        model_dir = Path(args.model_dir)
        text = Path(args.text_file).read_bytes().decode("utf-8")
        sp, bos_prefix = load_sentencepiece(model_dir)
        ids = sp.encode(text)
        chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, prefill)
        groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)

        started = time.perf_counter()
        model = load_model(model_dir)
        language = model.language_model
        if args.no_yarn:
            drop_yarn(language, nn)
        log(
            f"loaded {model_dir} with mlx-vlm {mlx_vlm.__version__} in "
            f"{time.perf_counter() - started:.0f}s; arm={arm}"
        )
        results = trace(lambda inputs: language(inputs).logits, groups, topk, log)
        for chunk in chunks:
            if not all(math.isfinite(v) for v in results[chunk.index][0]):
                raise OracleError(f"chunk {chunk.index} produced a non-finite NLL")
    except OracleError as err:
        log(f"error: {err}")
        return 1

    import sentencepiece

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    out.write(
        f"# oracle\t{arm}\t{mlx_vlm.__version__}\tmlx\t{mx.__version__}"
        f"\tsentencepiece\t{sentencepiece.__version__}\n"
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
