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
"""Image-prompted teacher-forced logit trace, computed by mlx-vlm.

The out-of-band reference arm for the image arm of `examples/logit_trace.rs`
(`MLXCEL_TRACE_IMAGE`, #2388). It renders the same one-image user turn with
mlx-vlm's own chat template and processor, appends the continuation tokens of
an existing mlxcel image trace (so both arms score the identical token stream),
runs one causal forward, and writes the TSV `logit_trace` writes, so
`scripts/compare_logit_traces.py` reads the pair unmodified. Nothing in the
runtime imports this file.

Usage:
    mlx_vlm_image_logit_trace.py MODEL_DIR IMAGE MLXCEL_TRACE.tsv
        [--prompt "Describe the image."] [--topk 8] [--float32]
        [--mlxcel-prompt] > reference.tsv

Needs `mlx`, `mlx-vlm`, `mlx-lm` and `numpy`. By default mlx-vlm runs in the
checkpoint's own activation dtype, a same-precision reference. With
`--float32` every quantized layer is dequantized (`mlx_lm.utils.dequantize_model`)
and the model cast to float32, a float32 oracle of the same weights; it is also
the only practical mode on a CPU-only `mlx` build, where quantized matmuls are
orders of magnitude slower than dense ones.
The prompt length after image expansion is printed as `# prompt_tokens` and
the ids as `# prompt_ids`, the two lines the mlxcel trace writes; a mismatch
means the two processors disagree before the tower runs. Where mlxcel follows
Hugging Face and mlx-vlm does not (the cases `docs/benchmarks.md` lists),
`--mlxcel-prompt` scores the mlxcel trace's own `# prompt_ids` instead of
mlx-vlm's render, so the arms differ only in the forward pass: mlx-vlm still
preprocesses the image, its image-token count must equal the trace's, and
MiniCPM-V's `image_bound` is recomputed from the substituted ids. mlx-vlm's
own prompt length is then printed as `# mlx_vlm_prompt_tokens`.
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs


def read_targets(path: str) -> list[int]:
    targets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if cols[0] != "0":
                break
            targets.append(int(cols[2]))
    if len(targets) < 2:
        raise SystemExit(f"{path}: fewer than two traced rows")
    return targets


def read_prompt_ids(path: str) -> list[int]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# prompt_ids\t"):
                return [int(x) for x in line.split("\t", 1)[1].strip().split(",")]
    raise SystemExit(f"{path}: no `# prompt_ids` line; regenerate it with a current logit_trace")


def image_bounds(ids: list[int], tokenizer) -> np.ndarray:
    """MiniCPM-V feature spans: the tokens between each `<image>`/`<slice>` and its close."""
    starts = {tokenizer.convert_tokens_to_ids(t) for t in ("<image>", "<slice>")}
    ends = {tokenizer.convert_tokens_to_ids(t) for t in ("</image>", "</slice>")}
    open_at, bounds = None, []
    for i, tok in enumerate(ids):
        if tok in starts:
            open_at = i + 1
        elif tok in ends and open_at is not None:
            bounds.append([open_at, i])
            open_at = None
    return np.array(bounds, dtype=np.int32).reshape(-1, 2)


def substitute_prompt(input_ids, inputs, ids: list[int], config, processor):
    """Score `ids` (the mlxcel prompt) with the pixels mlx-vlm preprocessed."""
    image_token = getattr(config, "image_token_index", None)
    if image_token is None:
        image_token = getattr(config, "image_token_id", None)
    own = np.array(input_ids[0]).tolist()
    if image_token is not None and own.count(image_token) != ids.count(image_token):
        raise SystemExit(
            f"image-token count differs: mlx-vlm {own.count(image_token)}, "
            f"mlxcel {ids.count(image_token)}; the prompts cannot share pixel features"
        )
    if "image_bound" in inputs:
        tokenizer = getattr(processor, "tokenizer", processor)
        inputs["image_bound"] = [image_bounds(ids, tokenizer)]
    return mx.array([ids], dtype=input_ids.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("image")
    ap.add_argument("mlxcel_trace")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument(
        "--mlxcel-prompt",
        action="store_true",
        help="score the mlxcel trace's `# prompt_ids` instead of mlx-vlm's own render",
    )
    args = ap.parse_args()

    model, processor = load(args.model_dir)
    if args.float32:
        from mlx.utils import tree_map
        from mlx_lm.utils import dequantize_model

        model = dequantize_model(model)
        model.update(
            tree_map(
                lambda p: p.astype(mx.float32) if mx.issubdtype(p.dtype, mx.floating) else p,
                model.parameters(),
            )
        )
    prompt = apply_chat_template(processor, model.config, args.prompt, num_images=1)
    inputs = prepare_inputs(
        processor,
        images=[args.image],
        prompts=[prompt],
        image_token_index=getattr(model.config, "image_token_index", None),
    )
    input_ids = inputs.pop("input_ids")
    pixel_values = inputs.pop("pixel_values", None)
    inputs.pop("attention_mask", None)
    mlx_vlm_prompt_len = input_ids.shape[1]
    if args.mlxcel_prompt:
        input_ids = substitute_prompt(
            input_ids, inputs, read_prompt_ids(args.mlxcel_trace), model.config, processor
        )
    prompt_len = input_ids.shape[1]

    targets = read_targets(args.mlxcel_trace)
    cont = mx.array([targets[:-1]], dtype=input_ids.dtype)
    full_ids = mx.concatenate([input_ids, cont], axis=1)
    if model.config.model_type in ("gemma3", "paligemma"):
        # Their `__call__` builds a 4D mask from the 2D attention mask, all
        # ones here, which would let every row see the whole continuation.
        # Merge the image features, then run the language model causally.
        ones = mx.ones(full_ids.shape, dtype=mx.int32)
        merged = model.get_input_embeddings(full_ids, pixel_values, mask=ones, **inputs)
        out = model.language_model(
            inputs=full_ids, inputs_embeds=merged.inputs_embeds, mask=None
        )
    else:
        out = model(full_ids, pixel_values, mask=None, **inputs)
    logits = getattr(out, "logits", out)
    rows = logits[0, prompt_len - 1 : prompt_len - 1 + len(targets)].astype(mx.float32)
    rows = np.array(rows)
    shifted = rows - rows.max(-1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    print(f"# model\t{args.model_dir}")
    print(f"# device\tmlx-vlm {mx.default_device()}{' float32' if args.float32 else ''}")
    print(f"# image\t{args.image}\tprompt\t{args.prompt}")
    print(f"# prompt_tokens\t{prompt_len}")
    print("# prompt_ids\t" + ",".join(str(int(i)) for i in np.array(input_ids[0]).tolist()))
    if args.mlxcel_prompt:
        print(f"# mlx_vlm_prompt_tokens\t{mlx_vlm_prompt_len}")
    print(f"# corpus\t{args.mlxcel_trace}\ttokens\t{len(targets)}")
    print(f"# chunks\t1\tchunk_tokens\t{len(targets)}\ttopk\t{args.topk}\tprefill\t0")
    print("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits")
    for j, target in enumerate(targets):
        row = rows[j]
        top = np.argsort(-row, kind="stable")[: args.topk]
        ids_s = ",".join(str(int(i)) for i in top)
        lg_s = ",".join(f"{float(row[i]):.6f}" for i in top)
        print(f"0\t{j}\t{target}\t{float(-lp[j, target]):.6f}\t{ids_s}\t{lg_s}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
