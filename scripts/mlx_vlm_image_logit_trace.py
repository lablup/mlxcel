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
        [--prompt "Describe the image."] [--topk 8] > reference.tsv

Needs `mlx`, `mlx-vlm` and `numpy`. mlx-vlm runs in the checkpoint's own
activation dtype, so this is a same-precision reference, not a float32 oracle.
The prompt length after image expansion is printed as `# prompt_tokens`; it
should equal the mlxcel trace's, and a mismatch means the two processors
disagree before the tower runs.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("image")
    ap.add_argument("mlxcel_trace")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    model, processor = load(args.model_dir)
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
    prompt_len = input_ids.shape[1]

    targets = read_targets(args.mlxcel_trace)
    cont = mx.array([targets[:-1]], dtype=input_ids.dtype)
    full_ids = mx.concatenate([input_ids, cont], axis=1)
    out = model(full_ids, pixel_values, mask=None, **inputs)
    logits = getattr(out, "logits", out)
    rows = logits[0, prompt_len - 1 : prompt_len - 1 + len(targets)].astype(mx.float32)
    rows = np.array(rows)
    shifted = rows - rows.max(-1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    print(f"# model\t{args.model_dir}")
    print("# device\tmlx-vlm")
    print(f"# image\t{args.image}\tprompt\t{args.prompt}")
    print(f"# prompt_tokens\t{prompt_len}")
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
