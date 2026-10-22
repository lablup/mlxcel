#!/usr/bin/env python3
"""Trace Qwen4-Exp with mlx-vlm, matching examples/logit_trace.rs.

Usage: python scripts/qwen4_exp_oracle_trace.py MODEL CORPUS START WIDTH
Run with mlx-vlm installed from the reference revision recorded in the results.
OptiQ wrapper keys, folded norm scales and the vision sidecar are normalized
to mlx-vlm storage and loaded strictly. No forward functions are patched. For the dense negative control use --indexer-budget 8192 with a total
context below that budget, and the same config override on the Rust side.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen4_exp import LanguageModel, Model, ModelConfig, VisionModel
from mlx_vlm.utils import load_model_weights, sanitize_weights

from laguna_oracle_trace import load_tokenizer


def load_reference(path: Path, indexer_budget: int | None = None):
    """Load the unchanged reference model, including OptiQ wrapper weights."""
    config = json.loads((path / "config.json").read_text())
    if indexer_budget is not None:
        config["text_config"]["indexer_budget"] = indexer_budget
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)
    weights = {}
    index = json.loads((path / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values())
    if sidecar := config.get("optiq_vision", {}).get("sidecar"):
        files.add(sidecar)
    files = sorted(files)
    for filename in files:
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"checkpoint index path must stay inside the model directory: {filename}")
        for key, value in mx.load(str(path / filename)).items():
            # OptiQ stores the text backbone without the VLM wrapper prefix.
            if key.startswith("model.") and not key.startswith(("model.language_model.", "model.visual.")):
                key = "language_model." + key
            # OptiQ folds the zero-centered norms' +1 into stored weights.
            # Undo that storage conversion in float32 for mlx-vlm's 1 + weight.
            if config.get("optiq_vision") and key.startswith("language_model.") and key.endswith((
                "q_layernorm.weight", "k_layernorm.weight", "q_norm.weight",
                "k_norm.weight", "hc_norm.weight", "norm_key.weight",
                "norm_query.weight", "norm_conv.weight",
            )):
                value = value.astype(mx.float32) - 1.0
            weights[key] = value
    weights = model.sanitize(weights)
    weights = sanitize_weights(VisionModel, weights, model_config.vision_config)
    weights = sanitize_weights(LanguageModel, weights, model_config.text_config)
    config["quantization"] = model_config.quantization
    model = load_model_weights(model, config, weights, path, strict=True)
    del weights
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("corpus")
    parser.add_argument("start", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("--indexer-budget", type=int)
    args = parser.parse_args()
    model_dir, corpus, start, width = args.model_dir, args.corpus, args.start, args.width
    if start < 0 or width <= 0:
        parser.error("start must be nonnegative and width must be positive")
    tok = load_tokenizer(Path(model_dir))
    ids = tok.encode(Path(corpus).read_text(), add_special_tokens=False).ids
    bos = tok.encode('', add_special_tokens=True).ids[:1]
    if len(ids) <= start + width:
        parser.error("corpus needs at least one target token after the traced span")
    print('loading', flush=True, file=sys.stderr)
    if args.indexer_budget is not None and not 0 < args.indexer_budget <= max(start + width, 8192):
        parser.error("indexer budget must be positive and no larger than max(context length, 8192)")
    model = load_reference(Path(model_dir), args.indexer_budget)
    lm = model.language_model
    cache = lm.make_cache()
    context = bos + ids[:start]
    t = time.monotonic()
    if context:
        warm = lm(mx.array([context]), cache=cache).logits
        mx.eval(warm)
        del warm
    print(f'prefilled {len(context)} in {time.monotonic()-t:.1f}s', flush=True, file=sys.stderr)
    rows = lm(mx.array([ids[start:start+width]]), cache=cache).logits[0].astype(mx.float32)
    rows = np.array(rows)
    print(f'# model\t{model_dir}')
    print(f'# start_token\t{start}')
    print(f'# bos_anchor\t{",".join(map(str,bos)) or "none"}\tchunks\t{"all" if bos else "none"}')
    print('# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits')
    for j, row in enumerate(rows):
        target = ids[start+j+1]
        nll = float(np.log(np.exp(row-row.max()).sum()) + row.max() - row[target])
        top = np.argsort(-row, kind='stable')[:8]
        print(f'0\t{j}\t{target}\t{nll:.6f}\t'+','.join(map(str,top))+'\t'+','.join(f'{v:.6f}' for v in row[top]))
    print(f'peak_gb={mx.get_peak_memory()/1e9:.3f}', file=sys.stderr)


if __name__ == "__main__":
    main()
