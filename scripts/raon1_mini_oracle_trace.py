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
"""Raon-1 mini reference arm, from the checkpoint's own Transformers code.

Two modes, both on `KRAFTON/Raon-1-mini` (revision `6691a076`) run through the
`modeling_raon1_mini.py` that ships in the checkpoint, in bfloat16 (the model
dtype the reference is written for), with `set_reasoning_effort("max")`. The
`max` alpha is 0, so the reasoning-effort steering vectors are not applied
and the forward is the steering-free architecture mlxcel implements (#3225;
steering is #3435).

    raon1_mini_oracle_trace.py trace MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256]
        [MAX_CHUNKS=4] [TOPK=8] [--no-prefix-sharing] > reference.tsv

writes the teacher-forced TSV `examples/logit_trace.rs` writes (PREFILL 0
only), so `scripts/compare_logit_traces.py` reads the pair unmodified.

    raon1_mini_oracle_trace.py generate MODEL_DIR PROMPT_FILE [MAX_TOKENS=64]
        [--thinking] > greedy.json

renders PROMPT_FILE as one user turn through the checkpoint's chat template
(`enable_thinking=False` unless `--thinking`), decodes greedily without the
generation config's presence penalty, and prints the prompt length, the
generated ids and their text as JSON.

The tokenizer and chunk plan come from `laguna_oracle_trace.py`, which
transcribes `examples/logit_trace.rs`. Needs torch, transformers 5.16.1 or
newer (the `mxfp8` quantizer), accelerate, tokenizers and safetensors. The
checkpoint's Python runs from a hashed temporary copy; only point this at a
checkpoint you trust.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laguna_oracle_trace import (  # noqa: E402
    OracleError,
    load_tokenizer,
    plan_chunks,
    plan_forwards,
    usize_arg,
)

CODE_FILES = ("configuration_raon1_mini.py", "modeling_raon1_mini.py")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_checkpoint_code(model_dir: Path):
    """Import the checkpoint's two `.py` files from a hashed temporary copy
    (see `laguna_oracle_trace.import_checkpoint_code` for why a copy)."""
    for name in CODE_FILES:
        if not (model_dir / name).is_file():
            raise OracleError(f"{model_dir / name} not found; fetch the checkpoint with `hf download`")
    package = "_raon1_mini_" + hashlib.sha256(str(model_dir.resolve()).encode()).hexdigest()[:12]
    digests: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="mlxcel-raon-oracle-") as tmp:
        tmp_dir = Path(tmp)
        for name in CODE_FILES:
            data = (model_dir / name).read_bytes()
            digests[name] = hashlib.sha256(data).hexdigest()
            (tmp_dir / name).write_bytes(data)
        module = types.ModuleType(package)
        module.__path__ = [str(tmp_dir)]
        sys.modules[package] = module
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            importlib.import_module(f"{package}.configuration_raon1_mini")
            modeling = importlib.import_module(f"{package}.modeling_raon1_mini")
        finally:
            sys.dont_write_bytecode = previous
    return modeling, digests


def build_model(model_dir: Path):
    import torch

    modeling, digests = import_checkpoint_code(model_dir)
    started = time.perf_counter()
    model = modeling.Raon1MiniForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()
    model.set_reasoning_effort("max")
    alphas = {layer.layer_idx: layer.effort_alpha for layer in model.model.layers if layer.is_moe}
    if any(alphas.values()):
        raise OracleError(f"steering is still on after set_reasoning_effort('max'): {alphas}")
    log(f"loaded {model_dir} in {time.perf_counter() - started:.0f}s (bfloat16, cpu, alpha 0)")
    return model, digests


def run_trace(args) -> int:
    import tokenizers
    import torch
    import transformers

    chunk_tokens = usize_arg(args.chunk_tokens, 256)
    max_chunks = usize_arg(args.max_chunks, 4)
    topk = usize_arg(args.topk, 8)
    model_dir = Path(args.model_dir)
    text = Path(args.text_file).read_bytes().decode("utf-8")
    tokenizer = load_tokenizer(model_dir)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos_prefix = tokenizer.encode("", add_special_tokens=True).ids[:1]
    chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, 0)
    groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)
    model, digests = build_model(model_dir)

    results = {}
    with torch.inference_mode():
        for n, (sequence, members) in enumerate(groups):
            started = time.perf_counter()
            logits = model(input_ids=torch.tensor([sequence]), use_cache=False).logits[0]
            for chunk in members:
                rows = logits[chunk.first_row : chunk.first_row + len(chunk.targets)].float()
                targets = torch.tensor(chunk.targets).unsqueeze(-1)
                nll = -torch.log_softmax(rows, dim=-1).gather(-1, targets).squeeze(-1)
                top = torch.topk(rows, topk, dim=-1)
                results[chunk.index] = (nll.tolist(), top.indices.tolist(), top.values.tolist())
            log(f"  forward {n + 1}/{len(groups)}: {len(sequence)} tokens, {time.perf_counter() - started:.0f}s")
    for chunk in chunks:
        if not all(math.isfinite(v) for v in results[chunk.index][0]):
            raise OracleError(f"chunk {chunk.index} produced a non-finite NLL")

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    out.write(
        f"# oracle\tmodeling_raon1_mini.py\tsha256\t{digests['modeling_raon1_mini.py']}"
        f"\ttransformers\t{transformers.__version__}\ttorch\t{torch.__version__}"
        f"\ttokenizers\t{tokenizers.__version__}\treasoning_effort\tmax\n"
    )
    out.write("# dtype\tbfloat16\tdevice\tcpu\n")
    out.write(f"# corpus\t{args.text_file}\ttokens\t{len(ids)}\n")
    out.write(f"# chunks\t{len(chunks)}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t0\n")
    out.write(f"# forwards\t{len(groups)}\tprefix_sharing\t{str(not args.no_prefix_sharing).lower()}\n")
    out.write("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits\n")
    anchor = ",".join(str(t) for t in bos_prefix) or "none"
    out.write(f"# bos_anchor\t{anchor}\tchunks\t{'all' if bos_prefix else 'none'}\n")
    for chunk in chunks:
        nll, top_ids, top_logits = results[chunk.index]
        for j, target in enumerate(chunk.targets):
            ids_s = ",".join(str(i) for i in top_ids[j])
            lg_s = ",".join(f"{v:.6f}" for v in top_logits[j])
            out.write(f"{chunk.index}\t{j}\t{target}\t{nll[j]:.6f}\t{ids_s}\t{lg_s}\n")
    return 0


def run_generate(args) -> int:
    import torch
    from transformers import PreTrainedTokenizerFast

    model_dir = Path(args.model_dir)
    max_tokens = usize_arg(args.max_tokens, 64)
    # The fast tokenizer class directly: `AutoTokenizer` would follow the
    # config's `auto_map` and prompt for remote code on stdout.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    prompt = Path(args.prompt_file).read_bytes().decode("utf-8")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=args.thinking,
        tokenize=True,
        return_dict=False,
    )
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    model, _ = build_model(model_dir)
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            torch.tensor([list(ids)]),
            max_new_tokens=max_tokens,
            do_sample=False,
            # 0.0, not None: None keeps generation_config.json's 1.5, which
            # `Raon1MiniForCausalLM._get_logits_processor` applies even greedy.
            presence_penalty=0.0,
            eos_token_id=[259202, 259200],
        )
    generated = out[0, len(ids) :].tolist()
    log(f"generated {len(generated)} tokens in {time.perf_counter() - started:.0f}s")
    json.dump(
        {
            "prompt_tokens": len(ids),
            "prompt_ids": list(ids),
            "generated_ids": generated,
            "text": tokenizer.decode(generated, skip_special_tokens=False),
        },
        sys.stdout,
        ensure_ascii=False,
        indent=1,
    )
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("trace")
    tr.add_argument("model_dir")
    tr.add_argument("text_file")
    tr.add_argument("chunk_tokens", nargs="?")
    tr.add_argument("max_chunks", nargs="?")
    tr.add_argument("topk", nargs="?")
    tr.add_argument("--no-prefix-sharing", action="store_true")
    ge = sub.add_parser("generate")
    ge.add_argument("model_dir")
    ge.add_argument("prompt_file")
    ge.add_argument("max_tokens", nargs="?")
    ge.add_argument("--thinking", action="store_true")
    args = ap.parse_args(argv)
    try:
        return run_trace(args) if args.mode == "trace" else run_generate(args)
    except OracleError as err:
        log(f"error: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
