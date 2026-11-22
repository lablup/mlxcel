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
"""Teacher-forced Arcee (AFM-4.5B) logit trace from Hugging Face transformers.

The out-of-band reference arm for `examples/logit_trace.rs` on the dense
Arcee family (`model_type: arcee`, #1719). Neither mlx-lm nor mlx-vlm
implements `arcee`, so the maintainer accepted transformers'
`ArceeForCausalLM` (`src/transformers/models/arcee/modeling_arcee.py`) as the
parity reference. The script writes the TSV `logit_trace` writes, so
`scripts/compare_logit_traces.py` reads the pair unmodified. Nothing in the
runtime imports this file.

Usage:
    arcee_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--arm reference|relu|no-yarn] [--dtype float32|bfloat16]
        [--device DEVICE] [--no-prefix-sharing] [--report-range] > oracle.tsv

Run it on CPU with, for example:

    uv run --python 3.12 --with torch --with transformers --with safetensors \\
        python scripts/arcee_oracle_trace.py models/afm-4.5b docs/architecture.md 256 8 8

The token stream and chunk plan are the ones `logit_trace` builds, through
the transcription in `laguna_oracle_trace.py`; that module's docstring says
what is and is not reproduced (each chunk is one causal forward over context
plus chunk, and chunks that are prefixes of one another share a forward
unless `--no-prefix-sharing` is passed).

Arms. `reference` is the checkpoint as transformers loads it. The two
negative arms each remove one Arcee-specific convention, so a comparison
against mlxcel shows that the trace can tell the convention apart:

- `relu`: every `ArceeMLP` uses plain ReLU instead of `relu2` (squared ReLU).
- `no-yarn`: `rope_scaling` is dropped, so RoPE runs at the unscaled
  frequencies with no YaRN attention factor.

Precision. `--dtype float32` (the default) widens the bf16 weights to
float32, which is exact, and runs every activation in float32.
`--dtype bfloat16` runs the checkpoint in its own dtype. `--report-range`
prints, to stderr, the largest absolute value the squared-ReLU activation and
the residual stream reach over the traced forwards; that is the evidence for
whether float16 can hold the activations.
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

ARMS = ("reference", "relu", "no-yarn")


def build_model(model_dir: Path, arm: str, dtype, device, log):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type != "arcee":
        raise OracleError(
            f"{model_dir} is model_type {config.model_type!r}, not 'arcee'"
        )
    if arm == "no-yarn":
        # transformers 5 keeps RoPE under `rope_parameters`; older releases
        # read `rope_scaling`. Reset whichever this release uses to the
        # default rotation at the checkpoint's base.
        # (In transformers 5 `rope_scaling` is an alias of `rope_parameters`,
        # so only one of the two may be written.)
        params = getattr(config, "rope_parameters", None)
        if isinstance(params, dict):
            theta = params.get("rope_theta", getattr(config, "rope_theta", None))
            config.rope_parameters = {"rope_type": "default", "rope_theta": theta}
        else:
            config.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, dtype=dtype)
    model.to(device)
    model.eval()
    if arm == "relu":
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.ReLU()
    rotary = model.model.rotary_emb
    log(
        f"rope_type={getattr(rotary, 'rope_type', '?')} "
        f"attention_scaling={float(rotary.attention_scaling):.6f} "
        f"inv_freq[0]={float(rotary.inv_freq[0]):.6g} inv_freq[-1]={float(rotary.inv_freq[-1]):.6g}"
    )
    return model


def install_range_hooks(model):
    """Record max |relu2(up_proj(x))| and max |residual| per forward."""
    peaks = {"activation": 0.0, "residual": 0.0}

    def on_down(_module, inputs):
        peaks["activation"] = max(peaks["activation"], float(inputs[0].abs().max()))

    def on_layer(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        peaks["residual"] = max(peaks["residual"], float(hidden.abs().max()))

    for layer in model.model.layers:
        layer.mlp.down_proj.register_forward_pre_hook(on_down)
        layer.register_forward_hook(on_layer)
    return peaks


def trace(model, groups, topk: int, device, log):
    import torch

    results = {}
    with torch.inference_mode():
        for n, (sequence, members) in enumerate(groups):
            started = time.perf_counter()
            ids = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = model(input_ids=ids, use_cache=False).logits[0]
            if logits.shape[0] != len(sequence):
                raise OracleError(
                    f"forward returned {tuple(logits.shape)} for {len(sequence)} tokens"
                )
            if logits.shape[1] <= topk:
                raise OracleError(
                    f"vocabulary {logits.shape[1]} is smaller than topk {topk}"
                )
            for chunk in members:
                rows = logits[
                    chunk.first_row : chunk.first_row + len(chunk.targets)
                ].to(torch.float32)
                targets = torch.tensor(chunk.targets, dtype=torch.long, device=device)
                nll = (
                    -torch.log_softmax(rows, dim=-1)
                    .gather(-1, targets.unsqueeze(-1))
                    .squeeze(-1)
                )
                top = torch.topk(rows, topk, dim=-1)
                results[chunk.index] = (
                    nll.cpu().tolist(),
                    top.indices.cpu().tolist(),
                    top.values.cpu().tolist(),
                )
            del logits
            log(
                f"  forward {n + 1}/{len(groups)}: {len(sequence)} tokens, "
                f"{len(members)} chunk(s), {time.perf_counter() - started:.1f}s"
            )
    return results


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
    ap.add_argument("--arm", choices=ARMS, default="reference")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument(
        "--no-prefix-sharing",
        action="store_true",
        help="one forward per chunk, to check how closely a shared forward agrees with it",
    )
    ap.add_argument(
        "--report-range",
        action="store_true",
        help="print the peak squared-ReLU activation and residual magnitude to stderr",
    )
    args = ap.parse_args(argv)

    chunk_tokens = usize_arg(args.chunk_tokens, 256)
    max_chunks = usize_arg(args.max_chunks, 4)
    topk = usize_arg(args.topk, 8)
    prefill = usize_arg(args.prefill, 0)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        import tokenizers
        import torch
        import transformers
    except ModuleNotFoundError as err:
        log(
            f"error: {err.name} is not installed here; run this with "
            "`uv run --python 3.12 --with torch --with transformers --with safetensors`"
        )
        return 1

    try:
        if chunk_tokens == 0 or topk == 0:
            raise OracleError("CHUNK_TOKENS and TOPK must be positive")
        model_dir = Path(args.model_dir)
        device = torch.device(args.device)
        dtype = getattr(torch, args.dtype)
        # Raw UTF-8, matching `examples/logit_trace.rs`'s `std::fs::read_to_string`.
        text = Path(args.text_file).read_bytes().decode("utf-8")
        tokenizer = load_tokenizer(model_dir)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        bos_prefix = tokenizer.encode("", add_special_tokens=True).ids[:1]
        chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, prefill)
        groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)

        log(f"loading {model_dir} on {device} in {args.dtype}, arm {args.arm}")
        started = time.perf_counter()
        model = build_model(model_dir, args.arm, dtype, device, log)
        peaks = install_range_hooks(model) if args.report_range else None
        log(
            f"loaded in {time.perf_counter() - started:.0f}s; "
            f"{len(chunks)} chunks in {len(groups)} forward(s)"
        )
        results = trace(model, groups, topk, device, log)
        if peaks is not None:
            log(
                f"range: max|relu2(up_proj)|={peaks['activation']:.1f} "
                f"max|residual|={peaks['residual']:.1f} (float16 max 65504)"
            )
        for chunk in chunks:
            if not all(math.isfinite(v) for v in results[chunk.index][0]):
                raise OracleError(
                    f"chunk {chunk.index} produced a non-finite NLL; refusing to write a trace "
                    "that cannot be compared"
                )
    except OracleError as err:
        log(f"error: {err}")
        return 1

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    out.write(
        f"# oracle\ttransformers-ArceeForCausalLM-{args.arm}\ttransformers\t"
        f"{transformers.__version__}\ttorch\t{torch.__version__}"
        f"\ttokenizers\t{tokenizers.__version__}\n"
    )
    out.write(f"# dtype\t{args.dtype}\tdevice\t{device}\n")
    out.write(f"# corpus\t{args.text_file}\ttokens\t{len(ids)}\n")
    out.write(
        f"# chunks\t{len(chunks)}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}\n"
    )
    sharing = str(not args.no_prefix_sharing).lower()
    out.write(f"# forwards\t{len(groups)}\tprefix_sharing\t{sharing}\n")
    out.write("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits\n")
    bos_anchor = ",".join(str(t) for t in bos_prefix) or "none"
    out.write(
        f"# bos_anchor\t{bos_anchor}\tchunks\t{'all' if bos_prefix else 'none'}\n"
    )
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
