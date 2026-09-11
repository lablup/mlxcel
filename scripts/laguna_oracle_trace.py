#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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
"""Teacher-forced Laguna logit trace from the checkpoint's own modeling code.

The out-of-band reference arm for `examples/logit_trace.rs` on the Laguna
family (#1765). It runs the `modeling_laguna.py` that ships inside the
checkpoint (for XS 2.1,
https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/blob/main/modeling_laguna.py),
through `transformers`, and writes the TSV `logit_trace` writes, so
`scripts/compare_logit_traces.py` reads the pair unmodified. Nothing in the
runtime imports this file.

Usage:
    laguna_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--device DEVICE] [--no-prefix-sharing] > oracle.tsv

Needs torch, transformers 5.x, tokenizers and safetensors, plus
`compressed_tensors_nvfp4.py` beside this file. It executes the checkpoint's
Python, exactly as `trust_remote_code=True` would, so only point it at a
checkpoint you trust.

What is reproduced exactly. The six positional arguments mean what they mean
to `logit_trace`, including its fallback to the default for a value that does
not parse. The token stream is the same: `tokenizer.json` with mlxcel's BOS
post-processor rule, the corpus encoded without special tokens, the BOS anchor
taken from `encode("", add_special_tokens=True)[:1]`, the per-chunk `seg`
slice, a BOS in front of every chunk after the first when PREFILL is 0, and a
context pass of `BOS + PREFILL` history tokens in front of every chunk after
the first when it is not (chunk 0 then has neither). Rows, targets, NLL and
the sorted top-k columns are the same.

What is not reproduced, on purpose. `logit_trace` runs the context pass and
the traced forward at CHUNK_TOKENS width because MLX picks a kernel by width.
This oracle has no width-selected kernels, so it computes each chunk from one
causal forward over context plus chunk, and chunks whose token sequence is a
prefix of another chunk's share that forward. The rows are mathematically the
ones `logit_trace` asks for. `--no-prefix-sharing` runs one forward per chunk,
which is how to check that claim on a real run.

Precision. Activations are float32 throughout. BF16 tensors are widened to
float32, which is exact. Every compressed-tensors `nvfp4-pack-quantized` plane
is dequantized as `code * (E4M3(scale) / global_scale)` in float32, which is
the compressed-tensors formula without its final cast to the checkpoint dtype.
The routed experts are decoded one MoE layer at a time, by a forward pre-hook
just before that layer's expert loop runs, and dropped by a forward hook after
it, so the modeling code runs unmodified on ordinary tensors while the dense
routed weights (31B parameters on XS 2.1) never exist at once. The packed
codes stay resident instead, about 25 GB on the device for XS 2.1.

Not emulated, and absent from mlxcel too: the checkpoint's NVFP4 activation
quantization (`input_activations`, `input_global_scale`) and its FP8 KV cache
(`kv_cache_scheme`, `k_scale` / `v_scale`). Both arms quantize weights only.

Arms this oracle cannot cover. The shipped `modeling_laguna.py` scores the
router with a sigmoid and nothing else, and it allocates `self_attn.sink`
without ever reading it in the forward. A config that sets
`moe_router_score_func` to anything but "sigmoid", `moe_router_use_sigmoid:
false`, or `swa_attention_sink_enabled: true` is refused rather than traced
as a different model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compressed_tensors_nvfp4 import (  # noqa: E402
    Checkpoint,
    CheckpointError,
    Nvfp4Decoder,
    PackedExperts,
    decode_linear,
    parameters_on_meta,
    stack_planes,
)

# Quantization state for arithmetic neither arm performs (see the module
# docstring). Every other checkpoint tensor must land in a parameter.
IGNORED_SUFFIXES = (".input_global_scale", ".self_attn.k_scale", ".self_attn.v_scale")


class OracleError(Exception):
    """A checkpoint, config or argument the oracle cannot trace faithfully."""


def usize_arg(raw: str | None, default: int) -> int:
    """`args.get(n).and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)`."""
    if raw is None:
        return default
    digits = raw[1:] if raw.startswith("+") else raw
    return int(digits) if digits and digits.isascii() and digits.isdigit() else default


# --- Tokenizer and chunk plan: a transcription of `examples/logit_trace.rs` ---


def load_tokenizer(model_dir: Path):
    """`tokenizer::load_tokenizer` for a checkpoint that ships `tokenizer.json`."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise OracleError(f"{path} not found; the oracle reproduces the tokenizer.json path only")
    tokenizer = Tokenizer.from_file(str(path))

    # `ensure_bos_post_processor`: install `<bos> $A` when tokenizer_config.json
    # asks for a BOS (or names a Gemma class) and the export does not add one.
    try:
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
    except (OSError, ValueError):
        return tokenizer
    add_bos = config.get("add_bos_token")
    if add_bos is False:
        return tokenizer
    is_gemma = config.get("tokenizer_class") in ("GemmaTokenizer", "GemmaTokenizerFast")
    if add_bos is not True and not is_gemma:
        return tokenizer
    bos = config.get("bos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    if not isinstance(bos, str) or not bos:
        return tokenizer
    bos_id = tokenizer.token_to_id(bos)
    if bos_id is None:
        return tokenizer
    probe = tokenizer.encode("bos probe", add_special_tokens=True).ids
    if probe and probe[0] == bos_id:
        return tokenizer
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos} $A", pair=f"{bos} $A {bos}:1 $B:1", special_tokens=[(bos, bos_id)]
    )
    return tokenizer


@dataclass(frozen=True)
class Chunk:
    index: int
    # `seg[1:]`: the corpus tokens the traced rows score.
    targets: tuple[int, ...]
    # Context pass followed by the traced input, as one causal sequence.
    sequence: tuple[int, ...]
    # Row of `sequence` whose logits predict `targets[0]`.
    first_row: int


def plan_chunks(
    ids: list[int], bos_prefix: list[int], chunk_tokens: int, max_chunks: int, prefill: int
) -> list[Chunk]:
    n_chunks = min((len(ids) - 1) // chunk_tokens, max_chunks)
    if n_chunks <= 0:
        raise OracleError(f"text too short: {len(ids)} tokens")
    chunks = []
    for c in range(n_chunks):
        seg = ids[c * chunk_tokens : (c + 1) * chunk_tokens + 1]
        traced = seg[:-1]
        if prefill > 0:
            # The context pass carries the anchor and the history, and chunk 0
            # has no history, so it gets neither.
            start = c * chunk_tokens
            ctx_from = max(start - prefill, 0)
            context = bos_prefix + ids[ctx_from:start] if ctx_from < start else []
        elif c == 0 or not bos_prefix:
            context = []
        else:
            context = list(bos_prefix)
        chunks.append(Chunk(c, tuple(seg[1:]), tuple(context + traced), len(context)))
    return chunks


def plan_forwards(
    chunks: list[Chunk], share_prefixes: bool
) -> list[tuple[tuple[int, ...], list[Chunk]]]:
    """Group chunks so that one causal forward over the longest sequence of a
    group yields every member's rows. Under PREFILL the chunks whose context
    starts at the corpus head are prefixes of each other."""
    groups: list[tuple[tuple[int, ...], list[Chunk]]] = []
    for chunk in sorted(chunks, key=lambda ch: len(ch.sequence), reverse=True):
        n = len(chunk.sequence)
        home = None
        if share_prefixes:
            home = next((members for seq, members in groups if seq[:n] == chunk.sequence), None)
        if home is None:
            groups.append((chunk.sequence, [chunk]))
        else:
            home.append(chunk)
    return groups


# --- The checkpoint's own modeling code, with float32 weights ---


def import_checkpoint_code(model_dir: Path):
    """Import `configuration_laguna.py` / `modeling_laguna.py` from the
    checkpoint directory as one package, without writing bytecode into it."""
    for name in ("configuration_laguna.py", "modeling_laguna.py"):
        if not (model_dir / name).is_file():
            raise OracleError(
                f"{model_dir / name} not found; the oracle runs the checkpoint's own modeling code"
            )
    package = (
        "_laguna_checkpoint_" + hashlib.sha256(str(model_dir.resolve()).encode()).hexdigest()[:12]
    )
    module = types.ModuleType(package)
    module.__path__ = [str(model_dir.resolve())]
    sys.modules[package] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        configuration = importlib.import_module(f"{package}.configuration_laguna")
        modeling = importlib.import_module(f"{package}.modeling_laguna")
    finally:
        sys.dont_write_bytecode = previous
    digest = hashlib.sha256((model_dir / "modeling_laguna.py").read_bytes()).hexdigest()
    return configuration, modeling, digest


def refuse_unimplemented_arms(config: dict) -> None:
    score = config.get("moe_router_score_func")
    if score not in (None, "sigmoid") or config.get("moe_router_use_sigmoid") is False:
        shown = score if score is not None else "softmax (moe_router_use_sigmoid: false)"
        raise OracleError(
            f"the config selects router scoring {shown!r}, but the shipped modeling_laguna.py "
            "scores with a sigmoid only, so its trace would be a different model"
        )
    if config.get("swa_attention_sink_enabled"):
        raise OracleError(
            "the config enables swa_attention_sink_enabled, but the shipped modeling_laguna.py "
            "allocates self_attn.sink and never reads it in the forward"
        )


def build_model(model_dir: Path, device, log):
    import re

    import torch
    from torch import nn

    configuration, modeling, digest = import_checkpoint_code(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    refuse_unimplemented_arms(raw)
    raw.pop("quantization_config", None)
    config = configuration.LagunaConfig.from_dict(raw)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with parameters_on_meta():
        model = modeling.LagunaForCausalLM(config)
    model.eval()

    ckpt = Checkpoint(model_dir)
    # The modeling file's own legacy-key mapping (the correction bias moved
    # from `mlp.experts` to `mlp.gate`), applied to every checkpoint key.
    renames = [
        (re.compile(src), dst)
        for src, dst in getattr(model, "_checkpoint_conversion_mapping", {}).items()
    ]
    key_for_param = {}
    for key in ckpt.keys():
        name = key
        for pattern, replacement in renames:
            name = pattern.sub(replacement, name)
        key_for_param[name] = key

    decode = Nvfp4Decoder(device)
    started = time.perf_counter()
    for name, param in list(model.named_parameters()):
        module_name, _, leaf = name.rpartition(".")
        module = model.get_submodule(module_name)
        if isinstance(module, modeling.LagunaExperts):
            continue
        if name in key_for_param:
            tensor = ckpt.tensor(key_for_param[name]).to(device=device, dtype=torch.float32)
        elif leaf == "weight" and ckpt.has(f"{module_name}.weight_packed"):
            tensor = decode_linear(ckpt, module_name, decode, device)
        else:
            raise OracleError(f"no checkpoint tensor for parameter {name}")
        if tuple(tensor.shape) != tuple(param.shape):
            raise OracleError(
                f"{name}: checkpoint shape {tuple(tensor.shape)}, "
                f"model expects {tuple(param.shape)}"
            )
        module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)

    staged = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, modeling.LagunaExperts):
            continue
        prefix = module_name.rsplit(".", 1)[0]
        experts = PackedExperts(
            decode,
            {
                "gate_up_proj": stack_planes(
                    ckpt, prefix, ("gate_proj", "up_proj"), module.num_experts, device
                ),
                "down_proj": stack_planes(ckpt, prefix, ("down_proj",), module.num_experts, device),
            },
        )
        experts.attach(module, module_name)
        staged.append(experts)
        log(
            f"  {prefix}: {module.num_experts} experts staged "
            f"({time.perf_counter() - started:.0f}s)"
        )

    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta:
        raise OracleError(f"{len(still_meta)} parameters were never loaded, first {still_meta[0]}")
    unread = sorted(k for k in ckpt.unread if not k.endswith(IGNORED_SUFFIXES))
    if unread:
        raise OracleError(
            f"{len(unread)} checkpoint tensors map to no parameter, first {unread[0]}"
        )
    model.to(device)
    return model, staged, digest


# --- Trace ---


def trace(model, staged: list[PackedExperts], groups, topk: int, device, log):
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
                raise OracleError(f"vocabulary {logits.shape[1]} is smaller than topk {topk}")
            for chunk in members:
                rows = logits[chunk.first_row : chunk.first_row + len(chunk.targets)].to(
                    torch.float32
                )
                targets = torch.tensor(chunk.targets, dtype=torch.long, device=device)
                nll = -torch.log_softmax(rows, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                top = torch.topk(rows, topk, dim=-1)
                results[chunk.index] = (
                    nll.cpu().tolist(),
                    top.indices.cpu().tolist(),
                    top.values.cpu().tolist(),
                )
            del logits
            if not any(experts.materialized for experts in staged):
                raise OracleError("no MoE layer ran; the routed experts were never decoded")
            log(
                f"  forward {n + 1}/{len(groups)}: {len(sequence)} tokens, "
                f"{len(members)} chunk(s), {time.perf_counter() - started:.1f}s"
            )
    return results


def default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    ap.add_argument("--device", help="torch device (default: mps, else cuda, else cpu)")
    ap.add_argument(
        "--no-prefix-sharing",
        action="store_true",
        help="one forward per chunk, to check that shared prefixes change nothing",
    )
    args = ap.parse_args(argv)

    chunk_tokens = usize_arg(args.chunk_tokens, 256)
    max_chunks = usize_arg(args.max_chunks, 4)
    topk = usize_arg(args.topk, 8)
    prefill = usize_arg(args.prefill, 0)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        import safetensors  # noqa: F401
        import tokenizers
        import torch
        import transformers
    except ModuleNotFoundError as err:
        log(
            f"error: {err.name} is not installed here; run this with a Python that has torch, "
            "transformers, tokenizers and safetensors (the mlx-lm benchmark venv has all four)"
        )
        return 1

    try:
        if chunk_tokens == 0 or topk == 0:
            raise OracleError("CHUNK_TOKENS and TOPK must be positive")
        model_dir = Path(args.model_dir)
        device = torch.device(args.device or default_device())
        text = Path(args.text_file).read_text()
        tokenizer = load_tokenizer(model_dir)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        bos_prefix = tokenizer.encode("", add_special_tokens=True).ids[:1]
        chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, prefill)
        groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)

        log(f"loading {model_dir} on {device} in float32")
        started = time.perf_counter()
        model, staged, digest = build_model(model_dir, device, log)
        log(
            f"loaded in {time.perf_counter() - started:.0f}s; "
            f"{len(chunks)} chunks in {len(groups)} forward(s)"
        )
        results = trace(model, staged, groups, topk, device, log)
        for chunk in chunks:
            for j, nll in enumerate(results[chunk.index][0]):
                if not math.isfinite(nll):
                    raise OracleError(
                        f"chunk {chunk.index} position {j} produced a non-finite NLL; refusing to "
                        "write a trace that cannot be compared"
                    )
    except (OracleError, CheckpointError) as err:
        log(f"error: {err}")
        return 1

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    out.write(
        f"# oracle\tmodeling_laguna.py\tsha256\t{digest}\ttransformers\t{transformers.__version__}"
        f"\ttorch\t{torch.__version__}\ttokenizers\t{tokenizers.__version__}\n"
    )
    out.write(f"# dtype\tfloat32\tdevice\t{device}\n")
    out.write(
        "# weights\tbf16 widened to float32; compressed-tensors NVFP4 decoded as code * E4M3 scale "
        "/ global in float32; activation and KV-cache quantization not emulated\n"
    )
    out.write(f"# corpus\t{args.text_file}\ttokens\t{len(ids)}\n")
    out.write(
        f"# chunks\t{len(chunks)}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}\n"
    )
    out.write(
        f"# forwards\t{len(groups)}\tprefix_sharing\t{str(not args.no_prefix_sharing).lower()}\n"
    )
    out.write("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits\n")
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
