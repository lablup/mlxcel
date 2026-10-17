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
"""Teacher-forced logit trace of an MLX checkpoint, computed by mlx-lm.

The out-of-band reference arm for `examples/logit_trace.rs` on any checkpoint
mlx-lm can load (#2351). It loads MODEL_DIR with `mlx_lm.load`, runs the same
chunk plan `logit_trace` runs, and writes the TSV `logit_trace` writes, so
`scripts/compare_logit_traces.py` reads the pair unmodified. Nothing in the
runtime imports this file.

Usage:
    mlx_lm_logit_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--no-prefix-sharing] > reference.tsv

Needs `mlx` and `mlx-lm` (with the `mlx-cuda-13` wheel on a CUDA host) plus
`tokenizers` and `numpy`. The tokenizer, the BOS anchor and the chunk plan
are the ones `scripts/laguna_oracle_trace.py` transcribes from
`examples/logit_trace.rs`, imported from it rather than copied, so the two
oracles cannot drift apart on which rows they score.

What differs from `logit_trace`. mlx-lm runs every forward in the checkpoint's
own activation dtype, as `mlxcel` does, so this is a same-precision reference,
not a float32 oracle; a disagreement at a decided position is a real
difference between the two runtimes. Like `laguna_oracle_trace.py`, each
chunk is computed from one causal forward over its context plus its traced
rows, and chunks whose token sequence is a prefix of another chunk's share
that forward (`--no-prefix-sharing` runs one forward per chunk).
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
    load_tokenizer,
    plan_chunks,
    plan_forwards,
    usize_arg,
)


def trace(model, groups, topk: int, log):
    import mlx.core as mx
    import numpy as np

    results = {}
    for n, (sequence, members) in enumerate(groups):
        started = time.perf_counter()
        logits = model(mx.array([list(sequence)]))[0].astype(mx.float32)
        mx.eval(logits)
        logits = np.array(logits)
        if logits.shape[0] != len(sequence):
            raise OracleError(f"forward returned {logits.shape} for {len(sequence)} tokens")
        if logits.shape[1] <= topk:
            raise OracleError(f"vocabulary {logits.shape[1]} is smaller than topk {topk}")
        for chunk in members:
            rows = logits[chunk.first_row : chunk.first_row + len(chunk.targets)]
            peak = rows.max(axis=-1, keepdims=True)
            lse = peak[:, 0] + np.log(np.exp(rows - peak).sum(axis=-1))
            targets = np.array(chunk.targets)
            nll = lse - rows[np.arange(len(targets)), targets]
            order = np.argsort(-rows, axis=-1, kind="stable")[:, :topk]
            values = np.take_along_axis(rows, order, axis=-1)
            results[chunk.index] = (nll.tolist(), order.tolist(), values.tolist())
        log(
            f"  forward {n + 1}/{len(groups)}: {len(sequence)} tokens, "
            f"{len(members)} chunk(s), {time.perf_counter() - started:.1f}s"
        )
    return results


def quantization_summary(model_dir: Path) -> str:
    """The top-level quantization record and how many per-layer entries carry
    `global_scale: true`, so a trace says which export it read."""
    try:
        config = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return "unknown"
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return "none"
    flagged = sum(1 for v in quant.values() if isinstance(v, dict) and v.get("global_scale"))
    top = {k: v for k, v in quant.items() if not isinstance(v, dict)}
    return f"{json.dumps(top, sort_keys=True)}\tglobal_scale_layers\t{flagged}"


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
        import mlx_lm
        import numpy  # noqa: F401
        import tokenizers
        from mlx_lm import load
    except ModuleNotFoundError as err:
        log(f"error: {err.name} is not installed here; run this with a Python that has mlx-lm")
        return 1

    try:
        if chunk_tokens == 0 or topk == 0:
            raise OracleError("CHUNK_TOKENS and TOPK must be positive")
        model_dir = Path(args.model_dir)
        # Raw UTF-8, matching `examples/logit_trace.rs`'s `std::fs::read_to_string`.
        text = Path(args.text_file).read_bytes().decode("utf-8")
        tokenizer = load_tokenizer(model_dir)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        bos_prefix = tokenizer.encode("", add_special_tokens=True).ids[:1]
        chunks = plan_chunks(ids, bos_prefix, chunk_tokens, max_chunks, prefill)
        groups = plan_forwards(chunks, share_prefixes=not args.no_prefix_sharing)

        log(f"loading {model_dir} with mlx-lm {mlx_lm.__version__} on {mx.default_device()}")
        started = time.perf_counter()
        model, _ = load(str(model_dir))
        log(
            f"loaded in {time.perf_counter() - started:.0f}s; "
            f"{len(chunks)} chunks in {len(groups)} forward(s)"
        )
        results = trace(model, groups, topk, log)
        for chunk in chunks:
            for j, nll in enumerate(results[chunk.index][0]):
                if not math.isfinite(nll):
                    raise OracleError(
                        f"chunk {chunk.index} position {j} produced a non-finite NLL; refusing to "
                        "write a trace that cannot be compared"
                    )
    except OracleError as err:
        log(f"error: {err}")
        return 1

    out = sys.stdout
    out.write(f"# model\t{args.model_dir}\n")
    out.write(
        f"# oracle\tmlx-lm\t{mlx_lm.__version__}\tmlx\t{mx.__version__}"
        f"\ttokenizers\t{tokenizers.__version__}\n"
    )
    out.write(f"# device\t{mx.default_device()}\n")
    out.write(f"# quantization\t{quantization_summary(model_dir)}\n")
    out.write(f"# corpus\t{args.text_file}\ttokens\t{len(ids)}\n")
    out.write(
        f"# chunks\t{len(chunks)}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}\n"
    )
    out.write(
        f"# forwards\t{len(groups)}\tprefix_sharing\t{str(not args.no_prefix_sharing).lower()}\n"
    )
    out.write("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits\n")
    bos_anchor = ",".join(str(t) for t in bos_prefix) or "none"
    anchored = "all" if bos_prefix else "none"
    out.write(f"# bos_anchor\t{bos_anchor}\tchunks\t{anchored}\n")
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
