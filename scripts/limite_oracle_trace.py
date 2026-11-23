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
"""Teacher-forced logit trace of a Limite checkpoint, computed by mlx-vlm.

The reference arm for `examples/logit_trace.rs` on Paradigma Limite (#2360),
which mlx-vlm implements (`mlx_vlm/models/limite`) and mlx-lm does not. It
runs the chunk plan `scripts/mlx_lm_logit_trace.py` runs and writes the same
TSV, so `scripts/compare_logit_traces.py` reads the pair unmodified. The
values traced are the sigmoid-softcapped head outputs, which is what both
runtimes hand to the sampler.

`--drop-mudd` is the negative arm: it clears the model's MUDD tap table, so
the mixing layers read `rms_norm(x)` and keep `x` as their residual base, like
every other layer. A trace that cannot tell this arm from the real model is
not measuring the mixer.

Usage:
    limite_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4]
        [TOPK=8] [PREFILL=0] [--drop-mudd] [--no-prefix-sharing] > reference.tsv

Needs `mlx`, `mlx-vlm` (any build with `mlx_vlm.models.limite`, for example
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
    ap.add_argument("--drop-mudd", action="store_true", help="negative arm without the MUDD mix")
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
        if args.drop_mudd:
            language.model.tap_idx = {}
            language.model.retained = set()
        log(
            f"loaded {model_dir} with mlx-vlm {mlx_vlm.__version__} in "
            f"{time.perf_counter() - started:.0f}s; drop_mudd={args.drop_mudd}"
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
    arm = "mlx-vlm-drop-mudd" if args.drop_mudd else "mlx-vlm"
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
