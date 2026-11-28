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
"""Teacher-forced logit trace of a LongCat-Flash-Lite-Sparse checkpoint, by mlx-vlm.

The reference arm for `examples/logit_trace.rs` on `longcat_flash_sparse`
(#2362), which mlx-vlm implements (`mlx_vlm/models/longcat_flash_sparse`) and
mlx-lm does not. It runs the chunk plan `scripts/mlx_lm_logit_trace.py` runs
and writes the same TSV, so `scripts/compare_logit_traces.py` reads the pair
unmodified.

Two negative arms, each of which a trace that measures this family must tell
apart from the real model:

* `--no-forced-columns` clears `index_init_tokens` and `index_local_tokens`
  on every indexer, so the top-k selection runs on scores alone. It only moves
  positions behind more than `index_topk` (2048) keys.
* `--ngram-fusion` replaces the embedding fusion `word + proj / (1 + n)` with
  the `longcat_flash_ngram` form `(word + proj) / (1 + n)`.

Usage:
    longcat_sparse_oracle_trace.py MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256]
        [MAX_CHUNKS=4] [TOPK=8] [PREFILL=0] [--no-forced-columns]
        [--ngram-fusion] [--no-prefix-sharing] > reference.tsv

Needs `mlx`, `mlx-vlm` (any build with `mlx_vlm.models.longcat_flash_sparse`,
for example the vendored `references/mlx-vlm` on `PYTHONPATH`), `tokenizers`
and `numpy`.
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
    ap.add_argument(
        "--no-forced-columns",
        action="store_true",
        help="negative arm: the indexer keeps no sink or local columns",
    )
    ap.add_argument(
        "--ngram-fusion",
        action="store_true",
        help="negative arm: fuse the n-gram embedding as longcat_flash_ngram does",
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
        from mlx_vlm.models.longcat_flash_sparse import language as sparse
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

        if args.ngram_fusion:
            original = sparse.NgramEmbedding.__call__

            def ngram_form(self, input_ids, cache=None):
                # `word + proj / (1 + n)` -> `(word + proj) / (1 + n)`, from
                # the real fusion's output and the word lookup alone.
                fused = original(self, input_ids, cache)
                word = self.word_embeddings(input_ids.astype(mx.int64))
                scale = 1 + self.num_embedders
                return (word + (fused - word) * scale) / scale

            sparse.NgramEmbedding.__call__ = ngram_form

        started = time.perf_counter()
        model = load_model(model_dir)
        language = model.language_model
        if args.no_forced_columns:
            cleared = 0
            for layer in language.model.layers:
                for attn in layer.self_attn:
                    if attn.indexer is not None:
                        attn.indexer.index_init_tokens = 0
                        attn.indexer.index_local_tokens = 0
                        cleared += 1
            if cleared == 0:
                raise OracleError("--no-forced-columns found no indexer to clear")
        log(
            f"loaded {model_dir} with mlx-vlm {mlx_vlm.__version__} in "
            f"{time.perf_counter() - started:.0f}s; no_forced_columns={args.no_forced_columns}"
            f" ngram_fusion={args.ngram_fusion}"
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
    arm = "mlx-vlm"
    if args.no_forced_columns:
        arm += "-no-forced-columns"
    if args.ngram_fusion:
        arm += "-ngram-fusion"
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
