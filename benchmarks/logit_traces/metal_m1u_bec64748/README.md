# Metal baseline logit traces for lablup/mlxcel#1809

Teacher-forced logit traces from `examples/logit_trace` on an M1 Ultra (Apple GPU generation 13, so no NAX path), measured at mlxcel `bec64748` with the MLX pin `81ba1c6a`. They are the Metal reference for the cross-backend correctness matrix in lablup/mlxcel#1809; the ROCm traces are produced from the same commit, corpus, and arguments. `METADATA.txt` records the host, OS, pin, and the metallib, binary, and corpus hashes. `RUNS.txt` has the exit status and row count of every run, and `SHA256SUMS` covers every trace.

## What was traced

Four affine 4-bit checkpoints: `qwen3-0.6b-4bit`, `meta-llama-3.1-8b-instruct-4bit`, `qwen3-30b-a3b-4bit`, and `mixtral-8x7b-instruct-v0.1-4bit`. Each is traced over `tests/fixtures/wikitext2_excerpt.txt` at three shapes (`CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL`):

| Tag | Arguments | Positions | Shape it measures |
|---|---|---|---|
| `w1` | `1 128 8 0` | 128 | decode |
| `w8` | `8 80 8 512` | 640 | verify-sized forward behind 512 tokens of context |
| `w256` | `256 2 8 0` | 512 | prefill |

`default` leaves `MLXCEL_FUSED_MOE` unset, so the two MoE checkpoints run the fused MoE kernel where it applies. `fused0` sets `MLXCEL_FUSED_MOE=0` and takes the `gather_qmm` path. Pick the variant that matches the path the ROCm build takes. Mixtral (top-k 2, 8 experts) sorts its expert indices only when `n_tokens * top_k >= 64` (`src/models/switch_layers.rs`), so its `w256` trace is the one that passes through the sorted `gather_qmm` path.

Absolute local paths in the trace headers were rewritten to repository-relative ones. `scripts/compare_logit_traces.py` reads only the position count and the target token of each row, so the rewrite does not affect a comparison.

## Comparing

```bash
python3 scripts/compare_logit_traces.py \
    benchmarks/logit_traces/metal_m1u_bec64748/metal_m1u_bec64748_<model>_<variant>_<tag>.tsv \
    <rocm trace for the same model, variant and tag>
```

Gate on disagreement at decided positions (`--decided 2.0`, the default), as lablup/mlxcel#1809 specifies. Byte identity is not expected across backends.

For scale, fused MoE against `gather_qmm` on this same host differs at one position out of 128 for `qwen3-30b-a3b` at `w1`, and that position is undecided. The other five MoE pairs agree at every position. A kernel swap within one backend moves almost nothing here, which is the context for reading a cross-backend disagreement.

`logit_trace` does not prepend BOS to chunk 0 (lablup/mlxcel#1785). Both backends run the same code, so this does not bias a comparison, but it does shift absolute perplexity.
