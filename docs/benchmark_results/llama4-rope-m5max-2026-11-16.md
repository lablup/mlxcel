# Llama 4 traditional llama3-scaled RoPE (M5 Max, 2026-11-16)

Issue #3120. Llama 4's attention rotated Q and K with `fast_rope(traditional = false, base = rope_theta, scale = 1.0)`. mlx-lm builds the same rope with `initialize_rope(head_dim, rope_theta, traditional=True, scaling_config=rope_scaling)` ([`mlx_lm/models/llama4.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama4.py)), and Scout's `rope_scaling` is `{rope_type: llama3, factor: 16, low_freq_factor: 1, high_freq_factor: 1, original_max_position_embeddings: 8192}`. The branch routes every Llama 4 RoPE call through `Llama4Rope` (`src/models/llama4_helpers.rs`), which pairs adjacent dimensions and takes the `llama3` table from the shared `rope_utils` path.

## Setup

- Host: Apple M5 Max, 128 GB, macOS 27. GPU shared with another job under a lock, so only one GPU process ran at a time.
- Checkpoint: `models/llama-4-scout-17b-16e-instruct-4bit`.
- Arms: `main` is `3efb3fb2` (includes #3117's chunked-decode fix); `branch` is this change on the same base. Both are release builds of `mlxcel` and `examples/logit_trace`.
- Reference: stock mlx-lm 0.31.3 on mlx 0.31.2, through `scripts/mlx_lm_logit_trace.py`. The installed transformers rejects Scout's config (`attn_temperature_tuning` is an int), so a wrapper replaces `mlx_lm.load` with `mlx_lm.utils.load_model`; the trace script tokenizes with `tokenizers` itself, so nothing else changes.
- Corpus: `docs/architecture.md` followed by `docs/installation.md` (35,061 tokens).
- Gate: `scripts/compare_logit_traces.py --decided 2`.

## Teacher-forced trace against mlx-lm

| Shape | Positions | Decided | main: decided disagreement | branch: decided disagreement | mlx-lm ppl | main ppl | branch ppl |
|---|---|---|---|---|---|---|---|
| Prefill width 1280, context up to 2304 | 5120 | 1965 | 83.82% | 0.05% (1) | 14.213 | 1868.2 | 14.217 |
| Prefill width 4608, context up to 9216 | 9216 | 3643 | 87.90% | 0.00% | 11.803 | 2392.9 | 11.800 |
| Decode width 1, context 512 to 2560 | 2048 | 750 | 83.33% | 0.00% | 16.351 | 1819.6 | 16.349 |
| Batched decode width 1, B=2, same rows | 2048 | 750 | 83.47% | 0.00% | 16.351 | 1815.7 | 16.367 |
| Decode width 1, context 4608 to 9216 | 4608 | 1886 | 92.42% | 0.00% | 10.268 | 3059.5 | 10.260 |

The 9216-token rows cross Scout's 8192 attention chunk and `original_max_position_embeddings`. The branch's one decided disagreement (prefill width 1280) has a reference gap of 3.06 and the reference's token at the branch's rank 2; every other branch disagreement sits where mlx-lm's own top two were less than 1.05 apart. Overall top-1 disagreement on the branch is 0.52 to 0.73%, all in the rounding class.

The decode rows use `MLXCEL_TRACE_BATCH_ROWS`, whose chunk `c` starts at token `PREFILL + c * CHUNK` with `PREFILL` tokens of history. With `PREFILL = CHUNK` that is the mlx-lm plan's chunk `c + 1`, so the reference was traced with one extra chunk and its rows renumbered by one before comparing. The single-row decode arms also set `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1` (`forward_with_sequence_id`); the B=2 arm runs `forward_batched_with_ids`, which reaches `CxxAttention::forward_batched_decode_rows` and so `Llama4Rope::apply_batched`.

A 9216-wide single forward (one chunk of 9216 tokens) aborted on `main` with a Metal out-of-memory command-buffer error, so the long prefill cell uses two 4608-wide forwards.

Per-cell numbers, including overall top-1 rates: `llama4-rope-m5max-2026-11-16/trace-summary.csv`.

## Greedy sample

`mlxcel generate -p "Explain in a few paragraphs how rotary position embeddings encode token positions in a transformer." -n 256 --temp 0`. Each arm produced the same text in both repetitions.

- main: "Embeddings, specifically token embeddings, play a crucial role in transformers, enabling models understand context words meaning words words, enabling models understand meaning words, enabling models understand understand words, words, understand understand, understand, understand, ..." (repeats "understand," to the end).
- branch: "Rotary Position Embeddings (RoPE) is a technique used in transformer models to encode token positions in a more efficient and effective way. Unlike traditional position embeddings, which add a fixed vector to each token's representation based on its position in the sequence, RoPE uses a rotation matrix to dynamically update the token representations based on their position. ..." (three coherent paragraphs).

## Decode throughput

Same command, 26 prompt tokens, 256 new tokens, 75 s cooldown before each run, order main, branch, branch, main.

| Run | Arm | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 1 | main | 66.93 | 40.73 |
| 2 | branch | 98.38 | 47.53 |
| 3 | branch | 93.65 | 45.60 |
| 4 | main | 97.98 | 46.93 |

Run 1 is the first load after the trace jobs and is slow in both phases. Leaving it out, main decodes at 46.9 tok/s and the branch at 45.6 to 47.5 tok/s, so the change has no measurable decode cost; `fast_rope_with_freqs` runs the same MLX kernel as `fast_rope` with a precomputed table. Raw values: `llama4-rope-m5max-2026-11-16/generation.csv`.
