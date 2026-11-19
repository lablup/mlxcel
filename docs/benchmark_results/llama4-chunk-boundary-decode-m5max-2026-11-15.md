# Llama 4 chunked decode across the chunk boundary (issue #3105), M5 Max, 2026-11-15

Llama 4's MoE (local) layers attend only to keys in the query's own `attention_chunk_size` block (8192 for llama-4-scout). mlx-lm rebuilds that chunk mask on every forward, decode included. mlxcel's decode step attended unmasked over the chunked cache's visible window, which after the front trim holds the last 8192 keys, so from position 8192 on it also attended to keys of the previous chunk. `ChunkedKVCache::attend` now slices the window to the query's chunk start for a decode step and keeps the #3100 per-chunk split for multi-token steps, for the single-row, batched-rows and batched-decode routes.

Host: Apple M5 Max, 128 GB, macOS 27. Two release builds (`metal,accelerate`): `main` at `ab2f1d07` and the branch on top of it, each copied aside before the other was built. Every GPU run held the shared GPU lock. Checkpoint: `models/llama-4-scout-17b-16e-instruct-4bit`.

## Teacher-forced trace across the boundary

`examples/logit_trace` with `MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1` (width-one `forward_with_sequence_id` decode, the CLI's decode path) and `MLXCEL_TRACE_BATCH_ROWS=2` (width-one `forward_batched_with_ids`, the server's batched decode, row 0 emitted), over the `make_prompt.py 28000` corpus behind an 8001-token context (BOS plus 8000 corpus tokens). The 448 traced query positions run 8001 to 8448, so the boundary at 8192 sits at traced row 191. Two references, both applying the mlx-lm chunk rule in a single prefill forward over 8449 tokens:

- mlxcel's one-pass prefill on the branch (`logit_trace ... 8448 1 8 8000`, no batch rows), the #3100 per-chunk path that #3104 matched against the dense chunk mask.
- mlx-lm 0.31.3 (`scripts/mlx_lm_logit_trace.py`) with its RoPE set to the one mlxcel's Llama 4 uses today (see below), so the two runtimes differ only in attention masking. Over all 8448 positions it agrees with the mlxcel prefill on every decided position (0 / 941, 60 undecided flips).

`scripts/compare_logit_traces.py --decided 2`, split at the boundary (`logit-trace-summary.csv`):

| Candidate | Query positions | Decided disagreement vs mlxcel prefill | vs mlx-lm (mlxcel RoPE) | Perplexity, reference to candidate |
|---|---|---|---|---|
| main, width-1 decode | 8001-8191 | 0 / 4 | 0 / 4 | 22.01 to 22.02 |
| main, width-1 decode | 8192-8448 | **20 / 75** (130 / 257 top-1) | **20 / 74** | 5.86 to **27.87** |
| branch, width-1 decode | 8001-8191 | 0 / 4 | 0 / 4 | 22.01 to 22.02 |
| branch, width-1 decode | 8192-8448 | 0 / 75 (2 / 257 top-1) | 0 / 74 | 5.86 to 5.86 |
| main, B=2 batched decode | 8192-8448 | **20 / 75** | | 5.86 to **27.88** |
| branch, B=2 batched decode | 8192-8448 | 0 / 75 | | 5.86 to 5.86 |

Main and the branch produce bit-identical trace rows up to row 190 and first differ at row 191, query position 8192: before the boundary the visible window is the query's own chunk and the branch runs the same unmasked SDPA over the same keys. From 8192 on, main's decode disagrees with both references at half the positions and its perplexity rises 4.8x, while the branch stays within the width-1-versus-prefill rounding class (largest gap at a disagreement 0.08 to 0.11).

## Generation and decode throughput

`mlxcel generate --temp 0 --profile` under `/usr/bin/time -l`, 75 s cooldown before every run, arm order reversed in the second repetition (`generate-ab.csv`).

| Cell | Decode tok/s, main (rep 1, rep 2) | Decode tok/s, branch | Peak, main to branch | Output |
|---|---|---|---|---|
| 554-token prompt, 128 tokens | 45.55, 44.94 | 45.56, 44.77 | 61.55 to 61.55 GB | byte-identical |
| 8077-token prompt, 400 tokens | 27.04, 27.12 | 28.09, 28.37 | 66.21 to 66.12 GB | diverges after the context passes 8192 |

The short cell (the pp512/tg128 shape with this corpus's tokenizer) never reaches the boundary and runs the same ops in both arms. In the long cell the context crosses 8192 after about 115 generated tokens; past it the branch's chunked layers attend to the keys of the new chunk only (at most 286 here, not 8192), so decode is 4% faster. Prefill is unchanged (6.05 / 6.04 s to 6.11 / 5.83 s). Each arm repeats its own text exactly; main and the branch part at the same character in both repetitions.

## Not fixed here: Llama 4 RoPE differs from mlx-lm

Stock mlx-lm and mlxcel disagree on 46% of decided positions across the whole trace, boundary or not (3685 / 8020 over positions 1 to 8448; perplexity 1.18 for mlx-lm against 14.05 for mlxcel), and both arms' greedy output on this corpus degenerates into repetition. Swapping mlx-lm's RoPE for mlxcel's removes the whole gap (0 / 941 above). mlx-lm builds the chunked layers' RoPE with `traditional=True` and the checkpoint's `llama3` scaling (`factor` 16, `original_max_position_embeddings` 8192); `CxxAttention::from_weights` in `src/models/llama4.rs` uses `fast_rope(..., traditional=false, ...)` with `rope_scale: 1.0` and ignores `rope_scaling`. That is a separate defect from the chunk boundary and is left for its own issue.
