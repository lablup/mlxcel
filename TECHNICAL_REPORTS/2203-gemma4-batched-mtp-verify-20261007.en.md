# PR #2203: Gemma 4 batched MTP verify decode-exact on the row-wise geometries

**Date**: 2026-10-07
**Status**: Implemented and verified on CUDA (GB10); Metal verification pending in #2158
**Risk**: Medium (a new per-batch-row branch in the shared Gemma 4 forward, gated on `mtp_verify` and the row-wise layer flag; batched MTP serving on CUDA 31B behind `MLXCEL_ENABLE_MTP_BATCH`)

## Summary

PR #2185 made the B=1 linear Gemma 4 MTP adapter byte-identical to classic decode on CUDA. The batched adapter (`Gemma4MtpBatchedTargetAdapter`) never set `sinks.mtp_verify`, so its verify skipped the row-wise path, and serving declined every B>1 window for the 31B and the CUDA 12B. `b1_batched_baseline_probe` failed on main: 31B rows 1, 2 and 3 at tokens 1, 9 and 17.

1. The batched adapter marks verify forwards and leaves prefill unmarked, as the linear adapter does.
2. On the row-wise geometries every row is prefilled on its own cache over the classic partition (`mtp_prefill_ranges`, last-row LM head via `prefill_mtp_chunk_explicit_cache`), and the row caches are stacked into the shared `[B, ...]` cache with each row's history first and any shortfall as a zero tail (`stack_prefilled_rows`). A short row is then a row whose valid end lags the shared offset, the layout a divergent accept already leaves, so ragged bursts need no left padding and every row keeps its standalone RoPE frame.
3. A B>1 verify runs each row as its own `[1, K]` call: projections, `head_rows_like_decode` RoPE at the row's logical offset, `attend_verify_rows` on keys physically cut to history `[0, ve[r])` plus the block, `o_proj`, the MLP (`finish_layer`) and the LM head. Only the K/V write into the shared cache stays batched, so finalize, rollback and the drafter slab contract are unchanged. Sliding layers use the ring cursor at the row's own length, `(cursor - (offset - ve[r])) mod window`.
4. `run_mtp_burst_batched` replaces the unconditional row-wise decline with `RowWiseBatchedWindow::decline_reason`, and batched windows carry per-row history-boundary splits.

## Design notes

- Rejected: widening the `b == 1` gates to a `[B, K]` call. `B * K >= 8` moves the quantized matmuls from qmv to qmm and the compiled GeGLU gate counts the same rows.
- Ceiling: the buffered sliding cache compacts against the shared offset (`buffered_planned_drop`), so after a compaction a row lagging that offset has lost the oldest keys of its own window. The forward reports that (`row_verify_inexact`) and the adapter returns `DraftFailed` rather than emit tokens classic decode would not. The serving gate keeps windows below it: `max_prompt_len + K * (max_tokens + 1) <= sliding_window + 32`, where `K * max_tokens` covers rows that finished early and keep advancing up to K positions a round.
- Gate: CUDA only (Metal 31B is row-wise but unmeasured, and ROCm builds are excluded for the same reason), B <= 4 (measured widths), dense FP16 caches (the stacker's requirement), prompts within the sliding window. Every other window declines before drafter IO, so no row reaches a client-facing error.
- The decision came from measurement: classic batched decode on the 31B barely scales with B on GB10 (8.3, 8.8, 8.9 tok/s at B=1, 2, 4), so a per-row verify that costs about B single-row verifies a round still wins.

## Verification on GB10 (CUDA, release, `--features cuda`)

Driver 580.178.04, kernel 7.0.0-1019-nvidia, MLX pin `81ba1c6a`.

- Full `speculative_parity --ignored --test-threads=1`: 9 passed, 0 failed, including `b1_batched_baseline_probe` on both pairs and the new `greedy_parity_mtp_gemma4_batched_matches_classic` (B=2 and B=4, equal-length and ragged, both pairs, 24 of 24 rows byte-identical to per-sequence classic greedy, no near-tie tolerance, a divergent accept round in every case). Main: 7 of 8 plus the failing probe.
- Lib `gemma4 speculative_burst`: 327 passed. The hermetic per-row test fails under a shared-ring-cursor mutation and with the row mode disabled.
- Throughput, three interleaved rounds per width, classic null arm, identical output bytes in every arm: 31B B=2 +75% to +77%, B=4 +53% to +55%, null -2.3% to +3.1%. A round with the shipped binary engaged batched windows with identical bytes.
- Graph cache at `MLX_CUDA_GRAPH_CACHE_SIZE=100`: 20 batches each of B=1 (4000 tokens) and B=2 (8000 tokens) served without the lifetime-miss abort.
- Clippy `-D warnings` and `cargo fmt --check` clean.

## Findings outside the change

A batched MTP window never forms in default serving: two or more queued requests take the classic batched prefill when `--max-batch-prefill` > 1 (the default); `--ignore-eos` gives every row a token bias, which the window collector refuses; unseeded requests draw different seeds. The 12B (`Gemma4Unified`) clamps `max_batch_size` to 1, so the decision is moot for it. In-process parity tests ran MLX's bare 400-entry graph cache instead of the server's 2000; they now apply the server defaults.

## Not verified on this host

Metal. Listed on #2158: the verify flag on the batched adapter, the per-row prefill and stacking, the per-row verify and LM head at B>1, and the `finish_layer` extraction. The serving gate keeps Metal declined.
