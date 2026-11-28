# PR #3419: Inkling-Small MTP exact at every width, and why it is still not faster

**Date**: 2026-11-25
**Status**: Draft PR open (closes #3280). Validated on an M3 Ultra (512 GB) with `inkling-small-mlx-4bit` and the native bf16 MTP head; narrow lib suites, clippy and the contract checks pass

## Executive Summary

Issue #3280 asked two things of first-generation Inkling MTP on Inkling-Small: make the verify block equal single-token decode, and either make MTP faster than classic decode or document that it is not. PR #3415 had made the block exact on the bf16 0.6B checkpoint but never ran the quantized Small target or its MTP head.

On Small the probe passed at widths 2 to 8 and declined at 10, the default MTP width. A second inexactness sat outside the probe, in how a partial accept was rolled back. Both are fixed, and drafter-on output now equals classic decode byte for byte. Removing the rollback's replay also makes width 2 about 1.5x faster. Classic decode still wins at every width, because the head's acceptance is low (1.43 to 1.63 tokens per round), the same as mlx-vlm measures. MTP stays opt-in.

## 1. Width 10: MLX's SDPA changes kernels past 8 queries

The survey harness from #3415 (`survey_block_vs_chain`) on `inkling-small-mlx-4bit`:

| Width | Before | After |
|---|---|---|
| 2, 3, 4, 8 | equal | equal |
| 10 | 150040 of 400116 logit bytes differ; layer 0 at row 3 (short draws), layer 5 (after 600 and 2100 tokens) | equal |
| 16 | not run | equal |

The row-invariant products from #3415 cover the dense projections, the convolutions and the experts, and they hold on the quantized target because the experts already run one position at a time. What changes at width 10 is attention: MLX's fused SDPA takes its per-query vector kernel for at most 8 queries and a tiled kernel above that, and the two order the key reduction differently. Rows 0 to 2 happened to match; row 3 did not.

The fix is in `InklingAttention::attend_row`. Inside a verify block longer than 8 rows the queries are split into groups of up to 8, each attending over the full key set, and the outputs are concatenated. The sliding-window row-by-row path from #3415 still runs first when it applies. Outside a verify block nothing changes.

## 2. Rollback: the replay was not the chain

After verifying K tokens and accepting `a` drafts, the adapter used to restore a snapshot taken before the verify and replay the `a + 1` kept tokens as one forward. That forward ran outside the verify scope, so a replay of two or more rows went through `conv1d` and the batched expert kernels, not the single-token step's kernels. The probe only compares consecutive full blocks with the chain, so it never saw this. At width 3 the old prose stream left classic decode.

The verify block now records each short convolution's padded input window `[kernel - 1 + K]` in `InklingLayerCache::block_tails` (only inside a verify block; any other forward clears them). `keep_verified_prefix` slices each convolution state to the `kernel - 1` rows before row `a + 1` and trims the KV tail by `K - a - 1`. A sliding layer keeps `K + W - 1` keys after the block, so `a + W` remain after the trim and the next step's front trim yields the chain's window. Because a row-invariant verify block equals the chain, the kept state is the chain's state.

`forward_speculative_for_sequence` skips the snapshot when the block is in the row-invariant scope (row kernel on, 2 to 16 rows), and `verify_finalize` takes the in-place path when no snapshot was captured. `MLXCEL_INKLING_ROW_INVARIANT=0` and wider blocks still use snapshot and replay. `InklingShortConv` moved to `inkling_layer/conv.rs` to keep `attention.rs` under 500 lines.

## 3. Throughput

128 greedy tokens, two prompts, ABAB, the same binary for both arms (before = previous code path):

| Prompt | Mode | Before tok/s | After tok/s | Mean accepted (after) |
|---|---|---|---|---|
| prose | classic | | 45.18, 45.01 | |
| prose | width 2 | 25.57, 26.15 | 39.82, 39.53 | 1.427 |
| prose | width 3 | 23.30, 23.94 | 34.53, 34.61 | 1.608 |
| prose | width 10 | 10.35, 10.38 | 12.71, 12.73 | 1.608 |
| code | classic | | 44.41, 44.39 | |
| code | width 2 | 27.06, 26.98 | 41.54, 41.68 | 1.512 |
| code | width 3 | 24.41, 24.42 | 34.85, 34.99 | 1.628 |
| code | width 10 | 10.50, 10.62 | 12.71, 12.77 | 1.628 |

A width-2 round costs about 36 ms against a 22 ms decode step and emits 1.43 to 1.51 tokens, so it reaches 0.88x (prose) and 0.94x (code) of classic. Almost every accepted draft is the first one: widths 3 and 10 add 0.1 to 0.2 tokens per round while drafting up to 8 more.

## 4. Acceptance

The drafter was compared with mlx-vlm 466069f5 `speculative/drafters/inkling_mtp`. Seed from block 0 over the accepted prefix, draft step `k` on block `min(k, 7)`, a snapshot restore before the accepted prefix goes back through block 0, the target's pre-norm hidden, and the target's final norm (upstream copies `model.llm.norm.weight` into its split drafter), muP scale, LM head and vocabulary trim all match. An uncommitted experiment that started the post-seed drafts at block 1 (DeepSeek-style depth indexing) left the accepted counts at 48 or 49 per 128 tokens. The low acceptance belongs to the head, and upstream reports the same 1.59 tokens per round.

## 5. Tests

- `mtp_partial_accept_keeps_the_single_token_chain_state_on_metal`: widths 3 and 10, every accepted count. After verify and finalize, the next two steps must equal, byte for byte, a model that decoded the kept tokens one at a time. It fails on the previous code at width 3, accepted 1.
- `the_verify_block_matches_the_chain_on_metal` now covers widths 10 and 16. It fails on the previous code at width 10 (layer 1, global).
- `mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly` now pins the replay path explicitly (row-invariant off).
- The ignored real-checkpoint probe test now includes width 10 and passes on both the 0.6B and the 4-bit Small checkpoints.

## 6. Not done

- No width gives a speedup, so the issue's "throughput gain" branch is answered by documentation. The remaining costs are the 2-row verify (the dense products read once, the experts twice) and the drafter's two bf16 LM-head passes per round. Neither changes acceptance.
- The #3417 adaptive verify width is a DFlash round-loop policy. Inkling MTP runs the MTP loop, which has its own adaptive controller, so `exact_at_every_verify_width` was not wired.
- `trim_state` still clears the convolution states. It serves padded prefill only, not prompt lookup: when this report was written prompt lookup refused first-generation Inkling at its gate, and issue #3422 later enabled it through `rewind_decode_appends`, which uses the same windows.
