# PR #2440: Clamp ragged batched DFlash accepts to one count

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; a per-row RoPE / divergent-mask port for Qwen 3.5 is not covered (see "Not addressed")
**Risk**: Medium (changes the committed accept count of every batched DFlash round on Qwen 3.5 and removes the per-row zero-tail rollback)

## Summary

Closes #2353. Batched (B > 1) DFlash speculative decoding on Qwen 3.5 / 3.6 / 3.8 (`qwen3_5`, `qwen3_5_moe`, and their VL wrappers) now cuts every round to the shortest live row's accept count before anything is emitted, so emit, bonus token, drafter hidden slice and cache rollback all see one uniform count. B = 4 output now matches four B = 1 runs for 128 tokens.

## Problem

Batched DFlash on Qwen 3.5 diverged from the B = 1 run whenever rows in a round accepted different numbers of draft tokens. Rollback trimmed the rectangular attention cache to the longest row and zeroed the shorter rows' rejected K/V tail. A zeroed key is still attended: `q . 0 = 0`, so it takes `exp(0)` weight in the softmax denominator and dilutes that row's attention on every later step. The output stays fluent, so nothing flags the drift. The GDN state replay and the drafter's hidden context were also sized by the longest row.

Per issue #2353, the Qwen 3.5 verify forward (`qwen3_5.rs:2832`, `:2844`) derives its mask from the shared cache offset and takes no per-row valid end or per-row RoPE offset. Both B > 1 production targets route to this rollback: `src/models/qwen3_5.rs:2905` and `src/vision/qwen3_5_vl.rs:226`, reached from the server through `src/server/batch/dflash_target.rs`.

Upstream fixed the same bug in Blaizzy/mlx-vlm#2113 (commit `3b8f727d`, "Clamp ragged batched speculative acceptance to fix phantom KV"). It clamps per-row acceptance to a uniform count before tokens are emitted, and raises if a ragged batch reaches a rectangular cache rollback.

## Technical decisions

- Clamp instead of masking. The issue rejected porting Gemma 4's masking (`mask_stale_key_gap`, `src/lib/mlxcel-core/src/utils.rs:618`). Gemma 4 can mask because it also keeps a contiguous valid prefix per row and rotates each row's window at its own logical position (`DivergentVerifyRows` at `src/models/gemma4.rs:1666`, `build_divergent_verify_mask` at `:4201`). Without that, later tokens are rotated at the shared offset and the gaps accumulate across rounds. Qwen 3.5 tracks none of it. If clamping later costs measurable throughput, the Gemma 4 layout can be ported in a follow-up.
- The clamp is exact for greedy decoding. The truncated last token of a longer row is an accepted draft token equal to the target argmax.
- Finished rows do not lower the count but are aligned to it, because they stay in the batch.
- Budget truncation happens inside the walk, so a row with one token of budget left yields a uniform accept of 0. Every row then emits one token, and the bonus for the next round is that token.
- The walk's true counts stay in `accept_lens`, so acceptance statistics stay honest. The committed counts and the number of clamped rounds are reported separately.

## Changes

- `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`: `clamp_uniform_acceptance` (port of Blaizzy/mlx-vlm#2113), a pure function next to the walk, cuts every round to the shortest live row's accept before the emit loop. `DFlashBatchedRunOutput` gains `committed_accept_lens` and `clamped_rounds`. The loop debug-asserts that the clamp left one accept count before it rolls back.
- `src/models/qwen3_5.rs`: `rollback_speculative_cache` asserts a uniform slice (`assert_uniform_batched_accepts`) instead of zeroing tails. `zero_per_row_kv_tail` / `rebuild_with_zero_tail` and their tests are removed, and the conv-state recovery uses one window for all rows.
- `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`: the trait doc of `rollback_partial_batched` now states the uniform-accept contract instead of per-row tail zeroing.
- `src/server/batch/dflash_target.rs`: `run_dflash_batched_on_target` logs the walk's and the committed counts at debug level.
- `src/vision/qwen3_5_vl.rs`: doc comment updated to say the rollback requires a uniform accept slice. `src/models/qwen3_5_tests.rs`: the zero-tail tests are replaced by the uniform-accept rollback tests.
- `tests/speculative_parity/qwen35_dflash_batched.rs` (new) and `tests/speculative_parity.rs`: ignored real-checkpoint parity test `greedy_parity_dflash_qwen35_batched_b4_matches_b1`.
- `CHANGELOG.md`: entry for the fix.

Coverage: the DFlash batched loop is reached only by `Qwen35Model` and `Qwen35VLModel`. LFM2, LFM2-VL, Muse Glimmer and Laguna report `supports_batched() == false` and keep the trait's B > 1 panic. Gemma 4 batched MTP runs `MtpBatchedGenerator`, a separate loop with per-row masking, and is not touched.

## Behavior change

Batched DFlash rounds on Qwen 3.5 commit fewer accepted drafts per round whenever rows disagree, in exchange for output that matches B = 1. `rollback_speculative_cache` now panics on a ragged accept slice with B > 1 instead of zeroing tails.

## Verification

GB10, `MLXCEL_SDPA_DETERMINISTIC=1`, under gpu-lock. Checkpoints: `~/models_archive/mlx/qwen3.5-4b-4bit` (target) and `~/models_archive/mlx/qwen3.5-4b-dflash` (drafter).

- Parity test, block 16, 128 tokens: B = 1 self-parity holds. On the unchanged code, rows 0, 1 and 2 first diverge at tokens 14, 54 and 37. With the fix, all four rows match.
- Mean accepted drafts per row-round on those prompts (a correctness run, not a benchmark):

| Run | Mean accepted | Rounds |
|---|---|---|
| B = 1, four prompts | 4.12 / 3.23 / 3.41 / 3.00 | n/a |
| B = 4 before | 1.08 | 101 |
| B = 4 after, walk | 4.11 | 57 (55 clamped) |
| B = 4 after, committed | 1.23 | 57 (55 clamped) |

- Server: `mlxcel-server --model <target> --model-draft <drafter> --draft-kind dflash --draft-block-size 4 --max-batch-prefill 1 --no-prompt-cache`. Two concurrent temperature-0 `/v1/chat/completions` ran as one B = 2 batched burst. Before the fix they differed from two sequential requests. After, they match (twice; 21 of 45 rounds clamped), and also at block 16.
- `rollback_speculative_cache_rewinds_uniform_batched_accepts` builds a two-layer synthetic model (gated-delta plus full attention). It checks that each row of a B = 2 rollback holds the same conv window and gated-delta state as a B = 1 rollback of that row, and that the KV offset matches.
- Revert checks: with the clamp call disabled, three synthetic round-loop tests fail. Without the assert, `rollback_speculative_cache_panics_on_ragged_batched_accepts` fails.
- `drafter::dflash` (94) and `models::qwen3_5_tests` (with ignored) pass, as do clippy `-D warnings` on mlxcel-core and mlxcel lib+tests, fmt, and the contract checks.

## Not addressed

- A per-row RoPE / divergent-mask port for Qwen 3.5 (out of scope in #2353).
- With the default `--max-batch-prefill 4`, two equal-length requests that queue together take classic batched prefill and never reach the batched DFlash burst. This is separate from the fix.
- The batched burst skips the block-versus-chain exactness probe that the B = 1 arm runs. This is separate from the fix.

## Statistics

9 files changed, 810 insertions, 333 deletions. Commits: `0b7ba553` fix(speculative): clamp ragged batched DFlash accepts to one count, `10b8bc29` test(speculative): compare batched Qwen 3.5 rollback with B = 1 rows, `6c6ddb74` docs: note the batched DFlash uniform-accept fix in the changelog.
