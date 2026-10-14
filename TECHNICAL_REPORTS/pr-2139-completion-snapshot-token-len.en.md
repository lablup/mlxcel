# PR #2139: Completion snapshots are keyed by the tokens their state holds

**Date**: 2026-10-06
**Status**: Verified on CUDA (GB10) across Gemma 4, Gemma 3, a GatedDeltaNet hybrid and a dense KV family. Speculative burst donations not measured.
**Risk**: Medium (changes donated entry lengths for every family; disables the decode lookahead for model-owned families)

## Summary

Decode forwards a sampled token on the following step. Every finish except a merged-EOS stop therefore leaves the last generated token out of the model state, but the completion donation keyed its entry by `prompt ++ generated`. An exact-entry restore then resumed prefill one position past the state it installed. The PR donates only the consumed tokens (`SequenceInfo::generated_in_state`). It also stops the decode lookahead pipeline from running model-owned families, whose teardown trim never reached their state. That trim gap pushed Gemma 3 snapshots the other way, holding one or two positions more than they claimed.

Closes #1754.

## 1. Problem

Issue #1754 predicted the off-by-one from code reading. A later comment reported llama-4-scout chat turns that were byte-identical to cold, which seemed to contradict it. Per-layer offsets read from each snapshot at capture settle the question:

| Finish | Before | After |
|---|---|---|
| `length`, `stop` string (Gemma 4; Gemma 3 with sync decode; Llama 3.2 KV) | offset = tokens - 1 | equal |
| merged EOS (Gemma 4) | equal | equal |
| `length`, default flags (Gemma 3, lookahead active) | offset = tokens + 1 | equal |

The chat routes hide the defect. Turn 2 re-renders the assistant reply (the template adds an end-of-turn scaffold, and re-tokenization is not always canonical), so `prompt ++ generated` is usually not a prefix of turn 2. The llama-4-scout comment's restore used the 1822-token warm-up entry rather than the 1817-token completion entry. The raw completion routes do reach the completion entry exactly. So does the warm-up extend, which restores the longest stored snapshot and forwards only the delta after it.

## 2. Change

- `SequenceInfo::eos_terminated` is set at the three merged-EOS sites in `decode_tick.rs`, where the EOS is sampled but never pushed. `SequenceInfo::generated_in_state()` returns all generated tokens in that case and drops the last one otherwise.
- The classic decode finalize and the finished-in-prefill donation pass `generated_in_state()`. The eos-at-prefill donation already passed an empty slice.
- `lookahead_params` returns `None` when `model.sequence_state_layout().backend == ModelOwned`. The earlier check read the allocated backend, which the paged override sets to `PagedKvCache` for Gemma 3, AFMoE and Llama 4 even though their caches live in `ModelOwnedSequenceState`. `apply_lookahead_trim` iterates `get_caches_mut`, which is empty for those families, so their speculative appends were never unwound.
- The speculative burst donations still pass the full committed tail. Whether a burst's final token is in the state depends on the drafter's verify rollback, which was not measured.

## 3. Decision: fix the donation, not the restore

The issue offered two repairs: donate one token fewer, or resume prefill from the restored offset. Recurrent families (GatedDeltaNet, Mamba) carry no offset to resume from, so only the donation side can be applied uniformly. The bookkeeping rule depends on whether the terminating sampled token was pushed, not on `FinishReason`. A structured-output stop also finishes as `Stop`, but it pushes its token.

## 4. Validation (GB10, CUDA, `mlxcel-server`)

Protocol: turn 1 on `/v1/completions`, then turn 2 = turn 1 + reply + suffix. Warm turn 2 runs in the same process with an exact completion-entry restore. Cold turn 2 runs in a fresh process. Both arms use `--prefill-chunk-size 1`, so forward-width rounding cannot differ. The pre-fix arm is the same binary with the old behavior toggled by a temporary switch, which was removed before commit.

| Checkpoint / finish | Before | After |
|---|---|---|
| gemma-4-12b `length` | diverges at token 2, first logprob -0.125 vs -0.625 | identical, max abs dlogprob 0.0 |
| gemma-4-12b stop string | same text, abs dlogprob up to 0.625 | identical, 0.0 |
| gemma-4-12b EOS | identical | identical |
| gemma-3-4b `length`, stop string | diverge at token 1 | identical, 0.0 |
| qwen3.5-0.8b `length` | same text, abs dlogprob 0.125 | identical, 0.0 |
| llama-3.2-1b dense KV `length` (`--parallel 1 --apc-block-size 1`) | diverges at token 1, first logprob -2.97 vs -2.07 | identical, 0.0 |

Lookahead on Gemma 3 4B (n=4 runs per arm, 200 tokens, default flags):

- Before the fix, output differed from `MLXCEL_FORCE_SYNC=1` and decode ran at 84.1 to 84.5 tok/s.
- After the fix, output is byte-identical to force-sync and decode runs at 74.1 to 75.2 tok/s.
- On Llama 3.2 (KV) the lookahead output equals force-sync, so the Gemma 3 difference came from the untrimmed appends and not from the pipeline itself.
- A sequence-aware trim hook (#1755) could re-enable the pipeline for these families.

Unit tests in `src/server/batch/scheduler_completion_snapshot_tests.rs` use the tiny real Gemma 3:

- `length` over decode steps
- the one-decoded-token boundary (finish inside prefill)
- an EOS stop
- a decoding model-owned sequence getting no lookahead params

Each test compares the stored entry's token count with the offset its snapshot carries. Three of the four fail under the old behavior.

## 5. Not verified here

- Speculative burst donations: no drafter run was measured.
- Cancelled finishes follow the same rule (the last pushed token is unforwarded) but were not exercised over HTTP.
- llama-4-scout exceeds this host's 40 GB per-model limit.

## 6. Related

- #1760 / PR #2136: the whole-prompt hit fix. Its measurements first showed the `length` off-by-one.
- #1755: the sequence-aware trim hook for model-owned prefill trims.
- #1346: the natural-versus-allocated backend lesson this gate applies.
