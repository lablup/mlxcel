# PR #2136: A whole-prompt cache hit no longer duplicates the last prompt token

**Date**: 2026-10-06
**Status**: Verified on CUDA (GB10); the M5 tile-padded abort path is not reproducible on this host
**Risk**: Low

## Summary

When a client replays an identical prompt, the prompt cache matches every prompt token. `try_adopt_cached_prefix` restored all `len` tokens, and admission then moved only the prefill cursor back to `len - 1` so the sampler would get a logit row. Prefill therefore forwarded the last prompt token onto a cache that already held it, at position `len` instead of `len - 1`. The PR caps every adopt at `len - 1` on both the snapshot and KV branches, so the re-run token lands where it belongs.

Closes #1760.

## 1. Problem

Issue #1760 left two explanations open: (a) a real duplicate in the cache, or (b) only the forward-width rounding effect, because warm forwards one token where cold forwards the whole prompt. Per-layer cache offsets read right after the admission clamp settle it as (a):

| Checkpoint | Path | prompt_len | offset after clamp |
|---|---|---|---|
| gemma-3-4b-it-4bit | snapshot (truncating restore) | 115 | 115 |
| gemma-4-12b-it-4bit | snapshot (truncating restore) | 119 | 119 |
| qwen3-4b-4bit | paged KV | 128 | 128 |
| llama-3.2-1b-4bit | paged KV | 160 | 160 |

On the KV path a whole-prompt hit needs a prompt length that is a multiple of the paged block (32 by default), because the clone path floors to whole blocks. That matches the 32- and 48-token replays that #2096 saw abort on M5, where the padded mask was built for `len - 1` cached rows over a cache holding `len`.

## 2. Change

- `whole_prompt_reuse_cap(matched_len, prompt_len) = min(matched_len, prompt_len - 1)` is applied right after each store lookup in `src/server/batch/scheduler/prompt_cache.rs`.
- Snapshot branch: when the cap shortens the match, the model is asked `snapshot_truncatable_to(snapshot, len - 1)`. If it agrees, the restore goes through `restore_sequence_state_truncated`. If it refuses (recurrent state, or a sliding ring that has wrapped), the request falls back to cold prefill and records a `layout_constraints` reject with the stored entry length. A partial restore is never installed.
- KV branch: the cap runs before the dense `truncate_to` and the paged block floors, which then install `len - 1` or fewer tokens. A cap below `min_prefix_tokens` declines as `prefix_too_short`.
- The multimodal whole-entry gate (#124 step c) keys on the store's match, not on the capped value. A whole-entry VLM replay therefore still adopts, and its one-token suffix is exactly what the old back-off already forwarded through the token path.
- The admission back-off in `admission.rs` stays as a backstop and logs at warn, since reaching it now means the invariant broke.
- Side effect: `usage.prompt_tokens_details.cached_tokens` and the shared-budget charge report the length actually restored, one less than before for a whole-prompt hit.

## 3. Decision: cap in the adopt, not in admission

The issue offered the same direction. Moving only the prefill cursor cannot work, because the restore has already installed the token. Restoring `len - 1` makes the forward land on the correct slot for every family, and the truncation predicate the store already uses decides whether that is possible. Recording the refusal under the existing `layout_constraints` reason avoids widening the fixed reject enum and the `/v1/cache/stats` schema for one decline site.

## 4. Validation (GB10, CUDA, `mlxcel-server`, prompt cache on, temperature 0, seed 0, 200 tokens)

| Checkpoint | Before | After |
|---|---|---|
| gemma-3-4b-it-4bit | warm diverged at char 226 | 3/3 byte-identical to cold |
| llama-3.2-1b-4bit | diverged at char 230 | 3/3 byte-identical |
| gemma-4-12b-it-4bit | diverged at token 71 | diverges at token 27, a bf16 tie (warm top-2 gap 0.0) |
| qwen3-4b-4bit | diverged at token 76 | diverges at token 82, a bf16 tie (warm top-2 gap 0.0) |
| qwen3.5-0.8b-4bit (recurrent) | identical (boundary-entry hit) | identical |

The Gemma 4 and Qwen3 residuals are the forward-width effect (b): the warm request forwards 1 token (Gemma 4) or 32 tokens (Qwen3, after the paged block floor) where cold forwards a wider chunk. A control that removes the width difference (`--prefill-chunk-size 1` in both arms, boundary snapshot disabled for Gemma 4) separates the two causes. After the fix, both checkpoints are token-identical with bitwise-equal logprobs. Before the fix, both still diverge, and Gemma 4 already differs at the first token's logprob (-0.5 vs -0.625).

Tests:

- `src/server/batch/scheduler_whole_prompt_hit_tests.rs` has three tests on the tiny real Gemma 3. They read the restored offset from the model's own snapshot for an identical replay (truncating), a whole-entry hit through admission (`prefill_start_offset == already_cached_tokens == len - 1`), and a sliding-window refusal that falls back to cold with the reject counted.
- `tests/prompt_cache_e2e.rs::identical_prompt_replay_restores_all_but_the_last_token` is ignored and runs qwen3-0.6b-4bit with `--apc-block-size 1`. It asserts `cached_tokens == prompt_tokens - 1` and byte-identical replies. It fails on the pre-fix binary (20 vs 19).

## 5. Not verified here

- `trinity-nano-preview-4bit` (AFMoE) is not present locally. Qwen3 and Llama 3.2 cover the KV branch instead.
- The M5 broadcast abort needs Metal hardware.
- A real-checkpoint refusal was not reachable through chat requests, because recurrent families hit their shorter boundary entry. The unit test covers that path.

## 6. Related

- #1754: completion-origin snapshot token counts versus the offsets the caches hold. The measurements above also show `offset == token_len - 1` for `length` finishes on Gemma 4 and the KV path.
- #2096 / PR #2097: the pool-backed trim fix that reported the M5 abort.
