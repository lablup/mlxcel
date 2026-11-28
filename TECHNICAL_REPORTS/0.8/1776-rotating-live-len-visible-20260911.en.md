# Technical Report: PR #1776 - fix(models): size sliding masks from visible_len, not seq_len

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed (exaone_moe validated by unit guard only; no checkpoint exists)
**Languages**: Rust
**Risk Level**: Low (mask width on FP16 rotating caches after restore or decode growth; real-checkpoint output unchanged)

---

## Executive Summary

Three sliding-window mask lookups still sized the mask from `RotatingKVCache::seq_len()`: `AnyKVCache::live_len` in exaone_moe and two copies of `first_cache_live_len` in Gemma 4. PR #1752 had already shown for Gemma 3 that this is the wrong accessor. `seq_len()` is the physical buffer length, and it runs ahead of `offset` after a decode step grows the ring by 256 slots, or after a `trim` rewinds `offset` and leaves the buffer as it was. The append, however, concatenates only `visible_len()` prior keys. All three sites now read `visible_len()`, the duplicated Gemma 4 lookup is gone, and the two guards that used to pass vacuously now fail on the old expression.

The issue asked for one question to be settled by test, not assumed: did Gemma 4 actually produce wrong output, given that its masks pass through `trim_mask_to_keys`? It did not. The right-crop of the too-wide mask equals the correctly sized one value for value, and a real multi-turn run on `gemma-4-12b-it-4bit` is token-identical before and after the fix.

---

## Problem Statement

Each of the three sites carried a comment saying `seq_len()` reported the live window, a claim that entered with #430. The two existing guards could not catch the error. Their fixtures only prefilled, so `physical == offset`. Even with a decode step, the old fixture's 4-token prefill in a 6-slot window landed at `offset == window - 1`, where the append's own clamp makes both accessors give the same mask.

Reachability differs by family. exaone_moe is latent: it declares no snapshot support, and its `LanguageModel` impl rebuilds fresh caches on every forward, so no rotating cache survives a call. Gemma 4 reaches the stale-buffer state through snapshot restore and model-owned caches. It never crashed, because `attend` crops every mask to the trailing key columns.

---

## Change Summary

- `exaone_moe.rs` `AnyKVCache::live_len` and `gemma4.rs` `first_cache_live_len` return `visible_len()` for a rotating cache. Their comments explain which states make `seq_len()` run ahead and scope the append's behaviour to FP16 caches.
- `Gemma4StageModel` had a private copy of the lookup. `execute_hidden` now calls the free function, whose parameter becomes `&[Cache]`, so one definition and one guard cover both Gemma 4 sites.
- The guards `sliding_cache_live_len_matches_returned_keys` and `first_cache_live_len_sliding_matches_returned_keys` use a 512-slot window, decode once before the multi-token append, and assert both `physical > offset` and `offset < window - 1`. Together these force the two accessors apart. With the fix reverted, both fail by name: "sized a sliding mask with 265 key columns, but the append returned 10 keys".
- `pre_fix_sliding_mask_right_crop_is_the_visible_band` builds the pre-fix mask, crops it with `trim_mask_to_keys`, and compares it against a band built from logical positions, independently of the mask helpers. It does this for a decode-grown buffer and a trimmed one, with appends inside the window and across it. A mutation that crops from the left fails it with a causal leak.

---

## Technical Decisions

**Establish the Gemma 4 outcome by test, then by a real run.** The band depends only on `k - q - kept_prior`, and `trim_mask_to_keys` keeps the trailing columns, so cropping the leading surplus reproduces the narrow mask exactly. The test proves this for the reachable states. The real run proves the model path reaches those states and that the output is unchanged.

**One consumer does change, and the change is a fix.** The forward's caller-mask rebuild check (`sliding_returned`) now sees the narrower count. An unbuffered FP16 cache whose buffer runs ahead of `offset` now keeps a correctly sized caller mask, such as a padding mask or a draft tree, instead of replacing it with plain causal masks. Nothing reaches this by default: Gemma 4 does not use padded prefill, cold batched prefill runs at offset 0, and tree verify runs on buffered caches unless `enable_speculative_buffer` fails on a non-FP16 rotating mode.

---

## Validation

- Workspace gate on the fix commit: 123 binaries, 11,016 passed, 0 failed, 359 ignored. Workspace clippy with `-D warnings`, fmt and the `make verify-*` contract checks are clean. The follow-up commit changes comments only.
- Real checkpoint: `gemma-4-12b-it-4bit` through `mlxcel-server` at temperature 0 and seed 0, four turns of 128 tokens with the assistant text fed back. The pre-fix (bb146efa) and post-fix binaries were run with the prompt cache on and off.
  - The trace shows T2 and T3 adopting restored prefixes of 311 and 517 tokens, under the 1023-token clamp, so the changed lookup ran.
  - Pre-fix and post-fix are token-identical on all four turns, with the cache on and off. They are also identical in a variant whose T3 append runs from 517 to 1,261 tokens across the 1,024 window.
  - Cache on and cache off differ at T3 identically in both binaries (94 vs 121 generated tokens). This is the existing model-owned-cache class tracked in #1346.

---

## Learning Points

- **A guard can be vacuous twice.** Adding the missing decode step did not make the old fixture discriminating, because its offset sat exactly at the clamp where both accessors agree. The new guard asserts the precondition that makes the two expressions differ (`offset < window - 1`), so a future fixture edit cannot quietly restore the vacuous case.
- **A wrong value behind a tolerant consumer is still worth fixing.** Gemma 4's output never depended on the wrong width. The caller-mask rebuild check did, however, and it silently discarded correctly sized masks in a path the default configuration does not exercise.

---

## Follow-ups

- A turbo-quantized sliding cache (`update_turbo4_concat`) returns the whole physical buffer from its append, including zero-filled growth slots and rolled-back draft keys. With this change its mask is narrower than the keys, so `trim_mask_to_keys` rebuilds it. The band is the same as before, plus one warning per sliding layer. The append reading the full physical buffer predates this PR.
