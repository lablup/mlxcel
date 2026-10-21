# PR #2493: Qwen3-Next multi-token cache-forward fix

**Date**: 2026-10-20
**Status**: Completed
**Risk Level**: Medium
**Languages**: Rust

---

## Executive Summary

Closes #2463. `Qwen3NextModel::forward_for_sequence` reset the model-owned GatedDeltaNet and KV caches on every multi-token forward with no sequence id, so one T=5 forward behind a populated cache ran from empty state while five T=1 forwards continued it. The reset is removed and the `supports_chunked_prefill()==false` override added by #2295 is dropped; chunked prefill for Qwen3-Next now works end to end with per-sequence state that continues across chunks.

---

## 1. Problem Statement

### 1.1 Background

#2295 added a no-ID multi-token reset inside `forward_for_sequence` (originating in #601, which protected cache-less callers reusing an instance across prompts) plus a `supports_chunked_prefill()==false` opt-out for Qwen3-Next. The reset fires on `seq_id.is_none() && seq_len > 1`, which is exactly the shape of a speculative burst and of chunked prefill through plain `LanguageModel::forward`.

### 1.2 Existing Issues

- **Symptom**: width-5 real-checkpoint trace read perplexity 987.4 (near-random) while the stepwise oracle read high-teens. Widths 1 and 256 were unaffected because the reset is T>1-only.
- **Latent wrongness**: one multi-token forward behind a cache started from empty state, so the first round's prefill and verify continuation were both wrong for Qwen3-Next on main.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Corrupted generation on any model-owned family using the fallback slot with T>1 | High | High (deterministic on Qwen3-Next) |
| Chunked prefill unavailable for Qwen3-Next long prompts | Medium | High (opted out by #2295) |

---

## 2. Technical Review

### 2.1 Security

Clean. The removed reset was `seq_id.is_none()`-gated and never executed on any server path; every server entry point passes a real SequenceId and cross-request isolation is structural (`Engine::open` inserts fresh per-sequence state, `Engine::close` releases it, `with_or_create_sequence_state(Some(id), ...)` never falls back to the shared internal slot). No cross-request state leak.

### 2.2 Performance

- The single-offset prefill mask (built from the first full-attention layer's own offset) is valid for every FA layer; model-owned layouts hold no dense pool caches, so FA offsets cannot diverge. GDN layers ignore the mask.
- GDN state is fixed-size (conv_state, state_cache) with no sequence-length growth; the per-sequence slot is created once at the first chunk and moved per forward, no per-chunk state reallocation.
- Chunked prefill turns `[1, prompt_len, vocab]` logit transients into `[1, chunk_len, vocab]`, so transient memory is equal or better than the old whole-prompt path at equal total lm-head work.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: no. The classic speculative walk's observable behavior on Qwen3-Next improves within a generation; `reset_runtime_state` semantics are preserved and now pinned by a test.
- **New Dependencies**: none.

### 2.4 Code Quality

- Test Coverage: the rewritten `qwen3_next_loaded_model_chunking_contract_is_safe` pins all three contract legs (supports_chunked_prefill true; no-ID continuation linear offset 5 after 2+3, which fails on main; reset_runtime_state -> 0; stable seq-id retention). Two new eight-layer tests pin logits equivalence through `LanguageModel::forward` (the offline CLI entry) and through the inherent forward with explicit masks.
- The fixture exercises both layer kinds (interval 4: layers 3 and 7 full attention), dense MLP, tied embeddings, and the established byte-decode readback pattern.

---

## 3. Technical Decisions

### 3.1 Remove the reset instead of narrowing it

**Context:** #601 added the reset to protect cache-less no-ID callers reusing a model across prompts; #2295 opted Qwen3-Next out of chunked prefill because of it.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep the reset, gate on prompt-lookup support | Smaller diff | Keeps the fallback slot broken for every multi-token caller; the repo's flow never reuses the fallback across prompts |
| **Chosen: remove the reset, drop the opt-out** | Restores first-round prefill and verify continuation; enables chunked prefill end to end; matches the qwen3_5 family pattern | Requires the stale-state protection to be provided by the per-sequence lifecycle instead |

**Rationale:** In the repo's real flow the fallback multi-token path has one production caller, the deprecated classic speculative walk (one generation per process). Per-sequence isolation is structural, so the reset's protection was dead code everywhere it mattered and a correctness bug on Qwen3-Next.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/models/qwen3_next.rs`**
- Removed the no-ID multi-token reset block in `forward_for_sequence`.
- Deleted the `supports_chunked_prefill()==false` override (trait default true applies).

**File: `src/models/qwen3_next_tests.rs`**
- Rewrote the chunking contract test; added the acceptance test `qwen3_next_multi_token_forward_behind_cache_matches_stepwise` and a state-update twin through the inherent forward with explicit masks.

---

## 5. Verification

| Check | Result |
|---|---|
| Acceptance test on main | FAILED (divergence 0.025385 > 0.002) |
| Acceptance test on branch | PASS |
| Real checkpoint qwen3-next-80b-a3b-instruct-4bit, width 5, docs/architecture.md, args 5 60 8 512, MLX_ENABLE_TF32=0 | 300 rows, mean NLL 2.9325, ppl 18.77 (reverted arm 987.4) |
| Widths 1 and 256, both arms | byte-identical |
| Scoped: qwen3_next / gated_delta | 16/16, 11/11 |
| Full gate at 2ff66014 | 161 test binaries build 0, clippy 0, doc 0, contract 0 |
| Comment-fix commit 008b8ad8 | fmt clean, clippy clean, 16/16 qwen3_next with TF32=0 |
| Security review | clean; no cross-request leak; no performance regression |

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 (+ comment fix in the same 2 files) |
| Lines added | +251 (main commit), +7 (comment fix) |
| Lines deleted | -23 (main commit), -7 (comment fix) |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `2ff66014` | fix | fix(models): continue Qwen3-Next cache across multi-token forwards |
| `008b8ad8` | docs | docs(models): fix orphaned qwen3_next comment and in_proj_qkvz formula |

---

## 7. Follow-up Actions

### Required

- [ ] Issue #2495 (filed): the deprecated classic speculative walk cannot rewind a model-owned family after a rejected round (pre-existing, out of this PR's scope).
- [ ] Record-only: pre-existing no-op `make_caches` comments in `src/server/batch/scheduler/prefill.rs:755-761` and `planned_prefill.rs:295-298` reset nothing for model-owned families.

### Monitoring Required

- Chunked-prefill behavior on Qwen3-Next in the server's planned prefill (first release after merge).

---

## Appendix

### A. Review verdicts

pr-reviewer: PASS, no CRITICAL/HIGH (three MEDIUM/LOW findings, none a regression; the MEDIUM became #2495). pr-security-checker: clean on security and performance. pr-finalizer applied the two comment-text fixes in `008b8ad8`.
