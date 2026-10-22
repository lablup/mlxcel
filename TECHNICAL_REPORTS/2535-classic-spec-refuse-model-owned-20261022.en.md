# PR #2535: refuse model-owned families in the classic speculative walk

**Date**: 2026-10-22
**Status**: Completed
**Risk Level**: Medium
**Languages**: Rust

---

## Executive Summary

The deprecated classic `--draft-model` speculative walk drops a rejected draft block by trimming the external KV caches of both models. On a family whose sequence state lives elsewhere (Qwen3-Next's recurrent GDN layers, model-owned caches) that trim rolls nothing back, so later rounds condition on stale positions, and the `can_trim_prompt_cache` asserts pass trivially on the placeholder caches. This PR takes fix option 1 from issue #2495: `SpeculativeGenerator::generate` now refuses such models up front with an explicit error instead of silently corrupting state. User-visible behavior change: the shared rule also refuses families that opt out of padded prefill (Gemma 4 and Llama 4), so those models now get an error on the classic path where they previously ran. No real model-owned checkpoint was run through the CLI.

---

## 1. Problem Statement

### 1.1 Background

Issue #2495 was found in the review of PR #2493 (Qwen3-Next multi-token cache-forward fix). The rejection path trims only the generator's external KV caches, which a model-owned family ignores. The path is deprecated (#2176) and runs one generation per process in the repo's CLI flow, but a library user reusing `SpeculativeGenerator` across prompts on a model-owned model can hit it.

### 1.2 Existing Issues

- After the first rejected round, internal GDN/KV state keeps the rejected tokens and later rounds condition on stale positions.
- The `can_trim_prompt_cache` asserts pass on the placeholder external slices, so there was no fail-fast.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Silent state corruption on model-owned families | Medium (wrong speculative output for library callers) | Library reuse on such models |
| Gemma 4 / Llama 4 now refused on the classic path | Medium (previously running configurations now error) | Deterministic for those families with `--draft-model` |

---

## 2. Technical Review

### 2.1 Security

No new input surface and no unsafe code. The refusal happens before any prefill or forward pass.

### 2.2 Performance

One rule evaluation per model at the start of `generate`, and one early target check in the CLI. No hot-path change.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: `generate` now returns `Result<_, SpeculativeGenerateError>`, so library callers must handle it (the type is already deprecated, #2176). The CLI behavior change is that families opting out of padded prefill (Gemma 4, Llama 4, per the rule's doc comment) are refused on the classic path.
- **New Dependencies**: none (`thiserror` derive on the new error type).

### 2.4 Code Quality

The gate reuses `prompt_lookup_unsupported_reason`, the rule `Engine::verify_rounds_unsupported` already applies, rather than adding a second definition. The existing `can_trim_prompt_cache` and empty-prompt asserts are unchanged.

---

## 3. Technical Decisions

### 3.1 Refuse instead of rewind

Fix option 2 (a model-aware `trim_state(None, n)`) would need Qwen3-Next to override `trim_state`, and GDN state cannot be rewound exactly by a trim. Refusing is the option the PR takes.

### 3.2 Reset before the gate

`generate` calls `reset()` first, then checks the target and the draft. Reset touches only the generator's own caches and counters, so a reused generator does not keep the tokens of an earlier successful call after a refused one.

### 3.3 CLI error text

The CLI refuses an unsupported target before it loads the draft checkpoint, and maps a refusal from `generate` (which also covers the draft) through the same text. The message points at plain generation without `--draft-model`; it does not suggest `--prompt-lookup` or the engine drafters, because they refuse the same models.

---

## 4. Implementation Details

- `src/lib/mlxcel-core/src/speculative/mod.rs`: new `SpeculativeGenerateError` (`TargetUnsupported`, `DraftUnsupported`, carrying the reason string); gate in `generate`; return type and early returns changed to `Ok(...)`; mock `StateKind` (Dense, Recurrent, ModelOwnedLayout, NoPaddedPrefill) and a forward-call counter for tests.
- `src/lib/mlxcel-core/src/speculative/distribution_tests.rs`: tests now `expect` the `Ok`.
- `src/commands/generate.rs`: early target check before the draft load, shared `classic_speculative_refusal`, and the `--draft-kind` error text now says the classic path refuses such targets.
- `src/lib.rs`: re-exports `SpeculativeGenerateError` next to `SpeculativeGenerator`.

---

## 5. Verification

| Check | Result |
|---|---|
| `cargo test --profile test-fast -p mlxcel-core --lib speculative::` | 178 passed |
| `cargo test --profile test-fast -p mlxcel --bin mlxcel commands::generate` | 101 passed, 1 ignored |
| Revert check (both gate calls replaced by `None`) | `speculative::tests`: 15 passed, 3 failed (the three new tests); restored afterwards |
| clippy `-D warnings` (mlxcel-core lib+tests, mlxcel lib+bin+tests) | clean |
| `cargo fmt --all --check` | clean |

New tests: `generate_refuses_a_model_owned_target_before_any_forward` and `generate_refuses_a_model_owned_draft_before_any_forward` loop over the three reasons of the rule (no batched decode, a model-owned layout on a batchable model, no padded prefill), assert the error names the refused model with the rule's reason, and assert neither mock's `forward` ran. `refused_call_clears_tokens_from_a_previous_success` runs a dense call and then a refused call on one generator.

**Not verified**: no real Qwen3-Next (or other model-owned checkpoint) run through `mlxcel generate --draft-model`. The refusal is exercised through mock models at the `SpeculativeGenerator::generate` entry point; the CLI's early target check is covered by compilation and code reading only. The `LoadedModel` wrapper delegates `supports_batching`, `sequence_state_layout`, and `supports_padded_prefill` to the model (read, not executed). In the revert check each refusal loop stops at its first failing state kind, so only that case was observed failing.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines | +324 / -103 |
| Tests added | 3 new, plus updated distribution tests |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `8f585e46` | fix | fix(speculative): refuse model-owned families in the classic walk |
| `751adfd1` | fix | fix(speculative): refuse unsupported targets before loading the draft |

---

## 7. Follow-up Actions

### Future Improvements

- Run a real model-owned checkpoint (Qwen3-Next) and a Gemma 4 or Llama 4 pair through `mlxcel generate --draft-model` to confirm the CLI error text end to end.
- The classic path is slated for removal in v0.8.0 (#2176); the refusal is a stopgap until then.

### Monitoring Required

- Reports of Gemma 4 or Llama 4 users hitting the new refusal on `--draft-model`.
