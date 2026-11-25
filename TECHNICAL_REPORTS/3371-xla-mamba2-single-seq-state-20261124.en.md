# PR #3371: OpenXLA single-sequence recurrent state in the shim; Mamba2 generates on xla

**Date**: 2026-11-24
**Status**: PR open (closes #3339, epic #564 Phase 1 unit U3). Validated on ROCm gfx1151 with IREE `hip` and `local-task`. Single-sequence only: continuous batching and `mlxcel-server` for recurrent models are #3340.

## Executive Summary

#3370 (#3338) emitted Mamba2 prefill and decode graphs that take and return recurrent state, but nothing could run them: the C shim and `IreeLlama` knew only the KV cache, `compile_vmfbs` failed on the embeddings prefill for a recurrent config, and the CLI could not even parse `mamba2-130m`'s `config.json`. This PR gives the shim a generic list of threaded state tensors with an optional KV cache, carries the prefill state into the decode module's globals through `@state_import` once per request, and wires the layout through `RuntimeConfig`, the bundle fingerprint and the CLI. `MLXCEL_BACKEND=xla mlxcel generate` now runs HF `Mamba2ForCausalLM` checkpoints single-sequence.

`AntonV/mamba2-130m-hf` @ `05e8773f` is token-exact against the HF fp32 greedy oracle for 40 tokens on `hip` at the default f16, on `hip` at f32 and on `local-task` at f32, through both the CLI path (prefix prefill plus decode) and `xla_oracle_check` (full prefill plus decode). A one-token prompt, which takes the new state-reset path, matches its 10-token oracle. Because f16 is token-exact, recurrent configs keep the per-device default precision instead of an f32 pin.

Most of the review effort went into cross-request isolation: every path that refuses or fails a request (a failed prefill, a failed state reset, a prompt refused in Rust before reaching the shim, a KV-only entry point called on a stateful bundle) now drops the stored state, so a decode cannot silently continue the previous request.

## 1. Problem Statement

After #3370 the emitter could produce a Mamba2 model, but the runtime around it assumed attention everywhere:

- **The shim had one state, the KV cache.** `xla_llama_create` allocated KV buffers, prefill popped `kcache`/`vcache` from the results, and decode seeded its globals through `decode_step.kv_import`. An attention-free bundle has no KV results and a `decode_step.state_import` entry instead, so the shim would have read the wrong outputs.
- **`compile_vmfbs` built `prefill_embeddings` for every dense config.** #3370 made that emitter return `Err` for recurrent configs, so a Mamba2 bundle could not be compiled.
- **`@state_import` had never run.** #3370's model check used an auxiliary single-entry module and zero-initialized globals, so the prefill-to-decode handoff, which is the core of this issue, was untested.
- **`real_len = 0` was accepted.** The dense one-token-prompt path prefills an empty prefix and takes logits from row 0 of padding. For a recurrent model that row is not a meaningful state boundary.
- **HF's bare `Infinity`.** Mamba2 configs carry `"time_step_limit": [0.0, Infinity]`. The emitter parsed it since #3370, but `RuntimeConfig::from_json`, `config_eos_token_ids` and the CLI's `xla_num_layers` parsed strictly, so the CLI failed on `mamba2-130m` before reaching the runtime, and the EOS lookup would have dropped the config's EOS ids silently.
- **Two open questions from the #3370 carry-over comment on #3339**: whether the f16 drift (0.42 max-abs logits) moves a greedy token, and how to handle the gated RMSNorm grouping for `n_groups > 1`, where HF's torch path and the `mamba_ssm` kernel disagree.

## 2. Change Summary

| File | Change |
|---|---|
| `csrc/xla_iree.c` | `xla_state_desc`, `XLA_MAX_STATE = 8`, `XLA_MAX_STATE_RANK = 8`; context fields `has_kv`, `n_state`, `state_desc[]`, `sscache[]`, `state_import_pending`, `state_zero[]`, `has_prefill_embeddings`; `xla_set_state_contract`, `xla_validate_state`, `xla_state_zero_ensure`, `xla_drop_state`, `xla_state_import_globals`; public `xla_llama_state_reset` and `xla_llama_state_drop`; `xla_require_kv` guard on the ragged, slot and embeddings entries; `xla_llama_create` / bucket create take the state layout; prefill pushes zeroed state inputs and adopts state results; test hooks under `MLXCEL_XLA_TEST_SUPPORT` |
| `src/state_ffi.rs` (new) | `StateFfi`: flattens `StateTensorSpec`s into `has_kv`, `n_state`, ranks, dims, element codes; repeats the shim's bounds; `identity()` for the bundle fingerprint |
| `src/hf_json.rs` (new, ungated) | `quote_non_finite_json` moved here from `emitter/recurrent_config.rs`; `parse_hf_config_json` and `parse_hf_config_json_tracked` (strict parse first, quoting only on failure) |
| `src/iree.rs` | `RuntimeConfig::{state_tensors, has_kv, has_recurrent_state, state_ffi}`; `check_ssm_norm_grouping` at load; `CompiledBundle.prefill_embeddings_vmfb: Option`; `compile_vmfbs` skips embeddings prefill for recurrent configs; state identity hashed into the fingerprint; `IreeLlama.has_state`, `invalidate_sequence_state` at every prefill entry, state reset for an empty prefix; `IreeRaggedLlama::load` refuses recurrent configs; test-support accessors |
| `src/lib.rs` | `XlaInferenceSession::begin_request` (drops state at every session request entry); `config_eos_token_ids` uses the tolerant parser; re-exports `parse_hf_config_json` |
| `src/emitter/config.rs`, `recurrent_config.rs` | `from_json_str` uses the tracked parser and refuses a quoted non-finite number for any non-Mamba2 type; `Config::check_ssm_norm_grouping` |
| `src/commands/generate.rs` | `xla_num_layers` uses the tolerant parser; `xla_model_has_no_kv_cache` suppresses the `KV cache mode` banner for Mamba2 on xla |
| `tests/xla_ssm_single_seq.rs` (new) | Synthetic two-layer Mamba2; 6 device tests (ignored without IREE) and 1 device-free test |
| `scripts/xla/validate_arch.sh` | `--single-only`: structural pre-gate plus token-exact gate 1, no serve gate |
| `README.md`, `FINDINGS_ssm.md`, `docs/supported-models.md`, `CHANGELOG.md` | State ABI section, single-sequence results, supported-model note, changelog entry |

15 files, +1687 / -209.

### Shim state contract

`xla_llama_create` gains `has_kv, n_state, state_ranks, state_dims, state_elems`. Dims are concatenated per tensor in `Config::state_tensors` order; element code 0 is f32, the only state element type today. `xla_set_state_contract` refuses `has_kv` outside {0,1}, `n_state` outside `[0, 8]`, a bundle with neither KV nor state, missing arrays when `n_state > 0`, ranks outside `[1, 8]`, non-f32 codes, and any dim that is non-positive or would push the running byte product past `IREE_DEVICE_SIZE_MAX` or `SIZE_MAX`. Bucket contexts copy the layout from the base context and allocate their own state buffers. `StateFfi::new` on the Rust side repeats the same bounds so an unrepresentable layout fails before any graph is compiled.

For Mamba2 the layout is `ssm_h [L, H, P, N]` then `ssm_conv [L, K-1, conv_dim]`, both f32, with `has_kv = 0`. For `mamba2-130m` that is `[24, 24, 64, 128]` and `[24, 3, 1792]`.

### Request lifecycle

| Step | Shim behavior |
|---|---|
| Prefill, `real_len >= 1` | `xla_state_zero_ensure` allocates each `state_zero[i]` once, then re-zeroes it on the device before every prefill; the zeros are pushed after `real_len`. KV is popped only when `has_kv`. Each state result is shape-validated, and the results replace `sscache[]` only when every one is valid; `state_import_pending = 1`. |
| Prefill, `real_len = 0` on a recurrent context | Refused. Dense contexts keep the existing `real_len = 0` seeding. |
| Empty prompt prefix (one-token prompt) | `IreeLlama::prefill_padded` calls `xla_llama_state_reset` instead of a prefill: `sscache[i]` points at the zeroed `state_zero[i]` and the import is pending. |
| First decode after either | `xla_state_import_globals` calls `decode_step.state_import` once with `sscache[]` and clears the flag only on success; a failed import leaves the flag set and the decode refuses. Later decodes advance the globals in place. |
| Any failed or refused prefill, failed reset, or `xla_require_kv` refusal | `xla_drop_state`: release `sscache[]`, clear the flag. A decode then refuses until a prefill or reset succeeds. |
| Start of every Rust request | `IreeLlama::invalidate_sequence_state` (every prefill entry, including prepared, DeepStack and Gemma3n ones) and `XlaInferenceSession::begin_request` (every session request entry) call `xla_llama_state_drop` before any validation. No-op without state. |

### Bundle shape

A recurrent bundle is the token prefill plus the decode module. `compile_vmfbs` skips `prefill_embeddings`, `CompiledBundle.prefill_embeddings_vmfb` is `None`, the C path receives a null pointer, and the shim treats the embeddings module as optional only when `n_state > 0`. Required entries become `prefill.main`, `decode_step.main`, the embeddings entry when present, and `decode_step.state_import` when the bundle threads state. The fingerprint hashes `state-abi-v1:has_kv=..:ranks=..:dims=..:elems=..`, so a cached bundle compiled for one state layout is never loaded against another.

### Config parsing

`hf_json.rs` is compiled without the `iree` feature so the `xla-backend`-only CLI build can use it. The strict `serde_json` parse runs first and the quoting pass runs only if it fails, so ordinary configs are parsed from unchanged text. `parse_hf_config_json_tracked` also reports whether quoting happened; `Config::from_json_str` uses that to refuse a quoted `Infinity`/`NaN` for every model type except `mamba2`, because the other readers treat a string where a number was written as an absent optional field and take its default.

## 3. Technical Decisions

**Generic state list in the shim, KV optional.** This follows the epic-level decision recorded in #3338 and implemented in the emitter by #3370. The shim now handles `has_kv = 0, n_state > 0` (Mamba2) and `has_kv = 1, n_state = 0` (every existing model) as distinct cases, and the combined case is representable for later hybrids (Nemotron-H #3344, Jamba). The cost is a `has_kv` check at every KV use in the shim; the dense path is otherwise unchanged, which the dense regression runs confirm.

**Import once per request, mirroring `kv_import`.** The alternative was to pass the state into every decode step as inputs and outputs. The module-global form keeps decode at one call per token with no per-step state transfer, and it reuses a pattern the shim already had for KV (`kv_import_pending`), so the two caches follow the same lifecycle rules.

**Adopt state atomically.** Prefill replaces `sscache[]` only after every state result validates, and KV and state are adopted together. A partial adoption would leave a decode importing a mix of new and old tensors with no error.

**Refuse `real_len = 0` and reset on the host instead.** The dense one-token path prefills an empty prefix and reads row 0 of padding, which is harmless for KV because decode overwrites the slot. For a recurrent model the prefill would advance the state over padding tokens. The shim refuses `real_len = 0` only for recurrent contexts, and `IreeLlama` calls `xla_llama_state_reset` for an empty prefix. The dense path keeps its seeding so no existing behavior moves. The token returned by the reset path is a placeholder the seed-then-decode loop discards, as it already discards the dense path's row-0 token.

**Drop state on every refusal, in two layers.** The shim drops state on its own failures. The security review found that requests refused in Rust (oversized or empty prompt) never reach the shim, so the previous request's `sscache` and the decode globals stayed live and a later `decode` or `decode_step` would continue the earlier sequence. Adding the drop at the top of every Rust entry, before any validation, closes that without depending on which check fails first. The alternative of checking state validity inside decode would not work, because a stale state is valid by shape.

**`xla_require_kv` refuses any state, not only `has_kv == 0`.** The first revision refused the ragged, slot and embeddings entries only for attention-free contexts. A future hybrid would then pass the guard and adopt a fresh KV next to the previous request's state. Refusing whenever `n_state > 0` keeps the guard correct for #3340 and #3344 without anyone having to remember to tighten it. Not reachable today, since `IreeRaggedLlama::load` refuses recurrent configs and recurrent bundles have no embeddings module.

**Keep the per-device default precision.** The carry-over comment asked for the token-exact gate to run at the hip default (f16) and to pin f32 for recurrent configs only if it failed. It passed 40/40, as did f32 on hip and local-task, so `effective_precision` is unchanged. This rests on one checkpoint and one 40-token prompt plus a one-token prompt; the 0.42 logit drift measured in #3370 is still present.

**Refuse `n_groups > 1` with a full-width norm.** HF's torch path normalizes the gated output over all of `d_inner`; the `mamba_ssm` kernel normalizes per group. They agree only for `n_groups = 1`. `check_ssm_norm_grouping` refuses a config with `n_groups > 1` and `norm_group_size == d_inner` at runtime load, so Codestral Mamba fails with a message instead of producing silently different text. A family that sets `norm_group_size = d_inner / n_groups` (Nemotron-H's form) passes. The emitter and the HF model check keep accepting such configs, so the question can be settled later against a reference.

**Refuse quoted non-finite numbers outside Mamba2.** Making the tolerant parser shared would otherwise let `"rope_theta": Infinity` in a Llama config parse as a string and silently fall back to a default. Refusing is safer than teaching every optional-float reader the quoted spelling.

**Test hooks behind `MLXCEL_XLA_TEST_SUPPORT`.** `xla_test_set_state_zero_failure`, `xla_test_state_status` and `xla_test_require_kv` exist only under the `test-support` feature. The failed-reset path cannot be triggered from outside without a forced failure, and the review fixes needed tests that fail with the fix reverted.

**Suppress the KV banner by model type.** `xla_model_has_no_kv_cache` checks `model_type == "mamba2"` on the xla backend rather than asking the runtime, because the banner prints before the session loads. It is documented as needing its own check for a hybrid.

## 4. Review Outcomes

**Implementation review** (e6a03cdb):

| Finding | Severity | Status |
|---|---|---|
| `xla_llama_state_reset` returned early when zero-state preparation failed and kept the previous `sscache`, so a following decode continued the old request | MEDIUM | Fixed: drops state like a failed prefill; `failed_state_reset_drops_the_previous_state` fails without the fix |
| Stray comment above the `SsmPrefillForm` re-export | LOW | Fixed |
| `xla_require_kv` refused only `has_kv == 0` | LOW | Fixed in e690800e (below) |
| CLI printed `KV cache mode: fp16 (applied to 24 of 24 layers)` for Mamba2 on xla | LOW | Fixed in 25eeb40f |
| `cargo clippy --features xla-iree,test-utils --all-targets` fails on `private_interfaces` in `IreeRaggedLlama` prepared-slot methods | LOW | Open, untouched by this PR |

**Security and performance review** (e690800e). Scope: the C/FFI state boundary, buffer lifetimes, cross-request isolation, the test hooks, and the HF JSON tolerance.

| Finding | Severity | Status |
|---|---|---|
| A request refused in Rust before reaching the shim left the previous state in `sscache` and the decode globals | MEDIUM | Fixed: `xla_llama_state_drop` at every engine prefill entry and session request entry; `request_refused_in_rust_drops_the_previous_state` |
| `xla_require_kv` admitted a hybrid (KV plus state) context | LOW | Fixed: refuses whenever `n_state > 0` and drops state; `kv_only_entry_points_refuse_any_recurrent_state` |
| A quoted bare `Infinity`/`NaN` reached the attention readers, which took the default silently | LOW | Fixed: refused for non-Mamba2 types; `attention_config_refuses_a_quoted_non_finite_number` |
| A `decode_step.main` failing mid-step may leave the globals partly advanced while `sscache` stays present, so a retried decode in the same request continues from them | LOW | Open. The next request re-imports, so isolation across requests holds; dense KV has the same behavior |

Checked without change: `xla_set_state_contract` validates rank before reading `state_dims`, reads only `sum(ranks)` entries and bounds the byte product, so `xla_state_elements` cannot overflow; all prefill failure paths go through one cleanup that drops state; state results are shape-validated so a short or mismatched result list fails cleanly; the test hooks compile only under `MLXCEL_XLA_TEST_SUPPORT`; `quote_non_finite_json` leaves string contents untouched and runs only after a strict parse fails. Each fix's test fails with the fix reverted.

**Carry-over from #3370**, as resolved here:

1. `compile_vmfbs` skips `prefill_embeddings` for recurrent configs, and the shim treats the module as optional only when `n_state > 0`.
2. `@state_import` runs for real in `tests/xla_ssm_single_seq.rs` and in the CLI gate.
3. `real_len = 0` is refused for recurrent contexts in the shim; the host resets state instead.
4. Precision: token-exact at the hip default (f16), so no f32 pin.
5. `n_groups > 1` with a full-width norm is refused (`check_ssm_norm_grouping`).

## 5. Tests and Verification

HF fp32 oracle (`spike/openxla/oracle_continuation.py`, unchanged for Mamba2), "The capital of France is", 40 greedy tokens:

```
247 2846 273 253 5112 952 13 285 253 5347 273 253 5112 4687 15 733 310 253 5347 273 253 4687 273 6181 13 285 253 5347 273 253 4687 273 6181 15 187 187 510 5347 273 253
```

| route | device | precision | result |
|-------|--------|-----------|--------|
| `mlxcel generate` (CLI, prefix prefill + decode) | hip | f16 (default) | 40/40 |
| `mlxcel generate` (CLI) | hip | f32 | 40/40 |
| `mlxcel generate` (CLI) | local-task | f32 (default) | 40/40 |
| `xla_oracle_check` (full prefill + decode) | hip | default | 40/40 |
| `xla_oracle_check` | local-task | default | 40/40 |
| `validate_arch.sh --single-only` | hip | default | PASS |

A one-token prompt ("The", 10 tokens), which takes the state-reset path, equals its HF oracle on hip: `1246 3688 7033 281 247 1332 323 253 3275 273`. The implementation review reproduced the 40-token CLI result on hip at the default precision. The 40 ids were re-checked on hip after the finalize commit.

`tests/xla_ssm_single_seq.rs` builds a synthetic two-layer Mamba2 and checks the state plumbing against itself, since three routes through prefill and decode compute the same recurrence:

- `prefill_to_decode_handoff_agrees_across_routes`: full-prompt prefill, prefix prefill plus one decode, and one-token prefill plus decodes through the rest of the prompt agree token for token.
- `consecutive_requests_start_from_a_fresh_state`: a second request on one session equals a fresh session.
- `prefill_at_full_capacity_matches_the_decode_route`.
- `empty_prefix_and_refused_prefill_never_reuse_state`.
- `failed_state_reset_drops_the_previous_state` (review fix).
- `request_refused_in_rust_drops_the_previous_state` (security fix).
- `kv_only_entry_points_refuse_any_recurrent_state` (device-free).

The six device tests are `#[ignore]` in ordinary CI and run with `cargo test --release --features xla-iree,test-utils --test xla_ssm_single_seq -- --ignored`. They pass on hip and local-task. A mutation that feeds the previous request's state into prefill fails 3 of the original 4 tests.

Unit tests added: 4 in `state_ffi.rs` (empty layout, flattening order, identity distinguishes layouts, bound refusals), 2 in `recurrent_config.rs` (bare `Infinity` parse, multi-group refusal including the per-group case that passes), 1 in `recurrent_tests.rs` (quoted non-finite refused for attention configs).

Dense regression on hip: `xla_ragged_hardening` 6/6, `xla_greedy_timing` 3/3, `xla_decode_scalar_reuse` 4/4, `xla_prepared_prefill` 4/4; `llama-3.2-1b-instruct-4bit` greedy ids (40 tokens) identical before and after.

Static gates: `cargo fmt --check`; clippy `-D warnings` on the mlxcel-xla lib and tests with and without `iree`, and on the root crate with `--features xla-iree --all-targets`. The `--features xla-backend --all-targets` clippy failures in `src/multimodal/host_preprocessor_molmo_tests.rs` and `src/worker_failfast.rs` reproduce on origin/main (ea0b45d8) and are pre-existing.

`cargo test --profile test-fast -p mlxcel-xla --lib`: the PR body reports 512 passed on an earlier revision. On the final head (25eeb40f, already up to date with origin/main when this report was written) it reports 513 passed, 0 failed, 21 ignored.

No throughput or latency numbers were measured for Mamba2 on xla in this PR.

## 6. Pending Work and Limitations

- **Single-sequence only.** `IreeRaggedLlama::load` refuses recurrent configs and `xla_require_kv` refuses the ragged, slot and embeddings entries for any stateful context. #3340 adds per-slot state, lifts those guards and enables `mlxcel-server`; `validate_arch.sh --single-only` skips the serve gate until then.
- **One real checkpoint, one long prompt.** Token exactness at f16 rests on `mamba2-130m` (`n_groups = 1`) with a 40-token and a 10-token continuation. The 0.42 max-abs logit drift at f16 from #3370 is still there; a longer generation or another checkpoint could flip a near-tie. Pinning f32 for recurrent configs in `effective_precision` remains the fallback.
- **`n_groups > 1` is refused,** not supported. Codestral Mamba needs the gated-norm grouping settled against `mamba_ssm`; Nemotron-H (#3344) normalizes per group and passes the check.
- **Mid-step decode failure** may leave the state globals partly advanced for a retry within the same request (LOW, open; dense KV behaves the same).
- **No performance measurement.** Decode and prefill throughput for Mamba2 on xla, and the cost of re-zeroing `state_zero` per request, were not measured.
- **CUDA not run.** Only `hip` (gfx1151) and `local-task`.
- **`private_interfaces` clippy failure** with `test-utils` in `IreeRaggedLlama` is pre-existing and out of scope.
- **Banner check is name-based.** `xla_model_has_no_kv_cache` matches `model_type == "mamba2"`; a hybrid family needs its own rule.

## 7. Learning Points

- When a runtime gains a second kind of per-sequence state, give it the same lifecycle as the first. Modelling `sscache` and `state_import_pending` on `kv_import_pending` made the import rules, failure handling and bucket behavior a known quantity instead of a new design.
- Cross-request isolation has to hold on refusal paths, not only on failure paths. A request refused before it reaches the native layer never runs the native cleanup; dropping state unconditionally at the top of every entry is simpler and harder to bypass than reasoning about which check fails first.
- A stale state is valid by shape, so validation at use time cannot catch reuse. The only defense is to make the state absent until a request establishes it, and to have decode refuse when it is absent.
- Guards should be written for the invariant, not for today's configs. `xla_require_kv` checking `has_kv == 0` was correct for Mamba2 but wrong for the hybrids the epic is heading toward; `n_state > 0` states what the entry points actually lack.
- Making a parser more tolerant widens what every caller accepts. Tracking whether tolerance was used, and refusing in readers that cannot interpret the result, kept the fix for Mamba2 from silently changing values in other families.
- Run the precision gate at the precision users get. #3370 measured f16 logit drift and predicted it would not move greedy tokens; this PR confirmed that on the real CLI path before deciding against an f32 pin.
