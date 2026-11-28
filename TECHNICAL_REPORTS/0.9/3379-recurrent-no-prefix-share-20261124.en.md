# PR #3379: Recurrent models stay out of prefix-hash caches on OpenXLA and MLX

**Date**: 2026-11-24
**Status**: PR open (closes #3341, epic #564 Phase 1 unit U5). Validated on ROCm gfx1151: the OpenXLA server with IREE `hip` on `mamba2-130m`, and the MLX ROCm server on `qwen3.5-0.8b-4bit` and `granite-4.0-h-tiny-4bit`. Falcon-H1 and PLaMo 2 are covered by config-detection unit tests only, since no local checkpoint exists.

## Executive Summary

A saved token prefix cannot restore recurrent state, so a recurrent model must never get a hit from the prompt-prefix store or from Automatic Prefix Caching (APC). Before this PR that held on OpenXLA only by accident: the XLA serve worker dropped `PromptCacheWarmup` and never read the store, but `ModelProvider` still attached one on its OpenXLA arm, and nothing asserted the property. On MLX, the APC opt-out list `HYBRID_SSM_MODEL_TYPES` was missing four recurrent families.

The PR adds an OpenXLA guard (`src/server/xla_prompt_cache_guard.rs`) that decides from the emitter's `Config::has_recurrent_state()` or membership in the shared list, and enforces the decision twice: `startup::resolve_prompt_cache_store` builds no store and clears `prompt_cache.enabled` and `apc.enabled`, and the provider's OpenXLA arm drops any store another caller passes in. A `config.json` the guard cannot read or parse fails closed. The MLX list gains `falcon_h1`, `plamo2` (+`plamo2vl`), `qwen3_5` (+`qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`) and `granitemoehybrid`, with matching architecture fragments. The startup detector now tolerates HF's bare `Infinity`, which used to make Mamba2 and Nemotron-H configs fail the strict parse and keep APC on, the opposite of what the check exists for.

On MLX the user-visible effect is limited to the startup warning and `apc_enabled: false`: the four families already keep their state outside the KV slice, so they never reached the block-hash store, and outputs are byte-identical to origin/main on both local checkpoints.

## 1. Problem Statement

Epic #564 brings SSM, hybrid and recurrent architectures to the OpenXLA/IREE backend, and its acceptance requires that the recurrent path never participates in prefix-hash sharing. Issue #3341 recorded the state on main:

- **OpenXLA, held by accident.** `xla_worker_admission.rs` drops `PromptCacheWarmup` because the XLA worker has no snapshot state, yet `ModelProvider`'s OpenXLA arm (`model_provider.rs`, then lines 938-953) set `provider.prompt_cache = prompt_cache_store`. The store existed, `/v1/cache/stats` reported it enabled, and the only reason no hit could occur was that the XLA engine never consulted it. A future change that wired the store into the XLA engine would have broken the property silently.
- **MLX list incomplete.** `HYBRID_SSM_MODEL_TYPES` (`src/server/prompt_cache/hybrid_ssm.rs`) omitted `falcon_h1`, `plamo2`, `qwen3_5` and `granitemoehybrid`, all of which carry recurrent state. APC defaults to on (`--apc-enabled`), so these families ran with APC nominally enabled.
- **Detector failed open on the families it targets** (found during implementation). `detect_hybrid_ssm_from_path` parsed `config.json` with strict `serde_json`. HF writes Python's bare `Infinity` into Mamba2 and Nemotron-H configs (`time_step_limit`), so the parse failed, `resolve_prompt_cache_store` matched only `Ok(Some(_))`, and APC stayed on for exactly those checkpoints.
- **Stale invariant in the module docs.** The `hybrid_ssm.rs` header said every listed family except Baichuan-M1 returns `supports_batching() == false` and so routes to `ModelOwned`. Mamba2, Nemotron-H and Qwen3.5 now batch on MLX, so the real load-bearing property had to be restated.

The issue named two risks: the MLX list change alters user-visible caching for four families, and its interaction with the #3047 turn-boundary state checkpoints had to be checked; and a recurrent family added later must also be added to the list.

## 2. Change Summary

| File | Change |
|---|---|
| `src/server/xla_prompt_cache_guard.rs` (new, `xla-iree` only) | `xla_serve_backend_selected()` (the same `MLXCEL_BACKEND == "xla"` test the provider uses); `xla_recurrent_family(model_path)` returns the family to log or `None`, failing closed with `UNDETECTED_FAMILY` (`"unreadable config.json"`) when the file cannot be read or parsed; `disable_prompt_cache_for_xla_recurrent` clears `prompt_cache.enabled` and `apc.enabled` and logs one `info!` line naming the family; `xla_provider_prompt_cache` drops a passed store for a recurrent model |
| `src/server/startup.rs` | `resolve_prompt_cache_store` calls the guard first on the XLA backend and returns `None` before any sizing or store construction; the inline APC opt-out is replaced by `disable_apc_for_hybrid_ssm` |
| `src/server/model_provider.rs` | The OpenXLA arm uses `xla_serve_backend_selected()` and filters `prompt_cache_store` through `xla_provider_prompt_cache` before `new_with_xla_worker` |
| `src/server/prompt_cache/hybrid_ssm.rs` | 8 new `model_type` strings and 4 architecture fragments (`falconh1`, `plamo2`, `qwen35`, `granitemoehybrid`); `detect_hybrid_ssm_from_path` runs `crate::models::sanitize_config_json` before parsing; new `disable_apc_for_hybrid_ssm` (the extracted startup opt-out) and `recurrent_prefix_sharing_family(config, backend_threads_recurrent_state)` (list first, then the backend's answer, logging `model_type` or `"recurrent"`); module docs restate the load-bearing guarantee as `SequenceStateBackend::keeps_state_outside_kv_slice` |
| `src/server/prompt_cache/mod.rs` | Re-exports the two new functions |
| `src/lib/mlxcel-xla/src/lib.rs` | `config_json_has_recurrent_state(text) -> Result<bool, String>`, gated `#[cfg(any(feature = "iree", test))]` like the emitter it calls; an unsupported family is `Err` and the caller's list decides |
| `src/server/batch/xla_worker_admission.rs` | Comment on the unchanged `PromptCacheWarmup` drop now points at the guard |
| `src/server/mod.rs` | Declares the guard module under `xla-iree` |
| `src/main.rs`, `src/bin/mlx_server.rs` | `--apc-enabled` help lists the recurrent families by name |
| `src/server/xla_prompt_cache_guard_tests.rs` (new) | 6 tests (Section 5) |
| `src/server/prompt_cache/hybrid_ssm_tests.rs` | 7 tests (Section 5) |
| `src/server/prompt_cache/apc_integration_tests.rs` | `state_checkpoints_do_not_depend_on_apc` |
| `src/lib/mlxcel-xla/src/emitter/recurrent_tests.rs` | `config_json_has_recurrent_state_matches_the_config` |
| `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `CHANGELOG.md` | The widened exclusion list and the OpenXLA no-store rule; the APC env section notes the override; a `Changed` entry describing the MLX behavior change |

17 files, +722 / -38, in six commits plus one merge of origin/main.

### Decision flow on OpenXLA

```
resolve_prompt_cache_store (startup, server and in-process)
  MLXCEL_BACKEND == "xla" ?
    xla_recurrent_family(model_path)
      config.json unreadable or unparseable   -> "unreadable config.json"  (fail closed)
      detect_hybrid_ssm(config) matches list  -> family
      config_json_has_recurrent_state == true -> model_type or "recurrent"
      otherwise                               -> None (may share)
    Some(family): log once, prompt_cache.enabled = false, apc.enabled = false, return None
  (MLX path continues: ctx-checkpoints, sizing, disable_apc_for_hybrid_ssm, store build)

ModelProvider OpenXLA arm
  xla_provider_prompt_cache(model_path, store): same decision, drops Some(store)
```

Because startup returns `None`, neither `AppState` nor the provider holds a store, and the `/v1/cache` routes report `enabled: false`. Router mode already passes `None`.

### What list membership does on MLX

The list has one consumer, the startup APC opt-out (now `disable_apc_for_hybrid_ssm`), which sets `apc.enabled = false` and nothing else. APC is read only by the dense KV store: the block-hash chain on insert, partial-match lookup, `apc_stats` and `/v1/cache/stats`. It does not touch snapshots, #3047 state checkpoints, paged KV or batching.

The four families already keep their state outside the KV slice: Falcon-H1, PLaMo 2 and Granite 4.0-H are `ModelOwned`, and Qwen3.5 is `SchedulerOwned` (its `LayerState`, since #3254). The scheduler's dense adopt and donate both decline such a sequence whether or not it batches. In practice the widening changes the startup warning and the reported `apc_enabled: false`.

## 3. Technical Decisions

**Decide by `has_recurrent_state()` or the list, not either alone.** The issue scoped the XLA decision to `Config::has_recurrent_state()`. That only answers for configs the emitter can load (Mamba2 today); `config_json_has_recurrent_state` returns `Err` for a family the emitter does not support (the test uses Jamba). The union with `HYBRID_SSM_MODEL_TYPES` refuses those too, so the 13 epic types are covered now and a family that becomes loadable later is already refused. The list half is the one an operator sees on both backends, so it is also the single place to add a new family. Using only the list would have missed a config that threads state under an unlisted `model_type`; the dedicated test `backend_recurrent_state_refuses_an_unlisted_family` pins the other half.

**Enforce at startup and again in the provider.** Startup is the only real constructor of the store (server and in-process both go through `resolve_prompt_cache_store`), so turning both flags off there is what makes `/v1/cache/stats` and `AppState` truthful. The provider-arm check is defense in depth for any caller that builds a provider with a store directly. The alternative, guarding only in the provider as the issue first scoped, would have left `AppState` holding a store and the stats route reporting an enabled cache for a model that can never hit it.

**Turn off the whole prompt cache on OpenXLA, but only APC on MLX.** MLX restores recurrent state through whole snapshots and #3047 turn-boundary checkpoints, which are keyed by the exact prefix and never truncate a recurrent state. The OpenXLA serve engine has no snapshot to adopt, so a store would only reserve memory and report a cache that can never serve. On MLX, disabling the store for these families would have removed working snapshot hits; the MLX server check shows 1 snapshot hit of 68 tokens with APC off.

**Fail closed on an unreadable config.** The first version returned "may share" when `config.json` was missing or unparseable. That was not exploitable, because the serve engine parses the same text with the same parser (`RuntimeConfig::from_json` -> `Config::from_json_str`) and such a model cannot load, and the XLA worker never reads the store anyway. The security review still changed it to fail closed: a guard that cannot rule out recurrent state should not leave the cache on, and the model load reports the parse error itself.

**Tolerant parse in the detector.** `detect_hybrid_ssm_from_path` now runs `sanitize_config_json` before `serde_json`, matching what the model loaders already do. The security review checked that the sanitizer only rewrites substrings that cannot appear in any listed family name, so the rewrite cannot change a detection result.

**Restate the invariant instead of keeping `supports_batching()`.** The old header tied safety to `supports_batching() == false`, which is no longer true for six of the listed families. The property that actually keeps a recurrent sequence out of the block-hash store is its state layout: `ModelOwned` or `SchedulerOwned` (`keeps_state_outside_kv_slice`) makes the dense adopt and donate paths decline it, independent of batching. Baichuan-M1 remains the documented dense exception.

**Fragments chosen not to match attention siblings.** `qwen35` matches `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForCausalLM` without matching `Qwen3ForCausalLM` or Qwen2.5; `falconh1` does not match Falcon. `new_fragments_do_not_match_attention_siblings` pins this, and the implementation review confirmed the fragments match no attention-only config among the local checkpoints.

**Extract `disable_apc_for_hybrid_ssm`.** The inline startup block became a named function so the MLX opt-out can be tested in isolation (`disable_apc_turns_apc_off_only_for_listed_families`) and so the test can assert that only APC changes, leaving the store, its snapshots and the checkpoint budgets as configured.

## 4. Review Outcomes

**Implementation review** (fix in 2bf5a579):

| Finding | Severity | Status |
|---|---|---|
| The startup wiring was untested: disabling the guard call in `resolve_prompt_cache_store` left every guard test green | MEDIUM | Fixed: `startup_resolver_builds_no_store_for_an_xla_recurrent_model` runs the resolver under `MLXCEL_BACKEND=xla` and fails with the call removed |
| The guard logs "prompt cache and APC disabled" even when the operator never enabled the prompt cache | LOW | Open, cosmetic |
| The `ModelProvider` OpenXLA-arm drop has no direct test; it needs a real XLA worker, and startup already passes `None` | LOW | Open |

The reviewer reverted each change locally: without the list and sanitizer changes 6 tests fail, and with the guard neutralized 3 guard tests fail.

**Security and performance review** (fix in c0566f66): no CRITICAL, HIGH or MEDIUM finding.

| Finding | Severity | Status |
|---|---|---|
| `xla_recurrent_family` returned "may share" for a missing or unparseable `config.json`, so the store stayed attached (fail-open, not exploitable) | LOW | Fixed: fails closed; `an_unreadable_or_unparseable_config_fails_closed` fails on the previous guard |

Checked with no further finding: cross-tenant paths (startup, in-process, router children, runtime settings PATCH, where `prompt_cache` is read-only, and the `/v1/cache` routes, which see `None`); crafted configs (`has_recurrent_state()` comes from the same `Config` the engine serves, layer counts are capped by `MAX_CONFIG_LAYERS`, `conv_kernel < 2` is rejected before `conv_kernel - 1`); env mismatch (the guard and the OpenXLA arm call the same `xla_serve_backend_selected()`, non-test code never sets `MLXCEL_BACKEND`, and a mismatch at worst turns the cache off on MLX). Performance: detection runs at startup and at provider construction only and reads one small file each time.

**Finalize** (58ed3023, a95dd225): merged origin/main without conflict; the resolver test restores `MLXCEL_BACKEND` through a drop guard so a panic cannot leak it into later tests. The branch was up to date with origin/main when this report was written, so no tests were re-run for the report.

## 5. Tests and Verification

All results are from the PR body and review comments, on ROCm gfx1151.

Unit tests added (15):

- `xla_prompt_cache_guard_tests.rs` (6): `every_epic_recurrent_type_gets_no_store_and_no_apc` (the 13 epic types: `mamba`, `mamba2`, `falcon_mamba`, `jamba`, `nemotron_h`, `falcon_h1`, `plamo2`, `qwen3_next`, `qwen3_5`, `kimi_linear`, `rwkv7`, `recurrent_gemma`, `granitemoehybrid`), `an_emitter_loadable_mamba2_is_refused_through_has_recurrent_state` (a tiny HF-format Mamba2 with a bare `Infinity`), `an_attention_model_keeps_its_store_and_apc`, `a_local_hf_mamba2_checkpoint_is_refused`, `startup_resolver_builds_no_store_for_an_xla_recurrent_model`, `an_unreadable_or_unparseable_config_fails_closed`
- `hybrid_ssm_tests.rs` (7): `every_epic_recurrent_type_is_refused_prefix_sharing_on_any_backend`, `backend_recurrent_state_refuses_an_unlisted_family`, `newly_listed_hybrids_are_detected_every_way` (by `model_type`, by VLM `text_config`, by architecture name alone), `new_fragments_do_not_match_attention_siblings`, `detect_from_path_tolerates_bare_infinity`, `disable_apc_turns_apc_off_only_for_listed_families`, `local_qwen35_and_granite_h_checkpoints_turn_apc_off` (skips when absent; `MLXCEL_REQUIRE_MODELS=1` makes a missing checkpoint fail)
- `apc_integration_tests.rs` (1): `state_checkpoints_do_not_depend_on_apc`, four Qwen3.5-style boundary turns, history edited after turn 2; both APC settings return a `CheckpointHit` with `matched_len` 20
- `mlxcel-xla` (1): `config_json_has_recurrent_state_matches_the_config` (Mamba2 `Ok(true)`, Llama `Ok(false)`, Jamba `Err`)

Suites, with `MLXCEL_REQUIRE_MODELS=1`:

| command | result |
|---|---|
| `cargo test --profile test-fast --lib prompt_cache` | 382 pass |
| same with `--features xla-iree`, plus `xla_prompt_cache_guard` | 386 pass |
| `mlxcel-xla` `config_json_has_recurrent_state_matches_the_config` | pass |
| `cargo fmt --check`; `cargo clippy --all-targets -D warnings` for default, `--features xla-iree`, and `-p mlxcel-xla` with and without `iree` | pass |

End-to-end checks:

| check | result |
|---|---|
| OpenXLA server (hip), `mamba2-130m`, `--prompt-cache-enabled=true --apc-enabled=true --metrics`, two completions sharing all but the last few of 68 prompt tokens | prompt-cache hits, misses and snapshot hits all 0; `/v1/cache/stats` `enabled: false`; both outputs equal `mlxcel generate` |
| MLX ROCm server, `qwen3.5-0.8b-4bit` and `granite-4.0-h-tiny-4bit`, three completions each (two sharing a prefix, then one extending request 1) | startup logs the APC opt-out; stats show `apc_enabled: false`, 0 dense lookups, 1 snapshot hit (68 tokens); every output byte-identical to origin/main with APC on; requests 1 and 2 equal `--no-prompt-cache` and the CLI; Granite request 3 also matches |
| Qwen3.5 request 3 (the snapshot hit) vs a cold run | parts at about token 9 on both origin/main and this branch: the known two-forward partition effect, not this change |

No throughput or memory measurements were taken; the change is a startup decision and adds no per-request work.

## 6. Pending Work and Limitations

- **Falcon-H1 and PLaMo 2 are untested end to end.** No local checkpoint; only config-detection unit tests cover them.
- **The list must grow with the model zoo.** A future recurrent family the OpenXLA emitter cannot load and that is not on `HYBRID_SSM_MODEL_TYPES` would pass the guard (its `has_recurrent_state` query returns `Err`, read as `false`). Such a model cannot load on OpenXLA, so no store would be read, but the MLX APC opt-out would also miss it. The enumeration tests guard only the 13 known types.
- **Guard compiled only with `xla-iree`.** A build without the feature has no OpenXLA backend, so nothing is lost, but the guard tests run only in `xla-iree` builds.
- **LOW, open:** the guard logs its disable line even when the prompt cache was never enabled.
- **LOW, open:** the `ModelProvider` OpenXLA-arm drop has no direct test.
- **No caching scheme for recurrent state on OpenXLA.** Out of scope by the issue; OpenXLA recurrent models re-prefill every request. Hybrids on OpenXLA (#3344) inherit the guard through both halves of the decision.
- **CUDA and Metal not run.** The guard is backend-agnostic Rust, but the server checks ran on ROCm only.

## 7. Learning Points

- An invariant that holds because a consumer happens to ignore a resource is not an invariant. The XLA worker never read the store, but the store existed and was reported enabled. Removing the resource at its single constructor, and asserting it there, makes the property survive future wiring.
- A wiring test should fail when the wiring is removed. Every guard unit test passed with the startup call deleted; only a test that runs the real resolver under the real env selection pinned it. Reverting each change and checking which test fails is a cheap way to find this gap.
- A detector that fails open on parse errors fails on the inputs it was written for. HF recurrent configs are the ones that carry bare `Infinity`, so a strict parse disabled the check precisely for Mamba2 and Nemotron-H. Any safety detector over HF JSON should use the same tolerant parse as the loaders and decide what a parse failure means explicitly.
- When a module's safety argument names a proxy (`supports_batching() == false`), check whether the proxy still holds after nearby changes. Here the real property was the state layout, and the proxy had quietly stopped being true for six families.
- When two backends restore state differently, the same "never share prefixes" rule maps to different switches: APC only on MLX, where exact-prefix snapshots still restore state correctly, and the whole store on OpenXLA, where nothing can restore it.
