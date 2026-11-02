# PR #2788: Close the KV mode reporting gaps left by #2776

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low (reporting and metadata changes, one startup panic turned into an FP16 fallback, and a warning that now fires per model; no change to what any cache stores in the aligned or non-Turbo cases)

## Executive Summary

PR #2788 (closes #2783) resolves the four LOW residuals that the #2776 reviews and its report (section 4) left on the model-owned KV surface. The architecture registry now advertises `fp16`, `int8`, `turbo4` for the seven families #2776 wired up, while LongCat stays FP16. A new `LanguageModel::kv_cache_storage_modes` reports what each cache really stores, so the server's `kv_cache_mode_applied_layers` count and `logit_trace` stop counting FP16 sliding-window layers as quantized: under `--kv-cache-mode int8`, Gemma 3 1B now reports 4/26 instead of 26/26, and GPT-OSS 20B 12/24 instead of 24/24. `fp16+turbo4` on a sliding window that is not a multiple of 32 no longer panics at startup; the sliding layers fall back to FP16 with one warning. The context-shift fallback warning moves from a process-wide static to a per-worker flag, so every affected model of a multi-model server logs it once and a wake from idle sleep does not repeat it.

## 1. Problem Statement

### Stale registry metadata

`ModelType::capabilities` sets `kv_modes` to `KV_TURBO` only for types matched by `supports_turbo_kv` in `src/models/registry.rs`. After #2776, ExaOne 4, GPT-OSS, Step 3.5, Step-3.7, Mellum, Laguna and NemotronNAS honor `int8` and `turbo4`, but only `Ministral3` was in that list, so `mlxcel arch --json` still reported `["fp16"]` for the other seven.

### The per-layer report overstated sliding layers

`RotatingKVCache::update_and_fetch` (`src/lib/mlxcel-core/src/cache.rs`) quantizes only for `Turbo4Asym`; under `Int8`, `Turbo4`, `Turbo4Delegated` and `Turbo3Asym` it stores FP16. Each family's `kv_cache_layer_modes()` returns the injected table, and both consumers of that table (`log_effective_kv_cache_application` in `src/server/batch/scheduler/paged_layout.rs`, and `examples/logit_trace/kv_mode.rs`) treated it as the storage. An operator who set `int8` on Gemma 3 saw `applied_layers=26/26` while 22 of those layers held FP16. The same applied to every family that builds a `RotatingKVCache` from the table: Gemma 3, Gemma 4, GPT-OSS, ExaOne 4, Laguna, Step 3.5, Ministral 3, Mellum and AFMoE.

### `fp16+turbo4` panicked on a misaligned window

`RotatingKVCache::new_with_mode_and_seed` asserts that a `Turbo4Asym` cache's `max_size` is a multiple of `turbo::BLOCK_SIZE` (32). All nine builders passed `config.sliding_window` straight in, so a checkpoint with, say, a 100-token window would abort at the first cache build. Gemma 3 already had this exposure; #2776 extended it to the newly wired families. No local checkpoint triggers it (windows are 128, 512, 1024 or 4096).

### The context-shift warning fired once per process

#2776 forced `--context-shift` off for model-owned layouts under a KV bound and warned behind a function-local `static WARNED: AtomicBool`. The static was there because the worker rebuilds the scheduler on every wake from idle sleep. Its side effect was that in a multi-model server only the first affected model logged the warning. The warning was also emitted inside `effective_context_retention`, so whether it fired depended on the order of `with_context_retention` and `with_max_kv_size` calls.

## 2. Technical Decisions

### Separate "injected table" from "storage" instead of changing `kv_cache_layer_modes`

The simplest fix would have been to make `kv_cache_layer_modes()` return FP16 for sliding layers. That would break its contract as the round-trip of `set_kv_cache_layer_modes`: `logit_trace` checks `applied == modes` after injection, and `model_owned_kv_mode_tests` compare the injected table with what comes back. The PR adds a second trait method, `kv_cache_storage_modes()`, whose default is `kv_cache_layer_modes()`. Only consumers that describe storage (the startup log and the logit_trace extra line) switch to it. Every wrapper that forwards `kv_cache_layer_modes` forwards the new method too (`LoadedModel`, the `&M` blanket impl in `language_model_ref.rs`, and 11 VLM wrappers); otherwise a VLM would silently fall back to the trait default and reintroduce the overstated count.

### One rule, stated once, next to the code it mirrors

`RotatingKVCache::stored_mode(max_size, requested)` is a `const fn` that returns `Turbo4Asym` only for `Turbo4Asym` on a positive multiple of 32, else `Fp16`. It lives next to `update_and_fetch` with a doc comment saying the two must change together. `KvCacheLayerModes::storage_modes(window_of)` maps sliding layers through it. Each of the nine families overrides `kv_cache_storage_modes` and passes the same sliding predicate its cache builder uses (`sliding_window_pattern` for Gemma 3, `layer_types` for GPT-OSS, `is_local` for ExaOne 4, `uses_sliding` for AFMoE, and so on), and the same window value, including ExaOne 4's `unwrap_or(512)` default.

A cache's own `mode` field was not used as the source of truth, because under `Int8` a sliding `RotatingKVCache` is still built with `mode == Int8` and only stores FP16 internally. Reading the field would reproduce the bug.

### Downgrade only the case that panics

`KvCacheLayerModes::rotating_mode_for_layer(layer_idx, window)` returns `Fp16` when the layer's mode is `Turbo4Asym` and the window is misaligned, and otherwise returns the table's mode unchanged. All nine builders call it for their `RotatingKVCache`; full-attention layers keep `mode_for_layer`. Other modes are deliberately not rewritten to `Fp16` even though they store FP16: the rotating cache's `mode` also drives `can_truncate_to` and snapshot reuse, and changing it would alter those paths for configurations that work today. The warning is guarded by a `Cell<bool>` on the `KvCacheLayerModes` instance (the same pattern as the LongCat refusal), because caches are rebuilt per sequence. `RotatingKVCache::new_with_mode` keeps its assert for direct callers, so a misuse outside the model builders still fails loudly.

### Move the warning out of the builder and scope it to the worker

`effective_context_retention` now returns `(policy, forced_off)` and logs nothing. The scheduler stores `context_shift_forced_off` and a `context_shift_fallback_warned: Arc<AtomicBool>` (fresh per scheduler by default). `warn_context_shift_fallback_once()` swaps the flag and logs the original text; `run` calls it first, after the builder chain has settled. `spawn_model_worker_with_batch_config` creates one flag per worker thread, outside its load/sleep/wake loop, and passes it on every rebuild through `with_context_shift_fallback_warned`. One worker serves one model, so this gives one warning per model regardless of how often it sleeps. `with_max_kv_size` ORs into `context_shift_forced_off` because a policy already forced off reads `context_shift: false` and would otherwise erase the record.

## 3. Change Summary

38 files, +648 / -78, one commit (`af0ca3b4`).

| Area | Files | Change |
|---|---|---|
| Core cache rule | `src/lib/mlxcel-core/src/cache.rs` | `RotatingKVCache::stored_mode` |
| Trait | `src/lib/mlxcel-core/src/generate.rs`, `language_model_ref.rs` | `kv_cache_storage_modes` with a default, forwarded through `&M` |
| Shared helper | `src/models/model_owned.rs` | `rotating_mode_for_layer` (with one warning), `storage_modes` |
| Families | `gemma3.rs`, `gemma4.rs`, `gpt_oss.rs`, `exaone4.rs`, `laguna.rs`, `step3p5.rs`, `ministral3.rs`, `mellum.rs`, `afmoe.rs` | builders use `rotating_mode_for_layer` for sliding layers; `kv_cache_storage_modes` override |
| Forwarders | `src/loaded_model.rs`, `src/vision/{mod, gemma4_vl, gemma4_unified, qwen3_5_vl, qwen4_exp_vl, lfm2_vl, step3p7, got_ocr, mage_vl, llmjp_vl, llmjp_vl_text}.rs` | forward `kv_cache_storage_modes` |
| Registry | `src/models/registry.rs` | seven types added to `supports_turbo_kv` |
| Scheduler | `scheduler/config.rs`, `mod.rs`, `run_loop.rs`, `paged_layout.rs`, `src/server/model_worker.rs` | forced-off record, per-worker flag, warn at `run`, startup count from storage modes |
| Example | `examples/logit_trace/kv_mode.rs` | prints `kv cache storage: {n}/{total} layers {mode}, {s} sliding layers fp16` when storage differs |
| Tests | `rotating_turbo_window_tests.rs`, `model_owned_kv_mode_tests.rs`, `scheduler_model_owned_slot_tests.rs`, `registry.rs`, `gemma3_tests.rs` (fixture visibility) | new tests listed in section 5 |
| Docs | `docs/supported-models.md`, `docs/llama-server-compat.md` | storage-count and misaligned-window note, `kv_modes` listing, per-model warning wording |

`recipes/registry/*.json` is untouched; those snapshots are regenerated by `make recipes-registry` at release.

## 4. Trade-offs and Residuals

- **The sliding predicate is written twice per family.** Each override repeats the predicate and window of its cache builder. A family that later changes how it picks sliding layers must update both, or the report drifts from storage. The tests pin Gemma 3 and GPT-OSS only; the other seven overrides are covered by the family test suites running green, not by a storage-specific assertion.
- **`stored_mode` and `update_and_fetch` are kept in step by a comment.** If the rotating cache gains quantized storage for `Int8` (out of scope here), `stored_mode` has to change in the same PR.
- **The misaligned-window fix has no real-checkpoint coverage.** No local checkpoint has a window that is not a multiple of 32; the unit tests with windows 100 (Gemma 3) and 4 (GPT-OSS) are the only evidence.
- **The logit_trace storage line assumes a uniform request.** It counts layers whose storage equals the requested `mode`; a mixed table (for example Boundary-V) would print a count that is correct but labeled with one mode.
- **Metal and ROCm were not run locally**; the gate is GB10 (CUDA).
- **Out of scope**, as stated in the issue: storing `Int8` or other modes in `RotatingKVCache`, Laguna's DFlash round-loop caches, a real per-layout context-shift trim, and the slot-release breadth note from the #2776 review.

## 5. Validation

Failing-first: the new tests call APIs that do not exist on main, so the branch was rebuilt with each fix replaced by main's behavior (the seven registry entries removed, `rotating_mode_for_layer` passing the table through, `storage_modes` returning the injected table, the warning flag back to a process-wide static). All six new mlxcel tests failed:

| Test | Failure on main's behavior |
|---|---|
| `model_owned_kv_families_advertise_quantized_modes` | `[Fp16]` vs `[Fp16, Int8, Turbo4]` |
| `gemma3_sliding_layers_report_fp16_storage_under_int8`, `gpt_oss_...` | `Some([Int8, Int8])` vs `Some([Fp16, Int8])` |
| `turbo4_asym_on_a_misaligned_window_falls_back_to_fp16` | panic at the `RotatingKVCache` constructor assert |
| `gemma3_storage_modes_follow_the_turbo4_asym_window_alignment` | same panic |
| `context_shift_fallback_warns_once_per_model` | the second model does not warn |

The mlxcel-core test `stored_mode_reports_fp16_except_for_aligned_turbo4_asym` covers all six modes on windows 512, 100 and 0, and checks that the aligned answer matches what the constructor builds. `context_shift_fallback_does_not_warn_when_nothing_was_forced_off` is the control.

Local gate on GB10 (`--profile test-fast --features cuda`, `--test-threads=1`, under `gpu-lock`): mlxcel-core `cache::` 586 passed; `models::registry`, `models::model_owned_kv_mode_tests`, `scheduler_model_owned_slot_tests`, `models::gemma3`, `models::gpt_oss`, `models::model_owned` 124 passed (including the #2776 tests `model_owned_layout_stops_at_the_kv_bound_under_context_shift` and `context_shift_fallback_holds_when_the_bound_is_set_last`); affected families (`exaone4 laguna step3p5 ministral3 mellum afmoe gemma4 mixed_cache vision::step3p7 kv_cache_mode`) 245 passed; `cargo fmt --check` and clippy `-D warnings` clean.

Real checkpoints (`MLX_ENABLE_TF32=0`, "before" is a main build at `29c2f0c2`):

| Run | Before | After |
|---|---|---|
| `mlxcel-server -m gemma-3-1b-it-4bit --kv-cache-mode int8` | applied 26/26 | applied 4/26 (the global layers) |
| `mlxcel-server -m gpt-oss-20b-mxfp4 --kv-cache-mode int8` | applied 24/24 | applied 12/24 (full-attention layers) |
| `mlxcel-server -m gemma-3-1b-it-4bit --kv-cache-mode fp16+turbo4` (window 512) | | applied 22/26, no fallback warning |
| `logit_trace` int8 on `exaone-4.0-1.2b-4bit` | | succeeds, no storage line (no sliding layers) |
| `logit_trace` int8 on `gemma-3-1b-it-4bit` | | prints `kv cache storage: 4/26 layers int8, 22 sliding layers fp16` |

The `fp16+turbo4` row is the aligned control: there the sliding layers do quantize, so the count rises to 22/26 rather than falling.

## 6. Learning Points

- **A getter that round-trips a setter should not double as a status report.** `kv_cache_layer_modes` answered "what was injected", and two consumers read it as "what is stored". Adding a second method with a default equal to the first kept every existing caller correct and let only the reporting paths change.
- **When a type's field does not describe its behavior, derive the behavior from the rule, not the field.** A rotating cache built with `mode == Int8` stores FP16; the report has to come from the same rule `update_and_fetch` applies.
- **Trait defaults hide forwarding gaps.** A new trait method with a default compiles cleanly in every wrapper that forgets to forward it. Any wrapper that forwards a sibling method needs the new one too, and grepping for the sibling is how to find them.
- **Narrow a fallback to the failing case.** Rewriting every unsupported sliding mode to FP16 would have looked tidier, but it would also have changed truncation and snapshot reuse for configurations that worked. Only the panicking combination needed a change.
- **Scope "warn once" to the entity the warning is about.** A process-wide static is the wrong scope for a per-model condition in a multi-model process; a flag owned by the per-model worker and shared across its rebuilds matches the lifetime exactly. Emitting the warning at `run` rather than inside a builder step also removes the dependence on builder order.
