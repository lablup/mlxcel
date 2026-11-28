# PR #2776: Honor KV mode, KV bound and release for model-owned caches

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low to medium (eight families start storing quantized KV when an operator asks for it, which changes their numerics under a non-FP16 mode; every model-owned family under `--context-shift` with a KV bound now stops at the bound instead of growing past it)

## Executive Summary

PR #2776 (closes #2766) closes the three residuals that #2748 left for families whose KV caches live inside the model. `--kv-cache-mode` now reaches the caches of ExaOne 4, GPT-OSS, Ministral 3, Step 3.5 (and the Step-3.7 VLM), Mellum, Laguna and NemotronNAS; LongCat-Flash and LongCat-Flash-Ngram refuse a quantized table with one warning and report FP16, because their MLA caches hold a latent pair no quantized mode supports. Under `--max-kv-size` (or the `--ctx-size` window) with `--context-shift`, the scheduler now serves any `ModelOwned` layout with context shifting off, so the bound holds through admission refusal and a `truncated: true` stop rather than the trim silently doing nothing. For the single-slot families, the scheduler records which sequence last reset the shared slot and frees that slot when exactly that sequence is released, so an idle server no longer keeps the last request's KV resident. ExaOne 4 Int8 was verified on a real checkpoint through `logit_trace`, `mlxcel generate` and `mlxcel-server`; NemotronNAS slot release was verified on the 49B 4-bit checkpoint by GPU memory; LongCat remains covered by tiny-config tests only.

## 1. Problem Statement

### Where the KV mode, the bound and the release normally act

The server and CLI resolve a KV mode per layer and apply it in two places: the scheduler's pool caches, and the model through `LanguageModel::set_kv_cache_layer_modes` (a default no-op in `src/lib/mlxcel-core/src/generate.rs`). `--max-kv-size` with `--context-shift` is enforced by `enforce_max_kv_size_for` in `src/server/batch/scheduler/paged_layout.rs`, which walks and front-trims the pool caches. A sequence's KV is freed when the engine closes it. All three mechanisms assume the scheduler's pool holds the caches. For a model with a `ModelOwned` sequence-state layout the pool entry is empty, and the caches live in `ModelOwnedSequenceState` (batching families) or in one `RefCell` slot on the model (NemotronNAS, LongCat after #2748).

### `--kv-cache-mode` was dropped without a warning

#1330 implemented the model-side hook for Gemma 3/4, Qwen 3.5/3.6/3.8, Qwen3-Next, LFM2, Llama 4 and AFMoE. The eight families in this PR never implemented it and built every cache with `KVCache::new()` / `RotatingKVCache::new(..)`, which are FP16. The pool caches received the mode, but they are empty for these layouts. A user who passed `--kv-cache-mode int8` got FP16 KV and full FP16 memory use, and `kv_cache_layer_modes()` returned `None`, so `examples/logit_trace` with `MLXCEL_TRACE_KV_CACHE_MODE=int8` failed with its "cannot reach the caches" error.

### `--context-shift` let model-owned caches grow past the bound

With context shifting on, the KV bound is enforced only by the trim, and the trim found no pool caches to cut. Generation on any `ModelOwned` layout (not only the eight families: Gemma 3/4 and Qwen 3.5 too) continued past `--max-kv-size` until the context window. With shifting off (the default), the bound already held, because admission refusal (`admission.rs`) and the finish step (`finish.rs`) count tokens rather than walk caches.

### A single-slot family kept the last request's KV

NemotronNAS and LongCat cannot batch and keep one shared slot. The scheduler reset it only at the next prefill (`prefill.rs`, `planned_prefill.rs`), so after a request finished its whole KV stayed allocated while the server idled; on the 49B NemotronNAS checkpoint that is about 250 MiB for a 973-token request. Resetting on every sequence release would be wrong: the #2748 security review pointed out that cancelling a queued request while another one decodes would then wipe the in-flight sequence's state.

## 2. Technical Decisions

### Follow the #1330 Gemma 3 pattern per family

Each of the six batching wrappers (ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum, Laguna) gains a `KvCacheLayerModes` field from `src/models/model_owned.rs`, a `make_caches_with_modes(Option<&KvCacheLayerModes>)` builder that calls `KVCache::new_with_mode` / `RotatingKVCache::new_with_mode` with `mode_for_layer(layer_idx)`, and a private `fresh_caches()` that every cache-creating path now uses: `reset_caches`, `prepare_sequence_state`, and the lazy builder in `with_or_create_sequence_state`. `set_kv_cache_layer_modes` stores the table and rebuilds the fallback slot; `kv_cache_layer_modes` returns it. The public `make_caches()` keeps FP16, so callers outside the model that build their own caches are unchanged.

The decision rule from the issue is that a family honors the mode when its attention reads K/V through `update_and_fetch`, which stores and dequantizes every mode. That holds for all seven honoring families. Routing every builder through `fresh_caches()` matters because the per-sequence path was the one the server uses; patching only the fallback slot would have made the CLI warmup quantized and every served sequence FP16.

### NemotronNAS indexes the table by block, not by slot

NemotronNAS keeps one cache per block that has an attention subblock, so slot `i` is not layer `i` once a `no_op` block appears. The server resolves the table over `num_layers()` (one entry per block), so `fresh_caches` now enumerates the blocks first, filters to attention-bearing ones, and takes `mode_for_layer(block_index)`. `nemotron_nas_kv_mode_table_is_indexed_by_block_layer` pins this: with block 1 a `no_op`, slot 1 belongs to block 2 and must carry block 2's mode (`[Int8, Int8, Fp16, Int8]` expected; main yields all FP16). A `replace_with_linear` block keeps a slot in the table's mode that it never writes, which costs nothing. The comment on `supports_batching()` drops its claim that the KV-mode table does not describe this family; only the paged layout still does not.

### LongCat stays FP16 and says so

LongCat's MLA stores `(kv_latent, k_pe)` through one `update_and_fetch`: the "K" slot is `kv_lora_rank` wide and the "V" slot is the RoPE key, so neither is a per-head K/V row. `mla::latent_layout_supports_mode` already restricts that layout to FP16, since a Turbo codebook calibrated on one slot would be applied to the other and an asymmetric mode would quantize the RoPE key while leaving the latent exact. The PR follows the `iquestloopcoder` precedent: `set_kv_cache_layer_modes` logs one warning (a `Cell<bool>` guard, because `DirectEngine` injects modes before every sequence) and `kv_cache_layer_modes` reports FP16 for every layer. The scheduler's startup line (`log_effective_kv_cache_application`) now prefers the model's own report over the configured table, so the operator sees FP16 for LongCat rather than the requested mode.

Two edits sit outside the issue's file list. `MLA_LATENT_CACHE_FAMILIES` in `src/lib/mlxcel-core/src/mla/cache.rs` gains `longcat_flash`: both `model_type`s load `LongcatFlashNgramModel`, but only the n-gram one was listed, so the CLI resolver mapped the plain variant to the requested mode instead of FP16. `src/vision/step3p7.rs` forwards both KV-mode hooks to its Step 3.5 backbone; without that the VLM kept the trait's no-op and ignored the mode even after the backbone honored it.

### Turn context shift off instead of adding a per-family trim

`BatchScheduler::with_context_retention` now stores `effective_context_retention(policy)`: when a KV bound is set, the policy enables `context_shift`, and the model's `sequence_state_layout().backend` is `SequenceStateBackend::ModelOwned`, it returns the policy with `context_shift: false` and logs one warning per process (a static `AtomicBool`, because the worker rebuilds the scheduler on every wake from idle sleep). `with_max_kv_size` reapplies the same function, so the result does not depend on whether the bound or the policy is installed first (`context_shift_fallback_holds_when_the_bound_is_set_last`).

The alternative, a per-family front-trim hook, was rejected in the issue: rotating, dense and latent layouts would each need their own trim, and a wrong trim corrupts attention silently. Refusal plus `truncated: true` is a visible, already-tested stop. The cost is that an operator who wanted unbounded rolling generation on Gemma 3/4, Qwen 3.5 or the eight families gets a stop at the bound; before this PR they got unbounded growth that ignored the bound they had set, which was not the rolling window either.

### Release only the sequence that filled the slot

`BatchScheduler` gains `model_owned_slot_holder: Option<SequenceId>`. Both prefill reset sites now call `reset_model_owned_slot_for(seq_id)`, which keeps the old `make_caches()` reset for a non-batching model and records the holder. `release_sequence_caches` calls `release_model_owned_slot_if_held_by(seq_id)`: if the id is the holder and the model does not batch, it calls `reset_runtime_state()`, clears the holder, and calls `mlxcel_core::clear_memory_cache()` so MLX returns the freed buffers rather than keeping them in its cache. Any other id is a no-op, which is the property the #2748 review asked for: a queued request that never prefilled cannot wipe the in-flight slot.

The check runs for every non-batching family the scheduler serves, not only NemotronNAS and LongCat. For Phi4MM, the OCR wrappers and Nemotron VoiceChat `reset_runtime_state()` only replaces fallback state. All of them are clamped to one active sequence, so the holder can never be a sequence that is still decoding.

### Make the dense control prove a shift happened

The first version of `dense_layout_keeps_context_shift_under_a_kv_bound` used the paged default decode storage, where a pool-backed FP16 cache's front trim is a no-op, so the test passed without any shift. Commit `28991823` runs it on `Dense` decode storage with a one-layer zero-projection Llama (`cycle_llama`, which maps token `t` to `t + 1` and never emits a stop) and asserts that the pool caches' live length reaches the bound of 16 and never exceeds 17. Without that, the control would not have distinguished "dense still shifts" from "nothing shifts anywhere".

## 3. Change Summary

26 files, +1267 / -96. Three commits (`91fe46c0` fix, `ef0b6c95` docs correction, `28991823` dense control) plus a merge of origin/main (`69650626`).

| Area | Files | Change |
|---|---|---|
| Batching model-owned families | `exaone4.rs`, `gpt_oss.rs`, `ministral3.rs`, `step3p5.rs`, `mellum.rs`, `laguna.rs` | `KvCacheLayerModes` field, `make_caches_with_modes`, `fresh_caches()` on every builder path, the two trait hooks, `#[cfg(test)] sequence_kv_modes` accessor |
| Single-slot families | `nemotron_nas.rs`, `longcat_flash_ngram.rs` | NemotronNAS: mode table indexed by block, `slot_live_len` / `slot_kv_modes` test accessors. LongCat: FP16-only refusal with one warning |
| VLM wrapper | `src/vision/step3p7.rs` | forward both KV-mode hooks to the Step 3.5 backbone |
| MLA resolver | `src/lib/mlxcel-core/src/mla/cache.rs` | `longcat_flash` added to `MLA_LATENT_CACHE_FAMILIES` and its test |
| Scheduler | `scheduler/config.rs`, `mod.rs`, `handoff.rs`, `prefill.rs`, `planned_prefill.rs`, `paged_layout.rs` | context-shift fallback, `model_owned_slot_holder`, holder-scoped release, startup log uses the model's reported modes |
| Engine comment | `src/lib/mlxcel-core/src/engine/direct.rs` | only NemotronNAS and LongCat keep one shared `RefCell` slot; the other six key state per sequence |
| Tests | `model_owned_kv_mode_tests.rs` (new), `scheduler_model_owned_slot_tests.rs` (new), `nemotron_nas_engine_tests.rs`, `longcat_flash_ngram_engine_tests.rs`, visibility changes in `gpt_oss_tests.rs` and `mixed_cache_batched_decode_tests.rs` | 34 new tests |
| Docs | `docs/supported-models.md`, `docs/llama-server-compat.md` | per-family KV-mode table with the sliding-layer note; the context-shift fallback |

Per-family outcome:

| Family | `--kv-cache-mode` |
|---|---|
| ExaOne 4, GPT-OSS, Ministral 3, Mellum, Laguna | honored per layer |
| Step 3.5 and the Step-3.7 VLM | honored per layer |
| NemotronNAS | honored, indexed by block layer index |
| LongCat Flash / LongCat Flash N-gram | FP16 only, one warning, reports FP16 |

## 4. Trade-offs and Residuals

- **Sliding layers stay FP16 under most modes.** These families build sliding-window layers as `RotatingKVCache`, which stores quantized data only for `fp16+turbo4`; under `int8`, `turbo4`, `turbo4-delegated` and `fp16+turbo3` it keeps FP16. So under `int8` the global layers of GPT-OSS, Mellum, Laguna, Step 3.5 and hybrid ExaOne 4 checkpoints are quantized and their sliding layers are not. Gemma 3/4 already have the same limit. The first commit's docs claimed otherwise; `ef0b6c95` corrected them.
- **The startup report overstates the sliding layers.** `kv_cache_layer_modes()` returns the requested table, so for those layers the server's `applied_layers` count and `logit_trace` report a mode the rotating cache does not store. Reporting the effective storage mode would need the rotating cache's fallback rule exposed to the model.
- **Laguna's DFlash round loop** keeps its own FP16 caches; the table does not reach them.
- **Behavior change for every model-owned family under `--context-shift`.** With a bound set, Gemma 3/4, Qwen 3.5 and the families here now stop at the bound with `truncated: true`. The warning names the cause. A real per-layout trim would remove the fallback; it is deferred.
- **The fallback is one-way within a builder chain.** Once `effective_context_retention` has forced `context_shift` off, a later `with_max_kv_size(None)` does not turn it back on. The scheduler is built once per wake, so this does not occur in practice.
- **LongCat is unverified on real weights** (560B, no host fits), as in #2748. Its FP16-only report and the resolver mapping are covered by tiny-config tests.
- **The NemotronNAS memory drop was not compared against main side by side.** The measurement below shows memory returning close to the loaded baseline after each request; main was not run under the same load.
- **Metal and ROCm were not run locally**; the gate below is GB10 (CUDA).

## 5. Validation

Failing-first: the new tests were run against main's behavior (hooks removed, fallback and release disabled, `longcat_flash` off the MLA list) with only the test accessors kept. Every new test failed except the two intended controls (`dense_layout_keeps_context_shift_under_a_kv_bound`, `context_shift_is_kept_without_a_kv_bound`). Representative failures:

| Check | main | PR |
|---|---|---|
| `kv_cache_layer_modes()` after injecting Int8 (all eight families) | `None` | `Some([Int8, ..])` (FP16 for LongCat) |
| NemotronNAS slot modes with a `no_op` block 1 and FP16 on block 2 | `[Fp16, Fp16, Fp16, Fp16]` | `[Int8, Int8, Fp16, Int8]` |
| `model_owned_layout_stops_at_the_kv_bound_under_context_shift` (bound 16) | sequence reaches 6 + 40 | stops at the bound, `truncated: true` |
| `completing_the_slot_holder_frees_the_single_slot` | slot live length 11 | 0 |
| holder after completion | `Some(SequenceId(0))` | `None` |

Each honoring family also has an 8-token greedy test under Int8 that asserts finite logits and the same first token as FP16.

Local gate on GB10 (`MLX_ENABLE_TF32=0`): 34 new tests pass; `--lib models::` 2491 passed; `--lib server::batch::` 587 passed; mlxcel-core `engine::`, `mla::` and `cache::turbo::allowlist` 130 passed; KV advisor, registry and turbo-args tests 109 passed; `cargo fmt --check`; clippy `-D warnings` on mlxcel (lib, tests, `logit_trace`) and mlxcel-core (lib, tests).

Real checkpoints on GB10, under `gpu-lock`:

- `logit_trace` with `MLXCEL_TRACE_KV_CACHE_MODE=int8` on `exaone-4.0-1.2b-4bit` succeeds; on main `install` fails with the cannot-reach error. Against the FP16 trace: top-1 disagreement 15/256 overall and 0/87 on decided positions, perplexity -0.3%, which shows Int8 reaches the caches and stays close to FP16.
- `mlxcel generate --kv-cache-mode int8` on the same checkpoint (1370-token prompt, greedy) completes, and the continuation diverges from FP16 after a few tokens, which it could not do if the mode were dropped.
- `mlxcel-server --kv-cache-mode int8` on ExaOne 4 completes a request and logs `kv_cache_mode_effective=int8 applied_layers=30/30`.
- `mlxcel-server --max-kv-size 128 --context-shift` on ExaOne 4 logs the fallback warning; a 10-token prompt with `ignore_eos` stops at 117 generated tokens with `truncated: true`, `stop_type: "limit"`.
- `Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit` server, two identical greedy 973-token requests: byte-identical output. GPU memory (nvidia-smi) 27006 MiB loaded, 28774 MiB peak during each request, 27046 MiB after the first and 27014 MiB after the second. One request's KV is about 250 MiB at this length, so the slot is freed on release.

## 6. Learning Points

- **A hook on the trait is only as good as the builders behind it.** A model-owned family has several paths that create caches: the fallback slot, `prepare_sequence_state`, and the lazy builder inside `with_or_create_sequence_state`. Each one must use the injected table, or the mode reaches some sequences and not others. A single `fresh_caches()` per wrapper is the cheap way to keep them in sync, and a test that reads the modes back from a real sequence (not from the fallback) is what catches a miss.
- **Index per-layer tables by the index the producer used.** The server resolves modes over `num_layers()`. A model whose cache list is a filtered subset of its layers (NemotronNAS, any family with `no_op` or linear-replacement blocks) must map through the layer index, and the test needs a gap in the layer list to tell the two indexings apart.
- **A no-op trim is a silent bound violation.** Any enforcement that walks the scheduler's pool will do nothing on a `ModelOwned` layout. When a mechanism cannot act on a layout, refusing it at startup with a warning is safer than accepting it.
- **A control test must be able to fail for the right reason.** The first dense control passed because the paged default turns the trim into a no-op, so it proved nothing about shifting. Assert the observable effect (live length capped near the bound) rather than the absence of an error.
- **Scope a release to its owner.** For shared single-slot state, record who filled the slot and free it only on that owner's release; a blanket reset on any release turns a cancelled queued request into a corruption of the running one.
