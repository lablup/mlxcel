# PR #2435: Use the attention dtype for mixed bf16/fp16 paths

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints before merge
**Risk**: Low (every changed decision is the same as before whenever q, k and v share a dtype; the snapshot change turns a silent no-op into a refusal; seven tests added)

## Summary

Issue #2422 collects four defects that review of #2417 found on the mixed bf16/fp16 surface: the CUDA query-chunk gate, the Gemma 2 softcap composite, a contradicting overlay header, and a wrong snapshot skip reason under turbo4. A bf16 model on an fp16 TurboQuant KV cache attends in `result_type(q, k, v)`, which is float32. This PR makes each of the four read that type instead of q's dtype or a wrong assumption. While writing the snapshot test it also found that a populated symmetric `turbo4` `KVCache` returned `Ok` from the snapshot path and stored nothing, which is fixed by the same change.

## Problem

All four items come from one fact: bf16 + fp16 promotes to float32, but four places still reasoned from q's dtype or from a comment.

1. **CUDA chunk gate (MEDIUM).** `cuda_sdpa_materializes_scores` judged cuDNN flash eligibility from `array_dtype(q)`, and `materializing_sdpa_query_chunk` sized score rows from it. For bf16 q over fp16 K/V, MLX runs the materializing fallback at 4 bytes per score. The gate either skipped chunking (head_dim <= 128) or budgeted 2 bytes per score, so long prompts could exceed the memory budget and OOM.
2. **Softcap composite (LOW).** `compiled_softcap_sdpa` and `_gqa` cast the mask to q's dtype and returned v's dtype. Fast SDPA already uses `result_type(q, k, v)` for both.
3. **Overlay header (LOW).** `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp` said bf16 + float16 gives bf16, while its table and a later note give float32.
4. **Snapshot skip reason (LOW).** `snapshot_standard` and `snapshot_rotating` checked that keys and values are both present before they checked the cache mode. Turbo4Asym keeps V in `v_packed`, so Gemma 4 logged `rotating cache has only one of keys/values` instead of naming the mode. `docs/turbo-kv-cache.md` says non-FP16 donation fails closed by design, so the reason was wrong, not the behavior.

### Found along the way: a silent snapshot gap

Testing item 4 with a populated symmetric `turbo4` `KVCache` (keys and values both `None`, offset 8) showed that the snapshot returned `Ok` and stored nothing, as a probe test confirmed on main. The keys/values pairing check treated "both absent" as an empty cache and skipped it. A populated quantized cache therefore looked the same as an empty one, and a donated snapshot silently lacked that layer's state. Checking the mode first closes it.

## Technical decisions

- **One bridge helper for the dtype.** `sdpa_attention_dtype(q, k, v)` sits next to `sdpa_mask_for_inputs` and reads the backend's promotion table. On CUDA that is the patched overlay table, so the Rust gate cannot drift from what the kernel will do. cuDNN eligibility and score row bytes both use it, and `cuda_sdpa_materializes_scores` now takes `k`.
- **Gate decision as a pure function.** The budget is read once through a `OnceLock`, which makes the real `attention` entry untestable at other budgets. The decision is factored into `sdpa_query_chunk_within_budget(.., budget)`, which the test calls with an explicit budget. The maskless causal path in `lib.rs` shares the gate.
- **Softcap uses the same helpers as fast SDPA.** Masks go through `sdpa_mask_for_inputs` and the output is cast to the attention dtype, including the grouped decode branch.
- **Mode check before the pairing check.** A populated non-FP16 cache is refused by mode first. Empty quantized layers (KV-shared) are still skipped, so the Gemma 4 donation path keeps working.
- **Overlay: fix the comment, pin the table.** Only the header was wrong. A test pins `result_type(bf16, fp16) == float32` on CUDA builds.

### Order-dependent three-way fold on the CUDA overlay

The new promotion-table test exposed one more fact, now noted in `dtype.cpp`. The patched fold is order-dependent on CUDA: `result_type(bf16, fp16)` is float32, but `result_type(fp16, bf16, bf16)` is bf16, because the first pairwise step lands on a type that the next step keeps. The gate and the softcap path call with `(q, k, v)` where k and v come from TurboQuant (always fp16), so the order is bf16, fp16, fp16 and the result is float32. TurboQuant never produces the fp16-first order that gives bf16. The note records this so a future caller that reorders arguments does not assume a symmetric fold.

## Changes

- `src/lib/mlxcel-core/cpp/mlx_cxx_internal.h`, `mlx_cxx_bridge.h`, `mlx_cxx_bridge.cpp`: `sdpa_attention_dtype`; softcap and softcap GQA use `sdpa_mask_for_inputs` and return the attention dtype.
- `src/lib/mlxcel-core/src/layers.rs`: `sdpa_query_chunk_within_budget`, gate and row sizing from the attention dtype; tests `query_chunk_gate_sizes_mixed_half_kv_from_the_attention_dtype` and `sdpa_attention_dtype_follows_the_backend_promotion_table`.
- `src/lib/mlxcel-core/src/lib.rs`: the maskless causal path shares the gate.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: `test_compiled_softcap_sdpa_mixed_half_kv_attends_in_float32`.
- `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`: corrected header and the order-dependence note.
- `src/models/kv_snapshot.rs`: mode checked before the pairing check in `snapshot_standard` and `snapshot_rotating`.
- `src/models/kv_snapshot_tests.rs`: `populated_turbo_caches_are_refused_by_mode`, `empty_turbo_caches_are_skipped_not_refused`.
- `src/models/gemma4.rs`: `gemma4_unified_mask_tests::turbo4_snapshot_donation_names_the_mode`, through `Cache::snapshot_into`.
- `docs/turbo-kv-cache.md`, `CHANGELOG.md`.

Total: 12 files, +471 / -30.

## Behavior change

When q, k and v share a dtype, the gate, the softcap path and the overlay behave as before. For bf16 over fp16 on CUDA, the gate now chunks at head_dim 128 and sizes rows at 4 bytes per score at head_dim 256. The softcap composite returns float32 for that mix instead of v's dtype. Every `kv_snapshot` user now refuses a populated non-FP16 cache with a mode-named error: Gemma 3, AFMoE, Gemma 4, Llama 4 and Muse Glimmer. Qwen3-Next/3.5, LFM2 and Bailing already checked the mode first. Jamba, Granite MoE Hybrid, Nemotron-H, Plamo2 and Falcon-H1 get no injected KV mode.

## Verification

- Chunk gate: `layers::tests::query_chunk_gate_sizes_mixed_half_kv_from_the_attention_dtype` also asserts the same-dtype decisions. `layers::tests::sdpa_attention_dtype_follows_the_backend_promotion_table` pins the table.
- Softcap: `ffi_tests::test_compiled_softcap_sdpa_mixed_half_kv_attends_in_float32` checks the output dtype and compares with a float32 reference. Its mask carries finite biases, so reverting only the mask cast also fails it.
- Snapshot: `models::kv_snapshot::tests::{populated_turbo_caches_are_refused_by_mode, empty_turbo_caches_are_skipped_not_refused}` and `gemma4_unified_mask_tests::turbo4_snapshot_donation_names_the_mode`.
- Revert check: with the fixes reverted, the chunk-gate, softcap, populated-turbo and Gemma 4 tests fail (`None` vs `Some(16)`, output dtype fp16, `has only one of keys/values`). The promotion-table test and the empty-cache test pass on main by design, since they guard behavior that was already correct.
- Run on GB10 (`--features cuda`, `--test-threads=1`, all passing): `-p mlxcel-core` `layers::tests` (102), `softcap` (18) and `ffi_tests` (112), plus root `models::kv_snapshot` (25), `models::gemma4` (121), `gemma3`, `gemma2`, `afmoe`, `llama4` and `muse_glimmer`. Clippy `-D warnings` (core and root `--lib --tests`), fmt and the `scripts/ci` checks are clean.
- Real checkpoints, all with `MLXCEL_SDPA_DETERMINISTIC=1` and temperature 0. "Before" is main `cadc6366`, and each before-arm showed self-parity.
  - `mlxcel-server -m ~/models/mlx/gemma-4-26b-a4b-it-4bit --kv-cache-mode fp16+turbo4`, 1987-token prompt, 300 tokens: coherent, response byte-identical to before. The skip reason changes from `rotating cache has only one of keys/values` to `rotating cache mode Turbo4Asym is not supported by model-state snapshots`.
  - `mlxcel generate`, 200 tokens: byte-identical before and after on `qwen3-0.6b-4bit`, and on `gemma-2-2b-it-4bit` with default KV and with `fp16+turbo4`. That checkpoint is fp16, so both of its arms are same-dtype.
  - The mixed softcap path needs a bf16 Gemma 2, and every mlx-community Gemma 2 is fp16. A local bf16 conversion of `gemma-2-2b-it-4bit` (not committed) is byte-identical with default KV. With `fp16+turbo4` it diverges at character 645 and stays coherent, which is expected from float32 masked scores.

## Not verified and follow-up

- The mixed softcap path was exercised only on a local bf16 conversion, not on a published bf16 Gemma 2 checkpoint, because none exists in mlx-community.
- The chunk-gate memory effect is verified by the gate decision test, not by an OOM reproduction on a long prompt.
- The three-way promotion fold is documented, not changed. Any caller that passes dtypes in another order should go through `sdpa_attention_dtype`.
