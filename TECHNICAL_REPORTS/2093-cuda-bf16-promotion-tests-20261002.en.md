# Technical Report: PR #2093 - Cast bf16 tables explicitly where the CUDA overlay demotes (issue #2087)

**Date**: 2026-10-02

**Status**: Implemented and validated on GB10 (`--features cuda`) and on a CPU-only Linux build; pending merge.

**Languages**: Rust (production code in the Nemotron VoiceChat TTS head, tests)

**Risk Level**: Low (two exact `astype` casts in the TTS path, no-ops on non-CUDA builds; the rest is test code)

## Executive Summary

Gating PR #2077 on GB10 left six lib-test failures that also failed on the base. Five of them were one cause: `src/lib/mlx-cpp/patches-cuda/dtype.cpp`, mlxcel's own CUDA overlay of MLX's promotion table, resolves bf16 with f32 to bf16 instead of upstream's f32. It is not an MLX bug at the pin and not an `mlxcel-core` wrapper: the overlay is deliberate (issue #636, the single-dtype bf16 decode graph) and stays. The tests and two production sites had assumed upstream promotion. The sixth failure, `gelu_approx_matches_mlx_nn_bit_for_bit`, was already fixed on main by #2080 and passes on CUDA unchanged.

## 1. Root Cause

`dtype::BFLOAT16 == 12` and `dtype::FLOAT32 == 10`, so the `left: 12, right: 10` assertions said a mixed op returned bf16. The overlay header states the change: the bf16 row's f32 column and the f32 row's bf16 column both read `bfloat16`. The table is compiled into the MLX library, so the rule holds for every device of a CUDA build, including `MLXCEL_DEVICE=cpu`. ROCm compiles upstream `dtype.cpp` and Metal never sees `patches-cuda/`, so only the `cuda` feature changes the rule.

Two consequences in the Nemotron VoiceChat TTS path were real output-dtype defects on CUDA, not only test artifacts:

- `RvqCodebooks::depthsum_embedding` documents an f32 sum, but added gathered bf16 codebook rows to an f32 `zeros`, so on CUDA the sum came back bf16.
- `MogHead::infer` gathered `proj_mus` and `low_mat` slabs (stored bf16 by design, see `promotes_to_f32`) and matmul'd them against the f32 activation, so `mu`, and with it the sampled latent, came back bf16.

## 2. Changes

- **Probe test** `audio::f32_weights::tests::mixed_bf16_f32_promotion_follows_the_build`: asserts the dtype of `add` (both operand orders), `matmul` (both orders) and `fast_rms_norm` for an f32 activation against a bf16 operand. It expects f32 without the `cuda` feature and bf16 with it, so a change to either rule fails loudly. The issue asked for f32 on both; that cannot hold while the overlay stays, so the probe pins the contract that does hold and the production code casts explicitly.
- **`rvq.rs`**: the gathered rows are cast to f32 before the add.
- **`mog_head.rs`**: the gathered `mus` and `low` slabs are cast to the activation dtype before their matmuls. bf16 to f32 is exact and is the same `astype` upstream promotion inserts, so non-CUDA output is bit-identical, and the cast is a no-op when the dtypes already match.
- **`f32_weights.rs` tests**: the "promoted" reference is MLX's implicit promotion off CUDA and an explicit in-graph `astype(weight, float32)` on CUDA (`as_promoted`). The module doc now says that on CUDA the load-time precast is required, not only faster.
- **`fastconformer/tests.rs`**: the reference encoder and projection are built from the bf16 weights cast to f32, which is what upstream promotion computes, instead of relying on implicit promotion.
- **`tts_tests.rs`**: `mog_infer_shapes_are_finite_with_guidance` loads its weights the way `RvqEarTtsModel` does (`promoted_subset` with `promotes_to_f32`), so the test exercises the production load path rather than an all-bf16 head that production never builds.

## 3. The gelu Test

`gelu_approx_matches_mlx_nn_bit_for_bit` hardcoded values measured on Metal. #2080 (`5486e404`, merged 2026-09-30) replaced them with a same-device transcription of `mlx.nn.gelu_approx` compared bit for bit over 4097 points. The 1-ulp gap was per-backend `tanh`/`power` kernel rounding. Neither side is compiled, there is no matmul, and the graph is element-wise, so fusion, FMA contraction across ops and TF32 are not involved. It passes on CUDA and on the CPU build with no change here.

## 4. Validation

GB10, `--features cuda --release`, each run under `gpu-lock` with `--test-threads=1`:

| Filter | Result |
|---|---|
| `audio::f32_weights` (incl. probe) | 4 passed |
| `audio::fastconformer` | 14 passed |
| `models::nemotron_voicechat::tts` | 13 passed |
| `models::gemma3_backbone` (incl. gelu) | 5 passed |

With the two production casts reverted and the tests kept, `depthsum_ignores_mask_index` and `mog_infer_shapes_are_finite_with_guidance` fail on CUDA, so the casts are what makes them pass.

CPU-only Linux build (no GPU feature, separate target directory): the same four filters pass with the same counts (4, 14, 13, 5), and there the probe asserts f32. `MLXCEL_DEVICE=cpu` on the CUDA build was not used as a substitute, because the overlay is compiled into the library and applies on every device of that build.

## 5. Open Items

- **The TTS backbone still demotes on CUDA.** `RvqEarTtsModel` keeps the Gemma norm weights (`1 + w` built in the stored dtype), `bos_emb`, `null_emb` and `audio_prompt_projection_w` as stored bf16, and each meets the f32 stream. Off CUDA they promote it to f32 as the reference does; on CUDA, by the rule the probe pins for `add`, `matmul` and `fast_rms_norm`, they demote it to bf16. The fixes here make the code embeddings and the MoG head f32, but the backbone between them is still bf16 on CUDA. Confirming the impact needs a real Nemotron VoiceChat checkpoint, which this host does not have, so it is left for a follow-up.
- **The plain Linux CPU lib-test build does not link on main.** `src/lib/mlx-cpp/turbo/kv_inplace_write.cpp` (#1961) references `mlx::core::copy_gpu_inplace`, which a no-GPU MLX build does not define, so `cargo test --release --lib` fails at the link step before running any test. The CPU results above were produced with a linker wrapper that ignores that one unresolved symbol (`-Wl,--unresolved-symbols=ignore-in-object-files`); the symbol is only reached on a GPU stream.
