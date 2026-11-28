# PR #2441: Load mlx-lm NVFP4 per-expert global_scale natively

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; Metal is left to the Metal nightly, and the items under "Not addressed" remain open
**Risk**: Medium (changes the `gather_qmm` FFI signature, the `SwitchLinear` scale handling and the Laguna sidecar key, and enables ModelOpt NVFP4 repacking for every text family)

## Summary

Closes #2351. mlx-lm's `quantize_model(..., mode="nvfp4", global_scale=True)` (ml-explore/mlx-lm#1844) writes `max|w[e]|` per expert as `{switch layer}.global_scale` and flags the layer with `"global_scale": true`. MLX's `gather_qmm` dequantizes with `global_scale / (448 * 6)`. mlxcel read the same key as the compressed-tensors output multiplier, so such an export decoded with every routed expert 2688 times too large. This PR keeps the two scales apart, passes the native one to `gather_qmm`, and extends ModelOpt NVFP4 expert loading beyond Gemma 4.

## Problem

- `SwitchLinear::from_weights_with_mode` accepted any `{prefix}.global_scale` as the in-process output multiplier (`1 / weight_global_scale`, applied after `gather_qmm`). The rank-1 shape check also accepted the mlx-lm tensor, so the load succeeded and the output was silently wrong. That key was produced only in process, by the Laguna transcode.
- The FFI could not pass a native scale: `gather_qmm` had no `global_scale` parameter and the C++ shim passed `std::nullopt`. MLX requires f32 dtype, shape `w.shape[:-2]`, `nvfp4` mode, indices present and a GPU stream.
- ModelOpt `weight_scale_2` handling was reached only for Gemma 4. The per-expert stacking loops in DeepSeek-V3, Mistral 4 and PhiMoE iterated only `weight`, `scales` and `biases` and dropped any scale sidecar.
- Two further problems blocked any mlx-lm nvfp4 MoE export from loading: `SwitchLinear::from_weights` declared bias-free block-float planes affine and refused them, and the quantized embedding lookup in the bridge requested a uint8 output dtype for block-float tables, which aborted the first lookup (`[dequantize] Only real floating types are supported but output dtype == uint8`).

## Technical decisions

- The on-disk key and the in-process key no longer share a name. The Laguna sidecar is renamed `inv_global_scale` (`laguna_sanitize.rs`, `laguna.rs` detection, tests). Laguna's dense shared-expert `[1]` `.global_scale` keeps its name because it feeds `UnifiedLinear`, and dense global scales are out of scope.
- `SwitchLinear::Quantized` carries two fields: `inv_global_scale` (the post-matmul multiplier) and `native_global_scale` (handed to `gather_qmm`). `quantized_parts` returns `None` when either is present, so the fused paths fall back to `forward`. `SwitchGLU::forward_with_expert_scales` passes through `forward_indexed` and gets the native scale too.
- The flag comes from the config, not from the key. `apply_native_global_scale_flags` reads the `quantization` / `quantization_config` entries (top level and `text_config`) and moves each flagged `{path}.global_scale` to `{path}.native_global_scale`. It is called by `load_text_weights` and, through `sanitize_gemma4_nvfp4_weights`, by every Gemma 4 loader. This was chosen over adding a parameter to `from_weights_with_mode` because about 55 family files construct `SwitchLinear` / `SwitchGLU`. The decision is still made from the per-layer config entry, once, before any family sanitizer renames keys.
- Failures at load, each naming the prefix: a flag without its tensor; a flag on a layer whose weight is not a `[num_experts, out, in]` stack; a `.global_scale` left at a switch prefix that was never flagged; a checkpoint that ships `native_global_scale` itself. The native scale is also validated against MLX's preconditions (nvfp4 mode, one f32 entry per expert, a Metal or CUDA GPU stream), so a violation is a load error naming the prefix and backend, not a mid-generation throw.
- `SwitchLinear::from_weights` infers bias-free block-float planes with uint8 scales (nvfp4 / mxfp4 / mxfp8, by words-per-group) from the shapes.
- ModelOpt detection reuses Gemma 4's `is_modelopt_nvfp4`, lifted to a module-level `pub(crate)` fn. `load_text_weights` runs `normalize_nvfp4_keys` and `repack_nvfp4_weights_to_quantized` for any text checkpoint that declares ModelOpt NVFP4 in `quantization_config` (top level or `text_config`) or in `hf_quant_config.json`, which is rewritten into the `config.json` shape before the check. `quant_method` and the producer name match case-insensitively.
- The repack decodes F8_E4M3 block scales that MLX's safetensors loader returns as raw uint8 bytes. A numeric cast would read the bytes as 0..255.
- A shared helper `stack_expert_leaf` plus `EXPERT_PLANE_LEAVES` in `switch_layers.rs` is used by the DeepSeek-V3, Mistral 4 and PhiMoE stacking loops and the generic per-expert stacker, so the per-expert `weight_scale_2` sidecar becomes a `[num_experts]` `inv_global_scale` vector in the same pass as weight/scales/biases. PhiMoE and Mistral 4 read the 16-wide block size from `quantization_config` (`modelopt_nvfp4_group_size`). DeepSeek-V3's expert path is affine-only; it now refuses a block-float plane by name instead of reporting a missing `.biases` key.

## Changes

- `src/lib/mlxcel-core/src/lib.rs`, `cpp/mlx_cxx_bridge.h`, `cpp/mlx_cxx_bridge.cpp`: `gather_qmm` gains a nullable `global_scale: *const MlxArray`; every existing caller passes null. A test-only `quantize_weights_with_global_scale` wraps `mlx::core::quantize(..., global_scale)`. The quantized embedding lookup no longer requests a uint8 output dtype for block-float tables.
- `src/models/switch_layers.rs`: the two scale fields, config-flag handling, load-time validation, mode inference, `stack_expert_leaf`.
- `src/models/sanitize.rs`: `apply_native_global_scale_flags`, ModelOpt repack for every text family, F8_E4M3 decode.
- `src/models/laguna_sanitize.rs`, `laguna.rs`, `laguna_layers.rs`, `laguna_tests.rs`: rename to `inv_global_scale`.
- `src/models/deepseek_v3.rs`, `mistral4.rs`, `phimoe.rs`: sidecar stacking through the shared helper.
- `src/models/gemma4.rs`: `is_modelopt_nvfp4` lifted to a module-level fn; flag resolution in the Gemma 4 sanitizer.
- `src/models/gpt_oss.rs`: refusal guard (see below).
- `scripts/mlx_lm_logit_trace.py`: an mlx-lm reference tracer in the `examples/logit_trace` TSV format, reusing the tokenizer and chunk plan from `scripts/laguna_oracle_trace.py`.
- `docs/supported-models.md`: documents both formats.
- Other model files and two ROCm test files get a one-line argument update for the new `gather_qmm` and `SwitchLinear` signatures.

## Verification

All on the GB10 CUDA host (sm_121), `--profile test-fast --features cuda`. MLX at the pin supports `gather_qmm(..., global_scale)` for nvfp4 on CUDA (qmv and qmm_naive apply it, qmm_sm80 and the grouped-GEMM path decline it), confirmed first with a Python probe on mlx 0.32.4 and then by the Rust tests. Metal is left to the Metal nightly.

New tests, all run on CUDA:

- `switch_layers::global_scale_tests`: the native path against `x @ dequantize(w[e]).T * gs/2688` in unsorted and sorted layouts, a relative-L2 check against the float weights, flag stripped fails to load, flag without tensor fails, reserved key refused, nvfp4 / shape / exclusivity checks, ModelOpt sidecar stacking and application, and a partial sidecar refused.
- `sanitize_tests::load_text_weights_repacks_modelopt_nvfp4_for_non_gemma_families` (config.json and hf_quant_config.json detection, E4M3 bytes preserved), `load_text_weights_resolves_flagged_native_global_scales`, `load_text_weights_leaves_undeclared_triplets_alone`.
- Per-family stacking tests in `phimoe`, `mistral4` and `deepseek_v3`.

Revert checks:

- With part 1 reverted (native scale applied as the old post-multiplier, bare key accepted, flags ignored), `native_global_scale_matches_dense_dequantized_reference`, `global_scale_without_the_config_flag_fails_to_load`, `config_flag_without_global_scale_tensor_is_a_load_error`, `native_global_scale_needs_nvfp4_and_one_entry_per_expert` and `load_text_weights_resolves_flagged_native_global_scales` fail.
- Without the gpt-oss guard, `gpt_oss_expert_linear_refuses_a_per_expert_global_scale` fails. Without the unstacked-sidecar guard, `a_plane_whose_family_left_the_modelopt_sidecars_unstacked_is_refused` fails.
- With part 2 reverted (no sidecar leaf, no F8 decode), the PhiMoE, Mistral 4 and DeepSeek-V3 stacking tests, `per_expert_sidecars_stack_into_the_output_multiplier`, `a_partial_per_expert_sidecar_is_refused` and `load_text_weights_repacks_modelopt_nvfp4_for_non_gemma_families` fail.

Module suites run with `--test-threads=1`: switch_layers, laguna, sanitize, phimoe, mistral4, deepseek_v3, deepseek, gemma4, olmoe, qwen3_moe, inkling, kimi_k3, deepseek_v4, gpt_oss, glm4_moe, step3p5, nemotron_h, hunyuan, llama4, exaone, ernie and qwen3_next all pass. `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo check -p mlxcel-core --lib --tests`, `cargo fmt --check` and the five contract checks pass.

Real checkpoints:

- Laguna regression, `laguna-xs-2.1-nvfp4` (`poolside/Laguna-XS-2.1-NVFP4`), now on `inv_global_scale`: `MLXCEL_SDPA_DETERMINISTIC=1 mlxcel generate ... -n 64 --temp 0 --show-reasoning` is token-identical to `main` (8744c87f, built in the same worktree before the change; main's self-parity checked first).
- ModelOpt path that already worked, `gemma-4-31b-it-nvfp4`: same command, token-identical to `main`.
- Real mlx-lm export, `olmoe-1b-7b-0924-nvfp4-gs`. Recipe: `allenai/OLMoE-1B-7B-0924` (HF bf16; its config lacks `rms_norm_eps`, so `1e-5`, the transformers default, was added for mlx-lm), loaded with `mlx_lm.utils.load(..., lazy=True)`, floats cast to bf16, `quantize_model(model, config, None, None, mode="nvfp4", global_scale=True)`, `save(...)`. mlx-lm 0.32.2 (mlx 0.32.4, `mlx-cuda-13` wheel) already carries the `global_scale` path, so it was used instead of a `ba8a08f` checkout. 48 switch layers carry the flag. `main` refuses it at load (`quantization.mode is "affine" but the checkpoint ships no .biases plane`). On this branch, `mlxcel generate -p "The capital of France is Paris. The capital of Germany is" -n 40 --temp 0` gives "Berlin. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C." (mlx-lm greedy: "Berlin. The capital of Italy is Rome. ..."). With the config flags stripped it fails to load naming `model.layers.0.mlp.switch_mlp.gate_proj`.
- Teacher-forced logit trace on that export (corpus `docs/architecture.md`, `256 4 8 0`, reference `scripts/mlx_lm_logit_trace.py` on mlx-lm 0.32.2 CUDA, compared with `scripts/compare_logit_traces.py`):

| Arm | Disagreeing decided positions (gap >= 2) | Disagreeing overall | Other |
|-----|------------------------------------------|---------------------|-------|
| Fixed | 0 of 396 | 19 of 1024, all at reference gaps under 0.5 | perplexity 20.15 vs 20.13 |
| Native scale replaced by the old post-multiply | 396 of 396 | 1024 of 1024 | greedy output is garbage ("ByrPLAYpackavelorschKK Television...") |

- ModelOpt non-Gemma: no ModelOpt NVFP4 MoE checkpoint of DeepSeek-V3, Mistral 4 or PhiMoE is published at a size this host fits (`mistralai/Mistral-Small-4-119B-2603-NVFP4` is in Mistral's consolidated format), so the synthetic triplet tests cover the stacking. `nvidia/Qwen3-30B-A3B-NVFP4` was also tried: the 18624 triplets now repack and the experts stack, but the model still aborts in the dense fused QKV projection (`FusedQKVLinear` hardcodes affine and has no global-scale path), which is outside this issue. `main` aborts on the same checkpoint with a matmul shape error.

## Changes during review

Three follow-up commits closed silent-wrong-output cases found while extending ModelOpt loading to all text families:

- `6e13adf1`: families with their own weight/scales/biases stacking loop that load through the shared `SwitchLinear` (OLMoE, dots1, Solar Open, Cohere2-MoE and others) stacked the codes and block scales, left the per-expert `weight_scale_2` sidecar unread, and loaded as plain nvfp4 through the new mode inference with every routed expert off by its `weight_scale_2`. On `main` those checkpoints failed to load; with the PR they would have decoded wrong silently. `SwitchLinear::load` now fails a bias-free uint8-scale plane that carries neither global scale while per-expert `.global_scale` sidecars remain under its MoE block, naming the leftover key. A new `global_scale_tests` case reproduces the OLMoE-style stacking: it loads without the guard and is refused with it.
- `19567aa5`: gpt-oss's own `ExpertLinear` calls `gather_qmm` with the config mode and no tensor scale, so an mlx-lm export with `global_scale: true` would load and decode every expert wrong. The loader now refuses a plane carrying `.native_global_scale` or a bare `.global_scale`, naming the key.
- `d61ae3a8`: a flagged path whose weight is not a `[num_experts, out, in]` stack would have its scale renamed and never read, so the load funnel now refuses it. The ModelOpt detection also matches `quant_method` and the `hf_quant_config.json` producer name case-insensitively, as `is_modelopt_nvfp4` already does.

A security and performance review of `git diff 8744c87f..HEAD` found no CRITICAL or HIGH issues. It checked the FFI preconditions, `apply_native_global_scale_flags`, `stack_expert_leaf`, the ModelOpt repack byte-length guards and the decode path, and found them clean. It listed LOW items, none fixed in the review:

- `refuse_unstacked_expert_sidecars` scans every key once per bias-free block-float plane, so a load costs O(planes x keys). The review estimated under 1 s even for a 61-layer, 384-expert per-expert map left unstacked.
- Only `load_text_weights` and the Gemma 4 sanitizer run the flag resolver, so other loaders do not refuse a checkpoint-shipped `.native_global_scale`. The value is still checked against MLX's preconditions, so this cannot abort.
- A flagged path that is not a shared `SwitchLinear` plane has its scale moved to `.native_global_scale` and nothing reads it. The review suggested a post-build check that fails on any unconsumed `.native_global_scale` key.
- `modelopt_quantization_config` compared `quant_method == "modelopt"` case-sensitively. The case-insensitive match in `d61ae3a8` addresses this.
- Pre-existing: the DeepSeek v1 private loader treats a plane with `.scales` but no `.biases` as `Regular`, which aborts in `gather_mm` at the first forward pass.

After review, Laguna XS 2.1 greedy output is still token-identical to `main`, and the OLMoE export's greedy output is unchanged.

## Not addressed

- Dense fused QKV with NVFP4 + `weight_scale_2` (blocks Qwen3-MoE ModelOpt checkpoints).
- VLM loaders other than Gemma 4 do not resolve the flag. An mlx-vlm export with per-expert global scales (Blaizzy/mlx-vlm#2437) fails to load where the bare key reaches a shared `SwitchLinear` prefix, but the Gemma 4 unified loader's `split_fused_moe_key` (`src/loading/vlm_gemma_unified.rs`) splits only `.weight` / `.scales` / `.biases` of a fused `experts.gate_up_proj` and would leave a `.global_scale` orphaned. This is pre-existing and in a loading file outside this issue.
- The LOW follow-ups listed above: load-time key scan cost, flag resolution only in the text and Gemma 4 funnels, and the DeepSeek v1 private loader.
- Out of scope per #2351: dense-linear global scales (`QuantizedWeight::global_scale`) and `mlxcel` quantize/convert producing `global_scale` exports.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 34 |
| Lines added | 1995 |
| Lines deleted | 137 |

| Commit | Message |
|--------|---------|
| `883839e4` | fix(moe): load mlx-lm NVFP4 per-expert global_scale natively |
| `6e13adf1` | fix(moe): refuse NVFP4 expert planes with unstacked ModelOpt sidecars |
| `19567aa5` | fix(gpt-oss): refuse expert planes that carry an NVFP4 global scale |
| `d61ae3a8` | fix(moe): refuse a global_scale flag on a non-expert layer |
