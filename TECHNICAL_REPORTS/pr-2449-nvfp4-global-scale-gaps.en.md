# PR #2449: Close NVFP4 global-scale load gaps in QKV, VLM and DeepSeek

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; PR open. The items under "Not addressed" remain open

## Summary

Closes #2444. #2441 left three load paths that mishandle an NVFP4 tensor scale. This PR fixes all three: the fused QKV projection for dense ModelOpt NVFP4, the VLM loaders and the Gemma 4 unified expert split, and the DeepSeek v1 bias-free expert plane. `nvidia/Qwen3-30B-A3B-NVFP4` (ModelOpt) now loads and generates; on `main` it failed to load.

## Problem

- **Dense ModelOpt NVFP4 in fused QKV.** `nvidia/Qwen3-30B-A3B-NVFP4` failed to load on `main` with `1-bit weights are affine-only ... (prefix: model.layers.0.self_attn.q_proj)`. `FusedQKVLinear::from_weights_separate` hardcoded `affine`, so the bias-free nvfp4 planes solved to a 1-bit affine layout, and the concatenated plane had no slot for each plane's `weight_scale_2` (the repacked `{plane}.global_scale`). `group_size()` also fell back to 64 for a ModelOpt-only config in Qwen3 and Qwen3-MoE.
- **VLM loaders and the Gemma 4 unified expert split.** `apply_native_global_scale_flags` ran only in `load_text_weights` and the Gemma 4 sanitizer. The Gemma 4 unified loader's `split_fused_moe_key` split only `.weight` / `.scales` / `.biases` of a fused `experts.gate_up_proj`, so a `.global_scale` would be orphaned.
- **DeepSeek v1 bias-free expert plane.** `SwitchLinear::from_stacked_parts` mapped `.scales` without `.biases` to `Regular`, so a ModelOpt or mlx-lm nvfp4 expert plane aborted in `gather_mm` at the first forward.

## Technical decisions

- **Split QKV when a plane has a scale.** `FusedQKVLinear::from_weights_separate` infers the mode from `q_proj` with `infer_quantization_mode`, as `UnifiedLinear::from_weights` does. If any of q/k/v has a `{plane}.global_scale` sidecar, the layer loads as `QkvProjection::Split`. Each plane is built by `UnifiedLinear::from_weights_with_mode` at its reconciled layout, applies its scalar after the matmul (key name and meaning unchanged, the `inv_global_scale` convention rather than MLX's native `global_scale`) and claims its own width-pinned runtime-LoRA terms (#2438). The issue rejects a fused per-row scale vector because the fused rope kernels take no scale.
- **No new fields on per-layer structs.** No per-layer struct gains a field, because the debug Pipeline CI smoke test sits at the 2 MiB stack edge (#2434). For the same reason the issue forbids adding fields to `QuantizedWeight`, `UnifiedLinear`, `QkvProjection` or `FusedQKVLinear`.
- **Fused block-float planes reconcile their group size.** The fused arm reconciles a block-float plane's group size from its shapes and checks the backend before storing it. Previously it stored the declared value, for example 64 for an mxfp4 plane packed at 32. (Added in the second commit, `abd936f6`.)
- **ModelOpt group size fallback.** Qwen3 and Qwen3-MoE `group_size()` fall back to `modelopt_nvfp4_group_size` (16), as PhiMoE does. This adds a `#[serde(default)] quantization_config` field to both `ModelArgs` and one line to each of three struct-literal test fixtures.
- **Flag pass in the VLM loaders.** `finish_vlm_weights_common` and `load_gemma4_unified` (raw keys, before the sanitizer) run the native global-scale flag pass. The filtered VLM loaders end in `finish_vlm_weights_common` too. There a subset variant skips a flagged plane with neither its weight nor its scale in the map, while a loaded flagged plane without its scale still fails.
- **Fused expert key split.** `split_fused_moe_key` handles `.native_global_scale` and `.global_scale`. The `[num_experts]` vector is copied unsplit to `gate_proj` and `up_proj` and renamed under `down_proj`, so an unflagged scale fails at its `switch_glu.gate_proj` prefix instead of being orphaned.
- **DeepSeek v1 half-quantized plane.** `SwitchLinear::from_stacked_parts` refuses a plane that has only one of `.scales` / `.biases`, naming the prefix, instead of building `Regular` and aborting in `gather_mm`.

## Changes

- `src/lib/mlxcel-core/src/layers.rs`: mode inference, `Split` path for scaled planes, group-size reconciliation in the fused arm, tests (223 lines changed).
- `src/models/qwen3.rs`, `src/models/qwen3_moe.rs`: `modelopt_nvfp4_group_size` fallback and the new `quantization_config` field; fixtures in `qwen3_tests.rs`, `qwen3_moe_tests.rs`, `qwen3_embedding_tests.rs`, `src/rerank/qwen3_generative_tests.rs` and the tensor-parallel `llama_runtime_tests.rs` updated for it.
- `src/loading/vlm.rs`, `src/loading/vlm_gemma_unified.rs`, `src/models/switch_layers.rs`, `src/models/sanitize.rs`: flag pass in the VLM loaders, subset variant, `split_fused_moe_key` extension.
- `src/models/deepseek.rs`: half-quantized plane refusal. `src/models/jamba.rs`: small signature update.
- Tests: `src/loading/vlm_gemma_unified_tests.rs`, `src/loading/vlm_tests.rs`, `src/models/deepseek_tests.rs`.

## Verification

GB10, CUDA, every GPU run under `gpu-lock`.

- `MLXCEL_SDPA_DETERMINISTIC=1 mlxcel generate -m ~/models_archive/hf-src/Qwen3-30B-A3B-NVFP4 -p "The capital of France is" -n 80 --temp 0 --show-reasoning` loads (main fails at load) and generates coherent text: "... Since the question is about the capital of France, the answer is Paris. ...". There is no mlx-lm teacher-forced trace, because mlx-lm 0.32.2 refuses this checkpoint (55968 unexpected parameters, unstacked ModelOpt keys).
- Greedy output is token-identical to main (`MLXCEL_SDPA_DETERMINISTIC=1`, same prompts) for `~/models_archive/mlx/laguna-xs-2.1-nvfp4` (with `--show-reasoning`), `olmoe-1b-7b-0924-nvfp4-gs` and `gemma-4-31b-it-nvfp4` (two prompts). This was re-checked on the final commit.
- 10 new tests, each failing with its fix reverted and passing with it restored:
  - layers: nvfp4 split QKV with per-plane scales, its forward checked against `x @ dequantize(w).T * s`; mxfp4 QKV fused at group 32.
  - qwen3 / qwen3_moe: ModelOpt-only `group_size() == 16`.
  - Gemma 4 Unified: flagged and unflagged fused experts.
  - VLM: a shipped `.native_global_scale` is refused, a flagged scale is resolved, and a filtered load skips planes outside the filter.
  - DeepSeek: the half-quantized plane.
- These filters pass: `layers::`, `runtime_lora`, `one_bit` (mlxcel-core) and `qwen3 vlm deepseek:: switch_layers sanitize` (mlxcel). Clippy `--lib --tests -D warnings` is clean for both crates. fmt and the five contract checks pass.

## Not addressed

Both are pre-existing.

- The Kimi K3 VLM loader bypasses `finish_vlm_weights_common`, so a flagged export is refused cleanly at `SwitchLinear`.
- A dense `.global_scale` is not shape-validated, which is the same as `o_proj` and MLP on main.

The issue also lists the O(planes x keys) sidecar scan cost as out of scope.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 17 |
| Lines added | 672 |
| Lines deleted | 24 |

| Commit | Message |
|--------|---------|
| `f1a74b25` | fix(nvfp4): close global-scale load gaps in QKV, VLM and DeepSeek |
| `abd936f6` | fix(nvfp4): guard fused block-float QKV and filtered VLM flag pass |
