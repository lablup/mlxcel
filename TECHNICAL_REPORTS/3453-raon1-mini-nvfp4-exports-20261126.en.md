# PR #3453: Load the Raon-1 mini MLX and ModelOpt NVFP4 exports (issue #3437)

**Date**: 2026-11-26

**Status**: PR open (closes #3437). Validated on an M3 Ultra. 17 files, +1618 / -100.

**Risk Level**: Medium (extends the shared ModelOpt NVFP4 detector; new load-time validation of untrusted quantized planes)

## Executive Summary

KRAFTON publishes two NVFP4 exports of Raon-1 mini, a vendor MLX layout (`KRAFTON/Raon-1-mini-mlx`) and an NVIDIA ModelOpt layout (`KRAFTON/Raon-1-mini-nvfp4`). #3225 refused the MLX export by name, and mlxcel's ModelOpt detector did not recognize the second. This PR loads both. Resident memory drops from 30.56 GB to about 16.4 GB and decode rises from 33.4 to about 39 tok/s. Top-1 agreement with the bf16 original is 87 to 90% with a perplexity increase of 2.5 to 3.6%, and the two exports agree with each other on every decided position.

## 1. Problem Statement

### 1.1 Background

Both exports quantize the routed experts and most dense linears to NVFP4 (E2M1 values, E4M3 scales per 16 inputs) with an extra f32 tensor-level scale, and keep `lm_head` in MXFP8 g32. They differ in how that is written down:

- **MLX export**: MLX `quantization {nvfp4, group 16}` with per-module overrides. Experts carry a per-expert `global_scale`. Dense linears listed in `raon_global_scaled_linears` carry a per-tensor `global_scale` applied to the output, `bf16(f32(y) * g)` (the vendor's `GlobalScaledLinear`).
- **ModelOpt export**: `quant_method: modelopt_fp4`, `quant_algo: MIXED_PRECISION`, with a per-module `quantized_layers` map (`NVFP4`, `W4A16_NVFP4`, `MXFP8`), per-expert tensors to stack, and `weight` / `weight_scale` / `weight_scale_2` triplets.

### 1.2 Existing gaps

- `is_modelopt_nvfp4` required `quant_method == "modelopt"` and `config_groups`, so this export was not detected and its triplets were never repacked.
- The Raon loader assumed the original bf16 + MXFP8 layout for the head and dense linears.
- Quantized planes from a checkpoint were trusted: a `scales` tensor with fewer rows than its `weight` would let a GPU kernel read past the scales buffer.

## 2. Technical Decisions

### 2.1 Infer each plane's layout from tensor shapes

Rather than trusting per-module config entries to say which format each linear uses, the Raon loader (`raon1_mini_quant.rs`) derives the layout from the `weight` and `scales` shapes. The MXFP8 head is then not misread as NVFP4 or MXFP4, and the same code loads the original checkpoint, both exports, and the tiny test models. Config entries are still cross-checked: `raon_global_scaled_linears` must match the tensors that carry a `global_scale`, and a `"global_scale": true` flag on a dense linear outside that list is refused instead of silently dropped.

### 2.2 Reuse existing NVFP4 global-scale paths

- Per-expert `global_scale` goes through the existing config-flag resolution into `gather_qmm`'s native NVFP4 tensor scale, the path Inkling 2 already uses.
- The per-tensor dense `global_scale` is the `UnifiedLinear` output multiplier.
- ModelOpt triplets go through the existing `sanitize_modelopt_nvfp4_weights` repack. Raon then stacks the repacked per-expert planes.

### 2.3 Keep MXFP8 conversion Raon-only

| Option | Pros | Cons |
|---|---|---|
| Convert MXFP8 layers to MLX `mxfp8` planes for every family | One rule | Families whose loaders assume one format would misread them |
| **Convert only for families in `MXFP8_LAYER_FAMILIES` (Raon)** | Raon reads layouts from shapes; other families get a named error | A second family needs an explicit opt-in |

The first version converted for every family; pr-reviewer flagged that as HIGH and it was narrowed in `474d6fee6`. An all-NVFP4 mixed-precision export still loads for every text family. Any other per-layer algorithm, an MLX quantization entry other than NVFP4 g16 or MXFP8 g32, or a `kv_cache_scheme` fails by name.

### 2.4 Validate every untrusted plane at load

The security review found the out-of-bounds read above (HIGH). `check_scales_match_weight` now requires `scales` (and `biases`) to match `weight` on every axis but the last, for every Raon quantized plane: dense, routed experts, RDA, MLA and the head (`f1989648a`, `f6ad29ee5`). A dense `global_scale` must be one float element. The same check in the shared `UnifiedLinear` constructors, which affects every mlx-lm quantized family and predates this PR, is left for a follow-up.

## 3. Validation

Reference for traces: mlxcel's own trace of the bf16 original, which matches the Transformers reference (#3441, #3445). Full numbers: `docs/benchmark_results/raon-nvfp4-m3ultra-2026-11-26.md`.

| Export | Tokens | Top-1 agreement | Perplexity delta |
|---|---|---|---|
| MLX | 512 | 88.67% | +2.48% |
| ModelOpt | 512 | 86.91% | +2.59% |
| MLX | 4096 | 89.77% | +3.63% |
| ModelOpt | 4096 | 89.92% | +3.49% |

Every disagreement is at a reference top-two gap of 2.375 or less.

| Checkpoint | Load | Resident | Decode |
|---|---|---|---|
| Original | 1.4 s | 30.56 GB | 33.35 tok/s |
| MLX export | 0.7 s | 16.34 GB | 39.88 tok/s |
| ModelOpt export | 67 to 75 s | 16.39 GB | 38.76 tok/s |

- `--reasoning-effort none` answers `42` on both exports; the default effort gives coherent thinking then `42`.
- Vendor runtime (`raon_mlx_model.py --no-dflash`, `RAON_EXACT_KERNELS=0`, greedy 64): identical at the default effort. With thinking off it diverges at token 50 on a 0.25 logit near-tie (34.25 vs 34.0), where a wider mlxcel forward agrees with the vendor.
- Unit tests: a tiny model quantized into each layout, written to disk and loaded through `Raon1MiniModel::load`, matches its dequantized twin at two efforts (covering the per-expert scale in the gather path and f32 steering vectors); global-scaled linear equals dequantize-then-matmul times the scale; the detector accepts this config and rejects unknown algorithms; the MXFP8 head is a byte view; malformed scales and global scales are refused. 345 scoped tests pass and the full local `make verify` gate passed.

## 4. Follow-up

- Add the scales-row check to the shared `UnifiedLinear::from_weights_with_mode` and `from_stacked_parts`.
- The ModelOpt export loads in about 70 s because the NVFP4 repack runs on the CPU.
