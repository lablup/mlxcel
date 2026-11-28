# PR #3331: Load unquantized DeepSeek-V3 MoE checkpoints

**Date**: 2026-11-23
**Status**: PR open as a draft (closes #3319). Narrow unit selectors, clippy, fmt and the CI scripts pass. Real-checkpoint traces against mlx-lm are recorded below.
**Risk Level**: Low (the quantized expert path is unchanged and traces byte-identical; the float path is new)

## Executive Summary

The `deepseek_v3` routed-expert block (`SwitchGLU` in `src/models/deepseek_v3.rs`) required quantization `.scales` and `.biases` on every expert plane, so an unquantized export failed at the first MoE layer. On the bf16 GigaChat3-10B-A1.8B export the load stopped with `Weight not found: model.layers.1.mlp.switch_mlp.gate_proj.scales`. The block is now an enum decided per block: the quantized variant is the previous struct, and a new float variant runs `gather_mm` through the shared `SwitchLinear`, with the same clipped SwiGLU activation. The bf16 checkpoint now loads and traces with 0 of 2771 decided top-1 disagreements against mlx-lm over 8192 positions, which matches the noise between mlx-lm in bf16 and in f16.

## 1. Problem Statement

Sanitize already stacked `mlp.experts.{e}.*` into `mlp.switch_mlp.*` for whatever leaves exist, and dense layers and shared experts load through `UnifiedLinear`, which accepts float weights. Only the routed-expert block was quantized-only: it stored nine required tensors and ran three `gather_qmm` calls. Every model type routed to this decoder (`deepseek_v3`, `kimi_k2`, `joyai_llm_flash`) and the other callers of `MoEBlock` (the Kimi-VL text model, Xing4.0, the V3 pipeline stage executor) had the gap. PR #3306 had to quantize GigaChat3 to 8-bit to trace it.

## 2. Change Summary

| Area | Change |
| --- | --- |
| `SwitchGLU` | Now `enum SwitchGLU { Quantized(QuantizedSwitchGLU), Float(FloatSwitchGLU) }`. `from_weights` partitions the three planes by `.scales` presence: none means float, all means quantized, and a mix is refused. |
| `QuantizedSwitchGLU` | The previous struct, renamed. Body, NVFP4 refusal and per-plane width solving unchanged; the plane loop uses the shared `SWITCH_GLU_PLANES` constant. |
| `FloatSwitchGLU` | Three `switch_layers::SwitchLinear` planes, each required to load as `Regular`. `forward` builds the gather indices once with `prepare_gather_indices` and calls `forward_indexed` on each plane, then `clipped_swiglu`, then the down projection with MLX's default lhs. |
| Load refusals | Mixed planes; a plane with `.biases` but no `.scales`; a float plane whose dtype is not f16, bf16 or f32. |
| Docs | `Used by:` on `forward_indexed` and `prepare_gather_indices`; `docs/supported-models.md` DeepSeek entry; CHANGELOG `Fixed`. |

## 3. Technical Decisions

### 3.1 Keep the block local instead of using the shared `SwitchGLU`

The shared `switch_layers::SwitchGLU` has no clipped activation. Kimi-VL and Moonlight need `clip(silu(gate), -100, 100) * up` to keep f16 activations finite (issue #525), so the block stays family-local and only its planes move onto the shared `SwitchLinear`. This reuses the existing `gather_mm` call (`SwitchLinear::Regular`) rather than adding a second wrapper.

### 3.2 Decide per block from `.scales`

The variant is chosen from the weights, not from the config's `quantization` block, because mlx-lm exports can quantize some modules and not others. Choosing per block also means a fully quantized block reaches exactly the same code as before. Mixed blocks are refused by name with the planes listed, for example `p: DeepSeek-V3 routed experts mix quantized and float planes (gate_proj quantized, up_proj/down_proj float)`.

### 3.3 Refuse integer planes on the float path (review)

Choosing the float variant from missing `.scales` alone would let a quantized export that lost its scales load its packed `uint32` weights as float planes. `gather_mm` would then throw across the cxx boundary at the first routed forward pass and abort the process, where main failed cleanly at load. The float loader therefore refuses a plane with orphan `.biases` and a weight dtype outside f16, bf16 and f32.

### 3.4 No sorted prefill route

The shared `SwitchGLU` sorts token-expert pairs when there are 64 or more, to reach the sorted `gather_mm` kernel. The float variant follows the issue's design and the quantized DeepSeek-V3 path, which run unsorted. The bf16 trace prefills 512-token chunks in 430 to 575 ms on an M3 Ultra. Adding the sorted route to both variants is left as a follow-up.

## 4. Verification

Teacher-forced `examples/logit_trace`, 16 chunks of 512 (8192 positions), top-8, against mlx-lm 1bf8da0 with mlx 0.32.4. mlx-lm needs a shim that drops the MTP layer at index 26, because upstream strips only index 61. mlxcel converts bf16 weights to f16 at load, so the reference ran twice: native bf16, and with its bf16 parameters cast to f16. Those two reference arms against each other give the noise floor.

| Arms | Decided disagreements (gap >= 2) | Top-1 disagreements | Largest gap at a disagreement | Perplexity ref / cand |
| --- | --- | --- | --- | --- |
| mlx-lm bf16 vs mlx-lm f16 (noise floor) | 0 / 2771 | 324 / 8192 | 0.875 | 14.557 / 14.550 |
| mlx-lm bf16 vs mlxcel | 0 / 2771 | 312 / 8192 | 0.875 | 14.557 / 14.546 |
| mlx-lm f16 vs mlxcel | 0 / 2718 | 130 / 8192 | 0.734 | 14.550 / 14.546 |

- Before the fix, `mlxcel generate -m models/gigachat3-10b-a1.8b-bf16` failed with the `scales` error. After it, generation is coherent at about 45 tok/s with 19.9 GB resident.
- Quantized regression, binaries built from main and from this branch: the `gigachat3-10b-a1.8b-mlx-8bit` trace (8192 positions) and the `moonlight-16b-a3b-instruct-4bit` trace (4 x 256 positions) are byte-identical, and greedy `generate` output on both is identical. The 8-bit trace against the #3306 mlx-lm 8-bit reference still has 0 of 2774 decided disagreements.
- Unit tests: a tiny f32 float block against a hand-computed f64 per-expert clipped SwiGLU within 1e-4 relative, with one expert's gate scaled so the clip is reached, and a 3D input giving the same values; the mixed-plane error; refusal of packed planes with and without orphan `.biases`. Selectors (test-fast, under the GPU lock): `deepseek_v3`, `xing4_0`, `coverage_tests` with `--test-threads=1`, `kimi` and `switch_layers`. `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` is clean.

## 5. Unverified and Open Items

- The quantized regression traces and the bf16 trace were taken before the review commit that added the dtype and orphan-bias refusals. That commit changes only load-time checks; the bf16 checkpoint was loaded and generated again with it.
- No unquantized Kimi-K2, JoyAI, Kimi-VL or Xing4.0 checkpoint was run; they share `MoEBlock` and are covered by the same code path.
- Expert shape checks for this block are #3320.
- The broad gates (`cargo test` across the workspace, the full verification list in the issue) are left to the orchestrator.

## 6. Learning Points

- A same-precision reference is not enough when the runtime changes the weight dtype at load. Running the reference in both dtypes gave a noise floor that the candidate could be measured against.
- Selecting a code path from a missing key also opens it to inputs that only lost that key. The selection needs its own validation of what it admits.

## Related Commits

- `efe57825` fix(models): load unquantized DeepSeek-V3 MoE checkpoints
- review follow-up: refuse integer and orphan-bias float planes, docs and CHANGELOG
