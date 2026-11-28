# PR #2438: Pin runtime LoRA term widths on quantized linear bases

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint (qwen3-0.6b-4bit); fusing into a dequantized base is not added (see "Not addressed")
**Risk**: Low to Medium (adds a width check at linear claim time and a new load-time refusal on the fused path for quantized linear bases)

## Summary

Closes #2437. A runtime LoRA pair on a quantized linear base is now checked against the layer's real input width when the layer claims it, and fusion refuses a quantized linear base by its `.scales` plane. Both reuse the mechanisms #2434 added for embedding tables.

## Problem

- Runtime path (`mlxcel-server --lora`, `mlxcel serve --lora`): `validate_pair_shapes` in `src/lora/runtime.rs` checks a quantized base only through the scales group count, so any width with `in / groups` in {16, 32, 64, 128} passes. On qwen3-0.6b-4bit a `q_proj` `lora_a` of width 512 loads and then aborts the process at warmup with `[matmul] Last dimension of first input with shape (1,1,1024) must match ... (512,8)` (SIGABRT, exit 134).
- Fused path (`mlxcel generate --adapter`, `--lora-fuse`, the pipeline stage loader): the shape guard compares the delta with the packed plane. A delta of width `in * bits / 32` matches it and is added to the packed words; the plane becomes f32 and the first forward aborts with `[quantized_matmul] The weight matrix should be uint32 but received float32`.

PR #2434 had already closed the embedding case with the same two mechanisms: `UnifiedEmbedding::from_weights` pins the width at claim time, and fusion refuses a quantized embedding table by its `.scales` plane.

## Technical decisions

- Check at claim, before the claim. `claim_linear(prefix, in, out)` checks every staged term for the prefix against `a_t == [in, r]` and `b_t == [r, out]`, modeled on `EmbeddingLoraTerms::new`, and only then claims them. The tied-head form of an embedding term staged for the prefix is checked the same way. The check runs before anything leaves staging, so a caller that maps a constructor error to an absent layer (Nemotron-H builds four optional projections with `UnifiedLinear::from_weights(..).ok()`) still fails the load as unclaimed instead of serving a model without the layer.
- Real widths per layout. For a packed weight `in = scales_cols * group_size`, using the reconciled group size in `UnifiedLinear::from_weights_with_mode` and the group size the concatenated plane carries in `FusedQKVLinear`. `out` is the weight's rows. `FusedQKVLinear` checks each projection against its own planes.
- Dense bases too. `Linear::from_weights` checks the dense columns, which the loader already enforces exactly, so this changes nothing for dense bases and also covers a tied head built from the embedding prefix. The width work is skipped when nothing is staged. No struct gains a field.
- Up-front refusal on the fused path. The `.scales` refusal that #2434 added for the embedding table moves into a pass next to the Conv1D verdict (`reject_quantized_base_fusion`) and covers linear weights too, so a refused adapter leaves the base weight map untouched. It also refuses a pair that resolves to a quantization side plane itself (`q_proj.scales.lora_a` resolves to `q_proj.scales`; its delta matched the scales plane and was added to it). The message names the layer and points to `--lora`.

## Changes

- `src/lib/mlxcel-core/src/runtime_lora.rs`: `claim_linear`, with the width check on the staged terms and on the tied-head form of an embedding term.
- `src/lib/mlxcel-core/src/layers.rs`: every linear claim goes through `claim_linear` (`Linear::from_weights`, the quantized arm of `UnifiedLinear::from_weights_with_mode`, and per projection in `FusedQKVLinear`).
- `src/lora/loader.rs`: `reject_quantized_base_fusion`, run up front.
- `src/lora/runtime.rs`: a 4-line change (5 lines touched).
- `src/lora/loader_tests.rs`: loader tests for the up-front refusal.
- Docs: `docs/server-features.md` states that only the unfused path applies adapters on a quantized checkpoint, and that it pins each pair's width at load.

## Behavior change on the fused path

A valid adapter on a quantized base never worked through the fused path: there is no dequantize step before fusion (`load_model_with_adapter` fuses into the raw safetensors map), and the `[out, in]` delta fails the shape guard against the `[out, in * bits / 32]` plane. On qwen3-0.6b-4bit with the `mlx_lm.lora`-trained q_proj adapter, main 8744c87f answers `LoRA shape mismatch fusing adapter layer model.layers.26.self_attn.q_proj ...: base is [2048, 128], delta is [2048, 1024]`. So the refusal breaks no working case. It replaces that message with one that says why and where to go, and it closes the packed-width corruption.

## Verification

GB10, `--features cuda`, test-fast profile, every GPU run under `gpu-lock`, `MLXCEL_SDPA_DETERMINISTIC=1`.

Real checkpoint `qwen3-0.6b-4bit` (affine 4-bit, group 64). Adapters were derived from an `mlx_lm.lora` adapter (rank 8, scale 20, q_proj of layers 26 and 27): `valid_qproj` (both pairs as trained), `bad_w512` (layer 27 `lora_a` cut to `[512, 8]`, passes the runtime loader's group-count check), `packed_w128` (only layer 27, `lora_a` `[128, 8]`, as wide as the packed plane). "Before" is the main 8744c87f binary, "after" this branch. Server request: `/v1/completions`, prompt "The capital of France is", 16 tokens, temperature 0.

| Command | Adapter | Before (main) | After |
|---|---|---|---|
| `mlxcel-server --lora` | `bad_w512` | abort at warmup, exit 134 | load error naming `model.layers.27.self_attn.q_proj` ("takes 1024 input features and produces 2048"), no abort |
| `mlxcel serve --lora` | `bad_w512` | abort at warmup, exit 134 | same load error, no abort |
| `mlxcel generate --adapter` | `packed_w128` | fuses into the packed plane, aborts at the first forward, exit 134 | refused: `model.layers.27.self_attn.q_proj.weight is a quantized linear weight`, exit 1 |
| `mlxcel generate --adapter` | `valid_qproj` | `LoRA shape mismatch` (base [2048, 128], delta [2048, 1024]), exit 1 | refused with the quantized-linear message, exit 1 |
| `mlxcel generate --adapter` | `bad_w512` | `LoRA shape mismatch`, exit 1 | refused with the quantized-linear message, exit 1 |
| `mlxcel-server --lora` | `packed_w128` | loader error (group count) | same loader error |
| `mlxcel-server --lora` | `valid_qproj` | " Paris. 1. If the capital is capital, then the capital is capital" | identical text |
| `mlxcel-server --lora` | #2434's embed_tokens + q_proj adapter | " Lyon. The capital of Japan is Osaka." | identical text |

After a load error the server stays up and answers 503, which is its existing behavior for any failed model load.

Real-checkpoint gate `real_checkpoint_embedding_adapter_parity` (ignored test) on the final commit: Llama-3.2-1B-Instruct-bf16 with #2434's adapter gives runtime vs fused 0.125 and stage vs fused 0.0000, scale 0 byte-identical to base, the same numbers #2434 reported. qwen3-0.6b-4bit runtime path: scale 0 byte-identical to base.

Tests:

- `-p mlxcel-core runtime_lora`: 11 passed, 3 new. `a_term_of_the_wrong_width_on_a_quantized_linear_fails_the_claim` stages widths 64, 128 and 512 (all accepted by the loader) on a 4-bit group-64 `[8, 256]` `UnifiedLinear` and expects `Err` with the refused term still staged, plus a wrong output width, then loads the right width and checks the forward value. `every_quantized_linear_layout_pins_the_term_width` covers affine at bits 2, 3, 4, 5, 6, 8 and group sizes 32, 64, 128, mxfp4, mxfp8 and nvfp4 (each block-float mode only where the backend runs it natively), and two layouts whose declared group size the reconciler overrides (mxfp4 stored at 32 under a declared 64, the 4-bit group-16 fallback stored at 64). `fused_qkv_pins_each_projection_term_to_its_own_plane` checks the fused q/k/v path per plane and the forward of the right width.
- `lora::`: 69 passed, 3 new: `the_fusion_loop_refuses_a_packed_width_delta_on_a_quantized_linear` (a real 4-bit packed plane, delta as wide as it), `a_quantized_target_is_refused_before_any_dense_pair_is_written`, `a_pair_aimed_at_a_quantization_side_plane_is_refused`. `loading::adapter_`: 9 passed, 1 ignored. `llama::adapter_tests`: 4 passed. `partial_loading_adapter`: 16 passed.
- Revert check: with the check in `claim_linear` and `reject_quantized_base_fusion` disabled, the 3 new core tests and the 3 new loader tests fail (along with the existing #2434 embedding refusal test, which shares that refusal); restored, they pass.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings` for the `mlxcel` and `mlxcel-core` packages, `cargo fmt --all -- --check`, and the contract scripts (`check_llama_compat_manifest`, `check_kernel_dtype_keys`, `check_kernel_port_dispatch`, `check_crate_versions`, `insert_apache_header --check`) pass.

## Changes during review

- The implementation review found that claiming before checking consumed the term, so Nemotron-H's `.ok()` constructors would have turned a wrong-width term into a silently missing projection. The check now runs before the claim.
- The security review found that a pair aimed at `q_proj.scales` fused into the scales plane. The up-front refusal now covers quantization side planes, and it also moved ahead of the first write.

## Not addressed

- Fusing into a dequantized base is not added.
- Integer-typed `lora_a` / `lora_b` tensors still reach `matmul` uncast on the fused path, where MLX throws for non-floating types. This predates the PR, applies to dense bases as well, and is a separate fix.
- Nemotron-H's optional projections treat any constructor error as an absent layer. With this PR a bad adapter still fails the load (as unclaimed), but the message names the layer as unclaimed rather than giving the width mismatch.

## Statistics

6 files changed, 520 insertions, 23 deletions. Commits: `9ad38d49` fix(lora): pin runtime LoRA term widths on quantized linear bases, `94071d0d` fix(lora): check staged terms before claiming and refuse packed fusion up front.
