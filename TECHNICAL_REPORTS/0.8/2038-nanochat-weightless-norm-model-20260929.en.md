# Technical Report: PR #2038 - nanochat Text Model

**Date**: 2026-09-29

**Status**: Implemented and validated locally on M1 Ultra against a HuggingFace oracle; pending merge (GB10 CI runner unavailable).

**Languages**: Rust

**Risk Level**: Low (new model family; the only shared change is a load-time dtype exclusion keyed on `model_type == "nanochat"`)

## Executive Summary

Issue #1368 asked for the `nanochat` speedrun models (d20 0.56B, d32 1.9B). The architecture is GPT-2-shaped with four choices that each produce fluent but wrong text when skipped, so acceptance was greedy-id parity against an oracle, not fluency. The PR adds `src/models/nanochat.rs` and registers the family end to end. Greedy ids match an independent HuggingFace `NanoChatForCausalLM` oracle on both validation checkpoints, up to logit near-ties below 0.08.

## 1. Problem Statement

`model_type: "nanochat"` hit the unsupported arm. The model is not a config variant of an existing family: it has no norm weights at all, rotates q and k the opposite way from the standard RoPE, squares a ReLU in the MLP, and soft-caps the output logits.

## 2. Change Summary

- `src/models/nanochat.rs`: attention (separate `c_q` / `c_k` / `c_v` / `c_proj`), `relu_squared` MLP, weightless blocks, `NanoChatModel` with a last-position logits override, `sanitize_weights` and `load`. `src/models/nanochat_config.rs` holds `ModelArgs` and bounds validation for the untrusted `config.json`.
- Registration: `ModelType::NanoChat`, `LoadedModel::NanoChat`, the `model_metadata.rs` row, registry family and id tables, the memory estimator, the tensor-parallel arch string (not TP-enabled), and detection on `model_type`.
- `src/models/sanitize.rs`: nanochat is excluded from the bf16 to f16 load conversion and listed as f16-fragile for pre-Ampere CUDA.
- `docs/supported-models.md`: entry naming the four choices, both layouts, the BOS requirement and the validated checkpoints.
- 11 unit tests in `nanochat_tests.rs` and one detection test.

## 3. Technical Decisions

### Mirrored RoPE as a negated frequency table

`fast_rope_with_freqs` divides the position by each table entry, so `-(base ** (arange(half) / half))` with `traditional = false` gives the angle `-p / f_i` in the half-split layout. This matches HuggingFace's `rotate_half` returning `[x2, -x1]`, the sign-flipped version of the usual one. A unit test checks the rotated pair against the closed form and that `fast_rope` disagrees.

### Two softcap keys as two fields

The MLX conversions write `logits_soft_cap` and `logits_softcap` together. A serde `alias` turns that into a duplicate-field error, so each is a double-option field and `soft_cap()` prefers the primary key, then the alias, then 15.0. An explicit `null` disables the cap.

### bf16 is required, f16 is not an option

The first real-checkpoint run on the bf16 model produced greedy id 0 everywhere. Per-layer statistics showed the residual stream at max 64000 after layer 15 and NaN from layer 16: every norm is weightless and only feeds the sublayers, so the residual is never renormalized. bf16 has the f32 exponent range. The 8-bit checkpoint was unaffected because quantized models already stay bf16.

### QK-norm ordering cannot be tested by magnitude

The issue asks for a test that norm-after-RoPE differs from norm-before-RoPE. A rotation preserves the per-head L2 norm, so the two orders agree up to the epsilon. The test instead compares the output with a CPU reference of RoPE followed by norm. The order is still implemented as specified.

## 4. Validation

Oracle: HuggingFace `transformers` 5.17 `NanoChatForCausalLM` on CPU, weights dequantized from the same safetensors. Prompts were built with the checkpoint's chat template or the literal `<|bos|>` prefix, and mlxcel's `/tokenize` returned identical prompt ids. 48-token greedy decode, ids read from the server `/completion` `tokens` field.

| Checkpoint | Prompt | Match vs bf16 oracle | Match vs fp32 oracle |
|---|---|---|---|
| q8 (chat template) | transformer | 43 of 45 | 43 of 45 |
| q8 | France | 7 of 7 | 7 of 7 |
| q8 | haiku | 41 of 41 | 17 (near-tie 0.004) |
| bf16 (`<|bos|>`, no template) | transformer | 17 (fp32 oracle agrees with mlxcel) | 13 (0.008) |
| bf16 | France | 9 of 9 | 9 of 9 |
| bf16 | haiku | 48 of 48 | 30 (0.075) |

Every divergence is a logit gap under 0.08 between the two candidate tokens, measured teacher-forced in the fp32 oracle. Unit tests, `fmt`, `clippy -D warnings` and the three contract tests pass. No throughput numbers are reported: other units shared the GPU.

## 5. Deferred

The d32 checkpoint (`karpathy/nanochat-d32` ships only a `.pt`), the `<|python_start|>` tool loop, and CI on the GB10 runner.
