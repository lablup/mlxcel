# Technical Report: PR #3243 - IFM K2-Horizon (`k2_horizon`) dense support

**Date**: 2026-11-23
**Status**: Open (draft PR, written before merge)
**Risk Level**: Low (a new model type on the shared Llama decoder; the only shared-code changes are an optional `attention_factor` field on `RopeScalingSpec` that no other family reads and `yarn_get_mscale` becoming `pub(crate)`)

## Executive Summary

`mlxcel` exited with `Unsupported model type: k2_horizon` on `IFM/K2-Horizon-0.9B` (#2358). The 0.9B is a plain Llama-shaped decoder (GQA, SwiGLU, pre-norm RMSNorm, untied head, Llama weight keys) with YaRN RoPE, so the PR adds no new decoder. `src/models/k2_horizon.rs` translates and screens the config and hands `llama3::ModelArgs` to the shared `Llama3Model`, registered as a config-backed text family so the CLI, `mlxcel-server`, the adapter path and `mlxcel arch` all reach it. Every K2-Horizon feature the shared decoder lacks (grouped RMSNorm, QK-norm, MoE and MoVA, gated attention, sliding window, partial rotary, a non-`silu` activation, a custom YaRN magnitude) is refused at load with an error naming the field. The loader also adds the chat template's turn terminator `<|ifm|im_end|>` to the EOS ids, since the checkpoint declares only `<|endoftext|>`. On the real bf16 0.9B, a 24576-position teacher-forced trace against mlx-vlm's `k2_horizon` has 0 decided disagreements and a perplexity 0.31% below the reference, while dropping `rope_scaling` gives 9.62% decided disagreements and +89.1% perplexity.

## 1. Problem Statement

K2-Horizon is a family, not a single shape. The HF `modeling_k2_horizon.py` supports grouped RMSNorm (`layernorm_num_groups` 2 / 4 / 4 on the 3.7B / 7B / 32B), per-head Q/K RMSNorm, MoE and MoVA (mixture of value attention) experts, a gated attention output, sliding window, and partial rotary through `rope_head_dim`. Only the 0.9B uses none of them. Mapping `k2_horizon` to the plain Llama arm would have loaded every variant: the larger dense checkpoints would normalize with the wrong group count, and a QK-norm checkpoint would silently ignore its `q_norm` / `k_norm` weights. Both decode wrongly without an error. So the work was to accept exactly the subset the shared decoder computes correctly and refuse the rest by name.

Two smaller problems sat on the 0.9B itself. RoPE is written under the transformers 5 `rope_parameters` block, which the flat `rope_theta` / `rope_scaling` keys the Llama path reads do not see. And `config.json`, `generation_config.json` and `tokenizer_config.json` all declare only `<|endoftext|>` as EOS, while every assistant turn the chat template renders ends with `<|ifm|im_end|>`, so the server decoded that marker into `content` before the model emitted EOS.

## 2. What Changed

The PR has four commits: the initial family (`e782a982`), the turn terminator (`4adb83c6`), HF-matching defaults for omitted screened keys (`c1b62d05`), and tighter geometry and activation screening (`c7a28def`).

- `src/models/k2_horizon.rs` (new): `K2HorizonConfig`, `validate()`, `to_llama3_args()`, the directory `load`, the adapter `from_weights`, and the turn-terminator lookup. The struct uses `#[serde(remote = "Self")]` with `mlxcel_core::impl_rope_lifting_deserialize!`, so the `rope_parameters` block is copied into `rope_theta` / `rope_scaling` before field deserialization: YaRN factor 16 (beta_fast 128, beta_slow 4, original context 8192) at base 1e6 on the 0.9B, base 1e7 with no scaling on the 7B, whose top-level `rope_theta` is null.
- Screening (`validate`): `layernorm_num_groups != 1`, `query_key_norm`, `num_experts` or `mova_num_experts` above 0, a non-null `attention_gate_func`, `use_sliding_window` (null read as false), `hidden_act` other than `silu`, `rope_head_dim != head_dim`, a YaRN `attention_factor` that differs from `0.1 * ln(factor) + 1`, a non-finite or non-positive `rope_theta` or `rms_norm_eps`, and invalid quantization parameters. Geometry checks reject zero sizes, heads not divisible by KV heads, odd or oversized `head_dim`, scalars above fixed ceilings, and `heads * head_dim` or `kv_heads * head_dim` that would overflow the decoder's i32 casts.
- `src/models/rope_utils.rs`: `RopeScalingSpec` reads an optional `attention_factor`; `yarn_rope_setup` does not use it. `yarn_get_mscale` is `pub(crate)` so the K2-Horizon check can call the same function the YaRN path uses.
- Registration: `ModelType::K2Horizon` in `models/mod.rs` (enum, `ALL_MODEL_TYPES`, description), `detection.rs`, `registry.rs`, `model_metadata.rs` (config-backed, `wrap: LoadedModel::Llama`), `execution/memory_estimate.rs`, and `distributed/tensor_parallel/inference.rs`, where the fallback architecture is `"k2_horizon"` rather than `"llama"`.
- Tests: `src/models/k2_horizon_tests.rs` (14 tests: both Hub configs through the lift, every refusal, HF defaults for omitted keys, the attention-factor check, the terminator lookup and merge, and the real checkpoint's tokenizer when present), a detection test, and a registry entry in `rope_parameters_lift_tests.rs`.
- Docs: `CHANGELOG.md` and `docs/supported-models.md`, including the refused variants and the `<ifm|think>` limitation.

## 3. Technical Decisions

**A config translation on `Llama3Model`, not a new decoder.** The 0.9B matches the Llama layout key for key, so a new model file would duplicate the decoder for no behavioral difference. This follows the Helium and TeleChat3 pattern. `to_llama3_args` is a full struct literal, so a field added to `llama3::ModelArgs` breaks this file at compile time instead of silently defaulting. `head_dim` is passed explicitly because the 0.9B's `32 * 64 = 2048` differs from its `hidden_size` of 1536.

**Omitted screened keys take the HF class defaults.** The first commit defaulted an absent `query_key_norm` to false and an absent `num_experts` to 0, which reads an incomplete config as dense. `configuration_k2_horizon.py` defaults them to true and 128, so such a config actually describes a QK-norm or MoE model. `c1b62d05` aligns the serde defaults with HF, so leaving a key out is refused the same way as declaring it. The published 0.9B declares both keys and is unaffected. `head_dim` follows the same rule (HF default 128, not `hidden_size / heads`); an explicit `null` still falls back to the division.

**Check `attention_factor` instead of applying it.** The shared YaRN path derives the attention magnitude as `yarn_get_mscale(factor, 1) / yarn_get_mscale(factor, 0)`, which is `0.1 * ln(factor) + 1`, or 1.27726 at factor 16. That equals the 0.9B's declared value, and the floor/ceil correction range already matches `truncate: true`. So the YaRN code is unchanged, and a declared value that differs (relative tolerance 1e-6) is refused rather than half-honored. The alternative, threading `attention_factor` into `yarn_rope_setup`, would change a code path every YaRN family shares to support a value no published checkpoint uses.

**Keep the TP fallback name off `"llama"`.** The tensor-parallel runtime parses `config.json` directly into `llama3::ModelArgs`. Routing `k2_horizon` there would bypass `validate()` and shard a grouped-norm or MoE checkpoint. With its own name, `runtime_kind_for` has no arm for it and TP is refused.

**Resolve the terminator by name, not by id.** `<|ifm|im_end|>` is looked up in `tokenizer.json` `added_tokens`, then in `tokenizer_config.json` `added_tokens_decoder`, and appended to the declared EOS ids only when found and not already present. This follows the InternVL and LLM-jp-VL loaders and leaves shared EOS resolution for other families untouched. A hard-coded id would break on any retokenized checkpoint.

## 4. Measurements

`models/k2-horizon-0.9b`, bf16, M3 Ultra.

- `mlxcel generate -p "The capital of France is" -n 50 --temp 0`: coherent output ("... The capital of France is Paris. ..."), 182 tok/s. With a chat question and `-n 200` it stops on the terminator after 68 tokens. It still loads and generates after the final screening commit.
- `mlxcel-server` `/v1/chat/completions`, streaming and non-streaming: `finish_reason: stop`, 68 tokens, no `<|ifm|im_end|>` in the output.
- Teacher-forced logit trace against the vendored mlx-vlm `k2_horizon` (d998a620, bf16), 2 chunks of 12288 tokens, `compare_logit_traces.py --decided 2`:

| arm | positions | top-1 disagreement | decided (gap >= 2) | perplexity vs ref 15.123 |
|---|---|---|---|---|
| YaRN (this PR) | 24576 | 2.73% | 0 / 7279 | -0.31% |
| YaRN, pos >= 8192 | 8192 | 2.94% | 0 / 2354 | -0.43% |
| `rope_scaling` dropped | 24576 | 37.64% | 700 / 7279 (9.62%) | +89.1% |
| dropped, pos >= 8192 | 8192 | 43.21% | 352 / 2354 (14.95%) | +152.2% |

Every YaRN-arm disagreement sits where the reference's top-two gap is below 0.25, consistent with mlxcel running the weights in f16. The dropped arm is the control: it shows the trace is sensitive to RoPE scaling, and the damage grows past the original 8192-token context, which is where YaRN matters.

## 5. Not Measured

- Only the 0.9B runs. The 7B's `rope_parameters` lift (base 1e7, no scaling) is pinned by a unit test, but the 7B itself is refused for grouped RMSNorm, so that RoPE path has never decoded a real checkpoint.
- No quantized K2-Horizon checkpoint was traced; quantization reaches the shared Llama path unchanged and is screened only for parameter validity.
- No CUDA or ROCm run. The code is backend-neutral, but no number exists off Metal.
- The adapter route (`from_weights`) runs `validate()` but does not add the turn terminator, so an adapter-loaded model can still leak `<|ifm|im_end|>` into `content`. The config-backed `weight_builder` hook receives only weights and config, with no model directory or tokenizer.
- The `<ifm|think>` block (and the `think_fast` / `think_faster` variants) is not split into `reasoning_content`; reasoning text and `</ifm|think>` stay in `content`.

## 6. Learning Points

**A serde default is a claim about the model.** Defaulting a screening field to the "safe" value (false, 0) made an omitted key mean "dense", the opposite of what the upstream config class means. For a field whose job is to refuse, the default must be the upstream default, or a sparse config passes the screen.

**A control arm makes a parity trace meaningful.** A 2.73% top-1 disagreement means little alone. The dropped-`rope_scaling` arm, with 700 decided disagreements, shows the comparison would catch a RoPE mistake, and the 0 / 7279 decided result on the YaRN arm is evidence rather than an absence of signal.

**Every dispatch table is an entry point.** The TP fallback name is easy to set to `"llama"` because the model runs on Llama. That would have created a load route that skips validation. When a family reuses a decoder through a screening translation, each table that maps the model type has to keep pointing at the translation.

## 7. Follow-up Actions

- Grouped RMSNorm (`layernorm_num_groups` > 1) in the shared decoder would unlock the 3.7B / 7B / 32B dense checkpoints, which would also put the base-1e7 RoPE path under a real trace.
- Register the `<ifm|think>` / `</ifm|think>` pairs (and the effort variants) in the five shared reasoning-marker tables so the server fills `reasoning_content`.
- Give the config-backed `weight_builder` hook access to the tokenizer, or add the terminator at a later stage, so the adapter route stops on `<|ifm|im_end|>` too.
- QK-norm, MoE and MoVA support would be separate work on a new decoder, not an extension of this translation.
