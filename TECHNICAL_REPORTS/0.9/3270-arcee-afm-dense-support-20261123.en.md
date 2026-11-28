# Technical Report: PR #3270 - Arcee AFM dense (`arcee`) support

**Date**: 2026-11-23
**Status**: Open (draft PR, written before merge)
**Risk Level**: Low (a new model type; shared code changes are documentation-only `/// Used by:` and roster lines)

## Executive Summary

`mlxcel` exited with `Unsupported model type: arcee` on `arcee-ai/AFM-4.5B` (#1719). The issue was blocked because neither mlx-lm nor mlx-vlm implements `arcee`, so there was no parity target. On 2026-11-23 the maintainer accepted Hugging Face transformers `ArceeForCausalLM` as the reference. The PR adds `src/models/arcee.rs`, which reuses the shared Llama attention (GQA, YaRN, fused QKV, cache-owned dispatch, split-attention batched decode) and adds a gate-free squared-ReLU MLP, plus `scripts/arcee_oracle_trace.py`, which scores the `logit_trace` chunk plan with transformers 5.24.0 on CPU in float32. On the real bf16 checkpoint the teacher-forced trace has 0 decided-position disagreements over 2048 positions and over 8192 positions reaching position 8192, while reference arms with plain ReLU or with YaRN dropped fail.

## 1. Problem Statement

AFM-4.5B is Llama-shaped except for one block: `hidden_act: relu2` and no `gate_proj`, so the MLP is `down_proj(relu(up_proj(x))^2)` and 7.2x the hidden width. RoPE is YaRN with `factor` 20 over an original 4096 context. Mapping `arcee` to the plain Llama arm would fail to load (no `gate_proj`); hand-rolling a decoder would duplicate the YaRN, KV-cache and batched-decode machinery. Three AFM names exist in the tree or its issues (AFMoE, Apple AFM-7, this one), so the registration also had to keep them apart.

The second problem was the reference. transformers derives YaRN differently from the shared mlx-lm-style path in a few corners (a declared `attention_factor`, `mscale` without `mscale_all_dim`, `truncate: false`, a missing `factor`, a top-level `original_max_position_embeddings`). For AFM-4.5B the two agree; for configs where they would not, the port has to refuse rather than decode differently.

## 2. What Changed

- `src/models/arcee.rs` (new): `ArceeConfig` with its own `Deserialize` (the shared `rope_parameters` lift plus a `RopeScreen` capturing `truncate`, `partial_rotary_factor`, a top-level original context and a `type` / `rope_type` conflict), `validate()`, `to_llama3_args()` (full struct literal), `ArceeMLP`, `ArceeBlock` (`llama3::Attention` + `ArceeMLP` + two RMSNorms with `fused_add_rms_norm`), and `ArceeModel` with `last_row_logits`, native batched decode, batched and maskless padded prefill and the paged decode backend.
- Registration: `ModelType::Arcee` (enum, `ALL_MODEL_TYPES`, description under the "Arcee" heading, metadata test list), `detection.rs`, `registry.rs`, `model_metadata.rs` (config-backed), `LoadedModel`, `execution/memory_estimate.rs`, the TP fallback (`"arcee"`, not shardable), and the batched-decode coverage list.
- `scripts/arcee_oracle_trace.py` (new): transformers reference arm with `--arm relu`, `--arm no-yarn`, `--dtype bfloat16`, `--no-prefix-sharing`, `--report-range`; safetensors only, no remote code.
- Roster comments: `relu_squared`, `fused_add_rms_norm`, `forward_split_attention`, and the RoPE-override family list name Arcee.
- Docs and data: `docs/supported-models.md`, `CHANGELOG.md`, `benchmarks/metal_m3ultra_2026-11-23_afm-4.5b.csv`.
- Tests: `src/models/arcee_tests.rs` (16) and a detection test.

## 3. Technical Decisions

**Own block on the shared attention, not a change to `llama3::MLP`.** `llama3::MLP` has public `gate_proj` fields read by the tensor-parallel runtime and several tests; making the gate optional would touch every Llama consumer. A separate block is about 100 lines and leaves the Llama graph byte-identical.

**Refuse where transformers and the shared path disagree.** The screen compares transformers' `_compute_yarn_parameters` attention factor with the shared `yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim)` and refuses a mismatch. It also refuses what the shared resolver would silently degrade to the unscaled rotation (unknown RoPE type, a factorless `linear` / `llama3` / `yarn` block) and the keys the shared path ignores. Threading these into the shared YaRN code would change a path every YaRN family uses, for configurations no Arcee checkpoint has.

**Screen inside `Deserialize`.** The first version read `truncate` in `parse_config`, which the adapter route bypasses. The review moved it into the `Deserialize` impl so every route that builds `ArceeConfig` sees the same keys.

**f16 activations are safe here.** mlxcel casts bf16 weights to f16 on load. BitNet and nanochat skip that cast because squared-ReLU or unnormalized residuals overflow f16; on AFM-4.5B the reference's peak squared-ReLU activation is 165 and peak residual 363, so the cast is kept.

## 4. Measurements

M3 Ultra, `models/afm-4.5b` (bf16 on disk, f16 at load). Reference: transformers 5.24.0, torch 2.15.0, tokenizers 0.23.3, CPU, float32. `compare_logit_traces.py --decided 2`.

| Arms | Positions | Decided disagreement | Perplexity |
|---|---|---|---|
| ref vs mlxcel, `docs/architecture.md 256 8 8` | 2048 | 0 / 686 | 39.808 vs 39.824 |
| ref vs mlxcel, 4 chunks of 2048 behind up to 6144 tokens | 8192 | 0 / 2942 | 17.855 vs 17.859 |
| ref vs mlxcel `MLXCEL_TRACE_BATCH_ROWS=1` / `=4` | 2048 | 0 / 686 each | +0.05% / +0.07% |
| ref `--arm relu` vs mlxcel | 2048 | 44 / 44 | 16459 vs 39.8 |
| ref `--arm no-yarn` vs mlxcel | 2048 | 11 / 618 | 42.2 vs 39.8 |
| ref `--arm no-yarn` vs mlxcel, long | 8192 | 126 / 1637 | 140.1 vs 17.9 |

Every positive-arm disagreement has a reference top-two gap below 0.5 and lands on rank 2. Generation: 60 to 63 tok/s decode, 1557 tok/s prefill on a 195-token chat prompt; `mlxcel-server` served two concurrent requests and a streaming one.

## 5. Limitations

- No quantized Arcee export was tested.
- Tensor parallelism is refused.
- The review-driven screening commits do not change the forward path; they were covered by unit tests, not by a second real-checkpoint trace.
