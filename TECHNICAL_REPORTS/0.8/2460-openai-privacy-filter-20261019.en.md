# PR #2460: OpenAI Privacy Filter (openai_privacy_filter) PII tagger

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with the bf16 and 8-bit mlx-community checkpoints; PR open

## Summary

Closes #2387, part of epic #2391. Adds the OpenAI Privacy Filter, a bidirectional sparse-MoE token classifier (8 layers, 128 experts, top-4) that tags PII with BIOES labels, to `mlxcel tag`. It implements `TokenClassifierModel` from #2383 (PR #2456) and adds a constrained Viterbi decoder with per-checkpoint calibration, selected with `--operating-point`.

## Problem

`model_type: openai_privacy_filter` had no arm, so every entry point failed with `Unsupported model type`. The model also needs pieces the #2383 pipeline did not have: banded attention with sinks over long inputs, the GPT-OSS expert layout and YaRN RoPE, and a decoder that only emits legal BIOES paths with calibrated transition biases.

## Technical decisions

- **Encoder.** `src/models/openai_privacy_filter.rs` reads the config (including the transformers-5 `rope_parameters` lift), sanitizes the HF fused layout (`gate_up_proj` split and transposed, `down_proj` transposed and renamed, sinks never cast or quantized, router quantized at 8 bits / group 64 when the checkpoint quantizes it). `openai_privacy_filter_layers.rs` holds the band mask, query-chunked banded attention with sinks, and an f32 top-4 MoE with token chunks.
- **Shared GPT-OSS helpers.** `yarn_rope_freqs` (with a `truncate` switch), `ExpertLinear` and `gpt_oss_swiglu` moved into `src/models/gpt_oss_common.rs`; `gpt_oss.rs` calls them. A teacher-forced trace on gpt-oss-20b shows the move changes nothing.
- **Viterbi decoder.** `src/token_classification/viterbi.rs` ports `mlx_vlm.privacy_filter.ViterbiDecoder` and `TransitionBiases`. Calibration comes from `viterbi_calibration.json` (`operating_points.<name>.biases`, six keys); a missing file means all zeros, and an unknown operating point or missing key fails at load. `decode_modes()` lists Viterbi first, so it is the default, and `--argmax` still works.
- **Contract kept.** One `TokenModelKind::PrivacyFilter` variant and one detector arm. `TokenClassifierLoadOptions` and `load_token_classifier_with` are additive; `load_token_classifier` wraps them. No trait signature changed.
- **Detection and refusal.** `get_model_type` fails with "openai_privacy_filter is a token classifier; use `mlxcel tag`", and the type is kept out of `ENCODER_ONLY_MODEL_TYPES`, so embedding detection does not claim it. `--operating-point` on an argmax-only family is refused before weights load.
- **Unsorted expert gather for dense planes.** CUDA `gather_mm` with `sorted_indices = true` over f32 activations and bf16 weights returned results that depended on allocation history: after an 8-bit model ran first in the same process, bf16 output drifted on some tokens, deterministically per predecessor. Unsorted gather is exact, and `gather_qmm` sorted (the 8-bit and GPT-OSS path) is exact in every order, so dense planes gather unsorted (about 1.5x the 8-bit forward on 7.5k tokens).
- **Review hardening.** Configs and planes that would wrap an i32 or index out of bounds are refused at load.

## Changes

- `src/models/openai_privacy_filter.rs`, `openai_privacy_filter_layers.rs`, `openai_privacy_filter_tests.rs` (new).
- `src/models/gpt_oss_common.rs`, `gpt_oss_common_tests.rs` (new), `gpt_oss.rs`, `detection.rs`, `mod.rs`.
- `src/token_classification/viterbi.rs`, `privacy_filter.rs` and their tests (new), `loader.rs`, `mod.rs`.
- `src/commands/tag.rs` (`--operating-point`), `src/main.rs` (help examples).
- `tests/fixtures/token_classification/`: Privacy Filter config, mlx-vlm fixture (mlx-vlm 0.7.8 git `73d366ec`, mlx CPU) and its generator.
- `docs/token-classification.md`, `docs/supported-models.md`, `CHANGELOG.md`.

## Verification

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, test binaries under `gpu-lock` with `--test-threads=1`: `privacy_filter` 20, `viterbi` 13, `gpt_oss` 11, `token_classification` 64, `models::openai_privacy_filter` 10, all passing; bin tests `extract_tests` 5 and `tag` 4 pass. Clippy and fmt are clean.

| Check | Result |
|---|---|
| `mlxcel tag -m models/mlx/openai-privacy-filter-bf16 "Alice Smith can be reached at alice@example.com."` | `private_person` (0..11), `private_email` (30..47), `<PRIVATE_PERSON> can be reached at <PRIVATE_EMAIL>.` |
| `--argmax`, `--operating-point default` | same spans |
| `--replacement "[REDACTED]"` / `--keep private_email` | `[REDACTED] can be reached at [REDACTED].` / `<PRIVATE_PERSON> can be reached at alice@example.com.` |
| `--operating-point nope` | refused, naming the defined `default` point |
| 8-bit checkpoint | same spans |
| `mlxcel generate -m models/mlx/openai-privacy-filter-bf16` | token-classifier refusal |
| Spans against mlx-vlm, six sentences plus replacement and keep, Viterbi and argmax, bf16 and 8-bit | identical |
| 4536-token input, default chunks vs unchunked (bf16) | max abs dp 0.00000, 0 argmax flips |
| GPT-OSS trace, gpt-oss-20b-mxfp4, `5 60 8 512`, main vs branch | 0/300 top-1, 0/100 decided, logit delta max 0.0000, rows byte-identical |

Log-prob parity on the acceptance sentence (10x33): max abs dp 6e-5, max abs dlogp 0.19 on labels above -10 nats, 5.4 on tail labels below -18 nats. The oracle's own f32 forward differs from its bf16 run by 4.5 nats in those tails, so the issue's 2e-2 bound is applied in probability space, plus 0.5 nats on labels above -10 nats; the issue body records the deviation. The first German test sentence flipped an argmax at a 0.19-nat margin between bf16 CUDA and bf16 CPU, so fixtures require a 0.5-nat margin and that sentence was replaced.

## Follow-ups

- The CUDA sorted `gather_mm` result that depends on allocation history (f32 activations, bf16 weights, `M == 1 && right_sorted_` branch in the overlay `matmul.cpp`) deserves its own issue: other bf16 MoE models may hit it.
- The `-4bit`, `-5bit`, `-6bit`, `-mxfp4`, `-mxfp8` and `-nvfp4` exports are not on this host; they load through `ExpertLinear` like GPT-OSS.
- Low review items: Viterbi sums in f32 (upstream adds f64 biases), the band mask is rebuilt per chunk per layer, and malformed tensor shapes abort inside MLX instead of erroring at load.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 24 |
| Lines added | 5071 |
| Lines deleted | 286 |

| Commit | Message |
|--------|---------|
| `7b362477` | feat(models): support OpenAI Privacy Filter PII tagger in mlxcel tag |
| `35078b81` | fix(models): satisfy clippy on the Privacy Filter port |
| `87c041ae` | docs: state the Viterbi tie order as the candidate order, not label order |
| `2a088636` | fix(models): refuse Privacy Filter configs and planes that would wrap or index out of bounds |
| `7b7ecd99` | docs: document Privacy Filter public items and the unsorted expert gather |

## Related

- Epic #2391; built on #2383 (PR #2456); sibling #2390 (GLiNER2.5).
