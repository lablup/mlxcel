# PR #2457: Upstream numeric fixes for Gemma 3n, PhiMoE, Gemma 4 audio and Phi-3-small

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints for all four families; PR open

## Summary

Closes #2379. Ports four upstream numeric fixes in families mlxcel ships, in one PR as the issue asks: the Gemma 3n KV-shared RoPE offset, PhiMoE sparsemixer routing and LongRoPE, the Gemma 4 audio attention mask, and the Phi-3-small padding-token suppression.

## Problem

- **Gemma 3n** (ml-explore/mlx-lm#1886): KV-shared layers rotated their queries at the owner cache's offset after the owner had already advanced it in the same forward, so every multi-token forward put shared-layer queries one chunk too far.
- **PhiMoE** (ml-explore/mlx-lm#1939, #1889, Blaizzy/mlx-vlm#2419): routing used softmax over the top-k logits instead of the reference sparsemixer, and the checkpoint's `longrope` scaling block was ignored, so long prompts ran on plain RoPE.
- **Gemma 4 audio** (Blaizzy/mlx-vlm#2385): the causal-valid mask admitted `dist == max_past`, one position past the intended left context.
- **Phi-3-small** (ml-explore/mlx-lm#1938): the 91 padding ids below `vocab_size` could be sampled.

## Technical decisions

- **Gemma 3n offset read once.** The RoPE offset is read once per forward from the first owner cache and threaded through `DecoderLayer` into `Gemma3nAttention::forward`, so owner and shared layers rotate at the same position.
- **PhiMoE through the Phi-3 LongRoPE code.** `sparsemixer` replaces softmax-over-top-k. The `longrope` block goes through `phi3::SuRope` via a new `pub(crate) SuRope::from_scaling`; the rotary-prefix scale moves onto `SuRopeTable`, and `rope_table_regime` keys the prompt cache on the short/long table. A null `rope_scaling` keeps plain RoPE. `num_experts_per_tok > num_local_experts` and `num_experts_per_tok == 0` fail at load.
- **Fused decode receives the scores unchanged.** The fused decode kernel (`run_fused_moe_two_kernel`) casts scores to f32 and multiplies without renormalizing (the normalizing code near `mlx_cxx_kernels.cpp:1491` is the DeepSeek `fused_moe_forward`, a different entry point). A test pins fused output equal to the graph path with scores summing to 1.5025.
- **Gemma 4 audio mask from distance.** `causal_valid_mask` builds the mask from `dist = q + max_past - c`, which excludes `dist == max_past`.
- **Phi-3-small suppression as a bias.** `output_suppressed_token_ids` returns the 91 padding ids, applied by the sampler; forward logits stay raw so traces and scoring are unaffected.
- **Audio arm in logit_trace.** `logit_trace` gains `MLXCEL_TRACE_AUDIO` and `MLXCEL_TRACE_AUDIO_PROMPT` so the audio fix can be traced.

## Changes

- `src/models/gemma3n.rs`, new `src/models/gemma3n_tests.rs`: offset threading and chunking tests (the xla-diagnostics call sites updated to the new signature).
- `src/models/phimoe.rs`, `src/models/phimoe_tests.rs`, `src/models/phi3.rs`: sparsemixer, LongRoPE, config validation.
- `src/audio/encoder.rs`, `src/audio/encoder_tests.rs`: mask.
- `src/models/phi3small.rs`, `src/models/phi3small_tests.rs`: suppressed ids.
- `examples/logit_trace/`: audio arm. Docs and CHANGELOG.

## Verification

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, test binaries under `gpu-lock` with `--test-threads=1`: 42/42 passed across `models::phimoe`, `models::gemma3n::tests`, `audio::encoder`, `models::phi3` (including phi3small). The Gemma 3n chunking test and the mask predicate test fail on main. Clippy (`--lib --tests` and `--example logit_trace`) and fmt are clean.

mlxcel's Gemma 3n MLP and AltUp input round to bf16, so a single 12-row pass differs from narrower passes by one bf16 ulp (2.7e-3 fixed, 2.7e-2 to 6.7e-2 reverted). Multi-pass splits are gated at 1e-3 (1e-6 fixed, at least 5.6e-3 reverted) and the single pass at 1e-2. This is a deviation from the issue's tolerance and was measured on CUDA only.

Real-checkpoint traces, fixed against reverted, `--decided 2`:

| Checkpoint | Width | Context | Decided disagreement | PPL fixed / reverted |
|---|---|---|---|---|
| gemma3n-e2b-4bit | 1 | 512 | 0/151 | 26.44 / 26.20 |
| gemma3n-e2b-4bit | 5 | 512 | 3/177 (1.7%) | 28.48 / 27.36 |
| gemma3n-e2b-4bit | 32 | 512 | 14/334 (4.2%) | 31.75 / 33.09 |
| phi-3.5-moe-4bit | 5 | 512 | 2/141 (1.4%) | 7.91 / 9.62 |
| phi-3.5-moe-4bit | 1 | 512 | 1/48 (2.1%) | 15.65 / 19.50 |
| phi-3.5-moe-4bit | 5 | 4200 (long table) | 118/136 (86.8%) | 7.70 / 3008.7 |
| gemma-4-e2b-it-4bit + audio | prefill | 100 audio tokens | 0/78 | 1.657 / 1.657 |

Gemma 3n at width 1 has no decided disagreement because shared layers write no KV, so the only error there is a one-position query shift in 10 of 30 layers; the effect grows with width. The audio clip is a deterministic 4 s synthetic chirp, tone and noise WAV; the arms differ (max logit delta 0.875) but not at decided positions. `mlxcel generate` on Phi-3.5-MoE is fluent at 128 tokens. On `phi-3-small-8k-instruct-aq4_64` (substitute for mlx-community Phi-3-small-8k-instruct-4bit), 256 sampled tokens through the suppression bias emitted no listed id. A transformers float32 reference for Phi-3.5-MoE (42B) is not feasible on this host.

## Follow-ups

- `mlxcel-xla` `gemma3n_emit_ops.rs` mirrors the old shared-layer query offset, so MLX-vs-XLA Gemma 3n parity will now diverge.
- `phimoe::ModelArgs` reads `layer_norm_eps` (default 1e-6) while the checkpoint spells it `rms_norm_eps: 1e-5`.
- The #674 chunked-prefill opt-out on `Gemma3nModel` may have been this bug; flipping it needs the parity battery.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 14 |
| Lines added | 1200 |
| Lines deleted | 135 |

| Commit | Message |
|--------|---------|
| `3e942576` | fix(models): port upstream numeric fixes for four model families |
| `63a22410` | test(models): make the Gemma 3n chunking test robust to bf16 rounding |
| `ad4c60ae` | test(models): widen the Gemma 3n multi-pass tolerance and log the fix |
| `9374c11c` | docs: note EmbeddingGemma 2 in the Gemma 4 audio mask entry |
| `8778f29c` | fix(models): reject zero experts per token in PhiMoE configs |

## Related

- ml-explore/mlx-lm#1886, #1939, #1889, #1938; Blaizzy/mlx-vlm#2419, #2385.
