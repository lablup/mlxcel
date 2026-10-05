# Technical Report: PR #2115 - Keep the VoiceChat TTS backbone stream f32 on CUDA (issue #2109)

**Date**: 2026-10-05

**Status**: Implemented and validated on GB10 (`--features cuda`) with synthetic weights; pending merge.

**Languages**: Rust (Nemotron VoiceChat EAR-TTS loader, Gemma 3 backbone, tests)

**Risk Level**: Low (exact widenings only; no real checkpoint available for validation)

## Executive Summary

mlxcel's CUDA overlay of MLX's promotion table (`src/lib/mlx-cpp/patches-cuda/dtype.cpp`, issue #636) resolves bf16 with f32 to bf16. The EAR-TTS reference runs its backbone in f32 against bf16 weights, so every stored bf16 tensor that meets the f32 stream unconverted demotes it on CUDA. PR #2093 fixed the code embeddings and the MoG head; this PR fixes the remaining four sites. The overlay and the shared `GemmaRMSNorm` stay unchanged.

## 1. Demotion Sites

| Site | Op | Fix |
|------|----|-----|
| `bos_emb` | `multiply(bool, bf16)` then `add(f32, bf16)` | promoted at load (`promotes_to_f32`) |
| `audio_prompt_projection_W` | `matmul(f32, bf16)` | promoted at load |
| backbone norms (6 per layer + final) | `fast::rms_norm(f32, bf16 1 + w)` | `Gemma3Backbone::widen_norms_to_f32` |
| gated fusion text branch | `multiply(f32, bf16)` | `astype` to the audio dtype |

The fusion site was not in the issue. `null_emb` was, but it is concatenated with the bf16 subword condition before `text_proj`, so it never meets the f32 stream directly; promoting it would make the condition f32 on non-CUDA builds and change `text_proj`'s output. It stays bf16, and the fusion cast covers its path.

## 2. Norm Widening

`GemmaRMSNorm::new` builds `1 + w` in the weight's dtype and keeps it private. `widen_norm_to_f32` reads the built `a = 1 + w` through `adjusted_weight()`, casts it to f32 and constructs a new norm from `a - 1` in f32. For half-precision `w`, `a` is zero or a multiple of `2^-11` (near `w = -1` the bf16 ulp is `2^-8`, the f16 ulp `2^-11`), so for `|a| < 2^24` both `a - 1` and `1 + (a - 1)` are exact and the rebuilt norm holds the widened `a` bit for bit. All forward paths (plain, batched decode, the fused quantized QKV kernel) read only the adjusted weight, and the fused kernel calls `fast::rms_norm` too, so they all take the f32 weight. Promoting `w` before `1 + w` was rejected because it changes the rounding of `1 + w` on Metal and CPU.

## 3. Tests

- `tts_dtype_tests::backbone_stream_is_f32_with_bf16_weights`: a tiny EAR-TTS with every weight stored bf16 checks the stream dtype after the prompt assembly (projection and `bos_emb`), after the fusion, after the final norm and on a step. On 33c45053 under `--features cuda` it fails at the first check (`left: 12` bf16, `right: 10` f32).
- `gemma3_backbone_tests::widened_norms_hold_the_stored_one_plus_w_in_f32`: byte equality of every widened norm against the cast stored `1 + w`, with edge values `-1`, `-0.99609375`, `-1.0078125`, `0`, `+/-300`, `65504`, and an f32 stream through a norm and the full backbone.
- `f32_promotion_keeps_norms_and_the_subword_path_as_stored` now lists `bos_emb` and `audio_prompt_projection_W` as promoted and keeps `null_emb` as stored.

`warmup` was split so the prompt assembly (`prompt_embeds`) and the fusion (`backbone_inputs`) are testable on their own; behavior is unchanged apart from the latent shape check now running before the code embedding.

## 4. Validation

GB10, `--release --features cuda`, `--test-threads=1` under `gpu-lock`: `models::nemotron_voicechat` 28/28, `models::gemma3_backbone` 6/6, `audio::f32_weights` 4/4. `cargo clippy --release --features cuda --lib --tests -- -D warnings` and `cargo fmt --check` are clean.

Non-CUDA bit identity was argued from the code, not measured: a CPU-only Linux build does not link (issue #2108). Each change is the exact bf16 to f32 `astype` that upstream promotion inserts inside `matmul`, `add`, `multiply` and `fast::rms_norm`, so the kernels see identical values; the norm test pins the only non-trivial step.

No real Nemotron VoiceChat checkpoint was used; none is available on GB10. The fused quantized QKV path was not exercised (the published TTS weights are dense).
