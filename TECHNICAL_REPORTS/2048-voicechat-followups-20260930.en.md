# Technical Report: PR #2048 - VoiceChat mic check, front-end quantization, attention dedup

**Date**: 2026-09-30

**Status**: Implemented and validated on the 4-bit checkpoint; the live microphone exchange is not verified; pending merge.

**Languages**: Rust

**Risk Level**: Low. Loader signatures gain a parameter, and the attention refactor keeps the existing parity guards green.

## Executive Summary

Three leftovers of epic #1372, grouped in issue #2043 and delivered in one PR. The microphone example wraps input-device errors with the device name and a macOS permission hint and falls back to a supported config. The speech front end (perception, FastConformer, RNNT joint) now loads with the checkpoint's `(group_size, bits)` instead of `(64, 4)`. `RelPositionMultiHeadAttention::forward` and `stream` share one private `attend`.

## 1. Problem Statement

- The microphone example surfaced bare CoreAudio errors, and in an agent session without microphone permission it hung or failed with "Unknown property".
- Every front-end `UnifiedLinear` trusted group size 64. A checkpoint that quantizes those layers at another group size would be dequantized with wrong parameters and produce garbage without an error. The LM and TTS loaders already read the checkpoint value.
- `forward` and `stream` duplicated the head split, position term, SDPA call and output projection.

## 2. Change Summary

- `examples/voicechat_microphone.rs`: `input_error` builds `input device {name}: {err}.` plus a macOS-only hint; `input_config` falls back to `supported_input_configs()` at the maximum sample rate; `build_input` and `play` use the wrapper.
- Loaders: `quantization: (i32, i32)` added to `VoiceChatPerception`, `FastConformerEncoder`, `CausalDwStridingSubsampling`, `ConformerBlock` (and its feed-forward), `RelPositionMultiHeadAttention` and `RnntDecoder`. `model::front_end_quantization` supplies it to `load`; the LM call keeps `config.default_quantization()`.
- `attention.rs`: one `attend(q_in, kv_in, pos_emb, mask)`; `stream` keeps its shape and `pos_emb` row checks and error strings and validates `pos_emb` before attending.
- `rnnt/mod.rs`: `joint_logits` extracted from `step_frame`.
- Tests: `attention_loads_non_default_quantization`, `rnnt_joint_loads_non_default_quantization`, `front_end_quantization_follows_checkpoint_config`; existing callers pass `(64, 4)`, including `tests/nemotron_voicechat_stream_real.rs`, which the issue did not list.

## 3. Technical Decisions

### Tests compare against dequantized dense weights

A `(32, 8)` map is loaded with `(32, 8)` and its output compared, within 1e-4, with a module built from the same quantized triples dequantized through `mlxcel_core::dequantize`. The issue also asked to assert that loading with `(64, 4)` does not match. That path raises an MLX C++ `std::invalid_argument` inside `quantized_matmul`, which aborts the process and cannot be caught, so the tests assert the stored scale layout instead.

### The LM keeps its Option

The issue suggested reusing the unwrapped `(64, 4)` default for the LM. The LM loader takes an `Option` and only sets the text config's quantization when one is present, so passing `Some((64, 4))` for a dense checkpoint would change behavior. It still gets `config.default_quantization()`.

## 4. Validation

- `cargo test --release --lib -- models::nemotron_voicechat audio::fastconformer audio::rnnt`: 48 passed.
- Env-gated `nemotron_voicechat_front_real` and `nemotron_voicechat_streaming_real` pass on the 4-bit checkpoint.
- `mlxcel generate --audio question.wav --stream --seed 0` still prints "[user] What is the capital of France" and "The capital of France is Paris."
- Clippy (workspace lib, tests, examples, and the `voicechat-mic` example), `cargo fmt --check`, and the `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest` contract tests pass.
- The `voicechat-mic` example builds and `--list-devices` lists the input and output devices.

## 5. Known Limitations

- The live microphone exchange, the wrapped error with permission revoked, and the #1376 checkbox need a manual run from a terminal with microphone access (procedure in the PR body).
- Per-module overrides in the `quantization` map remain unsupported, as before.
