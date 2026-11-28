# Technical Report: PR (issue #1980), Gemma 3 KV estimator defaults

**Date**: 2026-09-26

**Status**: Implemented and validated on the real gemma-3-4b-it-4bit checkpoint; not yet merged.

**Languages**: Rust

**Risk level**: Low

## Executive summary

`kv_arch.rs::attn_dims` falls back to `head_dim = hidden_size / num_heads` with `num_heads` defaulting to 1 when a config omits the per-head fields. Gemma 3 `text_config` files routinely omit `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `sliding_window_pattern` (gemma-3-4b-it-4bit carries only `hidden_size`, `intermediate_size`, `num_hidden_layers`, `sliding_window`), so every caller of the estimator overstated the model's KV footprint by roughly 2.6x. `classify` now fills in the HF `Gemma3TextConfig` defaults for `model_type` `gemma3` / `gemma3_text` (`num_attention_heads` 8, `num_key_value_heads` 4, `head_dim` 256, `sliding_window_pattern` 6) whenever a field is absent from the config, leaving any explicit value untouched.

## Problem statement

The prompt-cache snapshot capacity default added in #1978, `estimate_total_memory`, and `mlxcel inspect` all read the same `kv_arch::classify` path. On the real 4B checkpoint the architecture was misclassified as plain standard attention (`num_heads=1`, `head_dim=2560`) instead of the true sliding-window shape (8 heads, 4 KV heads, head_dim 256, one global layer in every 6), which both inflated the per-token byte rate and hid the sliding-window savings the estimator exists to capture.

## Change summary

- Added `apply_gemma3_defaults(text, model_type)` in `src/execution/kv_arch.rs`: for `model_type` `gemma3` / `gemma3_text`, it clones the config's `text_config` object and inserts `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `sliding_window_pattern` only for whichever of those keys is absent. `classify` now runs every downstream lookup (`attn_dims`, the `sliding_window_pattern` global/windowed split) against this merged value.
- Two new unit tests: one reproducing the real 4B `text_config` shape and asserting `marginal_bytes_per_token == 34 * 2 * 4 * 256 * 2`, and one asserting an explicit `num_attention_heads` is kept while the still-absent `num_key_value_heads` / `head_dim` still take the Gemma 3 defaults independently.
- No other model family's classification path changed; the defaults apply only when `model_type` is `gemma3` or `gemma3_text`.

## Validation

- `cargo test --release --features cuda --lib -- kv_arch:: --test-threads=1`: 28 passed, including the two new tests.
- `cargo test --release --features cuda --lib -- execution:: server::prompt_cache --test-threads=1`: 324 passed.
- `cargo test --release --features cuda --lib -- kv_cache_advisor:: quant_advisor:: memory_estimate:: --test-threads=1`: 83 passed, confirming the other estimator callers are unaffected.
- `cargo clippy --release --features cuda --lib --tests -- -D warnings`: clean.
- On `gemma-3-4b-it-4bit`, the startup line changed from `architecture=standard attention (34 layers, full context)`, `kv_bytes_at_representative_tokens=2852126720` (348,160 bytes/token) to `architecture=sliding-window: 29 layer(s) capped at 1024 tokens, 5 global`, `kv_bytes_at_representative_tokens=1140850688` (139,264 bytes/token), matching the acceptance criterion's `34 x 2 x 4 x 256 x 2` figure exactly. `recommended_snapshot_capacity_bytes` dropped from 17,112,760,320 (15.9 GiB) to 6,845,104,128 (6.37 GiB).
- A 3-turn chat against the running server still hit the prompt cache on turns 2 and 3 (`cached_tokens` 3294/3309 out of 3303/3318 prompt tokens), confirming the smaller default did not break cache reuse.

## Remaining work

None identified for this issue; the fix is scoped to the Gemma 3 default-field gap.
