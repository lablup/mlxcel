# Causal-mode and native-window prefill for attention-sink models (issue #3098), M5 Max, 2026-11-15

The attention-sink decoders built a dense f32 prefill mask because the sink SDPA bridge took only an array mask. Issue #3098 adds `fast_scaled_dot_product_attention_with_sinks_causal`, which calls MLX's `scaled_dot_product_attention` with `mask_mode = "causal"`, `window_size` and the sinks and no array, and `mlxcel_core::causal_prefill_attention`, which uses it wherever MLX's own verdict (asked with sinks) says the call fuses. GPT-OSS, Falcon-OCR, MiMo v2 Flash and an all-local DeepSeek-V4 stack decide once per forward through `prefill_causal_mask` / `prefill_sliding_window_mask` with sink geometry; Laguna, which builds its band per layer, routes every multi-token layer through `causal_prefill_attention`. Padded, batched, caller-masked and speculative-verify forwards keep the array.

Host: Apple M5 Max, 128 GB, macOS 27, shared with other jobs (1-minute load about 4.5). Two release builds (`metal,accelerate`): `main` at `00cd2d69` and the branch on top of it. Greedy generation, 48 tokens, `--show-reasoning` for gpt-oss and laguna. Prompts come from `scripts/bench/metal_sdpa_chunk/make_prompt.py` (28000, or 9000 for falcon-ocr). Every run held the GPU lock alone after a 60 to 75 s cooldown; single pass ran main first, 2048-token chunks ran branch first. Peak is the `/usr/bin/time -l` footprint and prefill is from `generate --profile`. Per-run rows are in `prefill-sink-causal-mode-m5max-2026-11-15/runs.csv`.

## Results

| Model | Prompt tokens | Path | Peak, main to branch | Prefill, main to branch |
|---|---|---|---|---|
| gpt-oss-20b-mxfp4-q4 | 19773 | single pass | 22.54 to **17.86 GB** | 6.63 to **3.62 s** |
| gpt-oss-20b-mxfp4-q4 | 19773 | 2048-token chunks | 13.42 to 13.16 GB | 3.96 to 3.84 s |
| laguna-xs.2-4bit | 28747 | single pass | 35.84 to **29.25 GB** | 39.03 to **7.32 s** |
| laguna-xs.2-4bit | 28747 | 2048-token chunks | 22.32 to 21.99 GB | 9.71 to 7.49 s |
| falcon-ocr (text only, two repetitions) | 9067 | single pass | 5.11 to 4.70 GB | 0.79 / 0.94 to 0.59 / 0.28 s |

Generated text is byte-identical between main and the branch in every cell, so no teacher-forced trace was needed.

gpt-oss-20b has 64 query heads over 8 KV heads at head dim 64 with a 128-token window on its sliding layers; both layer kinds carry sinks, and both now take MLX's causal mode (full layers) or causal mode plus `window_size` (sliding layers) with the sinks. falcon-ocr has sinks on every layer and runs no chunked prefill; its prompt exceeds the checkpoint's 8192-token `max_seq_len`, which does not affect the A/B. laguna-xs.2-4bit does not enable `swa_attention_sink_enabled`, so it measures the per-layer route without sinks: before, each of its 40 layers built its own `[L, k_len]` band; now its full layers take causal mode and its 512-token sliding layers the native window entry. The laguna single-pass main prefill (39.03 s) is one run on a shared host and likely includes memory pressure from those per-layer masks; the peak difference does not depend on it.

## Not converted

The OpenAI privacy filter's attention is a symmetric (bidirectional) band over 128-token windows, which MLX's causal mode cannot express, and it already builds only per-window masks; it is unchanged. DeepSeek-V4's compressed and sparse layers extend the local mask with their pooled rows, so a stack with any of them keeps the shared array; only an all-local stack leaves it to MLX. MiMo v2 Flash has no local checkpoint; its sink layers decide the same way as GPT-OSS, and its full layers at head dims 192/128 stay on the array where MLX does not fuse.
