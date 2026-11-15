# Causal-mode prefill for the hybrid and Qwen-VL families (issue #3051), M3 Ultra, 2026-11-15

Issue #3051: the full-attention layers of Qwen 3.5, Qwen3-Next, GraniteMoeHybrid and the Qwen-VL models built an explicit f32 causal array mask for every prefill. Before M5, MLX fuses SDPA in causal mode at head_dim 256 but not with an array mask, so a single-pass prefill took the query-chunked fallback from #3049 and kept the 3.65 GiB mask alive for the whole pass at 28K tokens.

`mlxcel_core::utils::prefill_causal_mask` now leaves causality to MLX's causal mode when that is equivalent to the array mask (more than one query, one row, no caller mask, not a speculative verify block) and MLX's own verdict (`sdpa_mode_is_fused`, memoized per shape) says causal mode fuses at the call's shape. Elsewhere the model keeps its single array mask, shared by every full-attention layer, which is the earlier dispatch. On the M3 Ultra causal mode fuses for a cold prefill and for the first piece of a chunked one. It does not fuse for later pieces (head_dim 256 over `q_len != k_len`), so those keep the array mask.

Host: Apple M3 Ultra, 80-core GPU, 512 GB, pre-M5. One release build of branch `fix/issue-3051-causal-mode` (`metal,accelerate`) runs both arms. The base arm sets `MLXCEL_PREFILL_CAUSAL_MODE=0`, which restores the array mask everywhere; the new arm uses the default. Generation is greedy, 32 tokens, reasoning included in the comparison. Prompts come from `scripts/bench/metal_sdpa_chunk/make_prompt.py`. Everything ran in one GPU lock hold, with the arms interleaved A/B then B/A (`scripts/bench/prefill_peak/ab_prefill.sh` with `BASE_ENV`/`NEW_ENV`). Load average was 0.3 to 5.4 and is recorded per row in the CSVs.

The issue names `scripts/bench/metal_sdpa_chunk/verify.sh`, but that script's arms toggle `MLXCEL_METAL_ATTENTION_CHUNK`, not the mask, so the interleaved harness was used. Its `gated_calls` column counts the `query-chunking` debug lines of a CLI run, one per full-attention layer whose SDPA takes the materializing fallback. The server cells do not emit that log, so they read 0 in both arms.

## Results

28000-token prompt (28707 tokens after the chat template), means of two repetitions per arm (`ab-summary-28k.csv`):

| Path | Model | Peak, base to new | Prefill / TTFT, base to new | Gated calls |
|---|---|---|---|---|
| CLI single pass | qwen3.5-4b-4bit | 30.13 to **19.82 GB** (-34%) | 16.89 to **13.38 s** (-20.8%) | 8 to 0 |
| CLI single pass | qwen3.8-27b-4bit | 64.20 to **53.96 GB** (-16%) | 86.09 to **73.33 s** (-14.8%) | 16 to 0 |
| CLI 2048-token chunks | qwen3.5-4b-4bit | 8.97 to 8.97 GB | 13.87 to 13.85 s | 96 to 96 |
| CLI 2048-token chunks | qwen3.8-27b-4bit | 25.80 to 25.80 GB | 74.77 to 74.74 s | 208 to 208 |
| server, `--parallel 1` | qwen3.5-4b-4bit | 9.10 to 9.10 GB | 13.90 to 13.89 s | - |
| server, `--parallel 1` | qwen3.8-27b-4bit | 25.92 to 25.92 GB | 74.88 to 74.81 s | - |

4600-token prompt with 2048-token chunks and `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16`, so that the fallback gate fires at this length (`ab-summary-4600.csv`):

| Model | Peak, base to new | Prefill, base to new | Gated calls |
|---|---|---|---|
| qwen3.5-4b-4bit | 5.34 to 5.34 GB | 1776.8 to 1763.4 ms (-0.8%) | 24 to 16 |
| qwen3.8-27b-4bit | 19.85 to 19.85 GB | 10714.4 to 10664.8 ms (-0.5%) | 48 to 32 |

Repetitions agree within 57 ms at 28K and within 12 ms at 4.6K. Chunked peaks agree within 1 MB between arms.

Generated text is byte-identical between arms in every cell, for CLI and server. An image prompt on qwen3.5-4b-4bit (`tests/fixtures/test_image_shapes.png`, `--show-reasoning`, 48 tokens) is also identical in both repetitions.

The single pass removes every gated call and the 3.65 GiB mask. In a chunked prefill only the first piece changes, so at 4.6K one of the three gated pieces is gone. At 28K the first 2048-token piece is under the score budget in both arms, so the count does not move. These later pieces are the remaining gap, and on this hardware MLX falls back for them whatever mask they are given.

## Alternatives measured and rejected

Two other ways to handle the pieces after the first were measured at 28K, in the same interleaved setup (separate lock holds, CSVs kept):

| Variant | qwen3.5-4b-4bit chunked peak | qwen3.8-27b-4bit chunked peak |
|---|---|---|
| Causal mode everywhere; a falling-back call runs `chunked_causal_attention`, which slices K/V and builds a mask per query chunk (`rejected-chunked-causal-28k.csv`) | 8.95 to 9.19 GB (+2.6%) | 25.79 to 26.82 GB (+4.0%) |
| Causal mode everywhere; a falling-back call builds `create_causal_mask` itself and runs the array-mask chunker (`rejected-per-layer-mask-28k.csv`) | 8.95 to 9.01 GB (+0.6%) | 25.78 to 28.72 GB (+11.4%) |

Both variants regress the server's default chunked path. The per-layer mask also costs one f32 `[2048, 30K]` array per full-attention layer instead of one per forward. Asking MLX first, and keeping the shared mask where causal mode would fall back anyway, is the only variant tried that does not raise any peak.

## M5

Not run here. On an M5-generation GPU, MLX's verdict may fuse causal mode for the pieces after the first as well. In that case they take causal mode and the chunked cells should show fewer gated calls. The verification request with exact commands is on the PR.

## Rerun

```
BASE_BIN=target/release/mlxcel NEW_BIN=target/release/mlxcel MODELS_DIR=models \
  BASE_ENV=MLXCEL_PREFILL_CAUSAL_MODE=0 NEW_ENV=MLXCEL_PREFILL_CAUSAL_MODE=1 \
  TOKENS=28000 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
BASE_BIN=target/release/mlxcel NEW_BIN=target/release/mlxcel MODELS_DIR=models \
  BASE_ENV=MLXCEL_PREFILL_CAUSAL_MODE=0 NEW_ENV=MLXCEL_PREFILL_CAUSAL_MODE=1 \
  TOKENS=4600 CLI_CELLS="qwen3.5-4b-4bit:2048 qwen3.8-27b-4bit:2048" SERVER_MODELS="" \
  EXTRA_ENV=MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
```
