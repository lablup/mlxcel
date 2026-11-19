# MLA prefill with the rope score folded into SDPA (issue #3071), M5 Max, 2026-11-15

Issue #3071: the DeepSeek V3 family (DeepSeek V3, Kimi-VL, Kimi-K2, Youtu) and DeepSeek V3.2 computed the rope half of the MLA score as its own matmul, `(q_pe * scale) @ k_peᵀ`, added the causal mask to it (which promotes it to f32), and handed the dense `[B, H, L, S]` result to SDPA as an additive mask. The tensor exists before SDPA runs, so the query-axis chunk gate from #3049 could not bound it.

The change concatenates `[q_nope | q_pe]` and `[k | k_pe]` along the head dimension for every multi-token step, so SDPA computes the whole score itself and the mask is MLX's causal mode (or the caller's `[.., L, S]` mask, or V3.2's top-k mask, now `[B, 1, L, S]`). MLX does not fuse a query width (192, or 576 on the absorbed latent) that differs from the value width (128, or 512), so these calls take MLX's fallback, and the #3049 gate query-chunks it to the 256 MiB Metal budget (the `chunked_calls` column counts the gated calls). Single-token decode keeps the old rope-mask form and is unchanged. The absorbed multi-token step in `mla::decode::absorbed_attention` (DeepSeek V2 with `MLXCEL_MLA_ABSORBED=1`) takes the same folded call.

Host: Apple M5 Max, 128 GB, macOS 27. Main is `9c5ea4df`; the branch is that commit plus this change; both release builds. 28K-token prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`, greedy, 48 tokens, reasoning included in the compared text. Peak is the `/usr/bin/time -l` footprint, prefill is `--profile`. Each run follows a 75 s idle period; rep 1 ran main then branch per cell, rep 2 ran branch then main. Raw rows: `mla-prefill-rope-fold-m5max-2026-11-15/runs.csv`, identity per cell: `output-identity.csv`.

## Results

| model | prefill | peak main | peak branch | prefill main (rep 1 / rep 2) | prefill branch (rep 1 / rep 2) | greedy output |
|---|---|---|---|---|---|---|
| kimi-vl-a3b-thinking-4bit (19723 tokens) | 2048 chunks | 19.33 / 19.99 GB | **14.43 GB** | 40.92 / 28.07 s | 36.58 / 25.78 s | identical |
| kimi-vl-a3b-thinking-4bit | single pass | 76.32 GB | **32.03 GB** | 53.55 / 35.37 s | 40.78 / 30.88 s | differs (see trace) |
| deepseek-v2-lite-chat-4bit-mlx (28705 tokens) | 2048 chunks | 27.99 GB | 27.98 GB | 53.39 / 42.84 s | 54.83 / 42.81 s | identical |
| deepseek-v2-lite-chat-4bit-mlx | single pass | 54.32 GB | 54.32 GB | 73.08 / 52.70 s | 70.79 / 52.85 s | identical |
| deepseek-v2-lite-chat-4bit-mlx, `MLXCEL_MLA_ABSORBED=1` | 2048 chunks | 15.13 GB | 15.13 GB | - / 42.73 s | - / 42.64 s | identical |

Rep 1 ran while other jobs shared the host, so its absolute times are higher; compare arms within a rep. Kimi-VL prefill is 8 to 24% faster on the branch in both reps: the old path ran a fused array-mask SDPA but first built and added the f32 `[16, L, S]` rope tensor in every layer.

DeepSeek V2-Lite does not move. Its default (decompressed) path already concatenated the rope half (`deepseek_v2.rs`, as in mlx-lm's `deepseek_v2.py`), and since #3049 its unfused causal SDPA is query-chunked (324 gated calls at 2048 chunks, 27 single pass, on both arms). Main reproduces the issue's 27.98 / 54.44 GB with every attention call already bounded, so that remainder is not the attention score tensor and needs its own investigation. The Kimi-VL single-pass remainder includes the full-position logits tracked in #3070.

## Teacher-forced logit trace (Kimi-VL)

The fold changes the order in which the nope and rope halves of the score are summed, so the single-pass greedy output diverges at the first reasoning sentence (a near-tie phrasing choice). `examples/logit_trace` from each arm over `docs/architecture.md`, `docs/benchmarks.md`, `docs/adding-models.md`, `docs/code-guidelines.md`, compared with `scripts/compare_logit_traces.py main.tsv branch.tsv --decided 2`:

| shape | positions | top-1 disagreement | on decided positions (gap >= 2) | largest gap at a disagreement | worst rank | perplexity main / branch |
|---|---|---|---|---|---|---|
| width 2048, cold (`2048 4 8 0`) | 8192 | 163 (1.99%) | 1 / 2627 (0.038%) | 2.18 | 4 | 26.5426 / 26.5275 |
| width 2048 behind 2048 context (`4096 2 8 2048`) | 8192 | 172 (2.10%) | 0 / 2731 (0.000%) | 1.69 | 4 | 21.4102 / 21.4021 |

159 and 169 of the disagreements sit where the reference's top two were within 0.5, and 90 to 93% of them pick the reference's second choice. Logit delta on the reference's choice: p50 0.031, p99 0.45 (cold) and 0.43 (behind context). This is the reassociation class, with perplexity 0.04 to 0.06% lower on the branch.

## Reproduce

```bash
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 28000 > prompt_28000.txt
MLXCEL_PREFILL_CHUNK=2048 RUST_LOG=mlxcel_core::layers=debug /usr/bin/time -l ./target/release/mlxcel generate \
  -m models/kimi-vl-a3b-thinking-4bit -p "$(cat prompt_28000.txt)" -n 48 --temp 0 --profile --show-reasoning
# MLXCEL_PREFILL_CHUNK=0 for single pass; count "query-chunking" lines on stderr for chunked_calls.
./target/release/examples/logit_trace models/kimi-vl-a3b-thinking-4bit corpus.txt 2048 4 8 0 > arm.tsv
./target/release/examples/logit_trace models/kimi-vl-a3b-thinking-4bit corpus.txt 4096 2 8 2048 > arm-ctx.tsv
python3 scripts/compare_logit_traces.py main.tsv branch.tsv --decided 2
```
