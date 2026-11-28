# Llama 4 chunked attention without a dense mask (issue #3100), M5 Max, 2026-11-15

Llama 4's MoE (local) layers attend only to keys in the query's own `attention_chunk_size` block (8192 for llama-4-scout). Before #3100 every prefill forward built that as one dense f32 `[L, S]` chunk mask shared by all MoE layers and ran the masked SDPA over every visible key. A one-row prefill now leaves the MoE layers maskless: `Llama4Cache::attend` splits the step at chunk boundaries and runs MLX's causal-mode SDPA per piece over that piece's own keys, from the chunk start (or the cache's `start_position`) to the piece's last query. Batched rows, speculative verify blocks and `MLXCEL_PREFILL_CAUSAL_MODE=0` keep the dense mask.

Host: Apple M5 Max, 128 GB, macOS 27. Two release builds (`metal,accelerate`): `main` at `00cd2d69` and the branch on top of it. The CLI cells ran through `scripts/bench/prefill_peak/ab_prefill.sh` (28000-word prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py`, 19717 tokens with this tokenizer; greedy, 48 tokens) with a 75 s cooldown before every run, arm order alternated between repetitions, all runs under one GPU lock hold. The host was shared with other jobs' builds (1-minute load 1.6 to 30, per row in `ab-summary.csv`).

## Results

| Path | Peak, main to branch | Prefill, main to branch (rep 1, rep 2) |
|---|---|---|
| single pass | 72.88 to **68.41 GB** | 37.92 / 27.31 s to 38.50 / 27.73 s |
| 2048-token chunks | 65.86 to 65.70 GB | 22.25 / 27.48 s to 21.29 / 17.84 s |

Peak is the `/usr/bin/time -l` footprint and repeats within 8 MB per arm. The single-pass peak drops 4.46 GB: the dense mask and the masked SDPA over the full 19717-key window are gone, and each MoE layer now attends at most 8192 keys per query piece. With 2048-token pieces the mask was at most `[2048, 10240]`, so the peak moves by only 0.17 GB. Prefill timing is within the host's load noise in both paths (the first single-pass repetition ran at load 10 and 30, the main arm's second 2048-chunk run at load 8.5), so no prefill change is claimed.

Generated text is byte-identical between main and the branch in both cells and both repetitions.

## Arithmetic

Per-chunk causal mode replaces the dense additive mask inside MLX's SDPA, so the change can move the numbers. The teacher-forced trace (`examples/logit_trace`, `scripts/compare_logit_traces.py --decided 2`) compares the branch with `MLXCEL_PREFILL_CAUSAL_MODE=0` (the dense chunk mask, main's dispatch) against the branch default, over three 2048-token forwards behind 7168 tokens of context, so each traced forward spans positions 7168 to 9215 and crosses the 8192 chunk boundary: the piece starts mid-chunk behind cached keys of its chunk, which is the case the split has to index correctly.

| Positions | Top-1 disagreement | On decided positions (gap >= 2) | Max logit delta on the reference choice | Perplexity, mask to causal |
|---|---|---|---|---|
| 6144 | 0 | 0 / 756 | 0.0000 | 11.4503 to 11.4503 |

A single-pass trace at width 10000 ran out of GPU memory in both arms (the trace keeps full-vocabulary logits for every row), so the single-pass shape is covered by the CLI identity above and by the unit tests, which compare the maskless path with the dense mask at offset 0, mid-chunk, across a boundary and behind a trimmed front.
