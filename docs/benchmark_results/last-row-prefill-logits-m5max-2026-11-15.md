# Last-row prefill logits in every model (issue #3070), M5 Max, 2026-11-15

Issue #3070: the engine prefill samples one position, but a model without its own `forward_last_logits*` override ran the full forward and sliced one row, so every prompt position went through the LM head. `LanguageModel::last_row_logits` slices the hidden state at the sampled position before the final norm, the head and any softcap or scale, and every autoregressive model and VLM wrapper in `src/models` and `src/vision` implements it.

Host: Apple M5 Max, 128 GB, macOS 27. Main is `05312c60`; the branch is that commit plus this change; both release builds, each binary copied aside before the tree changed. 28K-token prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`, greedy, 48 tokens requested, reasoning included in the compared text. Peak is the `/usr/bin/time -l` footprint, prefill is `--profile`. Every run follows a 60 s idle period, and each cell ran main, branch, branch, main, so both arms have one early and one late sample. Raw rows: `last-row-prefill-logits-m5max-2026-11-15/runs.csv`; identity per cell: `output-identity.csv`.

## Results

| model | prefill | peak main | peak branch | saved | prefill main (rep 1 / rep 2) | prefill branch (rep 1 / rep 2) | greedy output |
|---|---|---|---|---|---|---|---|
| exaone-4.0-1.2b-4bit (28706 tokens) | 2048 chunks | 5.98 GB | **5.10 GB** | 0.88 GB | 5.35 / 3.92 s | 3.71 / 3.75 s | identical |
| exaone-4.0-1.2b-4bit | single pass | 15.70 GB | **9.77 GB** | 5.93 GB | 7.18 / 7.07 s | 6.50 / 6.60 s | identical |
| gemma-3n-e2b-it-4bit (28706 tokens) | 2048 chunks | 9.75 GB | **7.67 GB** | 2.08 GB | 8.10 / 7.86 s | 7.17 / 7.20 s | identical |
| gemma-3n-e2b-it-4bit | single pass | 38.17 GB | **23.09 GB** | 15.08 GB | 15.21 / 11.31 s | 10.51 / 11.32 s | identical |
| gpt-oss-20b-mxfp4-q4 (19773 tokens) | 2048 chunks | 15.88 GB | **13.79 GB** | 2.08 GB | 4.42 / 5.27 s | 4.30 / 4.93 s | identical |
| gpt-oss-20b-mxfp4-q4 | single pass | 30.61 GB | **22.55 GB** | 8.07 GB | 7.69 / 12.24 s | 8.19 / 6.71 s | identical |
| qwen2.5-vl-3b-instruct-4bit, text only (27679 tokens) | 2048 chunks | 6.55 GB | **5.00 GB** | 1.55 GB | 5.59 / 5.36 s | 5.13 / 5.08 s | identical |
| qwen2.5-vl-3b-instruct-4bit, text only | single pass | 25.13 GB | **16.74 GB** | 8.39 GB | 9.41 / 9.43 s | 9.01 / 9.16 s | identical |
| kimi-vl-a3b-thinking-4bit, text only (19723 tokens) | 2048 chunks | 14.43 GB | **12.84 GB** | 1.59 GB | 26.07 / 26.09 s | 25.81 / 25.76 s | identical |
| kimi-vl-a3b-thinking-4bit, text only | single pass | 32.04 GB | **25.57 GB** | 6.47 GB | 31.21 / 31.07 s | 30.84 / 30.98 s | identical |

Peaks are the larger of an arm's two runs; the two runs of an arm agree within 0.2 GB in every cell. The main column reproduces the issue's baselines (9.83 / 38.41, 5.94 / 15.65 and 15.76 / 30.96 GB) and the 14.43 / 32.03 GB Kimi-VL figures recorded after #3083.

The single-pass saving matches the full-logits tensor the issue estimated for each vocabulary (5.9 GB for EXAONE, 8.0 GB for gpt-oss, 15.0 GB for Gemma 3n). Gemma 3n's softcap copy was a second tensor of that size that did not add to the peak here, so the footprint drops by one copy, not two.

Greedy output is byte-identical in all ten cells (one distinct SHA-256 across the four runs of each cell). The change removes rows from the head projection and does not reorder any arithmetic on the sampled row, so no logit trace was needed. Gemma 3n stops after 5 tokens on this prompt on both arms.

Prefill time is lower or unchanged on the branch in every cell where the two reps agree: 8 to 11% on Gemma 3n at 2048 chunks, 3 to 9% on EXAONE and Qwen2.5-VL, about 1% on Kimi-VL, where attention dominates. The first run of the sweep (EXAONE main, rep 1) and gpt-oss single pass (7.69 and 12.24 s on main, 8.19 and 6.71 s on the branch) vary more between reps than between arms, so no time claim is made for them.

## Reproduce

```bash
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 28000 > prompt_28000.txt
MLXCEL_PREFILL_CHUNK=0 /usr/bin/time -l ./target/release/mlxcel generate \
  -m models/gemma-3n-e2b-it-4bit -p "$(cat prompt_28000.txt)" -n 48 --temp 0 --profile --show-reasoning
# Unset MLXCEL_PREFILL_CHUNK for the default 2048-token pieces.
```
