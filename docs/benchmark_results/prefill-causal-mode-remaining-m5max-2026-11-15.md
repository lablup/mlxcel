# Causal-mode and native-window prefill for the remaining model-built masks (issue #3072), M5 Max, 2026-11-15

Issue #3072 extends #3051's `mlxcel_core::utils::prefill_causal_mask` to the models that still built a dense f32 `[L, L + offset]` causal or sliding mask once per forward: the plain-causal hybrid and full-attention decoders, the GLM4V-family decoders, the global layers of the sliding-window families, and the tensor-parallel Qwen 3.5 runtime. The sliding layers of those families go through the new `prefill_sliding_window_mask`, which leaves the band to MLX's causal mode while every key is inside the window and to MLX's native `window_size` entry once the keys outgrow it, wherever MLX's own verdict says the call fuses; elsewhere the model keeps its one shared array, as before. Padded, batched, caller-masked and speculative-verify forwards keep the array.

Host: Apple M5 Max, 128 GB, macOS 27. Two release builds (`metal,accelerate`): `main` at `8a5651ff` and the branch on top of it. Generation is greedy, 48 tokens, reasoning included in the comparison. Prompts come from `scripts/bench/metal_sdpa_chunk/make_prompt.py` (28000 words, or 7500 for `c4ai-command-r7b-12-2024-4bit`, whose context is 8192). The CLI cells ran through `scripts/bench/prefill_peak/ab_prefill.sh` with a 60 s cooldown before every run and the arm order alternated between repetitions (main first on repetition 1, branch first on repetition 2), one GPU lock hold per model. The host was shared with other jobs (1-minute load 1.2 to 8.7, recorded per row in `ab-summary.csv`); the one outlier is the branch's first single-pass run of glm-4.1v-9b-thinking-4bit (21.3 s at load 1.75, against 17.46 s on its second run).

## Results

Means of two repetitions per arm (`ab-summary.csv`). Peak is the `/usr/bin/time -l` footprint; prefill is from `generate --profile`.

| Group | Model | Prompt tokens | Path | Peak, main to branch | Prefill, main to branch |
|---|---|---|---|---|---|
| hybrid (LFM2) | lfm2-8b-a1b-4bit | 19717 | single pass | 13.50 to **11.63 GB** | 4.06 to **2.06 s** |
| hybrid (LFM2) | lfm2-8b-a1b-4bit | 19717 | 2048-token chunks | 8.37 to 8.25 GB | 2.40 to 2.22 s |
| full attention | llama-4-scout-17b-16e-instruct-4bit | 19717 | single pass | 72.88 to 73.00 GB | 27.50 to 26.07 s |
| full attention | llama-4-scout-17b-16e-instruct-4bit | 19717 | 2048-token chunks | 66.85 to 66.79 GB | 19.64 to 19.41 s |
| global + local | exaone-4.0-1.2b-4bit | 28706 | single pass | 9.82 to **4.05 GB** | 6.07 to **2.87 s** |
| global + local | exaone-4.0-1.2b-4bit | 28706 | 2048-token chunks | 5.15 to 4.72 GB | 3.75 to 3.08 s |
| sliding (Cohere2) | c4ai-command-r7b-12-2024-4bit | 7508 | single pass | 8.57 to 7.99 GB | 2.30 to 2.06 s |
| sliding (Cohere2) | c4ai-command-r7b-12-2024-4bit | 7508 | 2048-token chunks | 7.46 to 7.31 GB | 2.09 to 2.06 s |
| GLM4V, text only | glm-4.1v-9b-thinking-4bit | 22755 | single pass | 18.10 to 18.08 GB | 17.40 to 19.38 s (17.40 to 17.46 s without the outlier) |
| GLM4V, text only | glm-4.1v-9b-thinking-4bit | 22755 | 2048-token chunks | 8.91 to 8.40 GB | 11.94 to 11.16 s |
| sliding + global (Gemma 3n) | gemma-3n-e2b-it-4bit | 28706 | single pass | 23.05 to **13.05 GB** | 10.48 to **4.53 s** |
| sliding + global (Gemma 3n) | gemma-3n-e2b-it-4bit | 28706 | 2048-token chunks | 7.65 to 6.82 GB | 7.19 to 4.74 s |

Generated text is byte-identical between main and the branch in every cell.

exaone-4.0-1.2b-4bit declares no sliding window, so all of its layers are global; c4ai-command-r7b-12-2024-4bit (window 4096) and gemma-3n-e2b-it-4bit (window 512) exercise the native window route past the window. Llama 4 keeps its chunked-attention mask on the MoE layers, so only the global layers change there and its peak is set by the chunked mask; its single-pass peak moves by +0.12 GB (0.17%) while prefill drops 5%. On glm-4.1v-9b-thinking-4bit the single-pass peak is not set at a point where the mask is live in either arm, so it does not move.

Gemma 2's softcapped attention now shares one causal mask per forward instead of `causal_attention` building it in every layer. The masked softcap SDPA it feeds is the same, so the outputs are identical by construction (`a_softcapped_prefill_shares_the_mask_causal_attention_would_build` checks it bit for bit), and on gemma-2-2b-it-4bit at 7515 tokens (context 8192) peak and prefill do not move: single pass 7.32 GB and 1.26 s in both arms after the first repetition (the main arm's first run took 1.80 s), 2048-token chunks 5.22 GB and 1.04 s. The single pass's peak is the softcap composite's chunked scores (26 gated calls in both arms), not the per-layer masks.

## Arithmetic

Causal mode and the native window entry replace an additive array mask inside the same MLX SDPA, so the change can move the numbers. The teacher-forced trace (`examples/logit_trace`, `scripts/compare_logit_traces.py`) compares the branch with `MLXCEL_PREFILL_CAUSAL_MODE=0` (the array masks, which is main's dispatch for these models) against the branch default, over three 2048-token forwards behind 2048 tokens of context: the width of a default prefill piece, and past the sliding window of the Cohere2 and Gemma 3n checkpoints. `logit-trace-summary.csv` has the per-model rows.

| Model | Positions | Top-1 disagreement | On decided positions (gap >= 2) | Max logit delta on the reference choice | Perplexity, array to causal |
|---|---|---|---|---|---|
| lfm2-8b-a1b-4bit | 6144 | 0 | 0 / 5173 | 0.0000 | 2.0379 to 2.0379 |
| llama-4-scout-17b-16e-instruct-4bit | 6144 | 0 | 0 / 942 | 0.0000 | 13.1888 to 13.1888 |
| exaone-4.0-1.2b-4bit | 6144 | 0 | 0 / 4582 | 0.0000 | 2.0149 to 2.0149 |
| c4ai-command-r7b-12-2024-4bit | 6144 | 0 | 0 / 5845 | 0.0000 | 1.1143 to 1.1143 |
| glm-4.1v-9b-thinking-4bit | 6144 | 0 | 0 / 5931 | 0.0000 | 1.1320 to 1.1320 |
| gemma-3n-e2b-it-4bit | 6144 | 0 | 0 / 5078 | 0.0000 | 1.4757 to 1.4757 |

Every position agrees, with no logit delta at the reported precision: on this hardware MLX's causal and window modes reproduce the array-masked kernels' results for these shapes.

## Maskless padded prefill

Cohere2, OLMo 3, MiniCPM3, Gemma 3n and Mistral 4 now answer `supports_maskless_padded_prefill`. `tests/maskless_padded_prefill_parity.rs` prefills a 40-token prompt unpadded, zero-padded to 64 with no mask, and padded with `create_padded_prefill_mask`, and compares the last real row. On c4ai-command-r7b-12-2024-4bit, olmo-3.1-32b-think-4bit, minicpm3-4b-4bit, gemma-3n-e2b-it-4bit and mistral-small-4-119b-2603-4bit both padded arms match the unpadded logits exactly (relative RMS 0). The Mistral 4 gate runs only with `MLXCEL_TEST_MISTRAL4=1`, since its checkpoint is a 119B MoE.
