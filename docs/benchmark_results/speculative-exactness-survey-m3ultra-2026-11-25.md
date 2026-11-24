# Speculative exactness survey across drafter-capable models (M3 Ultra, 2026-11-25)

Issue #3409. PR #3405 made an Inkling 2 verify block byte-identical to decode and found two causes the 8-token startup probe cannot see (a sliding window, a chunked prefill). This survey asks the same question of the other drafter-capable families: is a K-row verify block byte-identical to K single-token decode steps, at widths 2, 4 and 8, after a short prompt and after prompts past the sliding window and the 2048-token prefill chunk.

Host: Apple M3 Ultra, 512 GB, Metal. Every run is serialized under the shared GPU lock. Load average was 18 to 25 for the first 13 checkpoints (another build was running) and 2 to 3 for the rest; the verdicts are byte comparisons and do not depend on load.

## Method

`models::speculative_exactness_survey_tests` (ignored, `EXACTNESS_SURVEY_MODEL=<dir>`) loads one checkpoint and prints one verdict per row:

- **family**: the probe the model's own gate runs (`probe_block_chain_exactness`), first with the gate's own plan (`startup`), then one draw each at 600 and 2100 prompt tokens through `probe_block_chain_exactness_at`.
- **engine**: the forward the engine's token-only verify round runs (prompt lookup, a full-model drafter): the plain `LanguageModel` forward on external KV caches, a K-row block inside the speculative append scope against K single-token forwards, at 8, 600 and 2100 prompt tokens, prefilled in the classic 2048-token chunks.

A diverging row is measured again with `qmv_wide` off, which is the gate's own retry: `wide` is MLX's default quantized kernel selection, `narrow` the one the gate pins when that restores exactness. `equal` means every logit byte of every row matches. Divergences are given as the share of logit bytes that differ at block row 0.

The end-to-end check runs `mlxcel generate --temp 0 --max-tokens 128` with and without the drafter on a short prompt and on a long one (bytes 4000 to 14000 of `docs/supported-models.md` plus `Summarize the text above in three sentences.`, 2507 to 3627 tokens depending on the tokenizer) and compares the generated text.

## Families with an exactness gate

Widths 2, 4 and 8 gave the same verdict in every row, so one line stands for all three.

| Checkpoint | Family, dtype | Drafter path | Startup probe, wide | Startup probe, narrow | 600 tokens | 2100 tokens | Gate verdict |
|---|---|---|---|---|---|---|---|
| qwen3.5-4b-4bit | Qwen 3.5, affine 4-bit | DFlash, MTP | differs (40%) | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| qwen3.8-27b-4bit | Qwen 3.5, affine 4-bit | MTP | differs (29%) | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| Qwen3.8-Flash-Next-4bit-mtp | qwen4_exp, affine 4-bit group 32 | MTP, prompt lookup | differs (52%) | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| gemma-4-12b-it-4bit | Gemma 4 unified, affine 4-bit | MTP | differs (45%), layer 0 attention | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| gemma-4-31b-it-4bit | Gemma 4, affine 4-bit | MTP | differs (41%), layer 0 attention | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| gemma-4-26b-a4b-it-4bit | Gemma 4 MoE, affine 4-bit, 8-bit routers | MTP | differs (61%), layer 0 attention | differs (77%), layer 0 MLP | differs both | differs both | **declines** |
| glm-4.7-flash-4bit | GLM-4.7-Flash, affine 4-bit | MTP | differs (50%) | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| d1-3b-mlx-8bit | LFM2-VL text stack, affine 8-bit | DSpark | differs (49%) | equal | equal (narrow) | equal (narrow) | engages, pinned narrow |
| inkling-0.6b-a0.6b, before | Inkling, bf16 | MTP | differs (24%), layer 0 | differs (24%), layer 0 | differs both | differs both | **declined** |
| inkling-0.6b-a0.6b, after | Inkling, bf16 | MTP | equal | not needed | equal (wide) | equal (wide) | engages, wide |
| inkling-2-mini | Inkling 2, bf16 | DSpark | equal | not needed | see #3405 | see #3405 | engages (fixed in #3405) |

The only bf16 checkpoint with a gate besides Inkling 2 is Inkling 0.6B; every other gated family is available locally in quantized form only. `decider-2b` (the one bf16 Qwen 3.5 architecture on disk) and `clef-flash-mlx-8bit` are decision models, which `load_model` refuses, so they have no generation path to survey.

What the rows say:

- **No gated family has a long-context cause.** Every family that is exact after a short prompt is exact after 600 and 2100 tokens, which crosses the 1024-token Gemma 4 window, the 512-token Inkling window and the prefill chunk.
- **Quantized families are exact on the narrow kernel only.** The gate's `qmv_wide` retry is what makes them engage on this GPU generation, as #1199 found; the dense gemv/gemm cause of #3405 does not apply, because their projections are quantized and take `qmv`.
- **Gemma 4 26B A4B declines on both kernels.** Its own trace puts the narrow divergence at the layer 0 MLP output, row 0, which on this checkpoint is the MoE feed-forward over quantized weights. That is not a dense projection the row-invariant kernel takes, so `row_invariant::dense` does not apply and the checkpoint stays declined; which MoE product differs was not traced further.
- **Inkling was the one dense case**, and is fixed below.

End to end, drafter-on text equals classic text for every gated pair that has an offline round loop:

| Target | Drafter | Short prompt | Long prompt | Classic, drafter decode tok/s (long) |
|---|---|---|---|---|
| qwen3.8-27b-4bit | qwen3.8-27b-mtp-4bit | identical | identical (2601 tokens) | 35.47, 48.71 |
| Qwen3.8-Flash-Next-4bit-mtp | Qwen3.8-Flash-Next-MTP | identical | identical (2601 tokens) | 42.89, 41.81 |
| gemma-4-12b-it-4bit | gemma-4-12b-it-assistant-4bit | identical | identical (2714 tokens) | 62.50, 47.77 |
| gemma-4-31b-it-4bit | gemma-4-31b-it-assistant-bf16 | identical | identical (2714 tokens) | 26.25, 27.73 |

These are single runs, not an ABAB series; the tok/s columns are context, not a benchmark. So the chunked-prefill cause of #3405 does not show on these families. Not run end to end: Qwen 3.5 DFlash (the CLI has no offline DFlash round loop for that target; it runs in the server), GLM-4.7-Flash and LFM2 (no drafter checkpoint on disk), Inkling (the 0.6B checkpoint carries no `mtp.*` tensors).

## Models without a gate (prompt lookup, full-model drafter)

The engine's token-only verify round has no exactness gate and is documented as not byte-identical to plain decode (`docs/speculative-acceptance.md`). The survey confirms that for every model it takes: block row 0 differs at every width and prompt length, on both quantized kernels.

| Checkpoint | Family, dtype | 8 tokens (w2, w4, w8) | 600 tokens | 2100 tokens | Narrow kernel |
|---|---|---|---|---|---|
| gigachat3-10b-a1.8b-bf16 | DeepSeek V3, bf16 | 43%, 43%, 44% | 42%, 42%, 42% | 47%, 44%, 44% | same (no quantized product) |
| gigachat3-10b-a1.8b-mlx-8bit | DeepSeek V3, affine 8-bit | 52%, 51%, 52% | 49%, 49%, 44% | 51%, 53%, 53% | differs (47%) |
| Kimi-VL-A3B-Thinking-4bit | DeepSeek V3 text, 4-bit | 58%, 58%, 45% | 44%, 44%, 53% | 35%, 35%, 37% | differs (45%) |
| moonlight-16b-a3b-instruct-4bit | DeepSeek V3, 4-bit | 40%, 40%, 42% | 39%, 39%, 38% | 42%, 35%, 35% | differs (40%) |
| xing4.0-29b-a4b-mlx-8bit | xing4_0, affine 8-bit | 52%, 52%, 52% | 39%, 39%, 38% | 37%, 25%, 25% | differs (60%) |
| xing4.0-29b-a4b-optiq-4bit | xing4_0, mixed 4/8-bit | 48%, 51%, 51% | 26%, 26%, 24% | 41%, 43%, 43% | differs (53%) |
| glm-4.7-flash-4bit (engine path) | GLM-4.7-Flash, 4-bit | 2%, 5%, 5% | 24%, 24%, 35% | 30%, 36%, 36% | differs (44%) |
| afm-4.5b | arcee, bf16 | 45%, 45%, 44% | 36%, 36%, 35% | 40%, 40%, 43% | same |
| k2-horizon-0.9b | k2_horizon, bf16 | 50%, 50%, 52% | 29%, 29%, 29% | 44%, 44%, 49% | same |
| olmoe-1b-7b-0125-4bit | OLMoE, 4-bit | 44%, 44%, 44% | 68%, 68%, 68% | 44%, 44%, 44% | differs (37%) |

Not taken by this arm (sequence state outside the external KV caches): Qwen 3.5, Gemma 4, Inkling, Inkling 2, LFM2, `plamo-2-1b`, `limite-1b-violetto`, and qwen4_exp, whose prompt-lookup path rewinds its own state and is gated by the family probe above.

The per-layer diagnostic (`EXACTNESS_SURVEY_DIAG=1`) on the three DeepSeek V3 checkpoints at width 2 puts the first divergence at the **layer 0 output, row 0** (885 of 3072 bytes on bf16, 940 of 3072 on 8-bit, 1249 of 4096 on Kimi-VL). On that family a single query always takes the absorbed MLA path and a block the unabsorbed one (`use_absorbed_mla`), so the two arms differ before any dense product is reached; GLM's gated MTP path avoids this with its own row-wise verify, which is why the same checkpoint is equal through `family` and differs through `engine`.

`row_invariant` is not adopted for these models. They have no gate for it to satisfy, their divergence starts in attention paths the dense helpers do not cover, and the text result does not call for it: with `--prompt-lookup`, the output equalled classic decode on gigachat3 bf16 and Kimi-VL (short and long prompt) and on xing4.0 8-bit (short); it differed once, on xing4.0 8-bit after the 2816-token prompt, which the documented contract allows.

## Inkling: row-invariant verify block

The diagnostic placed Inkling's first divergence at the layer 0 output, row 0, on both quantized kernels (the checkpoint has no quantized product). The causes are the Inkling 2 ones plus one of its own:

| Cause | Fix |
|---|---|
| gemv for one row, gemm for several: attention q/k/v/r/o, dense MLP, MoE gate and shared experts, LM head | `row_invariant::linear` and `dense` inside a model scope |
| The short convolutions multiply a window by the taps at one token and call `conv1d` for several | inside a verify block, the one-token form per position |
| The routed experts switch kernel with the token count (fused single-token, down-combine up to 8, `gather_mm`) | `row_invariant::per_position` over the MoE |
| Past the sliding window, block row r also sees the r oldest keys, masked | per-row attention from each row's window start |

After the change the probe passes at widths 2, 4 and 8 at startup (which now includes a 544-token draw past the 512-token window), at 600 and at 2100 tokens. `MLXCEL_INKLING_ROW_INVARIANT=0` restores the old products and the old verdict.

Decode cost, since the kernel also serves the one-row decode step: `mlxcel generate --temp 0 --profile`, one locked block, ABAB, 8 pairs, load average 5.7 to 6.5. The checkpoint ends its reply at 47 tokens whatever the prompt, so each sample is 47 decode steps.

| Arm | Decode tok/s (8 runs) | Mean |
|---|---|---|
| row-invariant kernel (default) | 248.33, 236.91, 245.61, 235.42, 245.64, 246.97, 247.61, 248.68 | 244.40 |
| MLX `matmul` (`MLXCEL_INKLING_ROW_INVARIANT=0`) | 239.18, 238.76, 239.27, 236.84, 239.27, 241.63, 240.79, 238.90 | 239.33 |

The kernel arm is 2.1% faster on the mean and faster in 6 of 8 adjacent pairs; the other two pairs are within 1%. No regression, so the change is adopted.

Not measured: MTP throughput on Inkling. The only published drafter pairs with Inkling-Small, whose checkpoints this run did not load, and the 0.6B checkpoint has no MTP head. Inside a verify block the MoE now runs once per position and a sliding layer past its window attends once per row, so a verify round costs more than the inexact one did; #1549 measured MTP on Inkling-Small as slower than classic decode even then. Whether the quantized Inkling-Small checkpoints now pass the probe is also not measured: their quantized products take `row_invariant::per_row`, which is exact by construction, but the NVFP4 and affine expert kernels were not exercised here.

## Long-context probe draw

The shared plan (`models::speculative_exactness_draws`) appends one draw of `sliding_window + 32` prompt tokens for the families with a sliding layer: Inkling (new), Laguna and Muse Glimmer (new, not measured on a real checkpoint: none is on disk), Gemma 4 (unchanged, `min(window + 32, 1536)` on the geometries it already covered). Families without a sliding layer run the draws they ran before. `MLXCEL_MTP_PROBE_LONG_CONTEXT=<tokens>` forces a long draw on every family and `0` drops it.

## Skipped

- `deepseek-v4-flash-4bit` (141 GB): optional in the issue, and its family (`deepseek_v4`) goes through the same ungated token-only round as the DeepSeek V3 rows.
- `hy4-preview-4bit`, `deepseek-v3.2-4bit`, `inkling-small-*`, `inkling-2-small`: excluded by the run's memory rules.
- Laguna and Muse Glimmer: no checkpoint on disk.
- LFM2 text checkpoints with a DSpark drafter: none on disk; the LFM2 text stack was probed through `d1-3b-mlx-8bit`.
