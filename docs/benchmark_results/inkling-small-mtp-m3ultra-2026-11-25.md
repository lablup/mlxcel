# Inkling-Small native MTP: exactness and throughput (M3 Ultra, 2026-11-25)

Issue #3280, first-generation Inkling. Target `models/inkling-small-mlx-4bit` (`mlx-community/Inkling-Small-mlx-4bit`, affine 4-bit experts, bf16 dense projections), drafter the native bf16 `mtp.safetensors` head (8 MTP layers, 4.46 GB). The head on disk is the one shipped in `thinkingmachines/Inkling-Small-NVFP4`; a directory holding only its `config.json` and `mtp.safetensors` is passed as `--draft-model <dir> --draft-kind mtp`, the layout `mlxcel download thinkingmachines/Inkling-Small --include config.json mtp.safetensors` produces. Host: Apple M3 Ultra, 512 GB. Every run is serialized under the shared GPU lock; load average 3.6 to 9.1.

## Exactness

The block-versus-chain probe (`InklingModel::mtp_exactness_allows`, run through `models::speculative_exactness_survey_tests::survey_block_vs_chain`, short draws plus 600- and 2100-token prompts):

| Width | Before | After |
|---|---|---|
| 2, 3, 4, 8 | equal | equal |
| 10 (the MTP default, `num_nextn_predict_layers + 2`) | declines: 150040 of 400116 logit bytes differ at verify position 0; first divergent stage layer 0 (sliding) output at row 3 on short draws, layer 5 (global) after 600 and 2100 tokens; `qmv_wide` off does not help | equal |
| 16 | not run | equal |

The row-invariant products from #3415 already made widths up to 8 exact on this quantized target (the experts take one position at a time, as on the bf16 0.6B checkpoint). Past 8 queries MLX's fused SDPA leaves the per-query vector kernel a decode step runs for its tiled kernel, which orders the key reduction differently, so rows from the fourth on parted from the chain. Inside a verify block the attention now runs in groups of at most 8 queries.

A second inexactness sat outside the probe. After a partial accept the MTP adapter restored a pre-verify snapshot and replayed the kept prefix as one forward outside the verify scope: a 2-row or wider replay ran `conv1d` and the batched expert kernels, not the single-token step's, so the state classic decode would hold was not restored. With the old code at width 3 the prose stream left classic decode. The replay is gone: the verify block records each convolution's input window and a partial accept trims the KV tail and takes the convolution states from those windows, which equals the chain after the kept rows. This also removes a second target forward from every round.

## Throughput

128 greedy tokens, `--show-reasoning`, chat template. Prompt 1 (prose): `Explain why the sky is blue.` Prompt 2 (code): `Write a Python function that returns the n-th Fibonacci number, with a short docstring.` Two rounds in ABAB order, both values listed. Width 10 is the adaptive MTP controller with a ceiling of 10 (it ranged over blocks 2 to 10). "Before" is the code before this change in the same binary; its width 10 needs `MLXCEL_MTP_ALLOW_INEXACT=1`. "Mean accepted" is tokens emitted per round (accepted drafts plus the bonus).

| Prompt | Mode | Before tok/s | Before mean accepted | After tok/s | After mean accepted | After output vs classic |
|---|---|---|---|---|---|---|
| prose | classic | | | 45.18, 45.01 | | |
| prose | width 2 | 25.57, 26.15 | 1.443 | 39.82, 39.53 | 1.427 | identical |
| prose | width 3 | 23.30, 23.94 (differs from classic) | 1.587 | 34.53, 34.61 | 1.608 | identical |
| prose | width 10 | 10.35, 10.38 (differs) | 1.587 | 12.71, 12.73 | 1.608 | identical |
| code | classic | | | 44.41, 44.39 | | |
| code | width 2 | 27.06, 26.98 | 1.530 | 41.54, 41.68 | 1.512 | identical |
| code | width 3 | 24.41, 24.42 | 1.649 | 34.85, 34.99 | 1.628 | identical |
| code | width 10 | 10.50, 10.62 (differs) | 1.649 | 12.71, 12.77 | 1.628 | identical |

Dropping the replay makes width 2 about 1.5x faster on both prompts, and every width now reproduces classic decode byte for byte, but no width beats classic decode: the best is width 2 at 0.88x (prose) and 0.94x (code). A width-2 round costs about 36 ms against a 22 ms decode step and emits 1.43 to 1.51 tokens. Acceptance, not the verify cost, is the limit: almost every accepted draft is the first one (widths 3 and 10 add 0.1 to 0.2 tokens per round over width 2 while drafting up to 8 more), so a wider block costs more than it returns.

## Acceptance against mlx-vlm

The drafter matches mlx-vlm's `inkling_mtp` (upstream 466069f5) in the points that decide acceptance: the seed token comes from block 0 over the accepted prefix, draft step `k` of a round uses block `min(k, 7)`, each block keeps its own KV and convolution caches, which are restored to the round's snapshot before the accepted prefix is fed back through block 0, the hidden state handed to the drafter is the target's pre-norm last hidden at the same position, and the logits use the target's final norm, `1 / logits_mup_width_multiplier` scale, LM head and 200058-token trim (upstream copies `model.llm.norm.weight` into the split drafter). The low acceptance is therefore shared with the reference, which measures 1.59 tokens per round on the same pair. Starting the draft steps after the seed at block 1 instead of block 0 (DeepSeek-style depth indexing) left the accepted drafts at 48 or 49 per 128 tokens on both prompts at widths 3 and 4, as before (an uncommitted experiment), so the chain indexing is not what holds acceptance down.

## Outcome

MTP does not help Inkling-Small on this host. It stays opt-in: it engages only when `--draft-model` (server `--model-draft`) names the head, and nothing selects it by default. When it is used, its temperature-0 output now equals classic decode at every width the gate admits. The adaptive verify width from #3417 does not apply: it belongs to the DFlash round loop (`SpeculativeTarget::exact_at_every_verify_width`), while Inkling MTP runs the MTP round loop and its own adaptive controller.
