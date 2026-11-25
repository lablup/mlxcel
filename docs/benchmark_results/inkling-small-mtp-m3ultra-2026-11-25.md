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

## Classic-decode fallback (issue #3425)

### Profile of a round

`RUST_LOG=info` round-loop diagnostics and macOS `sample` on the main thread (Instruments could not attach on this host). A 2-row exact round costs 36 to 37 ms against a 21.7 ms engine decode step and emits 1.40 to 1.51 tokens, so its break-even acceptance at width 2 is about 0.70 (measured 0.35 to 0.51). Of the round, the verify forward is about 31 ms and the drafter step and accept hook the rest; the 1-row snapshot and replay after a partial accept was already gone (#3280). The main thread spends about 74% of a drafted session in `async_eval` (encode 32%, GPU wait 36%), 15% waiting in the token read and about 13% building the graph; classic decode spends 91% in `eval` (wait 45%, encode 39%).

The round cannot be made cheaper than 1.6 decode steps without breaking exactness: the row-invariant products and the grouped attention that make it byte-identical are what it costs, and acceptance is the head's (mlx-vlm measures the same). So the change is to stop drafting when drafting loses.

### Mechanism

The MTP round loop now reuses the DFlash width policy (`ExpectedThroughputWidth`, #3417) for first-generation Inkling. Each round it picks the width in `1..=block_size` with the most expected tokens per measured millisecond, width 1 being classic decode. When classic wins, the loop runs pipelined classic steps (step `i + 1` queued on the lazy token of step `i` before step `i` is read, at most 16 per call so a server tick returns as often as a round does) through `MtpTarget::classic_step`, which on Inkling is the target's single-token forward with its pre-norm hidden. The drafter does not run during classic steps; the next drafted round first feeds their tokens and hidden rows to `Drafter::extend_history` in one forward (14 to 19 ms per session here). The drafter is re-tried after 8 classic steps; each re-try that loses doubles the period up to 64, and one that wins resets it (DFlash keeps its fixed period of 8). The fallback needs the target to declare `MtpTarget::exact_at_every_verify_width` (only the Inkling MTP target does) and the drafter `Drafter::extends_history_without_draft` (only the Inkling MTP head does), the default `MLXCEL_MTP_BLOCK_CONTROLLER` and no per-token logprobs; `MLXCEL_MTP_BLOCK_CONTROLLER=requested` turns it off. No other family, and not Inkling 2 DSpark (DFlash round loop), changes.

### Before and after

128 greedy tokens, the prompts above, one build, one locked block, 2 ABAB cycles, load average 2.9 to 9.0. "Fixed" is the requested width every round (`MLXCEL_MTP_BLOCK_CONTROLLER=requested`, the behaviour before this change); "fallback" is the default. Every arm produced the same 128 tokens as classic decode on both prompts.

| Prompt | Arm | tok/s (2 runs) | Mean | Against classic | Rounds | Classic steps |
|---|---|---|---|---|---|---|
| prose | classic | 46.26, 46.10 | 46.18 | 1.00x | | |
| prose | width 2, fixed | 39.68, 39.53 | 39.61 | 0.86x | 89 | |
| prose | width 2, fallback | 41.41, 41.43 | 41.42 | 0.90x | 5 | 120 |
| prose | width 10, fixed (the old default) | 12.27, 12.55 | 12.41 | 0.27x | 79 | |
| prose | width 10, fallback (the default) | 37.93, 37.73 | 37.83 | 0.82x | 6 | 116 |
| code | classic | 46.07, 46.03 | 46.05 | 1.00x | | |
| code | width 2, fixed | 41.67, 41.59 | 41.63 | 0.90x | 84 | |
| code | width 2, fallback | 41.31, 41.01 | 41.16 | 0.89x | 5 | 120 |
| code | width 10, fixed (the old default) | 12.62, 12.55 | 12.59 | 0.27x | 78 | |
| code | width 10, fallback (the default) | 37.54, 37.26 | 37.40 | 0.81x | 5 | 120 |

The default width goes from 0.27x to 0.81x classic and width 2 on prose from 0.86x to 0.90x; on code width 2 the fallback is 1% behind the fixed width. At width 10 the first two rounds run at the ceiling (134 ms each, plus 185 ms of drafting) before the policy has measured a narrow round, which is most of the gap to width 2 over 128 tokens.

The fallback does not reach classic decode. Its classic steps cost 23.7 ms each (22.1 ms steady after the first read) against 21.7 ms in the engine, and a pure classic run inside the MTP session (drafting disabled after the first round, an uncommitted experiment) ran at 40.4 tok/s against 45.8 over 256 tokens. Host time dominates: building and queuing a step takes 19.7 ms of the 23.7 (`classic_build_ms` 2364 over 120 steps), against about 2 ms of waiting. Each of these left the gap unchanged (all uncommitted, measured on this pair): carrying a queued classic step across calls, the decode command-buffer budget, the batched decode forward the scheduler uses, a dedicated generation stream, `MLX_MAX_MB_PER_BUFFER`, `MLX_MAX_OPS_PER_BUFFER` 300 and 4000, `MLXCEL_CACHE_LIMIT`, `MLXCEL_QMV_WIDE=0`, evaluating only the token, running the exactness probe before classic decode, and loading and binding the drafter before classic decode (the last two keep engine classic at 45.6 to 45.8 tok/s). Issue #3430 found the cause; see the next section.

### Classic steps inside the session (issue #3430)

The classic step read each token back through `dflash::materialize_argmax_i32_vec`, which called `array_to_raw_bytes`: it builds `contiguous(token)` and evaluates it. That op, and the event the read waits on, are queued behind step `i + 1`, which is already scheduled on the same stream, so every read waited for the whole next forward and the pipeline ran as synchronous decode. The engine reads its tokens with `array_evaluated_bytes`, which waits on the token's own event and enqueues nothing (#632). Host timing per step (scratch instrumentation, 128 tokens, prose):

| Path | Graph build | `async_eval` | Token read wait | tok/s |
|---|---|---|---|---|
| engine classic decode | 1.5 ms | 20.0 ms | about 0 | 46.3 |
| engine classic, forced synchronous after each step | 1.1 ms | 16.3 ms | 6.7 ms (sync) | 41.6 |
| session classic step, before | 1.1 ms | 18.5 ms | 4.1 ms | 41.6 |
| session classic step, after | 1.1 ms | 20.6 ms | 0.27 ms | 45.0 |

The session's classic steps ran at the speed of the engine forced synchronous. Building the step from the engine's own forward with the speculative append scope, evaluating only the token, `MLXCEL_KV_INPLACE_WRITE=0` and runs of 16 or unlimited steps each left them at 41.4 to 41.9 tok/s. `materialize_argmax_i32_vec` now reads through `array_evaluated_bytes` (a non-contiguous array still takes the copy), and the DFlash classic steps (#3417) pass their token directly instead of a fresh `reshape`, which had the same effect.

Pure classic inside the session (drafting disabled, an uncommitted switch), one build with the old read behind a scratch switch, ABAB in one locked block per length, load average 7.5 to 12.5:

| Tokens | Prompt | Engine classic | Session, old read | Session, new read | Session, new read, no 16-step cap |
|---|---|---|---|---|---|
| 128 | prose | 46.48, 45.38 | 40.41, 41.58 | 44.49, 45.18 | 45.52, 45.69 |
| 128 | code | 46.53, 46.15 | 41.59, 41.84 | 45.37, 44.95 | 44.85, 45.34 |
| 256 | prose | 45.26, 45.65 | 41.38, 41.25 | 44.56, 44.63 | 45.00, 44.89 |
| 256 | code | 45.57, 45.46 | 41.23, 41.31 | 44.59, 44.82 | 44.89, 44.86 |

Every arm produced the same text as classic decode. The new read reaches 0.97x to 0.99x the engine; what remains is the step at the start of each call of the round loop, which has no queued predecessor (34 ms of read wait over 128 tokens at the 16-step cap, 4 ms without it), and the cap stays so a server tick returns as often as a drafted round.

MTP with the fallback, 128 tokens, same build, ABAB, load average 6.2 to 8.4:

| Prompt | Classic | Width 10 (default), before | Width 10, after | Width 2, before | Width 2, after |
|---|---|---|---|---|---|
| prose | 46.12, 46.05 | 37.01, 37.71 | 40.30, 40.44 | 41.04, 41.04 | 43.95, 44.11 |
| code | 45.86, 45.81 | 37.06, 37.28 | 39.66, 39.76 | 40.95, 41.16 | 43.77, 43.86 |

The default width goes from 0.81x to 0.87x classic and width 2 from 0.89x to 0.95x, with output identical to classic decode. A classic step costs 20.6 to 20.8 ms against 22.1 to 22.4 ms before. The default width's remaining gap is mostly its first two rounds at the ceiling (135 ms each).

## Acceptance against mlx-vlm

The drafter matches mlx-vlm's `inkling_mtp` (upstream 466069f5) in the points that decide acceptance: the seed token comes from block 0 over the accepted prefix, draft step `k` of a round uses block `min(k, 7)`, each block keeps its own KV and convolution caches, which are restored to the round's snapshot before the accepted prefix is fed back through block 0, the hidden state handed to the drafter is the target's pre-norm last hidden at the same position, and the logits use the target's final norm, `1 / logits_mup_width_multiplier` scale, LM head and 200058-token trim (upstream copies `model.llm.norm.weight` into the split drafter). The low acceptance is therefore shared with the reference, which measures 1.59 tokens per round on the same pair. Starting the draft steps after the seed at block 1 instead of block 0 (DeepSeek-style depth indexing) left the accepted drafts at 48 or 49 per 128 tokens on both prompts at widths 3 and 4, as before (an uncommitted experiment), so the chain indexing is not what holds acceptance down.

## Outcome

MTP does not help Inkling-Small on this host. Since #3425 the default width stops drafting when classic decode is faster, which takes it from 0.27x to 0.81x classic (0.90x at width 2), and since #3430 its classic steps keep the decode pipeline, which takes it to 0.87x (0.95x at width 2), but it stays below classic decode. It stays opt-in: it engages only when `--draft-model` (server `--model-draft`) names the head, and nothing selects it by default. When it is used, its temperature-0 output now equals classic decode at every width the gate admits. The adaptive verify width from #3417 does not apply: it belongs to the DFlash round loop (`SpeculativeTarget::exact_at_every_verify_width`), while Inkling MTP runs the MTP round loop and its own adaptive controller.
