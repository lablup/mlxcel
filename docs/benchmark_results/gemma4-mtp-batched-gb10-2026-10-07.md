# Gemma 4 batched MTP on the row-wise geometries: exactness and the un-decline decision (GB10, 2026-10-07)

Issue #2190. PR #2185 made the B=1 linear Gemma 4 MTP adapter byte-identical to classic decode on CUDA; the batched adapter got none of that work, and serving declined every B>1 window for the 31B and the CUDA 12B (`mtp_requires_linear_singleton`). This record holds the exactness evidence for the batched adapter after #2190, the throughput measurement the issue asks for, and the resulting decision per pair and batch width.

## Host

NVIDIA GB10 (sm_121), driver 580.178.04, kernel 7.0.0-1019-nvidia, MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`, release profile, `--features cuda`, base `origin/main` `bbd05099`. Checkpoints under `/home/inureyes/models/mlx/`: `gemma-4-31b-it-4bit` + `gemma-4-31b-it-assistant-bf16`, `gemma-4-12b-it-4bit` + `gemma-4-12b-it-assistant-4bit`. Block width 4.

## Exactness

- `b1_batched_baseline_probe` (batched adapter at B=1 against the linear adapter, 4 prompts, 24 tokens): on `bbd05099` the 31B rows 1, 2 and 3 differ at tokens 1, 9 and 17. After #2190 all 4 rows are byte-identical on both the 31B and the 12B.
- `greedy_parity_mtp_gemma4_batched_matches_classic` (new): B=2 and B=4, equal-length (16 tokens) and ragged (7, 21, 13, 21 tokens) prompts, both pairs, 128 tokens per row, each row compared byte for byte with per-sequence classic greedy decode and no near-tie tolerance. 24 of 24 rows identical, and every case had a round in which live rows accepted different counts.
- Every timed batch below produced the same output bytes in every arm (classic open, MTP, classic close) of all three rounds.

## Throughput

Driver: `data/gemma4-mtp-batched-gb10-2026-10-07/harness/mtp_batched_rounds.py`, summarized by `harness/summarize.py`. The #2160 method (three interleaved rounds per width, each round classic, MTP, classic on fresh servers; the classic pair is the null arm; one discarded warm-up batch and two timed batches per arm; prompt cache off; 200 tokens) with B concurrent requests instead of one. Each row is the #1797 `prompt_retry.txt` behind a distinct opening line, so rows diverge. The classic arms run the branch binary. The MTP arm runs the same code with the `requires_singleton` decline removed, which is the change under decision, and sets `MLXCEL_ENABLE_MTP_BATCH=1` and `MLXCEL_ENABLE_MTP_BATCH_RAGGED=1`.

| 31B | Round | Classic open | MTP | Classic close | MTP vs classic | Null (close vs open) |
|---|---|---|---|---|---|---|
| B=2 | 0 | 8.78 | 15.34 | 8.79 | +74.6% | +0.1% |
| B=2 | 1 | 8.52 | 15.35 | 8.79 | +77.3% | +3.1% |
| B=2 | 2 | 8.80 | 15.43 | 8.83 | +75.1% | +0.3% |
| B=4 | 0 | 8.86 | 13.47 | 8.79 | +52.6% | -0.9% |
| B=4 | 1 | 8.85 | 13.55 | 8.65 | +54.8% | -2.3% |
| B=4 | 2 | 8.86 | 13.55 | 8.91 | +52.6% | +0.6% |

Aggregate tok/s: the rows' completion tokens over the time from the primer's completion (see below) to the last row's completion. Every timed MTP batch ran as one batched window (6 of 6 rows at B=2, 12 of 12 at B=4; the remaining B=1 bursts in each MTP log are the primers and the startup probe), and no arm hit `Cache thrashing` or an `NV_ERR_NO_MEMORY` increase.

Batched MTP beats classic by 75 to 77% at B=2 and 53 to 55% at B=4, against a null spread of -2.3% to +3.1%. Classic batched decode on this checkpoint barely scales with B (8.3 tok/s at B=1 in the #2160 record, 8.8 at B=2 and B=4), which is why a per-row verify that costs about B single-row verifies a round still wins. B>4 was not measured.

Host state, per arm in `run.jsonl`: the sustained-quiet gate of #1820 was replaced by one that waits for a minute without compiler or foreign model processes but ignores the CI runner's `Runner.Worker`, because the CI job queued on this host was itself waiting for the GPU lock these runs held. `ci_job_running` was true for 17 of 18 arms. One arm (B=4 round 0, classic close) recorded a clippy process starting right after its gate; its rate sits inside the null spread. `load1` was about 20 for the first seven arms with no compiler running and under 2 afterwards; the rates do not move with it.

### Harness deviations from #2160, and what they say about serving

A batched MTP window does not form in default serving, for three reasons found while setting this up. Each is pre-existing and none is changed here.

1. Two or more queued requests take the classic batched prefill (`execute_batched_prefill`) whenever `--max-batch-prefill` is above 1, the default, and that path never reaches the speculative window collector. The MTP arm runs `--max-batch-prefill 1`.
2. `--ignore-eos` suppresses EOS through each request's token bias, and the window collector (`sampling_config_eq`) refuses rows with a non-empty bias. No arm uses `--ignore-eos`; completions end at EOS or 200 tokens, and every row here reached 200.
3. A request without a seed draws a random one, and rows with different seeds cannot share a window even at temperature 0. Every request carries the same seed.

The collector also takes only rows already queued when it dequeues the head, and concurrent client threads do not land inside one tick. Every batch (warm-up and timed, every arm) therefore sends a short primer request first and the rows 0.25 s later; in the MTP arm the primer runs as a run-to-completion burst (`MLXCEL_MTP_TICK_SLICE=0`) and the rows dequeue behind it as one window.

### 12B

`gemma-4-12b-it-4bit` loads as `Gemma4Unified`, which does not support batched decode: the server clamps `--max-batch-size 2` to 1 (`clamp-12b-b2/server.b2.r0-mtp.log`), so neither classic nor MTP ever runs a B>1 window and the decline is moot. In that single-slot smoke run (64 tokens) classic served the two requests at 13.2 tok/s and B=1 MTP bursts at 16.1.

## Graph-cache budget

MLX keys its CUDA graph cache by topology and aborts the process once lifetime misses pass twice its capacity (#818). Per-row verify adds topologies that vary with each row's history length. The new parity test crossed that abort partway through its 12B half at MLX's bare default of 400; the shipped binaries raise it to 2000 at startup, and the test process now does the same (`tests/common::apply_server_mlx_cache_defaults`).

To see whether serving spends that budget faster, `harness/graph_cache_budget.py` started the undeclined server with `MLX_CUDA_GRAPH_CACHE_SIZE=100` (a 200-miss lifetime budget) and served 20 primer-plus-rows batches of 200 tokens each on the 31B (`graph-cache-31b/`). B=1 MTP served 4000 row tokens and B=2 batched MTP served 8000, and neither aborted. On this steady workload the per-row verify does not spend the budget at a rate that matters; the test process crossed it because it ran two models and dozens of prompt shapes in one process at MLX's bare default of 400.

## Decision

| Pair | B | Decision |
|---|---|---|
| 31B | 2 | Un-declined on CUDA, within the gates below |
| 31B | 4 | Un-declined on CUDA, within the gates below |
| 31B | >4 | Declined (not measured) |
| 31B, Metal | any | Declined (not measured; #2158) |
| 31B, ROCm | any | Declined (not measured) |
| 12B | any | Moot: the 12B serves single-slot, so no B>1 window forms |

`run_mtp_burst_batched` now declines a row-wise window to classic before any drafter IO unless all of these hold (`RowWiseBatchedWindow::decline_reason`): the backend is CUDA (not a ROCm build); B is at most 4; every cache is dense FP16 (the per-row prefill stacks caches); no prompt is longer than the sliding window; and `max_prompt_len + 4 * (max_tokens + 1)` stays within the sliding window plus the 32-token rollback buffer (1056). The last gate exists because the buffered sliding cache compacts against the shared offset: once it does, a row that lags that offset has lost the oldest keys of its own window, and the verify refuses the round rather than emit tokens classic decode would not. A row that already finished keeps advancing up to 4 positions a round, so the bound uses `4 * max_tokens`, not `max_tokens`. The measured configuration (prompts near 192 tokens, 200 tokens) is inside it; a 512-token generation is not.

The batched path itself stays behind `MLXCEL_ENABLE_MTP_BATCH` (default off), and the three serving conditions above still apply. Batched rows now carry their classic history-boundary splits into the per-row prefill, so the prompt cache does not move their prompt KV off classic's. The B=1 restrictions on tree rounds and buffered snapshot donation are unchanged: batched bursts run no tree rounds, and the measurement ran with the prompt cache off, so donation was not covered.

## Shipped binary

One more round at B=2 used the branch's own `mlxcel-server` for every arm (`shipped-31b-b2/`), with `MLXCEL_ENABLE_MTP_BATCH=1` and the serving conditions above for the MTP arm: classic 8.82 and 8.83 tok/s, batched MTP 15.27, all 6 timed and warm-up rows inside batched windows, and output bytes identical to classic and to the decision rounds.
