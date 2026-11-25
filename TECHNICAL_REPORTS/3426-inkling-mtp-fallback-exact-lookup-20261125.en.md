# PR #3426: classic-decode fallback for Inkling MTP, exact prompt lookup throughput, Inkling server lookup tests

**Date**: 2026-11-25

**Status**: Draft PR open (closes #3425). Validated on an M3 Ultra (512 GB). The narrow lib suites, the ignored server e2e test, clippy and the contract checks pass. One acceptance criterion of item 3 is unmet.

## Executive Summary

Issue #3425 bundled three follow-ups to the speculative exactness work.

- Item 1 measures prompt lookup with `MLXCEL_MLA_ROW_INVARIANT=1` on the four families #3421 added.
- Item 2 tests server prompt lookup on first-generation Inkling.
- Item 3 asks for Inkling-Small MTP to be no slower than classic decode.

Items 1 and 2 are complete. Item 3 adds an acceptance-aware fallback to classic decode, which takes the default MTP width from 0.27x to 0.81x classic and width 2 to 0.90x. It does not reach the 0.97x the issue asks for.

## 1. Exact prompt lookup throughput

The switch is free when acceptance is low and costs the most where a per-row block replaces a batched kernel. On a copy-heavy reply it costs OLMoE 34% (464.6 against 702.3 tok/s), DeepSeek V2 12%, Arcee 9% and K2 Horizon 5%. Even so, exact lookup stays 1.09x to 3.23x faster than plain decode on that prompt. On prose, acceptance is under 0.1, and lookup is slower than plain decode on DeepSeek V2 (0.84x to 0.86x) and OLMoE (0.85x) whether the switch is on or off. Text with the switch on and `MLXCEL_QMV_WIDE=0` equals plain decode in all eight checkpoint and prompt pairs. Section 4 of `docs/benchmark_results/speculative-exactness-followup-m3ultra-2026-11-25.md` has the full table.

## 2. Inkling server prompt lookup

`tests/pipeline_server_real_models/inkling.rs` starts a plain server and a prompt-lookup server (`--parallel 2`, `--prompt-lookup-max-batch 2`). It sends completion and chat requests, streamed and not, with a copy prompt and a prose prompt, and requires the same text from both servers. It then sends a concurrent pair. The test requires at least one copy reply to report `timings.draft_n > 0`, so it fails if the verify rounds never run.

This checkpoint answers most prompts with one memorized passage. The copy prompt therefore holds that passage followed by its opening again. The first prompts chosen produced replies of 1 to 9 tokens and drew no proposal, which the assertion caught. A tiny-model scheduler test covers a row alone and two rows in one batch. Neither test found a defect.

## 3. Inkling-Small MTP

**Profile.**

- An exact 2-row round costs 36 to 37 ms, while a decode step costs 21.7 ms. The round needs about 0.70 acceptance at width 2 to pay off, and it gets 0.35 to 0.51.
- The row-invariant products and the grouped attention that make the round exact are what it costs.
- Acceptance matches mlx-vlm's, so it belongs to the head and not to the port.

**Mechanism.** The fallback stops drafting when drafting loses.

- **Width choice.** `classic_fallback.rs` reuses `ExpectedThroughputWidth` in the MTP round loop. Width 1 runs pipelined classic steps through the new `MtpTarget::classic_step`: the target's single-token forward, returning its pre-norm hidden.
- **Drafter catch-up.** The drafter skips the classic steps. Before its next round, `Drafter::extend_history` feeds it the skipped positions in one forward.
- **Retry period.** The drafter is retried after 8 classic steps. A retry that loses doubles the period, up to 64, and a retry that wins resets it. `with_probe_backoff` adds this backoff, and the DFlash loop keeps its fixed period.
- **Opt-in.** Only the Inkling MTP target and head opt in, through `exact_at_every_verify_width` and `extends_history_without_draft`.
- **One-row blocks.** A one-row Inkling verify block is now finalized as a classic step, with no snapshot and replay.

**Result.** 128 tokens. Output equals classic decode on every arm.

| Prompt | Classic | Width 10, fixed | Width 10, default | Width 2, fixed | Width 2, default |
|---|---|---|---|---|---|
| prose | 46.18 | 12.41 | 37.83 | 39.61 | 41.42 |
| code | 46.05 | 12.59 | 37.40 | 41.63 | 41.16 |

**Gap.** Classic steps inside the MTP session cost 23.7 ms, against 21.7 ms in the engine. A pure classic run inside the session reaches 40.4 tok/s, against 45.8 for the engine. Eleven variations left the gap unchanged:

- carrying a queued step across calls
- the decode command-buffer budget
- the batched decode forward
- a dedicated stream
- the MLX buffer limits
- the cache limit
- the narrow kernel
- token-only evaluation
- running the exactness probe first
- loading the drafter first

The cause of the gap is the open lead for a follow-up.

## 4. Risks

- The fallback changes nothing unless a target and a drafter both opt in. Only first-generation Inkling MTP opts in.
- At width 10, the first two rounds run at the ceiling before the policy has measured a narrow round. That costs about 450 ms per session.
- Only the Inkling MTP pairing changes. The DFlash default of `ExpectedThroughputWidth` keeps its fixed period, and a test pins it.
