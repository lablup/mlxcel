# Technical Report: PR #2092 - Gate CUDA prompt-lookup drafting on shadow probes

**Date**: 2026-10-02

**Status**: Implemented and measured on GB10; pending merge. Issue #2091 stays open: one of its eight write and story targets (Qwen3-8B email, 0.97x against 0.98x) is not met, and Apple Silicon was not measured.

**Languages**: Rust (prompt-lookup governor and decode loop, CLI flag, tests, one example), Python (benchmark harness), Markdown and JSON (benchmark record)

**Risk Level**: Low. Only CUDA builds change behavior; the Metal and ROCm path makes exactly PR #2074's decisions. Output distribution is unchanged on every path (the acceptance rule is untouched); greedy output can differ from plain decoding only at near-ties, as before.

## Executive Summary

Prompt lookup (`mlxcel generate --prompt-lookup`, PR #2074) slowed prose 3 to 21% on GB10 because its governor, tuned on an M4 Pro, kept probing prose with verify blocks that are expensive on CUDA. This PR adds a second drafting policy, `DraftPolicy::Gated`, made the default on CUDA builds: it uses only a narrow or a full block, and it never spends a verify forward to find out whether a copy has resumed; while paused it checks the lookup's proposal against the tokens decoding emits next ("shadow probe"). On GB10, stories go from 0.83x-0.93x to 0.98x-1.00x of plain decoding, edit and summary gains are kept or raised, and every reply that stayed byte-identical to plain decoding still does.

## 1. Problem Statement

PR #2074's governor (now `DraftPolicy::Graded`) sizes the block as twice the recent accepted average plus two, up to 7 proposals (verify width 8), and after three drafted rounds land nothing it pauses 4 rounds, doubling to 32, then probes again with a drafted round. Each drafted round in the loop pays three costs: the verify forward itself, a drained pipeline (a proposal found while a plain step is in flight waits for that step to be read), and two synchronous plain rounds before pipelining resumes. On prose, where proposals rarely land, the probes are nearly pure loss.

How much loss depends on the backend, which the original tuning could not see. `examples/verify_width_cost.rs`, added here, measures a synchronous verify forward in pipelined one-token steps. On GB10 (release build, MLX pin `81ba1c6a`):

| Width | 2 | 3 | 4 | 7 | 8 |
|---|---|---|---|---|---|
| Qwen3-1.7B | 1.37 | 1.76 | 2.43 | 3.84 | 3.91 |
| Qwen3-8B | 1.15 | 1.59 | 2.10 | 3.85 | 3.30 |

Below 8 rows the affine path runs `qmv`'s multirow kernel, which `dispatch_multirow_width` instantiates at 2, 4 and 8 accumulator rows, so 5 to 7 rows pay for the 8-row instantiation; from 8 rows `quantized.cpp` switches to `qmm_sm80`. A block between narrow and full buys a few more proposals for nearly a full block's price, and on the 8B models for more than it.

## 2. Change Summary

- `DraftPolicy { Graded, Gated }` (`#[non_exhaustive]`) on `PromptLookupConfig`; `DraftPolicy::default_for_backend()` picks `Gated` when the crate is built with `cuda`, `Graded` otherwise.
- `DraftGovernor` keeps the Graded arithmetic byte-for-byte and adds the Gated state machine: `drafting`, `probation`, the shared accepted-proposal EMA, narrow (`GATED_NARROW_DRAFT` = 2) or full (`max_draft`) budget, `GATED_FULL_AT` = 1.5, `GATED_START_EMA` = 1.0.
- `ShadowProbe { start, proposal }` and `settle(&context) -> Option<bool>`; the decode loop settles the pending probe at the top of each round, looks a new one up on paused rounds (two tokens, `SHADOW_CONFIRM`), and calls `governor.shadow_confirmed()` when it comes true. A paused Gated loop pipelines its plain rounds at once.
- `PromptLookupStats::shadow_confirmations`, printed on the `[Prompt lookup]` line and in the end-of-decode trace.
- CLI: `--prompt-lookup-policy auto|graded|gated` (requires `--prompt-lookup`), `PromptLookupPolicyArg::resolve`, `mlxcel::DraftPolicy` re-export.
- Tests: Gated start, probation, timer-free pause, widths, max-draft below the narrow block, shadow settling, policy defaults, CLI parsing; the rollback parity matrix runs per policy; an exact script-model test pins probe alignment.
- `docs/benchmark_results/prompt-lookup-governor-gb10-2026-10-02.md` with the harness, prompts, three matrices, the load log and the width costs; `docs/benchmarks.md` links it.

## 3. How Gated Decides

1. A reply starts drafting, narrow, on probation. The first drafted round of a reply has no evidence behind it.
2. A drafted round updates the EMA. When the EMA reaches 1.5, the block is full; a narrow block that lands whole from the starting EMA gets there in one round.
3. Probation ends only on a round that lands a whole narrow block. A round that lands nothing while on probation pauses at once; off probation, three misses in a row pause.
4. While paused, `budget()` returns 0 indefinitely. Each round, if no probe is pending, the loop looks up a two-token proposal at the current context length and records it. When decoding has emitted those two positions, the probe settles: both came true, so drafting resumes (narrow, on probation, EMA reset); otherwise it is dropped and a new one is made.
5. Because nothing can be proposed until a probe settles, which takes at least two emitted tokens, paused rounds pipeline immediately rather than after two synchronous ones.

The probe's `start` is the context length when the lookup ran. In the pipelined path the context ends at `current_token` while `next` is in flight, so `proposal[0]` claims `next`; in the synchronous path it claims the token the round's forward will produce. Both line up with the next emitted token, which the script-model test pins: a probe recorded one position early or late never confirms there.

## 4. Measured Results

Final matrix on `34c627d7` (the shipped tree), greedy, `-n 400` (story `-n 800`), whole-call tok/s including prefill, median of 3, plain/graded/gated interleaved per repetition; load average median 3.1 on a shared host.

| Model | edit_fn | edit_json | summary | write | story |
|---|---|---|---|---|---|
| Qwen3-1.7B graded / gated | 1.16 / 1.22 | 1.39 / 1.41 | 0.82 / 0.99 | 1.07 / 1.02 | 0.83 / 0.98 |
| Qwen3-4B graded / gated | 1.27 / 1.39 | 1.71 / 1.64 | 1.38 / 1.42 | 0.94 / 0.98 | 0.87 / 0.98 |
| Qwen3-8B graded / gated | 1.41 / 1.51 | 1.90 / 1.91 | 1.59 / 1.62 | 0.98 / 0.97 | 0.91 / 0.98 |
| Llama-3.1-8B graded / gated | 1.54 / 1.66 | 2.02 / 1.97 | 0.94 / 1.02 | 0.96 / 0.99 | 0.93 / 1.00 |

Drafted rounds on stories fall from 30-70 to 12-18; shadow probes resume drafting 7 to 10 times per story. Edit rows often rise because a narrow block lands more tokens per unit of verify cost on GB10 than a full one. The largest edit/summary drop against Graded is Qwen3-4B edit_json (4%).

## 5. Technical Decisions

**Per-backend default instead of one retuned governor.** Graded was tuned on an M4 Pro, where a wide block is cheap; Gated was measured on GB10. No Apple Silicon machine was reachable, so changing Metal's behavior would have been unmeasured. Keeping Graded's decisions exact on Metal makes "no Metal regression" true by construction, and `--prompt-lookup-policy` lets anyone measure Gated there. The cost is two policies to maintain.

**Shadow probes instead of timed probes.** Graded learns whether a copy resumed by paying a verify forward; Gated learns it from a hash lookup and the tokens decoding was going to emit anyway. Requiring two confirmed tokens (not one) matters on prose, where the token after a two-token match is often a common one that recurs.

**Governor only; the round loop's verify path untouched.** The issue listed a pipeline-preserving verify (start the verify from the in-flight token) as its first direction. It changes where verify blocks start, and with that which near-ties flip, so it would put the greedy-parity criterion at risk for a measured gap of about one point on one row. The issue's own decision rule prefers the smaller change to the round loop.

**Rejected after measurement: a higher full-block threshold.** Requiring two clean narrow blocks before a full one (`GATED_FULL_AT` 1.75) aimed at the Qwen3-8B email, where a short restated fragment of the prompt widened the block just before the fragment ended. The third matrix showed the row unchanged at 0.97x and two email rows newly diverging from plain decoding, so it was reverted (`d678c940`).

## 6. Validation

- `speculative::prompt_lookup`: 39 tests pass. Deliberate mutations each fail them: probation ignored, a timed Gated pause, one-token confirmation, no shadow lookups, a full first block, probe start one position early or late, and probation lifted by a single landed token.
- CLI tests in the `mlxcel` bin: 96 pass; `dead_doc_pointers` passes; clippy `-D warnings` on lib, tests, bins and the example, and fmt, are clean.
- Real checkpoints on GB10: the matrix above, plus a round-by-round trace of the Qwen3-8B email from a debug build.

## 7. Residual Risks and What Was Not Verified

- **Qwen3-8B email at 0.97x** (0.96x to 0.97x across three matrices; Graded 0.97x to 0.98x). The trace attributes it to a restated prompt fragment; the pipeline-preserving verify is the remaining lever.
- **Metal not measured.** Safe by construction for the default; Gated on Metal is untested.
- **ROCm** keeps Graded, also unmeasured for prompt lookup.
- **Other CUDA architectures and NVFP4 targets** inherit Gated. The multirow and `qmm_sm80` boundaries are CUDA-wide on the affine path; NVFP4 runs `fp_qmv`, whose costs per width were not measured.
- **Shared host.** The final matrix ran at load median 3.1; arm interleaving spreads drift, and the largest within-arm spread was 6.3% (one repetition).

## 8. Learning Points

- **Measure the cost curve before tuning a speculation policy.** The governor's question is "is a miss cheap enough to probe with?", and on CUDA the answer is a step function of block width set by kernel instantiations, not a smooth curve. `examples/verify_width_cost.rs` answers it in a minute per model; rerun it after an MLX pin bump that moves the `qmv` or `qmm_sm80` boundaries.
- **A free signal beats a paid probe.** Decoding emits the tokens a proposal would have claimed anyway; comparing them costs a hash lookup. The same pattern applies to any drafter whose proposal can be produced without the target.
- **Speculative schedules move greedy near-ties.** Any change to which positions run in a multi-token verify can flip a near-tie, so "byte-identical where it was" has to be measured per row after every schedule change, as the reverted threshold showed.

## 9. Related

- Issue #2091 (this work), PR #2074 (prompt lookup, Graded governor), #2090 (plain decoding's stale penalty history, found during #2074).
- `src/cli/draft_block_policy.rs`: the same kernel-boundary reasoning for DFlash block width on GB10.
