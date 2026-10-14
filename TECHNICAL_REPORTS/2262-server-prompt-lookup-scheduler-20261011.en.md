# Technical Report: PR #2262 - Prompt lookup on the server without slowing plain decode

**Date**: 2026-10-11

**Status**: Implemented and verified on GB10; pending merge. Speed criteria and the two provisional constants are settled by the end-of-run measurement.

**Languages**: Rust, Python (benchmark client), Markdown (ADR, docs, CHANGELOG)

**Risk Level**: Medium. The feature is opt-in (`--draft-kind prompt-lookup`), but it changes the batch scheduler's lookahead gate and adds a second tick shape (synchronous verify ticks) next to the pipelined one. Output equals the flag-off server on real checkpoints except at exact logit ties; the lookahead and append-ownership rules are written down as invariants and pinned by tests.

## Executive Summary

Prompt-lookup decoding ran only in `mlxcel generate --prompt-lookup`. The server rejected `--draft-kind prompt-lookup`, and enabling any speculative dispatch turned off the lookahead pipeline for every token, which would have cost plain decode about 12 percent. This PR lets `mlxcel serve` and `mlxcel-server` run prompt-lookup verify rounds per sequence inside the batch scheduler, at most one round per tick per row, while ticks on which no row proposes stay pipelined exactly as without the flag. A batch-size gate (`--prompt-lookup-max-batch`, provisional default 2) and the drafter's governor decide when a row may verify, so verify forwards do not crowd out batched decode. The verify round itself moved out of the CLI client into `Engine::verify_round`, so the CLI and the server run the same code.

## 1. Problem Statement

- `SpeculativeDispatch::resolve` returned `Disabled` when no `--draft-model` was set, before reading `--draft-kind`, and rejected `prompt-lookup` as unknown when a draft model was set.
- `BatchScheduler::lookahead_params` returned `None` whenever `should_dispatch_speculative()` was true, so a naive prompt-lookup arm would have made every plain token synchronous.
- Each proposing row needs its own multi-token verify forward. With many concurrent rows that adds one forward per proposing row on top of the batched step, which can lower aggregate throughput.
- On paged storage a multi-token verify forward goes through the pool gather path, which ADR 0001 measured at two to three times contiguous SDPA past 4096 tokens.
- The maintainer asked that offering the feature must not slow plain decode, so the issue carries speed gates next to the correctness criteria.

## 2. Change Summary

| Area | Change |
|---|---|
| `server/speculative_dispatch.rs` | `SpeculativeDispatch::PromptLookup { config, max_batch }`, resolved before the draft-model early return. A draft model with it is `InvalidKind`. `--draft-block-size` sets `max_draft`. `is_kind_specific()` stays false, so the burst path is never entered; the scheduler reads `prompt_lookup_config()`. `--spec-type none` also drops it. `summary()` prints `speculative=prompt-lookup (max_draft=N, policy=P, max_batch=M)`. |
| Server CLI | `--prompt-lookup-max-batch N` on `mlxcel serve` and `mlxcel-server` (default `DEFAULT_PROMPT_LOOKUP_MAX_BATCH = 2`, provisional). |
| `engine/verify_round.rs` (new) | `Engine::verify_round` (verify forward, per-row sampling of every position in order, commit of the accepted prefix plus the target token, unwind of the rest), `Engine::verify_rounds_unsupported` (one eligibility rule: token-only sampler, trimmable state, model support), `Engine::warm_up_verify_widths`. `DirectEngine::speculate` now calls them; about 150 lines left `engine/speculative.rs`. |
| `server/batch/scheduler/prompt_lookup.rs` (new, 615 lines) | Drafter per row on `SequenceInfo`, primed at prefill completion; the per-tick ask; the verify gate; the switch between pipelined and synchronous ticks; verify-width warmup on the first eligible prefill. Module docs state invariants I1 to I8 (synchronous state, one append in flight, no prime past a proposal, drafter context, the gate, membership changes, errors, cancellation). |
| `decode_tick.rs`, `prefill.rs`, `finish.rs` | `lookahead_params` drops its blanket speculative check only for prompt lookup (MTP and DFlash keep it). `ContextBound::verify_room` caps proposals so a verify forward never appends past the context-bound stop. |
| `prompt_lookup_drafter.rs` | `observe_emitted` for plain-step tokens; the n-gram index is built at the first lookup instead of at prefill. |
| `scripts/bench_serving_concurrency.py` | `--prompt-style plain|copy`. |
| Docs | ADR 0009 (amended by #2255), `docs/CONTINUOUS_BATCHING.md`, `docs/llama-server-compat.md`, `docs/speculative-acceptance.md`, CHANGELOG Unreleased. |
| Tests | `speculative_dispatch_tests` (5 new), `engine::verify_round_tests` (3), `scheduler_prompt_lookup_tests` (11, on a dense Llama with zero-weight layers whose greedy next token is `(t + 1) % 16`, checking after every tick that each row's KV offset equals its committed tokens). |

48 files, +2433 / -141.

## 3. Technical Decisions

**One verify round per tick, not a burst.** A run-to-completion burst (the DFlash arm's shape) holds the worker for the whole request and stalls every concurrent row, the head-of-line block #734 removed for MTP. With one round per tick, other rows still get a token every tick, and the round's state (drafter, counters) lives on `SequenceInfo` between ticks.

**Lookahead switch rule.** The scheduler asks every gated-in drafter for a proposal before submitting the step n+1 prime. If nobody proposes, the tick is the unchanged pipelined tick. If anyone proposes, step n is read (through `try_eval`) and committed with no prime, the asks are retracted so no governor records a phantom miss, and the next tick is synchronous: proposing rows verify, the rest take the synchronous batched step. Re-priming waits until nobody proposes and every gated-in drafter reports `pipelines_plain_rounds()`. This mirrors the CLI's `pipelined_plain_round`, so the switch costs no forward. One consequence: after an admission, all rows stay synchronous until the new row's drafter has gone three asks without a proposal. The issue body requires this rule; measurements (a) and (c) check its cost.

**Verify gate.** A row verifies only when its governor grants a budget and the active decode batch is at most `--prompt-lookup-max-batch`. The provisional default 2 rests on the argument that one extra forward per proposing row at most doubles a step at B <= 2. A row the gate declines calls no `draft_block`, so its governor records no miss, and it keeps observing committed tokens so it resumes when the batch shrinks.

**Paged limit left open.** `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT` is `None` (no limit) until measurement (d) shows whether gather-path verify cost at 8192 tokens erases the gain.

**Review and security fixes.** The review added server-side verify-width warmup (the CLI warmed widths, the server did not, so early requests paid first-use kernel cost mid-decode) and moved the n-gram index build to the first lookup (gated-out rows no longer pay it at prefill). The security review capped proposals at the context bound (a prompt ending just under `--max-kv-size` could make a verify forward append up to 63 positions that the stop then threw away), applied the same bound to the warmup, and replaced an unchecked slice in the ask path, since the scheduler has no `catch_unwind` and a panic there would stop all serving.

## 4. Validation

- Local gate on the final branch (GB10, `--profile test-fast --features cuda`, `gpu-lock`, `--test-threads=1`): fmt, clippy `-D warnings` (root and mlxcel-core); mlxcel-core `engine::`, `speculative::`, `drafter::`, `session`, `sampling`, `generate::`; root `server::batch`, `server::speculative_dispatch`, `server::in_process`, `engine_probe`, `backend::`, `cli_input`, `commands::`, `mlxcel-server` bin tests; `cli_help_consistency`, `dead_doc_pointers`, `model_loader_stdout_guard`; Apache header check. All passed. `make verify-test-cuda` (workspace, 13,868 passed, 398 ignored) found one regression from this PR: `server_config_schema_classifies_all_111_fields`, because the new `prompt_lookup_max_batch` field was not classified in the runtime settings schema. Fixed in 9f5b5a87 (read-only, integer, worker-scoped like the other speculative knobs). The other two failures are the pre-existing SSM bf16 parity tests tracked in #2230.
- `mlxcel generate --prompt-lookup` on Llama-3.2-1B, greedy, 160 tokens: output identical to the build before this PR.
- Server pairs, release build, `MLXCEL_SDPA_DETERMINISTIC=1`, `/completion` greedy, flag on vs off: Qwen3-1.7B at 140 tokens (auto, dense and paged storage) and Llama-3.2-1B at 160 tokens (auto and paged) give identical copy and plain replies, concurrent copy and plain requests on the flagged server equal their solo outputs, and `draft_kind` is `prompt-lookup` with `draft_n_accepted > 0`. `mlxcel serve` gives the same result. Gemma-3-1B gives identical replies with prompt lookup inactive: its sliding-window `RotatingKVCache` cannot be trimmed, so the shared eligibility rule declines it, as it does on the CLI.
- At 160 tokens the Qwen3 copy reply diverges once at about token 150, after the copied paragraph. The flag-off server's own top two candidates there have the same logprob (-1.625), an exact tie.
- `mlxcel-engine-parity -n 400` on Qwen3-1.7B and Llama-3.2-1B: unchanged from #2256.

## 5. Residual Risks

- Speed is not yet measured. The end-of-run measurement covers (a) plain chat B=1 flag on vs off within -1.0 percent, (b) copy-heavy B=1 speedup and acceptance, (c) concurrency 2, 4 and 8 flag off, on and on with the gate opened, which sets `DEFAULT_PROMPT_LOOKUP_MAX_BATCH`, and (d) 8192-token prompts on paged and dense storage, which decides `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT`.
- The admission rule keeps every row synchronous for about three ticks per admitted request while the flag is on and the batch is within the gate.
- The n-gram index is about 120 bytes per prompt token, built on the scheduler thread at the first lookup (about 15 MB and tens of ms at 128K tokens).
- No scheduler-level test injects a verify eval failure; the engine-level failure paths are covered in `verify_round_tests.rs`.
- `stats.rounds` undercounts because rule 3 retracts the empty asks of rows that did not propose (the CLI does the same; clients do not see it).

## 6. Learning Points

- Speculative decoding and lookahead pipelining are not mutually exclusive. Asking the drafter before priming, and committing without a prime when it proposes, keeps proposal-free ticks fully pipelined at no forward cost.
- Writing the append-ownership invariants down before the switch logic (who holds which uncommitted position at each point, and who commits or unwinds it) made each test target one invariant, and the KV-offset-equals-committed-tokens check after every tick catches any leak.
- A synthetic benchmark prompt built by repeating one sentence is a copy-heavy prompt for prompt lookup. The no-loss measurement needs non-repetitive prompts.

## 7. Related

- Issue #2255 (split from #2229), epic #2166, PR #2256 (#2229), PR #2225 (`DirectEngine`, `PromptLookupDrafter`), ADR 0001, ADR 0009, #734, #822.
