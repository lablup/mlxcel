# Technical Report: PR #2274 - Server prompt-lookup break-even governor

**Date**: 2026-10-12
**Status**: Completed and verified on CUDA (GB10)
**Languages**: Rust, Python, shell
**Risk Level**: Medium
**Pull Request**: [#2274](https://github.com/lablup/mlxcel/pull/2274)

## Executive Summary

Issue #2264 adds a server-only break-even governor for prompt lookup. The CUDA server uses a four-token emission floor, seven-token shadow confirmation, immediate re-prime after a proposal-free ask, and a 4608-token paged admission cap. CLI defaults, graded adaptation, and nonadaptive operation remain unchanged.

Corrective commit `b274abf0` implements an intentionally narrow policy. A complete miss receives one additional probe only when the pre-update accepted-token EMA is at least four. Partial low-yield rounds do not receive this grace, consecutive misses still pause, and fully accepted or floor-meeting rounds reset the low-yield streak. The release build, full CUDA suite, focused tests, contracts, and unchanged attempt-4 matrix all pass.

## 1. Problem Statement

The historical server workload lost 3.49% on ordinary chat because weak prompt-lookup matches repeatedly interrupted lookahead and paid synchronous verify costs. The same feature delivered large gains on copy-heavy prompts, so disabling it broadly would violate the issue requirements.

Acceptance requires a chat median no worse than −1%, at least 90% retention of every historical copy gain, no disjoint concurrency-two regression, and a closed paged-8192 control.

## 2. Final Policy Under Evaluation

A verify round emits `accepted + 1` tokens. After at least two drafted samples, the server pauses when the emitted-token EMA is below four and the current proposal is incomplete and below the same floor. A current floor-meeting round repays EMA debt, and a fully accepted short proposal remains active because it exhausted the available copy rather than rejecting proposed tokens.

The drafter passes the actual proposal length. This distinguishes `2/2` from `2/7`: the former may widen to seven, while the latter is partial low-yield evidence. That fix corrected a real state-model defect but did not by itself restore the measured short-copy gains.

The selective boundary rule uses the accepted-token EMA before the current update:

- `accepted == 0` and pre-update accepted EMA at least 4: grant one more probe.
- Partial low yield such as `1/7` or `2/7`: no boundary grace once emitted EMA is below the floor.
- A second consecutive complete miss: pause.
- Full acceptance or a floor-meeting round: clear the low-yield streak.
- Shadow-confirmed resume: reset accepted EMA, emitted EMA, sample count, and low-yield history.

This zero-versus-partial distinction is an evidence-specific abrupt-boundary heuristic, not a general claim that zero acceptance is profitable. In the captured copy traces, complete misses followed accepted EMAs of 4.25, 4.981, and 6.982. Chat complete misses occurred below four except for q3's copy-identical `2/2 → 7/7 → 0/7` prefix. That prefix correctly received grace and added 7 proposed tokens, 0 accepted tokens, and 1 verify round; the other observed high-EMA chat lows were partial matches.

## 3. Supporting Decisions

### Actual proposal length

Correction `550107f56c6c570b0cd6419c12700d1d8b3429b4` preserved fully accepted short proposals. Attempt 2 showed that it did not address the benchmark root cause: isolated wide misses still ended profitable copy runs.

### Context admission

The nominal 4096-token workload tokenized to prompts of 4099, 4101, and 4102 tokens. Its maximum prompt-plus-continuation envelope was 4358 tokens. The paged cap therefore moves to 4608, the smallest 512-token boundary covering the observed envelope, while 4609 and the 8192 control remain excluded. Dense storage is exempt.

### Recovery path

The governor's existing paused state restores pipeline eligibility while shadow probes continue looking for a later copy span. Skipping the drafter was rejected because it would collect neither yield evidence nor recovery evidence.

## 4. Regression Coverage

The current tests cover:

- accepted-EMA grace threshold immediately below and at four;
- complete miss versus partial `1/7` and `2/7`;
- traced `2/2 → 7/7 → 0/7` copy-boundary behavior;
- consecutive misses at default and larger draft widths;
- floor-meeting repayment and full-short preservation;
- shadow-resume state reset and small-width floor capping;
- unchanged CLI defaults, graded policy, and nonadaptive behavior;
- scheduler pipeline recovery, concurrency-two isolation, re-prime cadence, and context boundaries.

Exact-counter diagnostics validate the selective policy. Across the eight-chat workload it produced 173 proposed tokens, 75 accepted tokens, and 32 verify rounds, compared with 226/83/43 under broad grace. The copy trajectory remained exactly 239/214/35. The chat totals include q3’s expected copy-identical boundary behavior.

## 5. Performance Evidence

Attempts are preserved separately with source, binary, environment, host-gate, raw output, and checksum evidence.

- **Attempt 1:** chat and concurrency passed; short-copy and paged 1024/2048 retention failed.
- **Attempt 2:** the five paired short-copy rounds still failed after the full-short correction; the run stopped before the remaining matrix.
- **Attempt 3:** the broad isolated-miss grace restored copy performance. Short-copy gains were +142.132%, +151.215%, +145.537%, and +146.083%. All copy, concurrency, and gate-control criteria passed, but chat failed at median −1.419%.
- **Attempt 4:** the selective grace passed the complete matrix. Chat median was −0.275%. Short-copy gains were +159.283%, +157.037%, +151.289%, and +148.602%. Paged ratios were 2.7349×, 2.8189×, and 1.5221×; dense 8192 was 2.3475×. Concurrency two passed and paged 8192 remained gate-closed.

Attempt 3 showed that broad grace added excessive ordinary-chat work: 60 proposals, 8 accepted tokens, and 12 verify rounds beyond the earlier behavior. The selective pre-update accepted-EMA condition retained the copy trajectory while reducing this chat work.

## 6. Final Validation

Corrective commit `b274abf0` passed 54 governor/drafter tests and 20 scheduler tests, including recovery and floor-boundary coverage. The release build passed in 6m16s. The test build, clippy, formatting, and three contract checks passed. The full CUDA suite passed 8,936 root tests and 2,093 core tests with zero integration failures.

Attempt 4's analyzer exited zero with no failures. The binary SHA-256 was `2388e9f8ef914e11ca0e621801a0cad7bf4101fc359a572a49d6a352afae9485`, the native MLX pin was `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`, and the host gate recorded no running CI job. Temporary runtime tracing was removed before the final build.

## Evidence Locations

- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/STATUS.md`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-1.md`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-2/`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-3/summary-corrected.txt`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-4/summary.txt`
- `docs/benchmark_results/server-prompt-lookup-break-even-gb10-2026-10-11.md`
