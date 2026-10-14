# Technical Report: PR #2272 - Bound DRY and sparse penalty step costs

**Date**: 2026-10-11
**Status**: Completed and verified on CUDA (GB10)
**Languages**: Rust
**Risk Level**: Medium

## Executive summary

This change removes two request-amplified costs from sampling. DRY suffix matching now uses reverse Z, and breaker recognition uses a reversed-token Aho-Corasick matcher, so one step takes expected `O(N + P)` host work for selected history length `N` and configured breaker-pattern tokens `P` instead of quadratic suffix scans or `N × P` breaker checks. DRY and rebuilt frequency/presence updates also scatter only touched token IDs rather than allocating and uploading vocabulary-sized host buffers.

The implementation preserves the exact dtype and element behavior of the prior paths. This required following the pinned CUDA MLX promotion table rather than the issue's original assumption: mixed bf16/f32 binary arithmetic can remain bf16 on CUDA. Scalar zero arithmetic resolves the same backend-specific dtype before sparse scatter, while the established incremental frequency/presence path retains its explicit f32 promotion.

## 1. Problem statement

The previous DRY implementation indexed token positions, then walked backward for every earlier occurrence of the current token. On a period-`p` stream with a full-history window of length `N`, one step performed roughly `N² / (2p)` comparisons. A server request can select full history, which includes the prompt, so this work increased with request-controlled history length.

Breaker handling also needed an explicit bound. Calling the preserved scalar breaker predicate at every position would scan exact-ID vectors and multi-token tail collections repeatedly, creating expected work proportional to the history window multiplied by breaker configuration size. Separately, DRY and rebuilt frequency/presence penalties constructed vocabulary-sized host vectors even when only a few token IDs received penalties.

## 2. Change summary

### 2.1 Bounded DRY matching

`dry_penalty_map` reverses the selected history and runs the Z algorithm once. For an earlier occurrence at `pos`, `Z[last_pos - pos]` gives the same common-suffix length as the former backward walk. A forward pass stores the most recent breaker strictly before each candidate, preserving the rule that a breaker at the candidate itself does not truncate its match.

Exact breakers become singleton patterns. A forward multi-token breaker `[head] + tail` is inserted into an automaton as `reverse(tail) + [head]`. A terminal reached at reversed position `r` always maps to original position `N - 1 - r`; propagating terminal state through failure links therefore marks overlapping and suffix patterns without enumerating every match. Trie construction is expected `O(P)` and scanning is expected `O(N)` with the sparse hash transitions.

### 2.2 Sparse device updates

DRY penalties are reduced to one maximum penalty per next-token ID and scattered across every logits row with `take_along_axis`, arithmetic, and `put_along_axis`. The rebuilt frequency/presence path now builds only touched IDs and values and uses the same sparse helper. Neither path allocates a host vector proportional to vocabulary size.

### 2.3 Exact dtype preservation

The dense DRY reference used `add(logits, f32_buffer)`, and the rebuilt frequency/presence reference used `subtract(logits, f32_buffer)`. On the pinned CUDA MLX patch, bf16/f32 arithmetic can produce bf16; f16 and unpatched backends promote differently. Each rebuild path first performs the corresponding operation against a scalar f32 zero, obtaining the same platform result dtype and unchanged values before sparse scatter.

The incremental `SamplerState` path behaved differently before this change: it explicitly cast logits to f32 when in-range IDs existed and returned an unchanged copy when all IDs were out of range. The shared sparse helper is fed that promoted array, and both branches retain their prior dtype behavior.

## 3. Technical decisions

### 3.1 Stateless per-step matching

A persistent DRY index could reduce repeated work further, but it would need to rebuild on history divergence, rebase sliding windows, and revisit breaker tails whose status changes when more tokens arrive. The stateless reverse-Z and automaton passes preserve request semantics with bounded work and no sampler-state synchronization contract.

### 3.2 Compiled matcher instead of repeated predicates

The first linear DRY draft still invoked `dry_breaks_at` at every position, which left an `O(N × P)` configuration amplifier. The final matcher preprocesses exact IDs and tails once per step. Direct per-position comparison against the unchanged scalar predicate covers negative IDs, duplicates, empty tails, overlapping suffixes, exact window-end matches, and tails that run past the window.

### 3.3 Backend-resolved promotion instead of forced f32

Forcing all sparse rebuild paths to f32 would have changed CUDA bf16 outputs. Scalar zero arithmetic keeps the old binary-operation contract without reconstructing a vocabulary-sized buffer. The pre-existing incremental f32 behavior remains explicit rather than being unified with rebuild semantics.

## 4. Verification

The measured source commit is `ecb89a308aecc468dd1886d9c9f4982f556f3c93` on base `73700aafa6a792f07830578c0c26137cea3a038b`. The CUDA GB10 gate completed a core build in 26.74 seconds, clippy in 9.08 seconds, and formatting with no Rust error. The sampling selection passed 210 tests with only `dry_step_cost_report` ignored, and the engine selection passed 63 tests.

PR publication rebased the implementation onto `3ae626c2a4d67e7628a4050f01cc62b7419296fb`; the resulting checkout is `327b6bf13a31d0f98e4cce717443d03b4be2cb73`. The `sampling.rs` blob remains `59fb6b02c6b54913aa549085fafc090e0b516d55`, and a scoped diff between the measured and publication checkouts across `src/lib/mlxcel-core`, `src/lib/mlx-cpp`, and `Cargo.lock` is empty. The raw evidence package archives the original two-commit format patch from the public measurement base plus hashes and the empty-diff proof.

The sampling coverage includes 512 deterministic parity cases across window sentinels, periodic and random histories up to 2,048 tokens, batch widths 1 and 2, f32 and bf16 logits, and exact, empty-tail, and nonempty-tail breakers. Frequency/presence rebuild parity covers window sizes 1 and 64, batch widths 1 and 2, and f32, f16, and bf16 logits. A separate regression confirms the incremental bf16 path still promotes to f32 and preserves its all-out-of-range early return.

Instrumented complexity tests hold reverse-Z work below `8N`, show near-doubling when the window doubles from 4,096 to 8,192, bound a 2,048-entry exact breaker set to additive build and scan work, and exercise a 128-pattern shared-trunk comb that forces long failure walks while staying below `8P` construction and `3N` scanning.

Independent implementation and security reviews reported no remaining findings.

## 5. Timing evidence

The manual test runs both the exact pre-change reference and the new implementation on a period-3 history with windows 1,024, 4,096, and 16,384, vocabulary 4, and no breakers. Reference iterations are 3, 3, and 1; the new path runs 20 iterations at every window. `black_box` receives lazy MLX arrays without evaluating them, so these figures measure host matching and MLX graph construction rather than GPU end-to-end latency.

| Window | Reference µs/step | New µs/step | Speedup |
|---:|---:|---:|---:|
| 1,024 | 619.751 | 23.607 | 26.25× |
| 4,096 | 4889.370 | 29.997 | 163.00× |
| 16,384 | 77199.293 | 111.433 | 692.79× |

Across the 16× window increase, reference cost grew 124.57× while the new path grew 4.72×; the 16,384-token speedup was 692.79×.

Raw command output, the inner and outer drivers, environment, host-gate output, build/test logs, checksums, and binary/source provenance are published under `docs/benchmark_results/data/dry-penalty-step-cost-gb10-2026-10-11/`.

## 6. Compatibility and limitations

No request field, default, or public API changes. Full-history, zero-window, short-history, low-base, breaker, exponent-cap, and max-per-token semantics remain covered by reference parity. The XLA sampler and windowed repetition penalty remain outside this issue.

The automaton is rebuilt each step, so total work includes configured pattern tokens as well as history length. Hash-map transitions make the stated bound expected rather than worst-case deterministic. The timing test measures one quiet GB10 host and is evidence for scaling and local cost, not a cross-platform throughput guarantee.

## 7. Related records

- Issue #2259: implementation requirements
- Issues #2229 and #2256: security-review context
- `docs/benchmark_results/dry-penalty-step-cost-gb10-2026-10-11.md`: dated timing record
- `docs/benchmark_results/data/dry-penalty-step-cost-gb10-2026-10-11/`: raw evidence package
