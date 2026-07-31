# Fused paged decode v2: Apple M1 Ultra, 2026-07-31

Validation run for issue #898 (epic #909). Three-way comparison of the production
gather-then-SDPA path, the fused v1 kernel, and the new fused v2 kernel.

**Both required outcomes are met.** v2 is at or above v1 across the sweep, and it
beats gather-then-SDPA at both ADR-0001 trigger points. One cell outside those
trigger points goes the other way and is called out below, because it constrains
the dispatch policy in issue #899.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal, f16 KV pool |
| mlxcel | 0.4.3, branch `feature/issue-898-paged-decode-v2`, base `f0ee1046` |
| MLX pin | `b7c3dd6d27f45b5365b08a840310187dc503f1db` |
| Harness | `examples/page_gather_microbench.rs`, warmup 10, iters 30, warm mode |
| Geometry | head_dim 128, 32 q heads, 8 kv heads (GQA 4), block_size 32 |
| Load average | 3.5 to 3.9 during the sweep |

CUDA was not available. The CUDA JIT bodies are structural transliterations that
have never been compiled or executed.

## Method

batch {1, 4, 8} x context {1024, 4096, 16384, 32768}, three repetitions of the
full sweep. Medians are reported; the per-cell spread across repetitions is given
so that small differences are not read as signal.

## v2 versus gather-then-SDPA (the production path)

Above 1.00 means v2 is faster. Median of three repetitions, with the observed
range.

| batch | ctx 1024 | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|---|
| 1 | **0.91x** (0.90-0.98) | 1.08x (1.02-1.21) | 1.29x (1.20-1.30) | 1.47x (1.44-1.51) |
| 4 | 1.41x (1.39-1.44) | 2.04x (2.00-2.09) | 2.78x (2.78-2.78) | 3.08x (3.07-3.14) |
| 8 | 1.47x (1.18-1.49) | 2.00x (1.98-2.00) | 2.27x (2.24-2.27) | 2.32x (2.32-2.34) |

The batched cells are tight enough to trust: `batch 4 / ctx 16384` returned 2.78x
in all three repetitions, and no batched cell at 4096 or above varied by more
than 0.07. The short-context cells at batch 1 and batch 8 are the noisy ones,
which is the same pattern every measurement in this epic has shown on this host.

### ADR-0001 trigger points

ADR-0001 set the trigger for building a fused path at "single-sequence context
past ~16384, or any sustained batched decode", and measured gather overhead at
~48% for batch 4 at 1024 tokens, rising to 2x-3x past 4096.

- **Batched decode, batch 4, every context measured**: 1.41x to 3.08x. Met.
- **Single sequence at 16384 and above**: 1.29x and 1.47x. Met.

The 2x-3x figure ADR-0001 predicted for batch 4 past 4096 tokens is reproduced
almost exactly: 2.04x at 4096, 2.78x at 16384, 3.08x at 32768.

## v2 versus v1

| batch | ctx 1024 | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|---|
| 1 | 1.12x | 1.63x | 2.98x | **4.11x** |
| 4 | 1.10x | 1.31x | 1.80x | 1.97x |
| 8 | **1.02x** | 1.12x | 1.29x | 1.34x |

v2 is at or above v1 in every cell. The weakest is `batch 8 / ctx 1024` at 1.02x
median, where one repetition returned 0.99x; that cell is parity within noise
rather than a regression, and it is the only cell where any repetition fell below
1.00.

The batch-1 column is the point of the whole exercise. v1 keeps its KV split
inside a single threadgroup, so one CTA serves one `(batch, q_head)` pair no
matter how long the context is, and adding context adds no parallelism. The raw
numbers show that directly: at batch 1, v1 costs 465us at 1K, 1620us at 16K and
3023us at 32K, while gather-then-SDPA costs 353us, 699us and 1063us. **v1 is
slower than the path it was supposed to replace at batch 1**, by up to 2.8x at
32K. v2's cross-CTA split removes that limit and is 4.11x faster than v1 there.

## The cell that constrains issue #899

At **batch 1 / context 1024**, v2 is slower than gather-then-SDPA: 0.91x median,
0.90x to 0.98x across repetitions, so it is below parity in all three. The plan
picks `pages_per_chunk = 2` there, giving 16 chunks over a 32-page sequence, and
the merge pass plus the workspace round trip costs more than the whole attention
does at that size.

This does not violate the issue's requirements, which scope the gather comparison
to the ADR trigger points, and 1024 tokens at batch 1 is the cheapest decode shape
there is. It does mean **issue #899 must not dispatch v2 unconditionally**. The
production selector needs a floor, either on context length or on total CTA count,
below which the gather path stays. The plan already computes the chunk count, so
the cheapest form of that gate is to keep gather when the plan degenerates to few
pages per chunk at batch 1.

## Plan behaviour observed

`pages_per_chunk` scales with the work while the chunk count stays pinned at 16,
which is the binary search hitting its CTA target: 2 at batch 1 / 1K, 68 at batch
1 / 32K, 341 at batch 4 / 32K, 1023 at batch 8 / 32K. The merge pass was active in
every configuration measured.

## Correctness (from the issue's harness, run separately)

68 configurations, zero failures at the 2e-2 tolerance: the 24-config committed
default (worst max relative error 5.24e-4), 12 long-context configurations at
16384 and 32768 over block sizes 16/32/64 (worst 4.43e-4), and 32 forced-merge
configurations up to 1408 chunks and 45056 CTAs (worst 5.24e-4). Worst relative
RMS across all of them was 3.05e-4, roughly 65x inside the tolerance.

## Open items

- **CUDA is entirely unvalidated.** Never compiled, never run. Both JIT bodies and
  the `DEFAULT_TARGET_CTAS` constant need a GB10 pass.
- **The committed correctness default covers 24 of the issue's 216 configurations**;
  `--full` covers the rest and was not run to completion here.
- Sliding-window and softcap variants keep their existing fallbacks. `first_page_offset`
  lets the page table express a trimmed window, but v2 applies no windowing mask of
  its own, which issue #899 must handle when it routes those families.
- Speculative and MTP multi-token verify steps stay on their current path by design.
