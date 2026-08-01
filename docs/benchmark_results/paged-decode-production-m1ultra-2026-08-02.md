# Production paged decode through fused v2: Apple M1 Ultra, 2026-08-02

Validation run for issue #899 (epic #909), which routes the server's batched
paged decode through the fused v2 kernel from #898 and retires gather-then-SDPA
as the default.

**Result: decode throughput improves in four of five scenarios, 1.15x to 1.57x,
with parity in the fifth and no regression anywhere.** Aggregate throughput moves
much less, for a reason explained below that is a property of the metric rather
than of the kernel.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-899-production-paged-v2-dispatch` |
| Model | `mlx-community/Llama-3.2-1B-Instruct-4bit` (2048 hidden, 32 q heads, 8 kv heads, head_dim 64, 16 layers) |
| Harness | `scripts/benchmark_paged_decode_production.sh`, `--parallel 4`, 128 decode tokens |
| Arms | `before` = `MLXCEL_PAGED_ATTENTION_NATIVE=0` (gather), `after` = default (fused v2) |
| Load average | 5.0 to 5.5 throughout, from unrelated concurrent work |
| Repetitions | 3 full sweeps, fresh server per arm per sweep |

CUDA was not available. The CUDA JIT bodies inherited from #898 remain
uncompiled and unrun.

## Dispatch was verified, not assumed

This is recorded first because the first attempt at this benchmark produced a
completely null result (0.98x to 1.07x everywhere) that looked like a legitimate
finding and was not. **Both arms had run the gather path**: the fused kernel never
launched once across the entire sweep, for two independent reasons (a token floor
whose shape excluded the measured scenarios, and a single-sequence decode path
that was never wired). A before/after comparison of gather against gather agrees
to within noise, which is indistinguishable from "the kernel does not help".

Every run below therefore carries proof of which kernel it executed, taken from
the server logs:

```
before: paged decode v2: gather: pinned by MLXCEL_PAGED_ATTENTION_NATIVE
after:  paged decode v2: fused v2 launch (batch 4, 3828 visible KV tokens, 16 chunks, merge on)
```

The harness now fails the run if the after arm never logs a fused launch, or if
the before arm does. A null sweep can no longer be produced silently.

## Decode throughput, after / before

The metric the kernel actually affects. Three independent sweeps.

| scenario | rep 1 | rep 2 | rep 3 | median |
|---|---|---|---|---|
| 4 clients, ctx 1024 | 1.33x | 1.82x | 1.57x | **1.57x** |
| 4 clients, ctx 4096 | 0.98x | 1.00x | 1.02x | 1.00x |
| 4 clients, ctx 16384 | 1.15x | 1.13x | 1.15x | **1.15x** |
| 1 client, ctx 16384 | 1.24x | 1.42x | 1.22x | **1.24x** |
| 1 client, ctx 32768 | 1.41x | 1.41x | 1.44x | **1.41x** |

Two cells are tight enough to quote without hedging: 4 clients at 16384 spans
1.13x to 1.15x, and 1 client at 32768 spans 1.41x to 1.44x. Both are large
relative to their own spread and both reproduce in every sweep.

The 4-client / 1024 cell is the noisiest (1.33x to 1.82x) but never drops below
1.33x, so the direction is not in doubt even though the magnitude is.

## Aggregate throughput, after / before

| scenario | rep 1 | rep 2 | rep 3 |
|---|---|---|---|
| 4 clients, ctx 1024 | 1.21x | 1.20x | 1.23x |
| 4 clients, ctx 4096 | 1.00x | 1.00x | 1.01x |
| 4 clients, ctx 16384 | 1.00x | 1.00x | 1.00x |
| 1 client, ctx 16384 | 1.09x | 1.26x | 1.18x |
| 1 client, ctx 32768 | 1.04x | 1.05x | 1.06x |

Aggregate includes time to first token, so at long context it is dominated by
prefill, which this change does not touch. At 4 clients and 16384 tokens the
server spends several seconds prefilling four prompts before decoding 128 tokens
each, so a 1.15x decode improvement is diluted to 1.00x. Read the decode column
for the kernel's effect and this column for what a user of that particular
workload shape would feel.

## Against the issue's required outcomes

- **"No scenario regresses beyond noise (3%)."** Met. The weakest cell is 4
  clients at 4096, whose median is exactly 1.00x (0.98x, 1.00x, 1.02x across
  sweeps), which is parity rather than regression.
- **"Batched decode aggregate throughput improves at 4K+ context."** Partially
  met, and stated precisely rather than rounded up: batched *decode* improves at
  16384 (1.15x, reproducible) but is at parity at 4096, and batched *aggregate*
  does not improve at either because prefill dominates it. ADR-0001's prediction
  that removing gather overhead is worth 2x-3x of SDPA time is an op-level claim,
  and #898 reproduced it at the op level (2.78x at this shape); it does not follow
  that end-to-end serving throughput moves by the same factor, because attention
  is one component of a decode step.

## The 4096 cell

Batched decode at 4096 tokens is the one scenario that gains nothing, sitting
between a 1.57x gain at 1024 and a 1.15x gain at 16384. That non-monotonicity is
not explained. It is reproducible (three sweeps within 4%), so it is a real
property of that shape rather than noise, and it is worth understanding before
the dispatch floor is tuned further. Recorded as an open question rather than
smoothed over.

## Open items

- The non-monotonic 4096 cell above.
- Batch 2 and 3 are interpolated in the dispatch floor: every multi-request cell
  #898 measured was batch 4 or 8.
- The 1-client / 4096 crossover cell has the least margin of any win (1.08x at
  the op level). If single-sequence 4K regresses in a future run,
  `MIN_SINGLE_REQUEST_KV_TOKENS` can be raised to 8192, which gives up only that
  cell.
- CUDA remains entirely unvalidated across #898 and #899.
- One model family measured end to end. Greedy parity covers three families
  (Llama3, Qwen2.5, Qwen3), all exact against the gather baseline, but throughput
  was measured only on Llama-3.2-1B.
