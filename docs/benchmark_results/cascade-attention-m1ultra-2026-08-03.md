# Cascade attention decode: Apple M1 Ultra, 2026-08-03

Validation run for issue #903 (epic #909).

**Outcome: cascade decode is slower than the flat #898 launch in every
configuration measured, so it ships available but unwired
(`MLXCEL_CASCADE_ATTENTION`, default off).** The implementation is correct and
its reuse of the #898 merge kernel is validated; what the measurement does not
support is enabling it.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-903-cascade-attention` |
| Harness | `examples/cascade_attention_bench.rs`, 200 steps, 40 warmup, 5 reps per cell |
| Geometry | q_heads 32, kv_heads 8, head_dim 128, page_size 32, tail 256 |
| Load average | 4.4 to 4.9 during both invocations |

No CUDA was written: both cascade levels reuse the existing v2 kernels.

## Dispatch is proven, not assumed

Every cascade row prints the launch stats it was measured on, and the harness
asserts `prefix_q_heads == q_heads * members` so an arm that failed to fold the
member queries panics rather than reporting a number. Observed
`prefix_q_heads=128` at 4 members and `256` at 8 members, both exactly
`32 * members`, alongside `shared_pages`, `shared_tokens`, `prefix_chunks` and
`suffix_chunks`. The cascade genuinely ran; these are not fallback timings.

This mattered because three earlier measurements in this epic turned out to be
comparing a path against itself.

## Results, two independent invocations

Speedup is flat median over cascade median, so above 1.0 favours cascade.

| shared tokens | batch | run 1 | run 2 |
|---|---|---|---|
| 2048 | 4 | 0.356x | 0.385x |
| 2048 | 8 | 0.408x | 0.784x |
| 8192 | 4 | 0.758x | 0.551x |
| 8192 | 8 | 0.654x | 0.428x |

**Eight of eight measurements are below 1.0.** The magnitudes are unstable across
invocations, but the direction never varies, and the best case observed is 0.784x.
Run 1 absolute medians for reference: flat 0.612 / 0.736 / 1.299 / 2.563 ms per
step against cascade 1.718 / 1.805 / 1.714 / 3.917 ms.

### Why

The decomposition adds roughly six extra small launches per layer-step: two
permutations in, two out, and the merge. At a 2048-token shared span the flat
launch already reads little enough that this fixed cost dominates, which is where
the worst ratios sit. Raising the shared span to 8192 improves the ratio at batch
4 (0.356x to 0.758x in run 1) without reaching parity, so the crossover, if it
exists, is beyond the span sizes the issue asks about.

## The no-sharing overhead gate

The issue requires that enabling cascade cost under 1% end to end when nothing is
shared. The two invocations disagree sharply, and only one of them is credible.

| invocation | batch 4 | batch 8 | flat range, batch 4 |
|---|---|---|---|
| run 1 | +62.3% | +17.9% | 0.804 - 1.317 ms |
| run 2 | **-0.4%** | **+2.6%** | 0.817 - 0.910 ms |

Run 1's flat arm spans a 64% range within a single invocation, which is wider than
the effect being measured, so its overhead figures are noise. Run 2's arm spans 11%
and is the number to read: detection costs about nothing at batch 4 and about 2.6%
at batch 8, so the gate is roughly met and mildly exceeded at batch 8.

Recorded in full rather than quoting only the favourable run, because the pair is
a useful reminder that a single invocation of this harness can produce a 62%
artifact at an apparently quiet load average.

## Correctness and the #898 merge contract

The merge kernel was reused with no edit to `paged_attention_v2.cpp`,
`paged_attention_v2_merge.cpp`, or the FFI signature, and its contract held.

The log2 LSE clause, which fails silently by returning a plausible wrong weighted
average, is satisfied structurally here: both cascade levels are v2 launches, so
their LSE is already the kernel's own `m + log2(l)` and **no unit conversion
happens anywhere on this path**. `merge_rejects_natural_log_lse_units_on_the_cascade_path`
restates #907's guard for cascade partials, and #907's original test still passes.

One design constraint is worth recording because it also fails silently: the
level-0 launch stacks member queries **KV-head major**, because the kernel maps
head `h` to KV head `h / NRep`. Member-major stacking launches successfully with
correct shapes and reads the wrong KV head.
`member_major_head_stacking_reads_the_wrong_kv_head` is the negative control.

`tests/cascade_decode_dispatch.rs` is a separate integration binary because the
gate and both thresholds are `OnceLock`s over environment variables, so no unit
test in a shared process can flip them deterministically. It asserts the outcome is
`FusedCascade` with the expected member count and span, and that the result matches
the flat library launch. Without it, a cascade correct in isolation but unreachable
from production would look exactly like #899's silent null.

## Decision

`DEFAULT_CASCADE_ENABLED` stays false. The kernel path, its tests, the harness and
the observability all land; only the wiring is off. This matches how #905, #907 and
#902 shipped in this same epic, and `MLXCEL_FUSED_QK_NORM` (#326) before it.

Flipping it is one constant once a configuration is found where it wins.

## Open items

- No configuration was found where cascade wins. Larger shared spans than 8192, or
  a batch composition with many more members, are the obvious places to look.
- Detection is not memoized across layers or steps, although the page table it
  reads already is. A `CascadePlan` slot in `PagedDecodeV2Cache` would remove the
  per-layer repeat.
- Whether production produces shared blocks often enough was traced through
  `clone_detached_paged_prefix` and reproduced in fixtures with `refcount == 3`
  asserted, but never observed on a live server. If `cascade_launches` stays zero
  while `prompt_cache_hits` climbs, look there rather than at the kernel.
- Thresholds (16 shared pages, 2 members) are reasoned rather than measured.
- Level-0 JIT specializes on member count, so an oscillating subgroup size would
  recompile repeatedly. Not measured.
