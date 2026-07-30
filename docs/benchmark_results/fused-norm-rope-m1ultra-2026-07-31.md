# Fused add-RMSNorm and fused RoPE + KV-append: Apple M1 Ultra, 2026-07-31

Validation run for issue #905 (epic #909).

**Outcome: neither fusion demonstrated a win, so both ship unwired (default off).**
The kernels, their parity tests and the microbench land intact; only the wiring
defaults are off. This follows issue #905's item-4 measure-then-keep policy and
matches how `MLXCEL_FUSED_QK_NORM` (#326) shipped after the same result.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal, activations f16 |
| mlxcel | 0.4.3, branch `feature/issue-905-fused-norm-rope-kernels`, base `be7afe26` |
| MLX pin | `b7c3dd6d27f45b5365b08a840310187dc503f1db` |
| Harness | `examples/fused_norm_rope_microbench.rs`, warmup 32, iters 256 |
| Load average | 4.0 to 5.9 during the sweep, from unrelated concurrent work |

CUDA was not available. Both CUDA JIT strings are written but have never been
compiled or executed.

## Method

Hidden {2048, 4096, 8192} x batch {1, 4, 8}, each fusion timed against the exact
graph it replaces. The sweep was run three times rather than once, because an
earlier single-run measurement in this epic proved to be noise.

## Results

Speedup is unfused / fused, so above 1.00 favours the fusion.

### Fused add-RMSNorm

| hidden | batch | rep 1 | rep 2 | rep 3 |
|---|---|---|---|---|
| 2048 | 1 | 1.00x | 0.98x | 1.01x |
| 2048 | 8 | 0.77x | 1.02x | 1.03x |
| 8192 | 1 | 1.13x | 1.03x | 1.14x |
| 8192 | 8 | 0.91x | 1.04x | 0.94x |

Full rep-1 sweep, all nine cells: 1.00, 0.92, 0.77, 1.03, 1.07, 0.88, 1.13, 1.14, 0.91.

### Fused RoPE + KV-append

| hidden | batch | rep 1 | rep 2 | rep 3 |
|---|---|---|---|---|
| 2048 | 1 | 1.01x | 0.95x | 1.03x |
| 2048 | 8 | 1.01x | 1.15x | 1.07x |
| 8192 | 1 | 0.94x | 0.90x | 0.89x |
| 8192 | 8 | 1.03x | 1.08x | 1.05x |

Full rep-1 sweep, all nine cells: 1.01, 1.03, 1.01, 1.04, 0.95, 0.96, 0.94, 1.09, 1.03.

## Reading

**The scatter is noise, not signal.** The worst cell in rep 1 (add-RMSNorm at
hidden 2048 / batch 8, 0.77x) re-measured at 1.02x and 1.03x. Cells move roughly
15% between repetitions with no consistent direction. Reporting the rep-1 numbers
alone would have produced a confident and wrong conclusion in either direction.

**The harness cannot resolve these ops on this host.** Per-iteration time stays
pinned near 280-410us across every configuration, even though hidden 2048 to 8192
is a 4x change in work and batch 1 to 8 another 8x. A measurement that does not
move when the work changes by 32x is measuring something other than the work:
here, fixed per-iteration dispatch, `eval`, and synchronization cost. Whatever the
fusion saves is well inside that floor. Amortizing the fixed cost, by looping many
op invocations inside one timed body before synchronizing, is the change that would
make this sweep informative.

**One cell is consistently below parity.** Fused RoPE + append at hidden 8192,
batch 1 measured 0.94x, 0.90x, 0.89x. Three repetitions in the same direction is
the only reproducible signal in the sweep, and it points against the fusion. That
is the stronger argument for leaving it unwired, independent of the noise
discussion above.

## Decision

Both defaults flipped to off in `src/lib/mlxcel-core/src/layers.rs`
(`FUSED_ADD_RMSNORM_DEFAULT`, `FUSED_ROPE_APPEND_DEFAULT`), and both env vars
documented as opt-in. Issue #905 states the fusion keeps its wiring only if the
numbers justify it; they do not. The kernels remain available, tested, and one
constant away from being enabled if a quiet host, an amortizing harness, or the
CUDA backend shows a win.

This is not a claim that the fusions are worthless. It is a claim that on this
hardware, with this harness, no win was demonstrated, and shipping a default-on
path on that basis would be asserting something unmeasured.

## Open items

- **CUDA is entirely unvalidated.** Never compiled, never run. Both JIT strings
  are line-for-line transliterations of the Metal versions. A GB10 compile plus
  the parity tests is the first thing to do when hardware is available.
- **Real-checkpoint greedy parity was not run.** `models/` is empty on this host.
  Parity coverage is the synthetic Llama in `src/models/llama3_tests.rs` (24 steps,
  pinned prompt, verified with switches on and off), which covers the wiring, not
  scale.
- **Harness amortization**, per the reading above, before any future attempt to
  measure these two ops.
- Gemma 2/3/4 are deliberately not adopted: they use sandwich norm
  (`norm(attn_out)` then `clip(x + normed)`), not an add-then-norm pair, so there
  is nothing to fuse.
