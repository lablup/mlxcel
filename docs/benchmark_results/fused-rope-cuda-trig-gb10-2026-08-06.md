# Fused RoPE + KV-append: CUDA trig accuracy and cost, GB10, 2026-08-06

Evidence for issue #1049. The CUDA branch of `fused_rope_append.cpp` called
`__cosf` / `__sinf`; MLX's own `mlx/backend/cuda/rope.cu` calls `cos(float)` /
`sin(float)`, which resolve to libdevice `cosf` / `sinf`. The two agree at small
angles and separate at large ones, which is why only the two large-offset cases
in `fused_rope_parity_tests` failed.

## Environment

| Field | Value |
|---|---|
| Hardware | NVIDIA GB10 (DGX Spark), compute capability 12.1 |
| Backend | CUDA, activations f16 |
| mlxcel | 0.4.3, branch `fix/issue-1049-cuda-rope-trig-accuracy`, base `018b5c16` |
| MLX pin | `2c46b953db88965c4270cc7306eda6887a3247f2` |
| Geometry | Hq 32, Hkv 8, head_dim 128, rope_dims 128, base 500000, scale 1.0 |

## Which side is less accurate

Measured against a float64 host reference, not against the other GPU path. The
clean measurement is the `p = 0` dimension pair: there `d = 0`, so `inv_freq` is
exactly 1.0 on host and device alike and `theta` equals the absolute position
with no representation error to confound the comparison. The applied phase is
recovered from the kernel's own output as `arg(r1 + i*r2) - arg(x1 + i*x2)`,
averaged over the 32 query heads, which share one `theta` at a fixed `(t, p)`.

Applied-phase error in radians against the exact float64 angle:

| offset | theta (rad) | `__cosf`/`__sinf` | `cosf`/`sinf` | theta * 2^-23 |
|---|---|---|---|---|
| 1024 | 1024 | -3.69e-5 | -7.36e-6 | 1.22e-4 |
| 2048 | 2048 | -6.91e-5 | -3.96e-5 | 2.44e-4 |
| 4096 | 4096 | -1.80e-4 | -5.00e-5 | 4.88e-4 |
| 8192 | 8192 | -2.88e-4 | -2.59e-5 | 9.77e-4 |
| 16384 | 16384 | -6.65e-4 | -1.09e-6 | 1.95e-3 |
| 32768 | 32768 | -1.33e-3 | -1.68e-5 | 3.91e-3 |
| 49152 | 49152 | -3.51e-3 | -1.40e-5 | 5.86e-3 |
| 65536 | 65536 | -2.66e-3 | 4.32e-6 | 7.81e-3 |
| 81920 | 81920 | -7.90e-3 | 1.15e-5 | 9.77e-3 |
| 98304 | 98304 | -7.01e-3 | -2.86e-7 | 1.17e-2 |
| 114688 | 114688 | -1.23e-2 | -2.47e-5 | 1.37e-2 |
| 131071 | 131071 | -1.16e-2 | 7.81e-6 | 1.56e-2 |

The libdevice column shows no trend across a 128x change in angle. It scatters
between 2.9e-7 and 5.0e-5 rad, order 1e-5 throughout, and its largest entry sits
at offset 4096 rather than at 131071. That is the measurement floor, and it is
the floor this method predicts: each head's two output components carry fp16
rounding of about 2^-11 relative, so a single head recovers the phase to roughly
5e-4 rad, and averaging the 32 heads that share one `theta` brings that under
1e-4 rad. The bound to quote for this arm is 5e-5 rad with no trend, not any one
cell.

The SFU column instead grows with the angle, about 300x over that same 128x
range, and every point stays inside the `theta * 2^-23` envelope the CUDA
documentation's `[-pi, pi]`-only accuracy bound predicts (the ratio to the
envelope runs 0.28 to 0.90). The growth is not monotonic point to point: 49152,
81920 and 114688 each read higher than the next offset up, by up to 24%. That is
expected, because the SFU error depends on where the angle falls inside the
reduction period and not on the angle alone. What the issue asked for is settled
either way: there is no threshold, no wraparound, and no sign change, only a
trend that tracks the magnitude of the angle. So the fused kernel was the less
accurate side and the MLX reference was right, which is what licenses changing
the kernel rather than the tolerance.

Whole-tensor deviation from the float64 reference (q, B=1 L=1, normalized rms /
normalized max, same statistic the parity test asserts on):

| offset | `__cosf`/`__sinf` | `cosf`/`sinf` | fp16 floor |
|---|---|---|---|
| 511 | 2.04e-4 / 9.78e-4 | 2.03e-4 / 9.78e-4 | 2.03e-4 / 9.78e-4 |
| 4096 | 2.24e-4 / 1.67e-3 | 2.10e-4 / 1.67e-3 | 2.03e-4 / 1.67e-3 |
| 32768 | 7.46e-4 / 8.34e-3 | 3.90e-4 / 5.17e-3 | 2.11e-4 / 9.59e-4 |
| 65536 | 1.43e-3 / 1.47e-2 | 7.27e-4 / 1.05e-2 | 2.07e-4 / 9.74e-4 |
| 131071 | 3.24e-3 / 3.61e-2 | 1.81e-3 / 2.27e-2 | 2.09e-4 / 9.66e-4 |

Both columns rise above the fp16 floor at large offsets. That residual is the
fp32 representation of `theta` itself, which both GPU paths compute identically
and which therefore cancels in the fused-vs-graph comparison the parity test
makes; the `p = 0` table above is the part that does not cancel.

## Agreement with the graph path after the change

Raw output bytes, fused kernel against `slice` / `reshape` / `transpose` /
`fast_rope`:

| shape | offset | q bytes differing |
|---|---|---|
| B=1 L=1 | 0 | 0 / 8192 |
| B=1 L=1 | 4096 | 0 / 8192 |
| B=1 L=1 | 131071 | 0 / 8192 |
| B=1 L=3 | 131069 | 1 / 24576 |
| B=1 L=17 | 65536 | 2 / 139264 |
| B=1 L=64 | 131000 | 26 / 524288 |

Single-token decode is byte-identical at every offset. Multi-token windows differ
on about 0.005% of elements at the last fp16 bit, which is the FMA-contraction
freedom nvcc has over `x1*costheta - x2*sintheta` in two separately compiled
kernels, and is the divergence class the tolerance was written for.

## Cost

Four arms interleaved inside one process (three trig choices plus the unfused
graph as a control), arm order rotated each round, 9 repetitions per shape.
Interleaving matters: an earlier split-across-builds A/B moved the *unchanged*
control arm by 4x between runs and was unusable.

The sweep covered eight sequence lengths and three of them are tabulated below:
the decode shape and the two prefill lengths where the arms were stable enough
across sessions for an arm-to-arm ratio to mean anything. The intermediate
lengths are measured but not reported, because the control arm was too noisy
there to quote a ratio from. They are not evidence either way; the numbers below
are the whole of what this section claims.

Per-call microseconds, median of 9, across three separate quiet-host sessions:

| seq | `cosf`/`sinf` | `__cosf`/`__sinf` | graph control | cosf vs SFU |
|---|---|---|---|---|
| 1 | 17.9 / 16.8 / 18.9 | 17.2 / 16.1 / 18.7 | 41.1 / 38.1 / 43.6 | 1.04x / 1.05x / 1.01x |
| 4096 | 1092 / 1028 / 961 | 1054 / 804 / 850 | 1390 / 1481 / 1453 | 1.04x / 1.28x / 1.13x |
| 8192 | 1701 / 1593 / 1544 | 1504 / 1354 / 1394 | 2251 / 2237 / 2186 | 1.13x / 1.18x / 1.11x |

At the decode shape the kernel is launch-bound and the change costs 0.2 to 0.7us
of 17-19us, which is smaller than the run-to-run drift of the control arm over
the same sessions (38.1 to 43.6us).

At prefill scale the kernel does real work (about 200 MB moved in 1.6 ms,
roughly half of this host's memory bandwidth) and the change costs 11-18% at seq
8192, where all four arms are tight. The seq 4096 row is looser: it reads 4%, 28%
and 13% across the three sessions, and its `__cosf`/`__sinf` arm alone moved 804
to 1054us between sessions, a spread wider than the effect being measured, so
that row bounds the cost rather than measuring it. Across every prefill cell
reported here the cost spans 4% to 28%; the figure to carry forward is the seq
8192 one, 11-18%. The fusion still beats the graph it replaces in every row.

`sincosf` was measured as a third arm and lands on the SFU arm's time (medians
925 vs 900 at seq 4096, 1469 vs 1403 at seq 8192, each pair from one session), so
it recovers essentially the whole cost. It was not adopted, and this is a trade
rather than a dismissal, so the terms are recorded here for whoever revisits it.

Against `sincosf`: it is not bit-identical to the two separate calls (one
differing q element at L=17, offset 65536). Byte-identical single-token decode
against the `fast_rope` graph is the property this fix is built on and the
cleanest thing the parity suite can assert, and `sincosf` would give it up.

For `sincosf`: the cost it recovers is not confined to prefill microbenchmarks.
`Attention::forward` in `src/models/llama3.rs` calls
`forward_fused_rope_append` for every window, prefill as well as decode, with no
`l == 1` guard, so the 11-18% at seq 8192 is paid on the wired path whenever
`MLXCEL_FUSED_ROPE_APPEND=1` and the checkpoint is non-quantized. At the decode
shape the same saving is 0.2 to 0.7us of 17-19us and is invisible.

Bit-identity was judged worth more while this path ships opt-in and off by
default, so the prefill percentage is currently paid by nobody. If
`FUSED_ROPE_APPEND_DEFAULT` is ever flipped, or a prefill-heavy profile makes
that percentage matter, `sincosf` is the first thing to re-measure and the table
above is the number to beat.

## Token-level impact

`qwen2.5-0.5b-bf16` (non-quantized, so it takes the fused path), 130967-token
prompt, greedy, 128 generated tokens at absolute positions 130967 to 131094.

| Measure | Value |
|---|---|
| Tokens identical, SFU vs libdevice | 128 / 128 |
| Chosen-token logprob shift | max 0.1875, median 0.0000, mean 0.0142 nats |
| Top1 minus top2 margin | min 0.125, p10 0.500, median 4.75 nats |
| Steps with margin below the largest observed shift | 2 / 128 |

The error is measurable in the logits and did not flip a token here, but two of
128 steps sat inside its range, so a flip is not structurally excluded.

The reason the effect is small in token terms: a phase error `phi` at frequency
`inv_freq` is indistinguishable from a position error of `phi / inv_freq`, and
the SFU error scales as `theta * 2^-23 = position * inv_freq * 2^-23`, so the
equivalent position error is about `position * 2^-23` for every dimension. At
position 131072 that is roughly 0.016 tokens. Measured per dimension at offset
131071 it ranges from 0.008 to 0.034 tokens over `p <= 32`, where the SFU error
is above the measurement floor.

Note for anyone reading this alongside issue #1049: `FUSED_ROPE_APPEND_DEFAULT`
is `false`, so nothing shipped ran the buggy kernel by default. The issue text
says the kernel defaults on; that is not what `layers.rs` does.

## Incidental: the fusion wins on CUDA

`examples/fused_norm_rope_microbench` on this host, which is the same harness
that measured 0.89-1.15x on M1 Ultra and led to `FUSED_ROPE_APPEND_DEFAULT =
false`:

| hidden | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| 2048 | 1.42x | 2.06x | 1.61x |
| 4096 | 1.02x | 1.67x | 1.81x |
| 8192 | 1.34x | 1.57x | 1.62x |

Every cell is at or above parity, unlike the Apple Silicon sweep. Flipping the
default is a separate decision with its own evidence bar and is not part of this
fix; recorded here so the question is not lost.
