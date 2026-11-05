# Fused tanh-GELU f32: measure-then-keep evidence

Op-level measurement for the fused `gelu_tanh_f32` Metal kernel against the two
chains it replaces. This follows the same measure-then-keep policy as the
`#905` fused decode kernels: the wiring keeps its place only if it beats the
graph it replaced at op level.

`examples/fused_gelu_tanh_microbench.rs` produces this evidence. It loads no
model and sweeps the three shapes the tanh-switched encoders feed.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Pro, 10 cores, 32 GB unified memory |
| OS | macOS 26.6.2 (Darwin 25.6.0) |
| Backend | Metal, activations bf16 |
| mlxcel | 0.8.1, worktree `mlxcel-2423`, base `630aaeba` |
| Harness | `examples/fused_gelu_tanh_microbench.rs`, warmup 64, iters 512, trials 7, min per arm |
| Load average | 10.0 to 13.1 during the sweep, from unrelated concurrent work |

The host was **not** quiet. An earlier single non-interleaved pass on this host
inverted the ordering (`compiled` measured `+96%` against `eager`), which is why
the harness interleaves its trials and reports the minimum: background load and
throttling only ever add time. The three sweeps below agree closely, but treat
an absolute microsecond as a warm-cache op-level figure on a loaded host, not as
a production latency.

No CUDA or ROCm measurement was taken. The port table sets `.cuda = nullptr`
and `.rocm = nullptr` for this kernel, so those backends keep the
`compile_shapeless` path by construction.

## Method

Dtype bf16, because that is what the tanh-switched encoders hand the activation
after the f32 GEMM: the tower runs the f32 formula and the result is stored
back in the input dtype. Input is a deterministic ramp —
`sin(i * 0.01) * 4.0` — which spans the saturated negative tail, the linear
region and the saturated positive tail without an RNG.

Every arm allocates its input once and reuses it on each timed iteration, so
this is a warm last-level-cache measurement; see "Warm vs cold last-level cache"
in `docs/benchmarks.md`. Each call is `eval`-ed and the timed region is
bracketed by `synchronize_default()`.

The sweep ran three times rather than once, for the reason given above.

```bash
caffeinate -i cargo run --release --features metal,accelerate \
    --example fused_gelu_tanh_microbench
```

## Results

Per-call microseconds, minimum of 7 interleaved trials; ratios are
`fused / baseline`, so **below 1.00 is the fused kernel winning**.

| Shape | eager | compiled | fused | fused / compiled | fused / eager |
|---|---|---|---|---|---|
| tower prefill `[64, 4304]` | 659.76 | 369.55 | 303.49 | **0.82x** | **0.46x** |
| decode row `[1, 4304]` | 340.16 | 273.32 | 251.41 | **0.92x** | **0.74x** |
| large prefill `[1024, 4096]` | 2824.82 | 1371.06 | 856.65 | **0.62x** | **0.30x** |

Sweeps 2 and 3, same binary, same host, to show the spread:

| Shape | fused / compiled (s2) | fused / compiled (s3) | fused / eager (s2) | fused / eager (s3) |
|---|---|---|---|---|
| tower prefill `[64, 4304]` | 0.83x | 0.82x | 0.48x | 0.45x |
| decode row `[1, 4304]` | 0.92x | 0.92x | 0.73x | 0.71x |
| large prefill `[1024, 4096]` | 0.74x | 0.69x | 0.34x | 0.32x |

Raw CSV for the first sweep:

```csv
shape,eager_us,compiled_us,fused_us,fused_vs_compiled,fused_vs_eager
"tower prefill [64, 4304]",659.76,369.55,303.49,0.8212,0.4600
"decode row [1, 4304]",340.16,273.32,251.41,0.9198,0.7391
"large prefill [1024, 4096]",2824.82,1371.06,856.65,0.6248,0.3033
```

## What this does and does not establish

The fused kernel wins on all three shapes against both baselines, in all three
sweeps, and the margin is widest where the work is largest — which is the image
prefill the fusion exists for. The narrowest result is the decode row against
the compiled chain at `0.92x`, so a single token still gains, but by the least;
that is expected, since a batch-1 row is launch-bound rather than
bandwidth-bound.

The op-level ratio is a **lower bound** on the end-to-end effect. The fused
kernel also removes the whole full-width intermediate from the MLX graph, which
shows up as allocator and dependency-tracking pressure rather than as kernel
time.

What this does not establish: no end-to-end prefill or decode number was taken
on a real checkpoint, because no tanh-switched vision checkpoint was loaded for
this measurement. The decision recorded here is the op-level one the
measure-then-keep policy asks for, and the wiring stays behind
`fused_gelu_tanh_f32_available()` so `MLXCEL_DEVICE=cpu` keeps the compiled
fallback.
