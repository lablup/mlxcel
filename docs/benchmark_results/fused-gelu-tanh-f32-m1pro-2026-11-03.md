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
| Host load | not sampled; `caffeinate -i` held, min of 7 interleaved trials per arm |

The host was **not** quiet. An earlier single non-interleaved pass on this host
inverted the ordering (`compiled` measured `+96%` against `eager`), which is why
the harness interleaves its trials and reports the minimum: background load and
throttling only ever add time. The three sweeps below keep the ordering but do
**not** agree closely on absolute microseconds: run-to-run figures moved by up
to ~30%, and the fused/compiled ratio on large prefill ranged 0.71x-0.85x.
Treat an absolute microsecond as a warm-cache op-level figure on a loaded host,
not as a production latency, and read the large-prefill ratio as a range.

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
| tower prefill `[64, 4304]` | 587.29 | 342.31 | 279.96 | **0.82x** | **0.48x** |
| decode row `[1, 4304]` | 314.52 | 252.87 | 220.77 | **0.87x** | **0.70x** |
| large prefill `[1024, 4096]` | 2497.79 | 959.47 | 738.92 | **0.77x** | **0.30x** |

Sweeps 2 and 3, same binary, same host, to show the spread:

| Shape | fused / compiled (s2) | fused / compiled (s3) | fused / eager (s2) | fused / eager (s3) |
|---|---|---|---|---|
| tower prefill `[64, 4304]` | 0.81x | 0.76x | 0.45x | 0.48x |
| decode row `[1, 4304]` | 0.96x | 0.97x | 0.77x | 0.73x |
| large prefill `[1024, 4096]` | 0.71x | 0.85x | 0.32x | 0.32x |

Raw CSV for the first sweep:

```csv
shape,eager_us,compiled_us,fused_us,fused_vs_compiled,fused_vs_eager
"tower prefill [64, 4304]",587.29,342.31,279.96,0.8178,0.4767
"decode row [1, 4304]",314.52,252.87,220.77,0.8731,0.7019
"large prefill [1024, 4096]",2497.79,959.47,738.92,0.7701,0.2958
```

## What this does and does not establish

The fused kernel was faster than both baselines on all three shapes in all three
sweeps. Against the eager chain the ratio falls as the shape grows (0.70x-0.77x
on the decode row, 0.45x-0.48x on the tower prefill, 0.30x-0.32x on the large
prefill). Against the compiled chain the tower and large-prefill ranges overlap
(0.76x-0.82x and 0.71x-0.85x), so these sweeps do not rank those two shapes. The
smallest gain is the decode row against the compiled chain, at 0.87x-0.97x.

The ratios are for this one operation. The activation is a fraction of an
encoder forward pass, so they do not predict an end-to-end change. The fused
kernel also removes intermediates from the MLX graph, and this harness did not
measure what that does to allocator or graph-construction time.

What this does not establish: no end-to-end prefill or decode number was taken
on a real checkpoint, because no tanh-switched vision checkpoint was loaded for
this measurement. The decision recorded here is the op-level one the
measure-then-keep policy asks for, and the wiring stays behind
`fused_gelu_tanh_f32_available()` so `MLXCEL_DEVICE=cpu` keeps the compiled
fallback.
