# Fused tanh-GELU f32 on CUDA: one kernel with the eager chain's bits, and the Gemma 3 / Gemma 4 image prefill against erf (GB10, 2026-11-11)

Issue #2780. PR #2858 made `GeluActivation::TanhF32Compiled` one Metal kernel
and left the port table's `.cuda` entry null, so on CUDA the variant still ran
through the `compile_shapeless` region that the
[GELU erf-vs-tanh record](versions/v0.8.0.md#gelu-erf-vs-tanh-2026-10-23)
found running as more than one kernel, and plain `GeluActivation::TanhF32`
ran as the eager chain. This is the CUDA port: the premise reproduced on the
pinned main, one launch for both variants, the measurement that the port
returns the eager chain's bits and why a naive port does not, the op-level
microbenchmark, and the image-prefill A/B the issue names as its acceptance.

The issue asks for a new section of
`docs/benchmark_results/gelu-erf-vs-tanh-2026-10-23.md`. #2949 condensed that
file into the v0.8.0 archive, so this is a standalone record under the
retention rule in `docs/benchmarks.md`.

## Environment

| Field | Value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 (sm_121), 121.7 GiB unified memory |
| Driver / toolkit | 580.178.04, CUDA 13.0 (`nvcc` V13.0.88), compute capability 12.1 |
| Kernel | `7.0.0-1019-nvidia` |
| mlxcel | 0.8.1 at `28d884a5` (upstream `main`, 2026-11-11), MLX pin `9f806b36` |
| Build | `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda`; `MLX_ENABLE_TF32=0` on every GPU run |
| Baseline binary | pinned `main`, sha256 `f2ef3324127393e0ec0877f67c31d5acdc157e4f413f318169e5bda9ae21dbd2` |
| Fused binary | this change on the same base, sha256 `0fd2764707ad35d57b4434fadf7ceb578e187ede8091e67deacd9c41ed00af88` |
| Host | not a CI runner; no `gpu-lock` on this host, runs are serialized by the driver script |
| GPU clock | held to 300-2100 MHz of the 3003 MHz maximum by the operator's clock-range manager, reapplied every 60 s, with a read-only readback every 15 s; left as found, so every number here is under that cap |
| PTX disk cache | enabled and warm; the overlay refuses a cache directory with a group- or other-writable ancestor, which this host's default umask produces (Gemma 3 prefill then reads 21.7 s instead of 0.86 s), so the workspace is `umask 077` |

Checkpoints, all public `mlx-community` repositories on Hugging Face:

| Name in this record | Repository | Revision |
|---|---|---|
| gemma-3-4b-it-4bit | `gemma-3-4b-it-4bit` | `93724907d4ed` |
| gemma-4-e2b-it-4bit | `gemma-4-e2b-it-4bit` | `238767527555` |
| minicpm-v-4.6-bf16 | `MiniCPM-V-4.6-bf16` | `993c9b1ac21a` |
| paligemma2-3b-6bit | `paligemma2-3b-ft-docci-448-6bit` | `2bab20b1962b` |
| internvl3-1b | `InternVL3-1B-4bit` | `7927bb6b8698` |
| qwen3-vl-2b | `Qwen3-VL-2B-Instruct-4bit` | `9c4f5209e57b` |

The two Gemma checkpoints were downloaded without a pinned revision; the
revision shown is the repository head read the same day, and both
repositories were last modified months earlier.

## The premise on the pinned main

`compiled_gelu_tanh_f32` run once per dtype (f32, bf16, f16) through
`compiled_tanh_f32_matches_the_eager_helper_to_f32_rounding` against an empty
`MLX_PTX_CACHE_DIR` leaves four compiled modules, each with the 42
contiguity/ndim specialisations MLX's compile backend emits:

| Module (name abbreviated) | Role |
|---|---|
| `Cf4IBroadcast…Multiply…_CV_f4_` | the cube and its coefficient (`x * x * x`, `0.044715 * x^3`), f32, shared by the three dtypes |
| `Gf4IAdd…Tanh…Multiply…V2OAsType…_VCCCVV_f4f4f4_` | the tail for bf16 output: `x + …`, `tanh`, `0.5 * (1 + …)`, `x * cdf`, cast back |
| `Gf4IAdd…Tanh…Multiply…f2OAsType…_VCCCVV_f4f4f4_` | the same tail with the f16 cast |
| `Ff4IBroadcast…Tanh…OMultiply…_VCCCV_f4f4_` | the same tail with no cast, f32 output |

So one `TanhF32Compiled` call is at least two kernel launches, the split the
2026-10-23 record saw. MLX's CUDA compile backend partitions the region and
this repository does not own that partitioning; the port table in
`src/lib/mlxcelverse/kernels/gelu_tanh_f32.cpp` had `.cuda = nullptr`, so
`gelu_tanh_f32_compiled_where_available` took the compiled chain on CUDA by
construction. Plain `TanhF32` was the eager helper, `gelu_tanh_approx`
between two casts, one AOT kernel per operation, about eleven launches: its
fold pin run alone against an empty cache passes and leaves the cache empty.

## The change

A `mlx::core::fast::cuda_kernel` port of the Metal body in
`gelu_tanh_f32.cpp`: one thread per element, `(float)x[i]`, the formula and op
order of `gelu_tanh_approx`, `tanhf`, store through `T`.

- **Grid guard.** MLX's CUDA launcher rounds the grid up to whole blocks
  (`mlx/backend/cuda/custom_kernel.cpp:339-340` at the pin:
  `dim3 block(std::min(tx, gx), …)`, `dim3 grid((gx + tx - 1) / tx, …)`), so
  257 elements with a 256-thread group launch 512 threads. The body returns on
  lanes at or past `x_shape[0]`, which the launcher passes as an argument when
  the source names it (`custom_kernel.cpp:60-62, 135`). Read as an argument,
  not a template constant, it leaves the JIT cache key as the dtype alone: one
  module and one disk-cache entry per dtype.
- **No contraction.** Every multiply and add in the CUDA body is `__fmul_rn` /
  `__fadd_rn`, which NVRTC never merges into a fused multiply-add (the CUDA
  Math API reference says so of each). "Numerics" shows what that buys.
- **Routing.** `TanhF32Compiled` reaches the kernel wherever
  `fused_gelu_tanh_f32_available()` is true, now Metal and CUDA, as #2858 set
  up. `TanhF32` reaches it where the port returns the eager helper's bits,
  asked through a new predicate, `fused_gelu_tanh_f32_matches_eager_bits()`,
  which reads a second port table listing the CUDA port only. On Metal, ROCm
  and the CPU device `TanhF32` is the eager helper, unchanged; #2858 kept it
  there because the Metal kernel may contract and the fold pin is bit for
  bit. On CUDA the pin passes unchanged through the kernel.

## One kernel

Fresh PTX cache each time. The fused sweep test leaves three modules, one
kernel each, signature `(const T*, Shape, T*)`:
`custom_kernel_mlxcel_fused_gelu_tanh_f32_float_2a4a3dc3e894499c`,
`…_bfloat16_t_16656dc2cd8b5e71`, `…_float16_t_ec24c2c5f6d7417d`. The `TanhF32`
fold pin run alone now leaves the same three and nothing else, where the
previous tree left the cache empty. In a real model, one `generate --image`
per row:

| Run | Modules | Of which GELU |
|---|---|---|
| Gemma 3 4B, baseline binary (`TanhF32Compiled` -> compiled chain) | 7 | 2: `Cf4IBroadcast…_CV_f4_` (cube), `Gf4IAdd…Tanh…_VCCCVV_f4f4f4_` (tail) |
| Gemma 3 4B, fused binary | 6 | 1: `custom_kernel_mlxcel_fused_gelu_tanh_f32_bfloat16_t` |
| Qwen3-VL 2B, baseline binary (`TanhF32` -> eager chain) | 6 | 0 (AOT kernels only) |
| Qwen3-VL 2B, fused binary | 7 | 1: the same custom kernel module |

The flatten before the launch and the reshape after it are views on a
contiguous input (`fused_kernel_stays_correct_on_a_non_contiguous_input`
covers the other case), so a call is one launch.
`tests/cuda_custom_kernel_disk_cache.rs` requires the bf16 entry and runs the
kernel in a second process on the same cache with no NVRTC compile.

## Numerics: the kernel returns the eager chain's bits

A probe evaluated the eager helper, the compiled chain and the fused kernel on
the same array and compared f32 bit patterns lane by lane (`+0` against `-0`
counts). Lanes that differ, fused against eager, fused against compiled and
compiled against eager alike:

| Input set | Lanes | f32 | bf16 | f16 |
|---|---|---|---|---|
| the tests' sweep (\|x\| up to 1e4) | 809 | 0 | 0 | 0 |
| 14 element counts from 1 to 65537, either side of every block boundary, and two 2-D shapes | 202499 | 0 | 0 | 0 |
| tower-shaped `[64, 4304]`, N(0, 4²) with outliers to ±170 | 275456 | 0 | 0 | 0 |
| dense `[1024, 1024]`, uniform in [-12, 12) | 1048576 | 0 | 0 | 0 |

1527340 lanes per dtype, none differing; a second run of the probe is
identical. Eleven special values (±0, ±the smallest normal, ±a subnormal,
±3.4e38, ±inf, NaN) agree in all three paths and dtypes; `-inf` gives NaN in
all three, the formula's `-inf * 0`.

**Why `__fmul_rn` / `__fadd_rn`.** The first draft of the port used plain
operators, as the Metal string does. `cuobjdump -sass` on its cached f32
module shows the polynomial as `FMUL`, `FMUL`,
`FFMA R7, R7, 0.044715, R4`: `x + 0.044715 * x^3` as one fused multiply-add.
The same probe then found f32 lanes one rounding step of tanh away from the
eager chain: 1171 of 275456 on the tower-shaped input (0.43 %), 2062 of
1048576 on the dense one (0.20 %), none on the 809-point sweep, at most 2^-22
absolute. On the cancelling negative tail that step is large against the
result (x = -4.24: -2.1105e-5 against -2.1231e-5, 0.6 %). bf16 and f16 showed
none, because the cast back hides a 2^-22 difference. The compiled chain
agreed with the eager chain bit for bit throughout: its multiply and its add
are in different modules, so nothing can contract them. With the body as
committed the same lines are `FMUL`, `FMUL`, `FMUL`, `FADD`, and every lane
agrees.

The limit of the claim: the multiplies and adds are IEEE operations and agree
with the eager chain's by construction. `tanhf` (the six FFMA left in the
kernel are inside it) agrees with the eager chain's `tanh` kernel on every
lane measured, which is what one expects when the AOT kernels and NVRTC come
from the same CUDA release, as here, and is not guaranteed across releases.
`fused_kernel_returns_the_eager_helper_bits_where_it_claims_to` is the test
that would say so.

**Tests.** `utils::gelu_tests`, release profile, `--test-threads=1`: 17
passed. New: `fused_kernel_returns_the_eager_helper_bits_where_it_claims_to`
(the sweep, seven element counts around the block boundary and the
tower-shaped input, in three dtypes, for the kernel and both dispatch arms; it
fails on the plain-operator body),
`tanh_f32_dispatch_keeps_the_eager_helper_on_the_cpu`,
`fused_tanh_f32_matches_the_eager_helper_to_f32_rounding_on_the_sweep` (the
compiled chain's tolerance, for ports that make no bit claim), and
`fused_kernel_is_reachable_on_cuda` in place of
`fused_kernel_is_unreachable_on_cuda`. The fold pins are unchanged and pass.
The disk-cache test passes with the new entry and
`tests/cpu_device_custom_kernel_gates.rs` reports both predicates false on the
CPU device. The issue's filter `utils_gelu` matches no test on current `main`;
the module is `utils::gelu_tests`.

**`compute-sanitizer` says nothing about the grid guard.** `memcheck` reports
0 errors on the probe and on the GELU test module, and also on a build with
the guard removed, whose outputs are unchanged. MLX's allocator rounds a
buffer up to a power of two below 16384 bytes and to whole 16384-byte pages
above (`mlx/backend/cuda/allocator.cpp:176-182`), and past one block either is
a whole number of 256-thread blocks for 2- and 4-byte elements, so lanes past
`n` land in the buffer's own slack. The guard rests on the launcher code
quoted above, not on the sanitizer.

## Op-level microbenchmark

`cargo run --release --features cuda --example fused_gelu_tanh_microbench`,
bf16, warmup 64, iters 512, minimum of 7 interleaved trials per arm, PTX cache
warm, GPU otherwise idle. Four runs of the harness on the fused binary;
microseconds per call as the median over the runs and (min..max); ratios per
run, below 1.00 is the fused kernel winning.

| Shape | eager | compiled | fused | fused / compiled | fused / eager |
|---|---|---|---|---|---|
| tower prefill `[64, 4304]` | 187.0 (118.1..216.9) | 84.1 (72.0..99.0) | 16.6 (14.8..17.8) | **0.19x** (0.17..0.22) | **0.09x** (0.08..0.14) |
| decode row `[1, 4304]` | 109.2 (94.0..114.2) | 30.6 (27.2..31.1) | 12.5 (10.8..13.2) | **0.41x** (0.36..0.48) | **0.12x** (0.11..0.12) |
| large prefill `[1024, 4096]` | 1571.2 (1557.7..1590.5) | 698.8 (657.6..707.3) | 184.4 (168.0..208.4) | **0.27x** (0.24..0.29) | **0.12x** (0.11..0.13) |

```csv
run,shape,eager_us,compiled_us,fused_us,fused_vs_compiled,fused_vs_eager
1,"tower prefill [64, 4304]",177.06,83.45,14.76,0.1769,0.0834
2,"tower prefill [64, 4304]",216.94,84.78,17.76,0.2095,0.0819
3,"tower prefill [64, 4304]",196.93,98.97,17.09,0.1727,0.0868
4,"tower prefill [64, 4304]",118.07,72.00,16.15,0.2243,0.1367
1,"decode row [1, 4304]",105.12,27.16,13.04,0.4802,0.1241
2,"decode row [1, 4304]",113.38,30.99,12.02,0.3881,0.1061
3,"decode row [1, 4304]",114.18,31.13,13.19,0.4237,0.1155
4,"decode row [1, 4304]",93.96,30.20,10.79,0.3573,0.1148
1,"large prefill [1024, 4096]",1579.85,690.49,188.44,0.2729,0.1193
2,"large prefill [1024, 4096]",1557.65,657.58,180.31,0.2742,0.1158
3,"large prefill [1024, 4096]",1562.64,707.01,168.04,0.2377,0.1075
4,"large prefill [1024, 4096]",1590.51,707.30,208.40,0.2946,0.1310
```

The ordering held in every run. The two small shapes are launch-bound and
move from run to run (eager on the tower shape spans 118 to 217 us), so read
those rows as ranges; the large-shape row is stable. These are warm-cache
op-level figures for one activation (see "Warm vs cold last-level cache" in
`docs/benchmarks.md`). `fused / eager` is the `TanhF32` route's before and
after, `fused / compiled` is `TanhF32Compiled`'s.

## Image prefill: tanh (fused) vs erf, a null arm for every arm

Method: `mlxcel generate --image tests/fixtures/test_image_shapes.png -p
"Describe the image." -n 8 --temp 0 --profile`, `MLX_ENABLE_TF32=0`, PTX cache
warm (a run printing the NVRTC notice is discarded and rerun; none did), one
process per run. The schedule is the one `scripts/engine_bench_rounds.py
--null-every-arm --hostgate` produces, driven by a small script because that
tool has no image input: each round runs every arm once and then every arm
again as its own null arm (the same binary, checkpoint path and flags), the
order rotates by one position each round, 8 rounds per family, and before
every run `wait_quiet()` from `scripts/bench_hostgate.py` (twelve quiet
samples five seconds apart) passes and the GPU's compute-process list is
recorded (a running `mlxcel` process does appear in that list on this host).

| Arm | Binary | Checkpoint |
|---|---|---|
| erf | fused | symlinked copy whose `config.json` declares the erf GELU (Gemma 3: `vision_config.hidden_act = "gelu"`; Gemma 4: `vision_config.hidden_activation = "gelu"`) |
| tanh | fused | the original checkpoint: `TanhF32Compiled` -> the CUDA kernel |
| compiled | baseline (pinned main) | the original checkpoint: `TanhF32Compiled` -> the `compile_shapeless` chain, the behavior before this change |
| erf-null, tanh-null, compiled-null | as the arm it repeats | as the arm it repeats |

Two things are added to the host gate, because of this host.

It serves an embedding model through ollama (`llama-server`, CPU work, never
in the GPU's compute-process list), started on demand by other clients and
absent from the gate's lists; when it wakes it burns five to six cores for a
minute or two. A 10 s sampler caught four starts in an hour and a half, and
the run in flight at the first of them, in an earlier block, read 764 ms
where its neighbours read 600. So a run starts only when that server is
absent or idle at the end of the gate's quiet window, and a run during which
it burned more than 0.5 s of CPU, or started or exited, is recorded,
discarded and rerun, like a cold-cache run.

A block that follows a rebuild starts with one discarded warm-up run per
distinct command: the first read of a checkpoint after the page cache has been
churned faults its weights in during prefill (a Gemma 3 run in that position
read 1731 ms with a 6.1 s wall, against 830 ms and 3.3 s). That applies to
the Gemma 3 block and the page-image block below; the Gemma 4 and 448x448
Qwen3-VL blocks ran back to back on warm caches and their first runs read
like the rest.

Prefill milliseconds, median (min..max); deltas are paired per round. The
issue's pass rule: the tanh - erf range overlaps the null range.

| Family (checkpoint, prompt tokens) | erf | tanh (fused) | compiled (baseline) | Paired tanh - erf | Paired compiled - erf | Paired tanh - compiled | Null arms: erf / tanh / compiled | Outcome |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 SigLIP (gemma-3-4b-it-4bit, 272) | 831.4 (824.8..847.4) | 810.9 (786.1..824.8) | 853.2 (838.6..868.8) | **-20.4 ms, -2.5 %** (-55.3..-4.9), negative in 8 of 8 rounds | +15.3 ms, +1.8 % (-2.0..+41.6) | **-45.7 ms, -5.3 %** (-78.0..-17.4), negative in 8 of 8 rounds | +0.3 ms (-23.0..+26.5) / -14.6 ms (-23.5..+27.2) / -2.7 ms (-26.1..+23.5) | pass: the tanh - erf range overlaps all three null ranges |
| Gemma 4 vision GeGLU (gemma-4-e2b-it-4bit, 278) | 608.5 (596.4..620.2) | 597.3 (586.7..610.1) | 615.4 (606.7..629.4) | **-10.9 ms, -1.8 %** (-24.7..+2.4), negative in 6 of 8 rounds | +4.3 ms, +0.7 % (-6.1..+32.6) | **-20.4 ms, -3.3 %** (-41.8..-4.0), negative in 8 of 8 rounds | -3.6 ms (-8.6..+13.3) / -0.4 ms (-11.3..+22.8) / +0.7 ms (-14.3..+11.6) | pass: the tanh - erf range overlaps all three null ranges |

Gemma 3: 48 valid runs after three warm-up runs, none discarded, the embedding
server absent throughout. Gemma 4: 48 valid runs; one further attempt (round
4, compiled-null) began as the embedding server started and was discarded and
rerun. The compute-process list was empty before every run.

```csv
family,round,erf,tanh,compiled,erf-null,tanh-null,compiled-null
gemma3,0,827.21,822.36,868.81,830.45,807.09,842.70
gemma3,1,831.17,811.07,846.36,828.55,787.57,864.38
gemma3,2,844.22,824.84,842.20,829.57,811.01,837.80
gemma3,3,842.69,792.95,842.33,828.23,790.33,843.80
gemma3,4,824.80,786.14,864.17,840.48,813.32,843.98
gemma3,5,847.35,792.03,862.68,824.40,789.33,845.66
gemma3,6,831.54,810.78,838.65,858.03,789.23,862.17
gemma3,7,826.60,815.08,859.99,842.38,793.43,859.08
gemma4,0,596.41,598.77,621.32,599.63,587.49,620.95
gemma4,1,605.07,586.74,606.68,610.36,602.37,610.51
gemma4,2,620.16,610.14,614.09,616.01,606.82,615.08
gemma4,3,619.42,595.78,616.65,616.36,602.73,618.69
gemma4,4,603.72,604.31,610.66,598.01,594.05,611.02
gemma4,5,611.94,600.18,610.98,603.30,602.78,622.59
gemma4,6,596.80,587.60,629.39,610.08,610.36,615.13
gemma4,7,615.75,591.05,622.98,608.76,587.52,610.08
```

Both families pass, on the side one would hope for: the tanh arm is not above
erf. On Gemma 3 it is below erf in every round, by a median 20 ms inside a
null floor of about 27 ms, so the size of the gap is not resolved and its
sign is consistent. On Gemma 4 the two are within 11 ms with a floor of about
23 ms. That the kernel can undercut erf fits "Erf arm considered" below: the
erf path is an eager six-op chain and this one is a single launch.

compiled - erf is the state before this change. The 2026-10-23 record
measured it at +2.8 % (Gemma 3, pooled) and +2.0 % (Gemma 4) and left it as an
unresolved residual; here it reads +1.8 % and +0.7 %, again inside the floor.
What does resolve is the kernel against the compiled chain it replaces:
negative in all 16 paired rounds.

Two earlier blocks the same day are not in the table and agree with it in
sign: one with the plain-operator body and the 2026-10-23 schedule (one null
arm; Gemma 3 tanh - erf -33.3 ms, Gemma 4 -7.0 ms), and one with this body
and schedule on Gemma 3 before the embedding-server condition existed
(-28.8 ms), whose erf null arm was positive in every round (+20.3 ms median),
which a repeat of the same command should not be. That block is what led to
the sampler and the condition; under the condition the same null reads
+0.3 ms.

### A family on the plain `TanhF32` route: Qwen3-VL, before and after

The Qwen3-VL vision blocks name `GeluActivation::TanhF32`
(`src/vision/encoders/qwen3_vl.rs`), so on the pinned main their GELU is the
eager chain and with this change it is the kernel. There is no erf arm; the
arms are the baseline binary and the fused binary on the same checkpoint,
each with its own null arm, same schedule, gate and conditions.

| Input (prompt tokens) | eager (baseline) | kernel (fused) | Paired kernel - eager | Null arms: eager / kernel | Outcome |
|---|---|---|---|---|---|
| `test_image_shapes.png`, 448x448 (210) | 320.0 (314.6..324.8) | 315.9 (310.9..320.9) | -1.8 ms, -0.6 % (-13.9..+4.0), negative in 6 of 8 rounds | -4.0 ms (-12.7..+2.9) / -4.3 ms (-14.1..+2.6) | unresolved: inside the null floor |
| `nemotron_parse_page.png`, 1240x1754 (2159) | 699.2 (694.2..705.1) | 623.0 (616.9..632.7) | **-76.8 ms, -11.0 %** (-84.6..-61.9), negative in 8 of 8 rounds | +10.9 ms (-8.6..+19.5) / -0.1 ms (-18.4..+9.4) | resolved: outside both null ranges |

```csv
input,round,eager,kernel,eager-null,kernel-null
448x448,0,316.62,315.95,314.61,309.47
448x448,1,318.78,315.94,312.01,316.97
448x448,2,314.58,318.60,310.67,306.55
448x448,3,319.97,319.63,307.30,317.42
448x448,4,321.29,313.00,316.08,306.38
448x448,5,322.70,314.63,319.63,312.68
448x448,6,324.83,310.93,320.76,313.49
448x448,7,320.06,320.94,323.01,306.82
1240x1754,0,701.55,621.51,704.69,630.88
1240x1754,1,702.33,626.73,706.58,612.08
1240x1754,2,695.85,624.57,713.98,624.11
1240x1754,3,703.78,619.19,695.16,625.73
1240x1754,4,696.76,617.12,710.66,622.17
1240x1754,5,705.12,632.71,712.95,620.02
1240x1754,6,694.23,632.31,708.84,613.91
1240x1754,7,694.95,616.89,714.50,617.18
```

32 valid runs per input, none discarded; the page-image block began with two
warm-up runs. At 448x448 the activation is too small a share of a 320 ms
prefill for the two arms to be told apart. With about ten times the tokens
the kernel is 77 ms faster than the eager chain, 11 % of the prefill, in
every round and well outside the null ranges. Greedy 32-token replies are
identical between the binaries on both inputs.

## Greedy replies

Greedy 32-token image replies (`-n 32 --temp 0`, with `--show-reasoning` so a
reasoning-channel checkpoint compares its tokens and not two empty content
channels), baseline binary against fused binary:

| Checkpoint | Tanh-GELU route in its vision tower | Baseline vs fused |
|---|---|---|
| gemma-3-4b-it-4bit | SigLIP MLP, `TanhF32Compiled` | identical |
| gemma-4-e2b-it-4bit | vision GeGLU, `TanhF32Compiled` | identical |
| minicpm-v-4.6-bf16 | encoder MLP and window merger, `TanhF32Compiled` | identical |
| paligemma2-3b-6bit | SigLIP MLP, `TanhF32Compiled` | identical |
| internvl3-1b | none (InternViT keeps erf): a control | identical |
| qwen3-vl-2b | vision blocks, `TanhF32` | identical |

The first five are the checkpoints the 2026-10-23 record lists. None of the
twelve compared runs printed the NVRTC notice. A seventh checkpoint says
nothing either way: `mlx-community/Molmo2-8B-4bit` (the Molmo2 ViT is the
other vision family on the plain `TanhF32` route) launches the kernel and
gives the same reply from both binaries, but that reply is 32 `!` tokens from
the baseline binary as well. Whatever is wrong there is on the pinned main and
is not the activation.

## Erf arm considered (acceptance criterion 2)

The Gemma 3 erf arm (`vision_config.hidden_act = "gelu"`, so the SigLIP MLP
takes `VisionMlpActivation::Exact` -> `GeluActivation::Erf` -> `gelu_approx`)
and the tanh arm were each run once against an empty `MLX_PTX_CACHE_DIR` with
the fused binary. The erf arm's cache holds five modules; the tanh arm's holds
the same five plus the fused kernel:

| Module | Kernels | In erf arm | In tanh arm |
|---|---|---|---|
| `CV2IBroadcast…Multiply…_VC_V2_` (a bf16 compiled elementwise region of the model) | 42 | yes | yes |
| `…7575056202914` (nested name, a second compiled region) | 42 | yes | yes |
| `gather_bfloat16_int32_1` | 22 | yes | yes |
| `gather_uint32_int32_1` | 22 | yes | yes |
| `qmm_sm80_r2_tn_bfloat16_m128_b4_g64_affine` | 1 | yes | yes |
| `custom_kernel_mlxcel_fused_gelu_tanh_f32_bfloat16_t` | 1 | no | yes |

No erf GELU module exists because the production erf path is not a compiled
region: `gelu_approx` in `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` is the
eager chain `erf(divide(x, sqrt2))`, `add(one, …)`, `multiply(half, …)`,
`multiply(x, …)`, `astype(…, x.dtype())`, whose f32 scalar constants promote a
bf16 input to f32 along the way. Each is one AOT kernel, so the erf GELU is
about six launches per call on CUDA. The `compile_shapeless` form,
`compiled_gelu_approx`, has no production caller. Per the issue this is
recorded and left for a follow-up, which should decide between routing
`GeluActivation::Erf` to `compiled_gelu_approx` (and checking whether MLX's
CUDA compile backend splits that region too) and an erf port of the custom
kernel.

## What this does and does not establish

Established on GB10: the compiled tanh-GELU region was at least two launches
per call and the eager one about eleven; the port is one launch for both
variants; it returns the eager chain's bits on every lane measured, so
nothing downstream of either variant moves; and on the Gemma 3 and Gemma 4
image prefill the tanh arm is inside the null floor of erf and below the
compiled chain in every paired round. On Qwen3-VL, a family on the plain
`TanhF32` route, the kernel is 77 ms (11 %) faster than the eager chain on a
page-sized image and indistinguishable from it at 448x448. Greedy image
replies are unchanged on six checkpoints.

Not established here:

- The StableHLO twin re-check under the `xla-iree` feature (no IREE toolchain
  on this host). What stands in for it is the bit comparison: on this host the
  kernel's output is the eager chain's and the compiled chain's on every lane
  measured, so the MLX side of each twin comparison is unchanged.
- Metal. The Metal port and its routing are untouched, but two backend-neutral
  tests added here
  (`fused_tanh_f32_matches_the_eager_helper_to_f32_rounding_on_the_sweep`,
  `tanh_f32_dispatch_keeps_the_eager_helper_on_the_cpu`) also run on Metal and
  were not run there.
- ROCm (no port; the compiled chain and the eager helper are unchanged there).
- Image-prefill timing for MiniCPM-V 4.6 and PaliGemma (the acceptance names
  Gemma 3 and Gemma 4; the 2026-10-23 record had these two inside the null
  floor with the compiled chain), and for Molmo2 and BERT on the `TanhF32`
  route. Molmo2 has no usable end-to-end check here at all.
- A host whose NVRTC and build toolkit are different CUDA releases.

The absolute prefill numbers are for this host, kernel and driver, under its
2100 MHz GPU clock cap and with `MLX_ENABLE_TF32=0`, and are not a serving
latency.
