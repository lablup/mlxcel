# HIP ports of the fused add-RMSNorm and RoPE + append kernels on gfx1151 (2026-10-05)

lablup/mlxcel#2063, part of #1814. `fused_add_rms_norm` (#905) and `fused_rope_qk_append` (#905) had Metal and CUDA ports only, so on ROCm `MLXCEL_FUSED_ADD_RMSNORM=1` and `MLXCEL_FUSED_ROPE_APPEND=1` quietly kept the graph path. Both fusions ship off on every backend (`FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` in `src/lib/mlxcel-core/src/layers.rs`), and the decode profile ([rocm-decode-profile-gfx1151-2026-09-30.md](rocm-decode-profile-gfx1151-2026-09-30.md)) put the most they could take over at 0.83 to 0.89% of Llama 3.1 decode GPU time. This page records what the ports match, what turning them on does to decode, and why the ROCm default stays off.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`, branch at `3f0e51af` on `main` `c05d5438`. Checkpoints `Meta-Llama-3.1-8B-Instruct-4bit`, `Qwen2.5-7B-Instruct-4bit`, `Qwen3-30B-A3B-4bit` (f16 activations).

## What each port matches

Each HIP body (`src/lib/mlx-cpp/turbo/fused_norm_hip.h`, `fused_rope_append_hip.h`) follows the ROCm graph it replaces rather than the Metal or CUDA kernel, so turning a fusion on changes no bit of the output. The launch, the template arguments and the dtype cache key are the CUDA ones, except that ROCm fixes the norm's `Threads` at 256.

| Kernel | Graph it replaces on ROCm | What the port copies |
|---|---|---|
| `fused_add_rms_norm` | `add`, then the overlay's `rms_norm_kernel<T, 256, 4>` | 256 threads per row, the same strided sweep and 32-wide `__shfl_xor` folds, `1.0f / sqrtf`, the row length read at run time (`weight_shape[0]`), and the gain multiply in `T` |
| `fused_rope_qk_append` | slices, reshapes, transposes, two `fast_rope` | the angle in `rope.hip` order with `sincosf`; per-shape FMA forms (`rope_single_1d` for one token of one sequence, `rope` otherwise); for f16, one rounding from the fused result |

Four of those details were found by tests failing, not by reading the code: the hipRTC-compiled port rounded differently from the hipcc-compiled graph wherever the expression left the compiler a choice. With the row length as a compile-time 4096, 44 of 4096 f32 rows came out a normalizer ulp off. Multiplying the gain in f32 and rounding once gives the same values but +0 where the graph writes -0, which moved Llama 3.1 logits at 5 of 128 decode positions. RoPE's second rotation output is fused differently by the graph's two kernels, so a port matching one of them failed batch-1 decode. And the graph rounds an f16 RoPE result to f16 once, where `(T)fmaf(...)` rounds through f32, which disagrees on about one element in 2^13. Each case is pinned by a test that fails with its fix reverted.

Tests, all on gfx1151 (`cargo test --release --features rocm -p mlxcel-core --lib -- --test-threads=1 fused_norm_parity_tests fused_rope_parity_tests`, 21 passed):

- The existing tolerance tests (f32 1e-6 / 1e-5, f16 2e-3 / 1.2e-2, bf16 1.6e-2 / 7e-2 normalized RMS / max for the norm; 2e-3 / 1.2e-2 for RoPE) now run on ROCm; before, they returned early while the predicate was false, and now they fail if a GPU backend's predicate is false.
- `fused_add_rms_norm_is_byte_identical_to_the_rocm_graph`: f32, f16, bf16 at widths 128 to 4096, plus 1024 rows of width 4096 with row scales spread over e^-8 to e^8. Fails with the row-sized thread count and with the compile-time row length.
- `fused_add_rms_norm_keeps_the_rocm_graph_sign_of_zero`: underflowing elements and zero weights in three dtypes.
- `fused_rope_append_matches_graph_rope_every_dtype` (new tolerance sweep, f32 / f16 / bf16, with the norm tests' bf16 budget for bf16) and `fused_rope_append_is_byte_identical_to_the_rocm_graph` (batch 1 and 2, windows of 1 to 512 tokens, offsets to 131071, both conventions). Fails with only the multi-token FMA form and with the f32 rounding of f16.
- A one-off stress run of 432 RoPE cases (batch 1 to 3, windows of 1 to 300 tokens, offsets to 131000, full and partial rotary dims, values scaled up to 181x) and a 4096-row norm run at four widths matched the graph bit for bit in all three dtypes. They are not committed tests.

## Correctness on models

Teacher-forced traces with each fusion off and on (`benchmarks/logit_traces/rocm_gfx1151_3f0e51af/`): every on/off pair is byte-identical, for Llama 3.1 8B with `MLXCEL_FUSED_ADD_RMSNORM=1` and for Qwen2.5 7B with both flags, at `w1`, `w8` and `w1ctx512`. A `rocprofv3` kernel trace of an 8-token generation confirms what runs: Qwen2.5 launches `custom_kernel_mlxcel_fused_add_rms_norm_*` and `custom_kernel_mlxcel_fused_rope_qk_append_*` and no graph RoPE kernel; Llama 3.1 launches the norm port and keeps `rope_single_freqs_1d` / `rope_freqs`, because its `rope_scaling` table routes around the RoPE kernel (the one-line notice on stderr says so).

Against Metal (`python3 scripts/compare_logit_traces.py <metal> <rocm> --decided 2.0`):

| Reference | Candidate | Top-1 disagreement | Decided mismatches |
|---|---|---|---|
| `metal_m1u_bec64748` Llama 3.1 `w8` | ROCm `addrms` `w8` | 0 / 640 | 0 / 230 |
| `metal_m5_d1128266` Qwen2.5 `w8` | ROCm `addrms-rope` `w8` | 4 / 640 (largest gap 0.047) | 0 / 274 |

These match the fusions-off rows exactly, since the traces are identical.

## Decode throughput

`scripts/bench_decode.sh` at pp512/tg128, one arm per run, arm order rotated each round, every run through `scripts/rocm_gpu_guard.sh --idle-secs 60` (all 44 runs clean on the first attempt; a parallel unit was using the GPU between runs, and the guard waited it out). Off is `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_fused-norm-rope-{off,add,rope,both}.csv`.

| Model | Arm | Decode tok/s, by round | Median | vs off | Paired (on minus off, same round), median |
|---|---|---|---|---|---|
| Qwen2.5-7B (7 rounds) | off | 46.84 46.40 45.75 46.95 46.73 46.53 46.36 | 46.53 | | |
| | add | 48.35 48.30 47.23 46.62 46.99 46.84 49.05 | 47.23 | +1.5% | +1.48 (6 of 7 positive) |
| | rope | 47.79 48.65 49.26 46.51 46.80 46.73 46.69 | 46.80 | +0.6% | +0.33 (6 of 7) |
| | both | 48.10 48.88 48.00 44.25 47.32 47.28 47.31 | 47.32 | +1.7% | +0.95 (6 of 7) |
| Llama-3.1-8B (5 rounds) | off | 37.29 38.05 38.16 37.85 37.73 | 37.85 | | |
| | add | 37.95 37.95 38.15 37.99 37.64 | 37.95 | +0.3% | -0.01 (2 of 5) |
| Qwen3-30B-A3B (3 rounds) | off | 62.36 61.76 62.21 | 62.21 | | |
| | both | 62.42 62.46 61.86 | 62.42 | +0.3% | +0.06 (2 of 3) |

Prefill (512 tokens) medians moved by +0.4 to +4.3%, inside per-run spreads as wide as that (Qwen2.5 off alone ranged from 1565 to 1651 tok/s), so prefill is not read as a result either.

How to read it:

- **Llama 3.1**: nothing. Only the norm port runs there, one add + RMSNorm join per layer, and the median and the paired differences sit at zero.
- **Qwen3-30B-A3B** calls neither kernel (`qwen3_moe.rs` has its own block), so its row is a control: +0.3%, noise.
- **Qwen2.5 7B** is the one model that reaches both ports. Every on arm is above off in 6 of 7 rounds, but the size does not hold still: rounds 1 to 3 gave +0.95 to +3.5 tok/s, rounds 4 to 7 (run about two hours later on the same binary) mostly -0.4 to +1 (with -2.7 for `both` in round 4 and +2.7 for `add` in round 7), and `both` is not above `add`, which it would be if both ports saved time independently. An earlier run on the branch's first commit gave the same picture (7 rounds: off 46.54, add 46.84, rope 46.95, both 47.58 tok/s). The off arm's own spread (45.75 to 46.95) is about as wide as the median gain. A gain of 1 to 2% on Qwen2.5 is plausible from the removed dispatches (the RoPE port alone stands in for three slices, three reshape and transpose pairs and two `fast_rope` calls per layer, and #2099 showed that GPU-time shares understate paths made of many small ops), but this data does not establish it.

## Recommendation

Keep `FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` at `false` on ROCm, as on Metal and CUDA. The measurement does not clearly support a flip: no change on Llama 3.1, and on Qwen2.5 a small positive median whose size changes with the time of day. Because the ports are byte-identical to the graph, opting in on ROCm costs nothing in output, so `MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_FUSED_ROPE_APPEND=1` is safe for a deployment that measures a win on its own model. What would justify a ROCm default of on: a longer interleaved run on a quiet GPU (20 or more pairs) that shows Qwen2.5-class models (no `rope_scaling`, so both ports run) gaining at least 1% with the paired differences consistently positive, and Llama-class models not losing.

## Not covered

Metal and CUDA (not available on this host); their kernel sources and table entries are untouched. Their predicates gained the GPU-device term: GPU behavior is unchanged, and `MLXCEL_DEVICE=cpu` now takes the graph path instead of throwing at the first launch. Gemma and IQuest Loop Coder also call `fused_add_rms_norm` but have no checkpoint here; the Gemma `(1 + w)` convention is covered by the tolerance tests only. Wave64 (CDNA) is untested.

## Reproducing

```bash
cargo build --release --features rocm --bin mlxcel --bin mlxcel-bench-decode --example logit_trace
cargo test --release --features rocm -p mlxcel-core --lib -- --test-threads=1 fused_norm_parity_tests fused_rope_parity_tests
cargo test --release --features rocm --test cpu_device_custom_kernel_gates
MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_FUSED_ROPE_APPEND=1 \
    scripts/rocm_gpu_guard.sh -- scripts/bench_decode.sh models/mlx/Qwen2.5-7B-Instruct-4bit --output /tmp/on.csv
```

The trace commands are in `benchmarks/logit_traces/rocm_gfx1151_3f0e51af/README.md`.
