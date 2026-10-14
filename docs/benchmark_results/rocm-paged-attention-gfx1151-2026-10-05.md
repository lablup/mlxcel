# HIP port of the paged-attention kernels on gfx1151 (2026-10-05)

lablup/mlxcel#2068, part of #1814. Before this change the three paged-attention port tables (v1 decode, v2 partial, merge) had no `.rocm` entry, so on ROCm the server's batched paged decode, MLA split-KV and the sparse paged decode all ran gather-then-SDPA. The ports (`src/lib/mlx-cpp/turbo/paged_attention_hip.h`) are the CUDA bodies compiled through hipRTC, with `__shfl_xor(v, o, 32)` for the lane fold. The decode profile ([rocm-decode-profile-gfx1151-2026-09-30.md](rocm-decode-profile-gfx1151-2026-09-30.md)) put paged attention at 0% of decode because `bench_decode.sh` decodes into a dense KV cache; the path these kernels serve is the server's paged decode, so that is what is measured here.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0. mlxcel `f4ca4b9a` (the PR branch on `origin/main` `c05d5438`), release build with `--features rocm`. The PR's later commits change only host-side shape checks, comments and the autotune runner label, not the kernel bodies or the dispatch.

## Method

The matrix of `scripts/benchmark_paged_decode_production.sh` (issue #899) with `Meta-Llama-3.1-8B-Instruct-4bit`: `mlxcel-server --parallel 4 --ctx-size 131072`, `scripts/bench_serving_concurrency.py` with 128 decode tokens per request, and the same binary for both arms:

- before: `MLXCEL_PAGED_ATTENTION_NATIVE=0`, the gather-then-SDPA path, which is also what every ROCm build ran before this change;
- after: unset, the fused v2 kernel.

Each case ran with a freshly started server per arm. The before and after arm of one case ran back to back inside one `scripts/rocm_gpu_guard.sh` window (55 s with `/sys/class/kfd/kfd/proc` empty and no compiler; the host was shared with other port units, and an attempt that saw a foreign GPU process or a compiler was rejected and rerun). The server log of every after arm announces `paged decode v2: fused v2 launch`, and no before arm does. The batch-4 cases ran three times; the two single-sequence cases once each, because a guarded window long enough for them was rare on the shared host.

## Results

`decode tok/s` is the mean per-request decode rate (completion tokens after the first, over the time after the first token); `aggregate tok/s` is all completion tokens over the level's wall-clock span, which includes prefill.

| Case | Decode tok/s, before | Decode tok/s, after | Decode, after / before (medians) | Aggregate tok/s, before | Aggregate tok/s, after |
|---|---|---|---|---|---|
| batch 4, ~1K prompt | 5.7 / 5.5 / 5.6 | 6.1 / 6.3 / 6.0 | 1.09x | 19.4 / 19.2 / 19.4 | 20.7 / 20.6 / 20.7 |
| batch 4, ~4K prompt | 10.1 / 10.6 / 10.2 | 10.4 / 10.6 / 10.5 | 1.03x (within run spread) | 10.8 / 11.2 / 10.9 | 11.0 / 11.3 / 11.2 |
| batch 4, ~16K prompt | 10.9 / 10.9 / 10.9 | 13.9 / 13.9 / 13.9 | 1.28x | 1.9 / 2.0 / 2.0 | 1.7 / 1.7 / 1.7 |
| batch 1, ~16K prompt | 17.2 | 24.1 | 1.40x (one run) | 2.0 | 1.7 |
| batch 1, ~32K prompt | 11.2 | 19.8 | 1.77x (one run) | 0.3 | 0.3 |

Time to first token is unchanged within a few percent in every case (prefill does not use these kernels): for example 155.1 s before and 151.1 s after (medians) at batch 4 and ~16K.

What the numbers say:

- The fused path is faster at decode wherever the context is long, and the gain grows with context: 1.28x at batch 4 and ~16K tokens per request, 1.40x and 1.77x for a single sequence at ~16K and ~32K. Run-to-run spread at batch 4 is at most 0.5 tok/s, so the ~16K and ~1K gains are outside it; the ~4K gain (1.03x) is not and is reported as no measurable change.
- The aggregate column does not follow decode at long context because it is dominated by prefill (the mean time to first token alone is about 150 s at batch 4 and ~16K) and depends on how many tokens each request generated before it stopped, which differs between the two numeric paths. It is not a measure of the kernel.
- Batch-4 decode on this server is far below the single-stream `bench_decode.sh` rate for the same model (35 tok/s): the per-request rate includes time spent waiting while the other three requests prefill. That is the server's scheduling, not this change, and is the same in both arms.

## Correctness

The 36 mlxcel-core tests that skipped on ROCm for lack of these ports (`skipping ... lablup/mlxcel#1814`) run and pass on gfx1151 with no test edits: the v1 decode against the gather path over 200 steps and a GQA and batch matrix, the v2 partial and merge against host references and the gather path (f32 and f16 pools, empty requests, trimmed windows, GQA head mapping, two pool dtypes at one geometry), cascade, sparse and MLA split-KV. `make verify-rocm` reports 11869 passed, 0 failed. With the HIP lane fold started at 8 instead of 16 and the HIP merge computed in base e instead of base 2, 29 of those tests fail.
