# ROCm allocator memory: Radeon 8060S (gfx1151), 2026-09-30

Where the ROCm backend's peak memory went on a UMA host, which knobs move it, and the bounded defaults that shipped with it (issue #2062, part of #1814). Before the change, a pp512/tg128 run of `Meta-Llama-3.1-8B-Instruct-4bit` (4.75 GB of weights once loaded) peaked at 20.60 GB, and `Qwen3-30B-A3B-4bit` (17.17 GB) at 23.56 GB. With the defaults the peaks are 6.14 GB and 18.58 GB, and decode throughput is unchanged.

Raw harness rows (three runs per model and configuration): [`data/rocm-memory-gfx1151-2026-09-30/`](data/rocm-memory-gfx1151-2026-09-30/).

## Environment

Same host and software as the [decode baseline](rocm-baseline-gfx1151-2026-09-30.md): AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), 96 GiB VRAM carve-out (`mem_info_vram_total` 103079215104 bytes), 31 GiB host RAM (`MemTotal` 32493820 kB), GTT 96 GiB, Debian 13, kernel 6.18.12, ROCm 10.0.0 with HIP 7.15.26333. mlxcel 0.7.0 at `c5a71cfa` (main) plus this change, MLX pin `81ba1c6a`, overlay NripeshN/mlx `rocm-support` at `75915908` plus `LOCAL_FIXES.md` items 1 to 28. Release build, `--features rocm`.

## Method

Every run is the harness's shape: `mlxcel-bench-decode -p "Hello, how are you today?" -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 512`, a 20-token warmup and one measured pass in one process. The before/after rows below ran through `scripts/bench_decode.sh <model>` itself (`BENCH_RAW_DIR` kept each runner log); the knob comparisons ran the same binary with the same arguments directly, which is the command the harness issues.

### What was read

| Counter | What it covers |
|---|---|
| `mlxcel_core::memory::active_memory()` | Bytes of live MLX buffers in the ROCm allocator: weights, KV cache, activations, and any temporary a command batch still holds. |
| `cache_memory()` | Freed buffers the allocator keeps for reuse. Not in `active`. |
| `peak_memory()` | High-water mark of `active` only, never of the cache. This is the `MLX peak memory` line. |
| `mem_info_vram_used` (`/sys/class/drm/card0/device`), sampled every 0.25 s | Device-wide bytes allocated from the 96 GiB VRAM carve-out, every process included. The table reports its maximum minus its value just before the run ("VRAM delta"). The ROCm allocator's fine-grained allocations (`hipExtMallocWithFlags`) land here. |
| `mem_info_gtt_used`, same sampling | GTT (host memory mapped for the GPU). It never rose by more than 0.01 GB in any run, so no allocation fell back to GTT or managed memory. |
| `VmHWM` of the bench process | Host RSS high-water mark: 0.75 to 0.87 GB in every run, the process's own host memory; the carve-out allocations are not in it. |

`bench_decode` now prints `active`, `cache` and the phase peak at three points: after load (weights are loaded lazily, so this is still 0), after the warmup pass (prefill plus 20 tokens), and after the measured pass. `generate_with_stats` calls `clear_memory_cache()` right after prefill, so the cache figures after a pass are what decode left behind.

### Keeping the GPU to itself

Another development unit was profiling on the same GPU. Every run went through a guard that waited for 15 to 20 idle seconds (nothing in `/sys/class/kfd/kfd/proc` except this run's own processes, no `rustc`, `cargo`, `cc1plus`, `clang`, `hipcc` or linker process), sampled both every 0.25 s during the run, and rejected the run if any sample saw another GPU process or a compiler. Rejected runs were rerun; none of their numbers is used. Every number below comes from a run in which no other GPU process and no compiler was seen.

## Where the peak went

Before the change (origin/main binary plus the counters above):

| Model | Weights (`active` after warmup) | Cache after warmup | Warmup-pass peak | Measured-pass peak | Cache after measured pass | VRAM delta |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 4.75 GB | 1.64 GB | 6.19 GB | 20.60 GB | 0.42 GB | 21.19 GB |
| Qwen3-30B-A3B-4bit | 17.17 GB | 0.62 GB | 17.56 GB | 23.56 GB | 0.50 GB | 24.09 GB |

The peak is live buffers, not cache: the allocator's peak never counts the cache, and the device-wide VRAM delta is within 0.6 GB of it. It is set in the measured pass's prefill (a 20-token and a 128-token measured pass reach the same 20.60 GB), not during decode, and not in the warmup pass, whose prefill overlaps the lazy weight load and so is split into many small batches by the cross-stream waits.

The cause is how long a command batch holds what its operations allocate. `CommandEncoder::add_temporary` keeps every input and scratch buffer of an operation alive until the completion handler of its batch runs. The eager path committed a batch only every 2000 operations (`MLX_MAX_OPS_PER_BUFFER`), and it never told MLX's scheduler about those commits, so neither the scheduler's cap of 10 outstanding tasks nor `set_memory_limit` ever made the host wait. The host could therefore encode far ahead of the GPU, and the transients of many batches were live at once. For the 8B, an f16 checkpoint, the largest of them is the f16 copy of each weight matrix that `QuantizedMatmul`'s dequantize-and-GEMM path allocates for a 512-row activation (117 MB for each MLP matrix): with that path off (`MLX_ROCM_QMM_DEQUANT_GEMM=0`) the peak was 6.78 GB, at a twentieth of the prefill speed. The MoE model is bf16, and bf16 affine 4-bit matmuls take the fork's fused WMMA kernel, which allocates no weight copy, so the same variable left its peak at 23.56 GB; its excess is other operations' outputs held the same way, and the in-flight bound below removes most of it too.

The remaining pieces of the fork's allocator hold nothing in these runs:

- **Decode arena** (`decode_arena_begin`, `MLX_GRAPH_NODEFER`): `decode_arena_begin` and `decode_capture_begin` have no caller in mlxcel or in the rest of the overlay, and `use_hip_graphs()` returns `false`, so the arena is never allocated (capacity and high-water mark 0) and graph deferral never runs.
- **Async pool** (`MLX_ROCM_USE_ASYNC_POOL`, `MLX_ROCM_FORCE_ASYNC_POOL`; `MLX_ROCM_NO_ASYNC_POOL` forces it off): off by default. Turned on, both models hit a GPU memory fault in `gather_rows_kernel` during the warmup pass and aborted, which is the failure the fork's comment warns about.
- **Managed/GTT fallback** (`MLX_ROCM_ALLOW_MANAGED_FALLBACK`, `MLX_ROCM_NO_MANAGED_FALLBACK`): only used when a device allocation fails, which never happened (GTT did not move).
- **Buffer cache**: 0.4 to 1.7 GB at every phase boundary. It is not what set the peak, but it had no working bound: `set_cache_limit` stored its argument and nothing read it, and the cache only reuses a buffer of exactly the requested size. Its default limit was the allocator's memory limit, 76.8 GiB here.

## Every knob, before the change

Measured pass peak, VRAM delta, and throughput, one run each:

| Setting | Llama peak | Llama VRAM delta | Llama prefill / decode tok/s | Qwen peak | Qwen VRAM delta | Qwen prefill / decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| none (default) | 20.60 GB | 21.19 GB | 1069.34 / 37.21 | 23.56 GB | 24.09 GB | 277.53 / 61.88 |
| `MLXCEL_CACHE_LIMIT=1GB` | 20.60 GB | 21.19 GB | 1066.17 / 37.10 | 23.56 GB | 24.09 GB | 261.64 / 61.63 |
| `MLXCEL_MEMORY_LIMIT=8GB` (Llama), `24GB` (Qwen) | 20.60 GB | 21.19 GB | 1060.90 / 36.91 | 23.56 GB | 24.09 GB | 273.75 / 61.95 |
| `MLX_ROCM_USE_ASYNC_POOL=1` | GPU memory fault | | | GPU memory fault | | |
| `MLX_ROCM_NO_ASYNC_POOL=1` | 20.60 GB | 21.19 GB | 1070.09 / 37.29 | 23.56 GB | 24.09 GB | 274.02 / 61.99 |
| `MLX_ROCM_NO_MANAGED_FALLBACK=1` | 20.60 GB | 21.19 GB | 1066.52 / 37.21 | 23.56 GB | 24.09 GB | 276.83 / 61.83 |
| `MLX_ROCM_ALLOW_MANAGED_FALLBACK=1` | 20.60 GB | 21.20 GB | 1066.96 / 37.10 | 23.56 GB | 24.09 GB | 269.75 / 61.82 |
| `MLX_ROCM_FINEGRAINED=0` (coarse `hipMalloc`) | 20.60 GB | 21.19 GB | 1071.94 / 37.33 | 23.56 GB | 24.09 GB | 270.32 / 61.77 |
| `MLX_GRAPH_NODEFER=1` | 20.60 GB | 21.19 GB | 1072.63 / 37.05 | 23.56 GB | 24.09 GB | 261.29 / 61.75 |
| `MLX_MAX_OPS_PER_BUFFER=50` | 20.60 GB | 21.19 GB | 1074.03 / 37.44 | 23.56 GB | 24.09 GB | 270.77 / 61.75 |
| `MLX_ROCM_QMM_DEQUANT_GEMM=0` | 6.78 GB | 7.16 GB | 51.75 / 37.19 | 23.56 GB | 24.09 GB | 263.29 / 61.14 |
| `MLXCEL_CACHE_CLEAR_INTERVAL=0` | 20.60 GB | 21.19 GB | 1063.32 / 36.96 | not run | | |

No knob bounded the peak without giving up prefill. MLX's two limits did nothing on ROCm: `set_cache_limit` was not enforced, and `set_memory_limit` only acts through the scheduler's task accounting, which the eager path bypassed. More frequent commits alone (`MLX_MAX_OPS_PER_BUFFER=50`) did nothing either, because the host never waited on them. The periodic cache clear does not fire in a 128-token run (its cadence is 256 tokens). Prefill on the MoE model varies widely from run to run (from 203 to 315 tok/s over the runs on this page), so its differences in this table are noise.

## What shipped

Two bounds, one per mechanism; both are documented in `docs/installation.md` (Linux with AMD ROCm, Memory footprint):

1. **In-flight bound (overlay, `LOCAL_FIXES.md` item 28).** `gpu::eval` counts what each operation allocates, the eager path commits a batch once it has allocated a quarter of `MLX_ROCM_MAX_INFLIGHT_MB` (default 1024), and the host waits for the oldest committed batch while the committed ones exceed the budget. The last batch of each eval is counted too. `0` restores the old behavior.
2. **Cache bound.** The overlay's `malloc_async` now trims the cache to three quarters of its limit on a cache miss once it is over the limit, so `set_cache_limit` works, and mlxcel applies `MLXCEL_CACHE_LIMIT=2GB` by default on ROCm builds (`src/execution/runtime.rs`); an explicit value, including `0` or `none`, still wins.

### Choosing the in-flight budget

With the 2 GiB cache default, one run each. This sweep and the next ran on the first version of the bound, whose host wait spun on `hipEventQuery`, which did not count the last batch of each eval, and whose cache trim went down to the limit rather than to three quarters of it; the shipped version is what the harness table further down measured.

| `MLX_ROCM_MAX_INFLIGHT_MB` | Llama peak | Llama VRAM delta | Llama prefill / decode | Qwen peak | Qwen VRAM delta | Qwen prefill / decode |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (off) | 20.60 GB | 21.19 GB | 1065.74 / 37.10 | 23.56 GB | 24.10 GB | 296.08 / 61.86 |
| 256 | 5.28 GB | 5.88 GB | 1042.60 / 37.67 | 17.82 GB | 18.40 GB | 303.34 / 61.84 |
| 512 | 5.75 GB | 6.34 GB | 1039.04 / 37.72 | 18.06 GB | 18.65 GB | 294.04 / 61.91 |
| 1024 (default) | 6.36 GB | 6.99 GB | 1038.93 / 37.82 | 18.57 GB | 19.15 GB | 202.66 / 61.64 |
| 2048 | 7.34 GB | 8.03 GB | 1052.68 / 38.20 | 19.64 GB | 20.30 GB | 307.57 / 61.74 |
| 4096 | 9.36 GB | 10.01 GB | 1054.57 / 37.31 | 21.66 GB | 22.33 GB | 302.42 / 61.67 |

The peak tracks the budget, as intended, and decode does not move. The 202.66 tok/s Qwen prefill at 1024 is the MoE model's run-to-run spread, not the setting (see the harness runs below). 1024 MiB is the default because a transient larger than a quarter of the budget commits its own batch, and a model with larger weight matrices than these two (a 70B MLP matrix is about 470 MB in f16, computed, not measured) keeps the GPU fed only if the budget holds a few of them; the cost below 1024 MiB is no lower anyway.

### Choosing the cache bound

With the in-flight default, one run each:

| `MLXCEL_CACHE_LIMIT` | Llama peak | Llama prefill / decode | Qwen peak | Qwen prefill / decode |
|---:|---:|---:|---:|---:|
| 128MB | 6.15 GB | 892.73 / 37.70 | 18.57 GB | 303.53 / 61.24 |
| 512MB | 6.36 GB | 1062.48 / 37.72 | 18.58 GB | 298.54 / 62.03 |
| 1GB | 6.39 GB | 1040.75 / 37.20 | 18.57 GB | 301.62 / 61.87 |
| 2GB (default) | 6.39 GB | 1027.63 / 37.50 | 18.60 GB | 301.98 / 61.97 |
| none | 6.30 GB | 1062.33 / 37.33 | 18.58 GB | 297.39 / 61.84 |

Decode stayed within 2% of the unbounded run at every value, down to 128 MiB, so decode alone would allow a very small bound. Prefill does not: at 128 MiB the 8B lost 16%, most likely because each 117 MB f16 weight copy no longer survives in the cache from one matrix of the same shape to the next and is reallocated every time. From 512 MiB up nothing was measurable. The default is 2 GiB, four times the smallest value with no measured cost on either model, to leave room for the larger weight copies of bigger models. In these runs the cache never reached 2 GiB, so the default did not bind; it bounds what a long-running process can accumulate between the periodic clears, which the exact-size cache otherwise let grow toward 76.8 GiB.

## Before and after, through the harness

Three runs each through `scripts/bench_decode.sh`, alternating. "Before" is the same binary with both bounds off (`MLXCEL_CACHE_LIMIT=none MLX_ROCM_MAX_INFLIGHT_MB=0`), which reproduced the origin/main figures exactly (20.60 GB and 23.56 GB peak, 21.19 GB and 24.09 to 24.11 GB VRAM delta). Medians; the peak and the VRAM delta varied by at most 0.14 GB across the three runs of each row:

| Model | Setting | MLX peak | VRAM delta | Steady state (active + cache after the measured pass) | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | before | 20.60 GB | 21.19 GB | 5.16 GB | 1064.83 | 37.02 |
| Llama-3.1-8B-Instruct-4bit | defaults | 6.14 GB | 6.73 GB | 5.18 GB | 1045.87 | 37.57 |
| Qwen3-30B-A3B-4bit | before | 23.56 GB | 24.11 GB | 17.67 GB | 300.07 | 61.79 |
| Qwen3-30B-A3B-4bit | defaults | 18.58 GB | 19.16 GB | 17.56 GB | 303.75 | 61.69 |

Decode changes by +1.5% and -0.2%, inside the 2% bound. Prefill changes by -1.8% on the 8B, the host now waiting on the GPU instead of queueing far ahead; the MoE model's +1.2% is noise (its prefill ranged from 264.33 to 315.02 tok/s over the three runs with the defaults and 291.52 to 306.43 without). The steady state was already close to the weights, because `generate_with_stats` clears the cache after prefill; the peak is what moved.

## The pre-load memory estimate

On ROCm the estimate behind `mlxcel inspect` and `--estimate-memory` reads the allocator's `memory_limit()` as available memory (#1805). Neither bound changes it: the cache default goes through `set_cache_limit`, and the in-flight budget is internal to the backend, so `memory_limit()` is still 76.80 GiB and every estimate is what it was. `runtime_tests::the_cache_default_leaves_the_memory_limit_the_estimator_reads` checks that runtime bring-up leaves it alone. What the estimate predicts is another matter, and the change moves reality toward it. `mlxcel inspect --max-tokens 640` (the pp512/tg128 context) estimates 5.58 GB for the 8B and 20.72 GB for the MoE model (weights and KV cache times 1.20 plus an activation term) and reports 76.80 GiB available for both, as before. The measured peaks were 20.60 GB and 23.56 GB before and are 6.14 GB and 18.58 GB now: the 8B still peaks 0.56 GB above its estimate (it was 15 GB above), and the MoE model now peaks below its estimate. Recalibrating the 1.20 factor for ROCm is not part of this change.

## Reproducing

```bash
cargo build --release --features rocm --bin mlxcel --bin mlxcel-bench-decode
BENCH_RAW_DIR=/tmp/raw MODELS_DIR=models/mlx ./scripts/bench_decode.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
MLXCEL_CACHE_LIMIT=none MLX_ROCM_MAX_INFLIGHT_MB=0 BENCH_RAW_DIR=/tmp/raw-before \
    MODELS_DIR=models/mlx ./scripts/bench_decode.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
grep -E '^\[Memory\]|MLX peak memory' /tmp/raw/*.log /tmp/raw-before/*.log
```

Check that `rocm-smi --showpids` lists no other process and nothing is compiling before each run.
