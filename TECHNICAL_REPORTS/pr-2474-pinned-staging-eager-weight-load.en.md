# PR #2474: Stage weights through pinned memory and load eagerly

**Date**: 2026-10-20
**Status**: Completed (PR open; unverified items listed in section 5)

## Executive Summary

PR #2474 (closes #1564) removes weight upload from the first token. `Load::eval_gpu` now stages tensors through four reused 64 MiB pinned host slots, and every loader's text weights are materialized before `Model loaded` prints. On GB10 with `qwen3.8-27B-4bit` the first token falls from 4.02 s to 0.67 s and load plus first token from 4.37 s to 2.24 s, with unchanged decode speed and under 0.1 GB of added peak memory.

## 1. Problem Statement

MLX `Load` arrays are lazy: `load_safetensors` returns unevaluated arrays, so a checkpoint is read and copied to the device inside the first `eval` that needs it. `mlxcel generate` printed `Model loaded` with 0.00 GB resident, then charged the whole upload to the first token (3.35 s of a 4.02 s first token on GB10; 78% on the V100 in the #1545 record). Two separable costs were involved:

- **Staging in `Load::eval_gpu`.** Each tensor went through a fresh pageable `malloc`, a `pread`, a synchronous pageable `cudaMemcpyAsync` and a `free`. On GB10 the DMA is the same 0.26 s either way (about 58 GB/s); the cost was host time: 0.89 s of `cudaMemcpyAsync` API time and a 2.20 s host staging leg (4.7-5.3 GB/s) that no CUDA profiler sees.
- **Policy.** Whether a loader charged materialization to load or to the first token was a per-loader accident (only the Gemma 4 loader evaluated its weights). A server paid the upload on its first request.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/load.cpp`, `load_staging.h` | `Load::eval_gpu` overlay with four pinned 64 MiB slots per device |
| `src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/load.cpp`, `load_staging.h`, `mlx/io/safetensors.cpp` | ROCm `Load::eval_gpu` parity (four 64 MiB `hipHostMallocPortable` slots per device, upload counters, `load_safetensors` on GPU stream) |
| `src/lib/mlxcel-core/src/weight_materialize.rs` | `MaterializationScope`, policy parsing, tower-tensor exclusion |
| `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`, `mlx_cxx_bridge.h`, `build.rs` | `array_weak_ref` / `array_weak_upgrade` and CUDA/ROCm `weight_load_transfer_stats` bridge functions |
| `src/loading/materialize.rs`, `src/loading/mod.rs` | `load_model` and adapter variants open and finish a scope |
| `src/lib/mlxcel-core/src/weights.rs` | weights loaders register returned tensors with the open scope |
| `generate.rs`, `engine/direct.rs`, `ttft_profile_tests.rs` | `[TTFT] load` line; upload counters appended to the first-token line |
| Tests | `weight_materialize_tests.rs` (355 lines), `ttft_profile_tests.rs` |
| Docs | benchmark record, `docs/environment-variables.md`, `CHANGELOG.md`, overlay README |

18 files, +1582/-7. Commits: `e56456258` (feat), `054792a5b` (benchmark record), `4a23cfbfd` (weak references), `8b5179819` (chunking test and error naming).

## 3. Technical Decisions

### 3.1 Pinned slot ring in `Load::eval_gpu`

**Behavior.** Four 64 MiB `cudaHostAlloc` slots per device are allocated once and reused. The read into one slot overlaps the `cudaMemcpyAsync` out of the previous one; each slot has a CUDA event so a slot is reused only after its copy finished. Chunks are `min(64 MiB, nbytes - done)`, so a tensor larger than a slot (embedding and untied `lm_head`, about 636 MB each) takes ten copies. Chunk boundaries are multiples of 64 MiB, so the endianness swap never splits an element. Per-tensor `malloc`, first-touch page faults and the `free` that unmapped the checkpoint are gone.

**Failure handling.** A failed `cudaHostAlloc` or event creation warns once, naming the failing call, and the process falls back to the upstream pageable path. `MLXCEL_CUDA_PINNED_LOAD=0` selects that path explicitly. Slots are never freed, so an in-flight copy cannot read freed memory.

**Trade-off.** 256 MiB of pinned host memory per device, outside MLX's allocator. Call count rises from 1,847 to 1,865 (18 extra chunks).

### 3.2 One materialization policy for every loader

**Alternatives.**

| Option | Pros | Cons |
|---|---|---|
| Each loader evaluates its own weights | No shared state | Already inconsistent (only Gemma 4); every new family must remember |
| Evaluate inside `weights.rs` loaders | One place | The loader does not know what the model keeps or drops |
| **Scope opened by `load_model`, loaders register tensors, `finish` evaluates the survivors** | Every family inherits the policy; sees the final model | Needs a registry and a lifetime rule |

`finish` calls `eval_all` then `detach_all` on registered tensors, excluding modality towers and dropped tensors. `MLXCEL_WEIGHT_MATERIALIZATION=lazy` restores first-use materialization. Loads outside a scope (pipeline stages, embedding loaders, tests) register nothing and stay lazy.

### 3.3 Weak references instead of strong handles (changed during review)

The first version held a strong handle to every registered tensor and used `array_use_count` to detect dropped tensors. Review found (HIGH) that this keeps every source buffer alive until the end of load for loaders that evaluate a repack mid-load and drop its sources: `fp8_block.rs` (FP8 block requantization) and `kimi_k3_sanitize.rs` (expert stacking). Their peak would roughly double.

The registry now holds weak references through `array_weak_ref` / `array_weak_upgrade`. Registering never extends a tensor's life, a dropped source is freed when the loader drops it, and a tensor that nothing holds upgrades to null and is skipped, so it is never read. A moved-from handle also upgrades to null and counts as unreferenced. The security review checked the bridge: `array` has the single `array_desc_` member, and the replaced descriptor is the static placeholder's (use count > 1), so bypassing `array::release` is safe. A regression test asserts that a dropped tensor is freed.

### 3.4 Vision and audio towers stay lazy

`is_multimodal_tower_tensor` matches `vision_tower.`, `visual.`, `audio_tower.` and similar leading names, plus projectors that only feed a tower. A text-only run would otherwise hold memory it never uses. On the benchmark checkpoint 333 of 2,180 tensors stay lazy; the 1,847 `language_model.` tensors (15.13 GB) are resident after load. Tower tensors load on the first image or clip.

### 3.5 What stays lazy

- Tensor-parallel loads (`--tp-size` > 1), pipeline-stage loads, embedding and reranker loads: they register no tensors.
- Tensors a loader dropped (MTP heads, tied `lm_head`, sanitized-away duplicates).
- Tensors derived from `Load` arrays (copies, repacks, dtype conversions): the source is materialized, the derivation runs at the first token, and MLX frees each source as its result is produced.

## 4. Measured Results (GB10)

`qwen3.8-27B-4bit`, 2-token prompt, warm page and JIT caches, medians of 6 (two interleaved rounds). Full record: `docs/benchmark_results/weight-materialization-gb10-2026-10-20.md`.

| Metric | Before | After |
|---|---|---|
| Model load | 0.35 s | 1.58 s (15.13 GB resident) |
| First token | 4.02 s | 0.67 s |
| Load + first token | 4.37 s | 2.24 s (-49%) |
| Process wall | 5.44 s | 3.30 s |
| `cudaMemcpyAsync` host API time (nsys) | 0.89 s / 1,847 calls | 0.02 s / 1,865 calls |
| Host staging leg | 2.20 s | 0.90 s |
| Decode slope (-n 40 vs 120) | 75.6 ms/token | 76.1 ms/token (within 1.4 ms repeat spread) |
| MLX peak after first prefill | 15.54 GB | 15.57-15.60 GB |

Attribution by arm: lazy + pinned alone gives load + first token 2.229 s, eager + pageable 2.635 s, eager + pinned 2.242 s. Pinned staging removes about 2.15 s of the 4.37 s. Eager only moves the remaining 1.25 s of upload from the first token to load; it does not reduce it. The arm with both switched off reproduces the baseline within 0.4%.

Second prefill in the same process takes 0.14 s, so the one-time cost in front of the first token is 3.88 s before and 0.53 s after. The remaining 0.53 s is not weight I/O (`eval weights=0`): it is JIT module loads, CUDA graph instantiation and cross-stream fences.

OLMoE-1B-7B (raw bf16, all 3,072 expert tensors restacked by the loader): peak 13.93 GB to 14.14-14.40 GB (+1.5 to 3.4%), not the 2x a naive reading predicts, because MLX frees each source as its repack is produced.

Correctness: greedy output (`-n 24 --temp 0`) is byte-identical across before, eager + pinned, eager + pageable and lazy; an image request is identical on both binaries.

### 4.1 Measured Results on ROCm (`gfx1151`, AMD Radeon 8060S Graphics)

Two ROCm-specific gaps were found and resolved while validating this PR on `gfx1151`:
1. **`mlx/io/safetensors.cpp` in the ROCm overlay checked only `cu::is_available()`** when choosing the default `Load` stream, so `load_safetensors` fell back to `Device::cpu` (`Load::eval_cpu`), bypassing `Load::eval_gpu` and introducing cross-stream CPU-to-GPU fences whenever a loader derived tensors from `Load` arrays. Checking `cu::is_available() || rocm::is_available()` schedules `Load` on `Device::gpu`.
2. **Per-tensor `hipHostMalloc` + `hipLaunchHostFunc(hipHostFree)` in `mlx/backend/rocm/load.cpp`.** Under eager `eval_all`, per-tensor page pinning paid 413 ms per 2 GiB in KFD page-pinning churn on `gfx1151` (standalone HIP benchmark for 16 × 128 MiB = 2 GiB: 617.22 ms total — `hipHostMalloc` 297.26 ms, `pread` 156.37 ms, `hipMemcpyAsync` 47.67 ms, `hipHostFree` 115.91 ms — vs **162.98 ms** with the 4 × 64 MiB `StagingPool`, **3.79× faster**) and kept duplicate pinned staging buffers alive alongside the device weights until batch completion. `mlx/backend/rocm/load.cpp` and `load_staging.h` now use the same 4 × 64 MiB pinned `StagingPool` (`hipHostMallocPortable` + `hipEventDisableTiming`) and `LoadTransferStats` counters as CUDA, with `MLXCEL_ROCM_PINNED_LOAD=0` (or `MLXCEL_CUDA_PINNED_LOAD=0`) selecting the pageable fallback.

End-to-end warm-cache measurements on `gfx1151` (`--no-chat-template -p "The capital of France is" -n 16 --temp 0 --seed 0 --profile`, 5 prompt tokens):

| Model | Arm | Model load | First prefill (TTFT) | Load + first token | Decode | Upload counters (`[TTFT]`) |
|---|---|---|---|---|---|---|
| `Qwen3-0.6B-4bit` (704 tensors, 0.34 GB) | `eager + pinned` | 0.304 s (0.31 GB res.) | **397.61 ms** | 0.702 s | 303.35 tok/s | load `eval=85.66ms pinned=704 memcpy=705`, prefill `weights=0` |
| `Qwen3-0.6B-4bit` | `lazy + pinned` | 0.169 s (0.00 GB res.) | 470.86 ms | 0.640 s | 318.86 tok/s | prefill `weights=704/0.34GB upload=68.34ms pinned=704` |
| `Qwen3-VL-2B-Instruct-4bit` (704 text / 315 skipped vision, 0.97 GB) | `eager + pinned` | 0.285 s (0.90 GB res.) | **395.79 ms** (-23%) | 0.681 s | 145.15 tok/s | load `eval=108.40ms pinned=704 memcpy=706`, prefill `weights=0` |
| `Qwen3-VL-2B-Instruct-4bit` | `lazy + pinned` | 0.169 s (0.00 GB res.) | 515.05 ms | 0.684 s | 143.63 tok/s | prefill `weights=704/0.97GB upload=105.01ms pinned=704` |
| `Meta-Llama-3.1-8B-Instruct-4bit` (743 tensors, 4.52 GB) | `eager + pinned` | 0.555 s (4.21 GB res.) | **444.79 ms** (-46% vs lazy+pinned, -50% vs lazy+pageable) | 1.000 s | 43.42 tok/s | load `eval=407.59ms staging=313.19ms pinned=743 memcpy=749`, prefill `weights=0` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `eager + pageable` | 0.610 s (4.21 GB res.) | 449.13 ms | 1.059 s | 43.27 tok/s | load `eval=459.69ms pageable=743 memcpy=743` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `lazy + pinned` | 0.154 s (0.00 GB res.) | 826.21 ms | 0.980 s | 43.19 tok/s | prefill `weights=743/4.52GB upload=373.87ms pinned=743 memcpy=749` |
| `Meta-Llama-3.1-8B-Instruct-4bit` | `lazy + pageable` | 0.151 s (0.00 GB res.) | 883.64 ms | 1.035 s | 43.61 tok/s | prefill `weights=743/4.52GB upload=479.22ms pageable=743` |
| `Qwen3-30B-A3B-4bit` (1,351 tensors, 17.17 GB) | `eager + pinned` | **9.409 s** (16.00 GB res.) | **634.01 ms** (-94.0% vs lazy+pinned, -95.3% vs lazy+pageable) | **10.043 s** (-26.9% vs lazy+pageable) | 61.37 tok/s | load `eval=9175.45ms staging=8537.43ms pinned=1351 memcpy=1499`, prefill `weights=0` |
| `Qwen3-30B-A3B-4bit` | `eager + pageable` | 13.610 s (16.00 GB res.) | 988.03 ms | 14.598 s | 61.06 tok/s | load `eval=13424.21ms pageable=1351 memcpy=1351` |
| `Qwen3-30B-A3B-4bit` | `lazy + pinned` | 0.188 s (0.00 GB res.) | 10,552.30 ms | 10.740 s | 59.27 tok/s | prefill `weights=1351/17.17GB upload=9908.50ms pinned=1351 memcpy=1499` |
| `Qwen3-30B-A3B-4bit` | `lazy + pageable` | 0.185 s (0.00 GB res.) | 13,555.83 ms | 13.741 s | 33.16 tok/s | prefill `weights=1351/17.17GB upload=12985.53ms pageable=1351` |

Greedy output (`-n 16 --temp 0 --seed 0`) is byte-identical between `eager` and `lazy` across all tested checkpoints on `gfx1151` (`Qwen3-0.6B-4bit`, `Qwen3-VL-2B-Instruct-4bit`, `Meta-Llama-3.1-8B-Instruct-4bit`, `gpt-oss-20b-MXFP4-Q4`, `Qwen3-30B-A3B-4bit`).

## 5. Unverified and Open

- **No V100 or PCIe device.** The V100 figures in the issue were not re-measured. GB10 and `gfx1151` are coherent/unified memory; the gain on a discrete PCIe device may differ.
- **Metal.** Eager is the default there too (it matches mlx-lm's default load behavior) but was not run. The pinned `Load::eval_gpu` overlay covers CUDA and ROCm.
- **Multi-GPU and tensor-parallel.** Not run on these one-GPU hosts. One process-wide mutex covers every device's pool, so a process driving several GPUs would serialize uploads (LOW).
- **Failure injection.** Pinned-allocation failure was not injected; the fallback was exercised only through `MLXCEL_CUDA_PINNED_LOAD=0` / `MLXCEL_ROCM_PINNED_LOAD=0`. `ensure_ready` relies on `cudaHostAlloc` / `hipHostMalloc` leaving `slot.host` null on failure before freeing (LOW). If `cudaEventRecord` / `hipEventRecord` fails after a successful copy the slot is not marked pending, reachable only after a GPU error that already fails the eval (LOW).
- **Remaining first-token cost.** `event_signal_kernel` fires 1,847 times on the first token in every arm, eager included: weights derived from `Load` arrays run on the generation stream at the first token. This is the next candidate inside the remaining 0.53 s and is not addressed here.
- **Minor.** Slot events use `cudaEventDisableTiming` / `hipEventDisableTiming` without blocking-sync flags on the event itself, so a slot wait spins a core on CUDA; `MLXCEL_CUDA_PINNED_LOAD` / `MLXCEL_ROCM_PINNED_LOAD` are case-sensitive and do not accept `no`. Pre-existing upstream, not this PR: the pageable path's `malloc` is unchecked and leaks on a read exception, and `safetensors.cpp` adds `offset + data_offsets[1]` without an overflow check.
- Why the pageable read is cheaper at load (eager, 0.95 s) than inside the first `eval` (2.20 s) on GB10 with the same code was not investigated; it does not affect the default.

## 6. Verification

- `weight_materialize_tests` (10/10) and `ttft_profile_tests` (7/7) on both CUDA (GB10) and ROCm (`gfx1151`), including a dropped-tensor-is-freed test and `pinned_upload_chunks_and_wraps_the_slot_ring` (352 MiB tensor, six chunks, four-slot ring), added in `8b5179819` after review noted the chunking path was covered only by the 636 MB embedding in the byte-identical output.
- `cargo test --release --features cuda -p mlxcel --lib -- loading::` (372 pass) and `cargo test --profile test-fast --features rocm -p mlxcel --lib -- loading::` (371 pass; 1 expected native-`nvfp4` unsupported test on ROCm).
- `scripts/ci/rocm_smoke.sh` (passed, 234.92 tok/s) and `make verify-rocm-overlay` (records + drift + ownership mock passed).
- `cargo clippy` with `-D warnings` on mlxcel-core and mlxcel lib and tests (CUDA and ROCm), `cargo fmt`, license header check.
- Reviews: implementation review (one HIGH, fixed in `4a23cfbfd`); security and performance review found no CRITICAL or HIGH items. Checked clean: slot bounds, untrusted checkpoint offsets (the overlay reads the same byte range as upstream), exception safety (`ParallelFileReader::read` joins every batch before throwing), no deadlock from `cudaEventSynchronize` / `hipEventSynchronize`.

## 7. Related

- Issue #1564 (closed by this PR); `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md` (#1545) measured the V100 baseline and named the three candidates.
- New environment variables: `MLXCEL_WEIGHT_MATERIALIZATION` (`eager` default, `lazy`), `MLXCEL_CUDA_PINNED_LOAD` (`0` disables), `MLXCEL_ROCM_PINNED_LOAD` (`0` disables). `MLXCEL_PROFILE_TTFT` gains a `[TTFT] load` line and upload counters on both CUDA and ROCm.
