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
| `src/lib/mlxcel-core/src/weight_materialize.rs` | `MaterializationScope`, policy parsing, tower-tensor exclusion |
| `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`, `mlx_cxx_bridge.h`, `build.rs` | `array_weak_ref` / `array_weak_upgrade` bridge functions |
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

## 5. Unverified and Open

- **No V100 or PCIe device.** The V100 figures in the issue were not re-measured. GB10 is coherent memory; the DMA cost is identical for pinned and pageable here, so the gain on a PCIe device may differ.
- **Metal and ROCm.** Eager is the default there too (it matches mlx-lm's default load behavior) but was not run. The pinned overlay is CUDA only.
- **Multi-GPU and tensor-parallel.** Not run on this one-GPU host. One process-wide mutex covers every device's pool, so a process driving several GPUs would serialize uploads (LOW).
- **Failure injection.** Pinned-allocation failure was not injected; the fallback was exercised only through `MLXCEL_CUDA_PINNED_LOAD=0`. `ensure_ready` relies on `cudaHostAlloc` leaving `slot.host` null on failure before `cudaFreeHost` (LOW). If `cudaEventRecord` fails after a successful copy the slot is not marked pending, reachable only after a CUDA error that already fails the eval (LOW).
- **Remaining first-token cost.** `event_signal_kernel` fires 1,847 times on the first token in every arm, eager included: weights derived from `Load` arrays run on the generation stream at the first token. This is the next candidate inside the remaining 0.53 s and is not addressed here.
- **Largest checkpoint.** The 15 GB qwen checkpoint is the largest that loads on `main` on this host; `DeepSeek-V2-Lite` (30 GB) and `gpt-oss-20b` do not load on either branch.
- **Minor.** Slot events use `cudaEventDisableTiming` without `cudaEventBlockingSync`, so a slot wait spins a core; `MLXCEL_CUDA_PINNED_LOAD` is case-sensitive and does not accept `no`. Pre-existing upstream, not this PR: the pageable path's `malloc` is unchecked and leaks on a read exception, and `safetensors.cpp` adds `offset + data_offsets[1]` without an overflow check.
- Why the pageable read is cheaper at load (eager, 0.95 s) than inside the first `eval` (2.20 s) with the same code was not investigated; it does not affect the default.

## 6. Verification

- `weight_materialize_tests` and `ttft_profile_tests` on CUDA, including a dropped-tensor-is-freed test and `pinned_upload_chunks_and_wraps_the_slot_ring` (352 MiB tensor, six chunks, four-slot ring), added in `8b5179819` after review noted the chunking path was covered only by the 636 MB embedding in the byte-identical output.
- `cargo test --release --features cuda -p mlxcel --lib -- loading::`: 372 pass.
- `cargo clippy` with `-D warnings` on mlxcel-core and mlxcel lib and tests, `cargo fmt`, license header check.
- Reviews: implementation review (one HIGH, fixed in `4a23cfbfd`); security and performance review found no CRITICAL or HIGH items. Checked clean: slot bounds, untrusted checkpoint offsets (the overlay reads the same byte range as upstream), exception safety (`ParallelFileReader::read` joins every batch before throwing), no deadlock from `cudaEventSynchronize`.

## 7. Related

- Issue #1564 (closed by this PR); `docs/benchmark_results/volta-ttft-fixed-cost-2026-09-01.md` (#1545) measured the V100 baseline and named the three candidates.
- New environment variables: `MLXCEL_WEIGHT_MATERIALIZATION` (`eager` default, `lazy`), `MLXCEL_CUDA_PINNED_LOAD` (`0` disables). `MLXCEL_PROFILE_TTFT` gains a `[TTFT] load` line and upload counters.
