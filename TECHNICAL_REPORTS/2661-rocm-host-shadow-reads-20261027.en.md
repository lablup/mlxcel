# PR #2661: Skip host-shadow flushes after CPU reads and count shadow memory

**Date**: 2026-10-27
**Status**: Complete on Windows gfx1151; Linux pool-buffer path covered by reasoning and unit guards only
**Languages**: C++, HIP, Rust, Markdown
**Risk Level**: Medium

## Executive Summary

On Windows, WDDM does not map device memory into the CPU address space, so the ROCm allocator gives each GPU buffer a pinned host shadow (#2603). PR #2661 (issue #2643) closes the two follow-ups #2603 left at MEDIUM.
- **Reads:** a host read no longer marks the shadow dirty, so the token read of each decode step no longer costs a full host-to-device copy before the next kernel.
- **Accounting:** pinned shadow bytes are counted, reported next to active memory, and reserved by the Windows ROCm memory estimate against the RAM Windows sees, not the device limit.

The measured gain is small (+0.2 to +2.7% on Qwen3-0.6B-4bit and Qwen3-1.7B-bf16) with byte-identical output, because most remaining uploads are genuine host writes.

## 1. Problem Statement

`Buffer::raw_ptr()` set `host_dirty = true` on every access, because `array::data<T>()` cannot tell a reader from a writer: its const overload `const_cast`s to the non-const one. `gpu_ptr()` then flushed the whole buffer before the next kernel, so each CPU read of a logits or token buffer paid an upload. Shadow memory came from `hipHostMalloc` and was counted nowhere, although pinned memory is capped by the roughly 28 GiB of RAM Windows sees on the validated host, against 92+ GiB of device memory.

## 2. Change Summary

| Area | Change |
|---|---|
| Read path | New `rocm::host_read_ptr(const array&)` (declared in `rocm.h`, stubbed in `no_rocm.cpp`). It synchronizes like `data<T>()`, refreshes a stale shadow and leaves it clean. Used only where the read is proven: the bridge's `item_*`, `array_to_raw_bytes`, `array_evaluated_bytes`, `try_array_to_raw_bytes`, `staged_write`, and `MaskedScatter`'s flatten check (now a pointer and offset compare). |
| Buffer state | `RocmBuffer::host_valid` means the shadow equals the device. `ensure_host_shadow(buf, for_write)` downloads only when the shadow is neither dirty nor valid. `gpu_ptr` clears `host_valid`. A cache hit in `malloc_async` clears both flags. Zero-byte buffers never get a shadow. |
| Free paths | `release_host_shadow` is the single release point: `free_async`, the scalar pool, arena wrapper reuse (which leaked before), and the shell-only frees of alien buffers and MoE workspace views. |
| Counters | `host_shadow_bytes_`, `host_shadow_downloads_`, `host_shadow_uploads_` as relaxed atomics, moved only after a successful `hipHostMalloc`. Exported through `rocm::host_shadow_stats()`, `MLX_ROCM_HOST_SHADOW_STATS=1` (printed at exit), `memory::snapshot()`, and the `Model loaded` line and load logs. |
| Estimate | `memory_estimate_host_shadow.rs` reserves `batch × (vocab × 4 + ctx × 4)` on Windows ROCm only, compared against `GlobalMemoryStatusEx` available RAM; `fits` needs both checks; the preflight error names the host-RAM case. Zero elsewhere. |
| Build | `mlxcel-core/build.rs` links `advapi32` on Windows; MLX's CPU `device_info.cpp` reads the registry and test binaries failed to link without it. |
| Tests | `tests/rocm_host_shadow.rs` (on-host): read then kernel adds no upload; write then kernel adds one and the kernel sees it; read, write, kernel flushes; a recycled buffer with a stale valid shadow serves device bytes; a kernel overwriting a just-read buffer in place invalidates the shadow; empty arrays get no shadow; the byte counter returns to start. A pre-existing env-var race in three estimator tests is fixed. |

## 3. Technical Decisions

### 3.1 Default dirty, mark proven readers clean

The issue preferred marking writes at the writer, with a content check as fallback. The writer audit showed every writer lives in MLX core or the CPU backend: CPU op outputs, `array::init`, the gguf reader, host fills. Marking at the writer would need `mlx/array.h` as a new core overlay and would still rely on constness, so a writer that casts const away would be missed silently. A content check at `gpu_ptr()` costs a pass over each kernel argument. The inverse design keeps every unknown access on the safe, dirty path and only skips the flush where a read is proven, so the failure mode of a missed case is a redundant copy, not corrupt data.

### 3.2 Why `gpu_ptr` is the invalidation point

The review grepped the whole built ROCm backend: every device write (kernels, hipMemcpy and hipMemset targets, JIT and custom-kernel arguments, `Load::eval_gpu`) takes its pointer from `gpu_ptr`, including the const overload. HIP graphs are disabled (`use_hip_graphs()` is false), so kernels launch when encoded, and `host_access` synchronizes the device before a synchronous download, as before. A read therefore cannot set `host_valid` while an earlier writer is pending. Donation and `copy_shared_buffer` share the `RocmBuffer`, so flags travel with the bytes, which the in-place donation test checks directly.

### 3.3 A small, explicit estimate term

Measured resident shadow bytes were 0 after load and 8 bytes at exit, because weights load through `gpu_ptr` and never create a shadow. The reserve covers one f32 logits row and the int32 context ids per sequence (623 KB for Qwen3 at 4096 tokens). It is compared against OS-visible RAM because that is what bounds pinned memory on Windows. It rarely changes a decision; its purpose is to put the right limit in the check and in the report.

## 4. Validation

All on Windows 11, Radeon 8060S gfx1151, ROCm 10.1.0.
- **Throughput** (`generate -n 256 --temp 0`, 98-token prompt, five runs, medians, baseline from `main` at cbe7919f): Qwen3-0.6B-4bit prefill 81.23 to 83.42 tok/s, decode 102.58 to 104.02; Qwen3-1.7B-bf16 prefill 70.62 to 72.38, decode 31.90 to 31.95. Output byte-identical in all ten pairs. Server chat (plain, streamed) and `/v1/messages` correct.
- **Counters** for an 11-token run: 913 downloads, 899 uploads; the 14 reads left clean are the gain, the rest are host-built scalars through `array::init` (about 80 per token).
- **Tests:** `rocm_host_shadow` 3 of 3 (all cases above), `memory_estimate` lib tests 72, fmt, clippy for mlxcel-core and the mlxcel lib, bins and the new test, overlay records, env-var docs, license headers.

## 5. Known Gaps

- **Most uploads remain:** about 80 host-built scalars and index arrays per decode token are real writes through `array::init`. Building them on the device, or batching them, is the next lever and a separate change.
- **Linux pool buffers:** shadows also exist on Linux for stream-ordered pool buffers when the async pool is enabled; the new read path applies to them but was not run on a Linux host.
- **Latent paths:** `decode_capture_replay` would write buffers without `gpu_ptr` (no callers today), and shadow allocation runs outside the allocator mutex, so two threads touching one buffer at once could double-allocate (nothing does that today).
- **Windows test-build gaps** predating this PR: the whole `cargo test --features rocm` cannot link on Windows, and mlxcel-core lib tests and workspace clippy do not compile there, so the Rust unit guard runs only on Linux CI.
- **Drift check on Windows:** `rocm_overlay.py` spawns `bash`, which resolves to a non-Git bash on this host; records pass with `--pin`, drift was run by the developer before the review commits only.

## 6. Learning Points

- **Constness is not a write signal:** when an API erases constness before the point you instrument, mark the proven readers, not the writers, so an unknown case costs performance rather than correctness.
- **Zero-size allocations:** `hipHostMalloc(0)` succeeds with a null pointer. A null check that treats null as failure turns every empty array into a crash.
- **Test the hazard directly:** the stale-read risk only shows up when a kernel overwrites a buffer whose shadow was just read; asserting that donation actually happened (no new shadow allocated) is what makes the test meaningful.
