# fix(rocm): set blocking-sync before the first HIP queue is

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-set-blocking-sync-before-the-first-HIP-queu.patch`](0001-fix-rocm-set-blocking-sync-before-the-first-HIP-queu.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none (the FFT PR depends on this one) |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 20 |
| Reproduction | `repro.hip` (standalone HIP program, no MLX) |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

`rocm::device()` calls `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`, but the allocator runs before it. The first unified allocation's host read calls `hipStreamQuery(nullptr)` in `allocator::Buffer::raw_ptr()`, which creates the null stream's queue while active wait is still on, and CLR gives that queue non-interrupt completion signals. After the flag flips, CLR (`HwQueueTracker::ActiveSignal` in `rocvirtual.cpp`) no longer swaps such a signal for an interrupt one when a command needs a completion handler, and ROCr rejects async handlers on non-interrupt signals (`hsa_amd_signal_async_handler() failed to set the handler!` under `AMD_LOG_LEVEL=4`). The marker a device-wide sync enqueues on the null stream then never signals its waiter: `hipFree` hangs in `std::condition_variable::wait` below `Device::SyncAllStreams`. rocFFT's plan destroy was the first code to hit it (a hipFFT plan cache hung at its first eviction), but any `hipFree` or device-wide sync with pending null-stream work can.

`rocm::ensure_device_flags()` in `device.cpp` sets the flag once per device index. It is thread-safe, touches only that device, and restores the caller's current device. It runs before every path that can create a queue ahead of `rocm::device()`: `unified_malloc`, both branches of `raw_ptr()` and the async-pool free streams in the `RocmAllocator` constructor (`allocator.cpp`), `gpu::init()` (`eval.cpp`) and the device sync in `staged_write` (`host_stage.cpp`); `rocm::device()` uses it too. If `hipSetDevice` fails the device is not recorded, so a later call retries; if `hipSetDeviceFlags` fails, one line goes to stderr naming the device and the HIP error, so a return of this hang is not silent.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. With the change, `hipSetDeviceFlags` precedes the first `Created SWq` line in the `AMD_LOG_LEVEL=3` log (before it came after the null stream's queue), and a hipFFT workload that hung at the first plan eviction with a 16- or 32-entry plan cache completes. Only gfx1151 with HIP 7.15.26333 and rocFFT 1.0.39 was checked.

## Reproduction

`repro.hip` shows the runtime behavior without MLX: `timeout 30 ./repro 1 200` (null stream touched before the flag) hangs at its first `hipFree`, and `./repro 0 200` (flag first) prints `done`. It may be worth a CLR report on its own; none has been filed.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
