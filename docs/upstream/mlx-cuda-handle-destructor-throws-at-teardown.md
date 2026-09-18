# MLX CUDA: `CudaHandle`'s destructor throws while the driver is unloading

A test binary on GB10 sometimes aborts after it has already printed `test result: ok`:

```
test result: ok. N passed; 0 failed; ...
terminate called after throwing an instance of 'std::runtime_error'
  what():  Destroy(handle_) failed: driver shutting down
error: test failed, to rerun pass `--lib`
Caused by: process didn't exit successfully: ... (signal: 6, SIGABRT)
```

Cargo then exits 101 with zero failed tests, so a green run reports red. Tracked as `lablup/mlxcel#1422`.

## Where it comes from

At the pinned MLX commit `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`:

`mlx/backend/cuda/cuda_utils.h:28-34` is the destructor of the RAII base every CUDA resource wrapper uses:

```cpp
~CudaHandle() {
  // Skip if there was an error to avoid throwing in the destructors
  if (cudaPeekAtLastError() != cudaSuccess) {
    return;
  }
  reset();
}
```

`reset()` at `cuda_utils.h:46-51` calls `CHECK_CUDA_ERROR(Destroy(handle_))`, and `check_cuda_error` throws `std::runtime_error` on failure. A throw that escapes a destructor calls `std::terminate`, which is the SIGABRT.

The guard above it does not prevent this. It tests for a *sticky* CUDA error left by an earlier call. Driver unloading is not that: once the runtime has begun tearing down, `cudaPeekAtLastError()` still returns `cudaSuccess`, the destructor proceeds into `reset()`, `Destroy` returns `cudaErrorCudartUnloading`, and the throw fires.

## Which object reaches the destructor that late

Upstream has already addressed this class once. `#4480` (`24c699ece`, 2026-09-08) changed `get_global_command_encoders()` to leak, with the comment that the encoders "would synchronize on process shutdown". That commit is the one immediately before mlxcel's pin. It covers the global map only, and two sibling owners remain:

- `get_command_encoders()` (`mlx/backend/cuda/device.cpp:614`) returns a `static thread_local std::unordered_map<int, CommandEncoder>`, destroyed at thread exit. Each `CommandEncoder` owns a `CudaStream stream_`, a `CudaGraph graph_` and an `LRUCache<std::string, CudaGraphExec> graph_cache_` (`device.h:147-158`), all deriving from `CudaHandle` (`cuda_utils.h:72`, `:79`, `:84`).
- `Worker::start()` detaches its thread (`worker.cpp:20`, with a comment citing a Windows join deadlock), and `~CommandEncoder` calls `worker_->stop()` (`device.cpp:215-218`), which only sets a flag and notifies without waiting. The detached thread owns a `CudaStream signal_stream_` (`worker.h:46`) and outlives the encoder by an unsynchronized amount. (`CudaEvent` is not a `CudaHandle`; it has its own destructor.)

`Device` itself is not a candidate: `device()` leaks its vector deliberately (`device.cpp:583`).

**Which of the two fires here was not instrumented.** The abort was intermittent and is no longer reproducible with the patch applied, so the specific owner is unproven. The fix does not depend on the answer, since both reach the same base-class destructor.

That either candidate exists explains the observed load dependence: both need a teardown ordering that an idle single run rarely produces and a shared GPU does.

## Why a host-side shutdown hook does not close it

`lablup/mlxcel#1422` first proposed an explicit `mlxcel_core::shutdown()` that synchronizes and clears caches before `main` returns.

MLX does expose a usable entry point for this: `gpu::clear_streams()` (`mlx/backend/cuda/eval.cpp:91-96`) calls `get_command_encoders().clear()` and, when `is_main_thread()`, `get_global_command_encoders().clear()`. So a hook is not inert, and it would make the main thread's and the global encoders deterministic.

It still does not close the window. `get_command_encoders()` is `thread_local`, so the call clears only the calling thread's map; a detached worker thread's encoders and its own `CudaStream` are unreachable from the main thread, and a thread that has already exited cannot be drained retroactively. `std::process::exit` bypasses the hook entirely.

A hook is therefore a partial mitigation with independent value for a deterministic server shutdown path, not a fix for this abort.

## The local fix

`src/lib/mlx-cpp/patches/mlx/backend/cuda/cuda_utils.h` replaces the destructor body so it releases the handle without `CHECK_CUDA_ERROR`:

```cpp
if (handle_ != nullptr) {
  Destroy(handle_);
  handle_ = nullptr;
}
```

`reset()` is untouched, so every non-destructor caller keeps its error checking. Nothing is lost by not reporting here: the handle cannot be reclaimed once the driver is gone, and the alternative is terminating the process.

This fixes the whole class rather than the one instance, since all three wrappers share the base.

## For upstream

Two changes would be worth making in MLX itself, either independently:

1. Make `~CudaHandle` non-throwing, as here. A destructor that can throw is a latent `std::terminate` wherever it is used. Submitted as a PR; the draft is `mlx-pr-cuda-handle-destructor.md` beside this file.
2. Extend `#4480` to the two owners it did not cover: leak the `thread_local` encoder map the way `device()` and `get_global_command_encoders()` already leak theirs, or join the worker thread instead of detaching it. Both have costs the maintainers are better placed to weigh, which is why the PR does not change them.
