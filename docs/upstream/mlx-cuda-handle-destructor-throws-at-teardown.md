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

`CommandEncoder` owns three `CudaHandle` subclasses: `CudaStream stream_` and `CudaGraph graph_` (`mlx/backend/cuda/device.h:147-148`) and `LRUCache<std::string, CudaGraphExec> graph_cache_` (`device.h:158`). `CudaGraph`, `CudaGraphExec` and `CudaStream` all derive from `CudaHandle` (`cuda_utils.h:72`, `:79`, `:84`).

MLX already recognises the teardown hazard for two of the three places it stores these, and deliberately leaks both:

- `mlx/backend/cuda/device.cpp:583-592`, the `Device` vector: *"The devices are leak intentionally as user code may still be accessing device after main thread teardown."*
- `mlx/backend/cuda/device.cpp:619-623`, the global encoder map: *"encoders are leaked intentionally as they would synchronize on process shutdown."*

The third is not leaked. `mlx/backend/cuda/device.cpp:614-617`:

```cpp
std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}
```

This `thread_local` has a non-trivial destructor and runs at thread exit. When a worker thread ends after the driver has released the primary context, the map destroys its `CommandEncoder`s, those destroy their stream, graph and cached graph execs, and the first one to call `Destroy` throws out of a destructor.

That also explains the observed load dependence: it needs a thread to exit late enough relative to driver teardown, which is why an isolated run on a quiet box rarely shows it and a shared GPU does.

## Why a host-side shutdown hook does not fix it

`lablup/mlxcel#1422` first proposed an explicit `mlxcel_core::shutdown()` that synchronizes and clears caches before `main` returns. That does not reach this: the objects are `thread_local` to MLX's worker threads, so a call on the main thread destroys none of them, and a thread that has already exited cannot be drained retroactively. `std::process::exit` bypasses it entirely.

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

1. Make `~CudaHandle` non-throwing, as here. A destructor that can throw is a latent `std::terminate` wherever it is used.
2. Leak the `thread_local` encoder map the way `device()` and `get_global_command_encoders()` already leak theirs, or drain it on thread exit before the driver can unload.
