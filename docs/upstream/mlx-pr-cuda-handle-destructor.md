# Upstream PR draft: `~CudaHandle` must not throw while the driver is unloading

Ready to open against `ml-explore/mlx`. Nothing below needs editing; paste the title and body as they are.

- **Base:** `ml-explore/mlx` `main`
- **Head:** `inureyes:fix/cuda-handle-destructor-no-throw` (commit `e48c7a4`)
- **Open it at:** https://github.com/ml-explore/mlx/compare/main...inureyes:mlx:fix/cuda-handle-destructor-no-throw
- **Diff:** one file, `mlx/backend/cuda/cuda_utils.h`, +12 −1
- **Verified against upstream `main` at `59d600b5e`** (#4507), which still carries the unfixed destructor.

---

## Title

```
[CUDA] Do not throw from ~CudaHandle while the driver is unloading
```

## Body

```markdown
### Problem

`~CudaHandle` calls `reset()`, which checks the result through `CHECK_CUDA_ERROR` and throws on failure. A throw that escapes a destructor calls `std::terminate`, so instead of reporting anything this aborts the process.

It is reachable at ordinary shutdown. The existing guard tests `cudaPeekAtLastError()`, which catches a sticky per-context error but not the CUDA runtime unloading: once teardown has begun the peek still returns `cudaSuccess`, `reset()` proceeds, `Destroy` returns `cudaErrorCudartUnloading`, and the throw fires.

The symptom is a `SIGABRT` after the program has otherwise finished successfully:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Destroy(handle_) failed: driver shutting down
```

We hit this on Linux / CUDA (GB10, sm_121, CUDA 13) as an intermittent abort in test binaries, after the test runner had already printed its success line, which turned green runs into a non-zero exit. It is load dependent: rare on an idle machine, reproducible when another process shares the GPU. Roughly one run in ten in our case.

### Fix

The destructor releases the handle without checking the result.

`reset()` is unchanged, so `operator=` and explicit callers keep their error checking, where throwing is legal and useful. Nothing is lost by not reporting in the destructor: the handle cannot be reclaimed once the runtime is gone, and the alternative is terminating the process.

### Why the destructor is reached that late

`#4480` already leaked the global `CommandEncoder` map for this class of problem, with the comment that the encoders "would synchronize on process shutdown". Two sibling owners are not covered by that change:

- `get_command_encoders()` in `mlx/backend/cuda/device.cpp` returns a `static thread_local std::unordered_map<int, CommandEncoder>`, which is destroyed at thread exit. Each `CommandEncoder` owns a `CudaStream`, a `CudaGraph` and an `LRUCache<CudaGraphExec>`, all deriving from `CudaHandle`.
- `Worker::start()` detaches its thread, and `~CommandEncoder` calls `worker_->stop()`, which only sets a flag and notifies without waiting. The detached thread owns a `CudaStream` and outlives the encoder by an unsynchronized amount.

Either can run after the primary context has been released. We did not instrument which one fires in our case, and this change does not depend on the answer: it makes late destruction safe rather than fatal for every `CudaHandle` owner.

Whether those two owners should additionally be leaked the way `#4480` leaked the global map, or joined instead of detached, seems worth deciding separately. We did not change them here because the detach carries an explicit comment about a Windows join deadlock, and leaking a `thread_local` has a different cost profile than leaking a single global.

### Testing

Built and run on Linux / CUDA (GB10, sm_121, CUDA 13.0). 40 runs of two test subsets with a sibling process holding the GPU throughout: no abort and no non-zero exit, against a baseline near one in ten before the change. A full suite of 11,369 tests passes with no failures and no aborts.
```

---

## Notes for the submitter

- The commit is authored as `Jeongkyu Shin <jshin@lablup.com>`.
- MLX requires a CLA on first contribution; expect the bot to ask.
- `pre-commit` in the MLX repo runs `clang-format`. The added lines respect the repo's `ColumnLimit: 80` (longest is 79), but `clang-format` is not installed on this machine, so it was not run. Worth running once before opening if convenient.
- If the maintainers prefer the narrower fix, the alternative is to leak the `thread_local` encoder map the way `#4480` leaked the global one. That fixes our observed abort but leaves the general hazard that a throwing destructor represents.
