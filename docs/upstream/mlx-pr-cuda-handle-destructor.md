# Upstream PR draft: `~CudaHandle` must not throw while the driver is unloading

Ready to open against `ml-explore/mlx`. Paste the title and body as they are.

- **Base:** `ml-explore/mlx` `main`
- **Head:** `inureyes:fix/cuda-handle-destructor-no-throw` (commit `82b6b1d`)
- **Open at:** https://github.com/ml-explore/mlx/compare/main...inureyes:mlx:fix/cuda-handle-destructor-no-throw
- **Diff:** `mlx/backend/cuda/cuda_utils.h`, +7 −1
- Verified against upstream `main` at `59d600b5e` (#4507), which still carries the unfixed destructor.

---

## Title

```
[CUDA] Do not throw from ~CudaHandle while the driver is unloading
```

## Body

```markdown
### Problem

`~CudaHandle` calls `reset()`, which throws through `CHECK_CUDA_ERROR`. A throw escaping a destructor calls `std::terminate`.

The `cudaPeekAtLastError()` guard above it does not cover driver unloading, which is not a sticky per-context error: the peek returns `cudaSuccess`, `reset()` proceeds, `Destroy` returns `cudaErrorCudartUnloading`, and the process aborts.

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Destroy(handle_) failed: driver shutting down
```

Seen on Linux / CUDA (GB10, sm_121, CUDA 13) as an intermittent abort in test binaries *after* the runner had already reported success, turning green runs into a non-zero exit. Load dependent: roughly one run in ten when another process shares the GPU, rare on an idle machine.

### Fix

Release the handle in the destructor without checking the result. `reset()` is unchanged, so `operator=` and explicit callers keep their error checking, where throwing is legal. The handle cannot be reclaimed once the runtime is gone, so there is nothing to recover and nothing to report.

### Why it is reached that late

`#4480` leaked the global `CommandEncoder` map for this class of problem. Two owners it did not cover can still destroy a `CudaHandle` after the primary context is released:

- `get_command_encoders()` returns a `static thread_local` map, destroyed at thread exit; each `CommandEncoder` owns a `CudaStream`, a `CudaGraph` and an `LRUCache<CudaGraphExec>`.
- `Worker::start()` detaches its thread, and `~CommandEncoder` only signals `stop()` without waiting, so the detached thread and its `CudaStream` outlive the encoder by an unsynchronized amount.

This change makes late destruction safe for every owner rather than fatal. Whether those two should also be leaked as in `#4480`, or joined instead of detached, looks like a separate decision: the detach carries a comment about a Windows join deadlock, and leaking a `thread_local` costs more than leaking one global.

### Testing

Linux / CUDA (GB10, sm_121, CUDA 13.0). `pre-commit run --all` is clean. `python3 python/tests/run.py` ran 905 tests with one failure, `test_linalg.TestLinalg.test_eigh`, an eigenvalue tolerance comparison against numpy. That failure reproduces identically with this change reverted and the extension rebuilt, so it is pre-existing on this configuration rather than a regression. No test aborted.

No automated regression test accompanies this, and I want to be explicit about why rather than leave it unsaid. Reproducing the failure requires a `CudaHandle` to be destroyed after the CUDA runtime has begun unloading. That ordering only exists during process teardown, after the test framework itself is gone, so it cannot be triggered deterministically from inside a test. Making it testable would mean injecting the `Destroy` result, a larger change than the fix it would cover.

It was instead verified by repetition under the condition that provokes it: 40 runs of two test subsets with a sibling process holding the GPU throughout, no abort and no non-zero exit, against a baseline near one in ten.

No benchmarks: the change is confined to a destructor and removes an error check, so no execution path gets slower.
```

---

## Notes for the submitter

- Commit is authored as `Jeongkyu Shin <jshin@lablup.com>`.
- MLX asks for a CLA on a first contribution.
- MLX's `pre-commit` runs `clang-format`. The added lines respect `ColumnLimit: 80` (longest 79), but `clang-format` is not installed here and was not run.
- Narrower alternative, if the maintainers prefer it: leak the `thread_local` encoder map the way `#4480` leaked the global one. That fixes the observed abort but leaves a throwing destructor in place.
