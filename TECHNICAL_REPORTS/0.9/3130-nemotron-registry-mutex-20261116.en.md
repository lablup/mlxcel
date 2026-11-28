# Technical Report: PR #3130 - guard the NemotronH decode-handle registry with a mutex

**Date**: 2026-11-16
**Status**: Completed (closes #3124)
**Languages**: C++, Rust
**Risk Level**: Low

---

## Executive Summary

#3124 reported intermittent process deaths when Metal test groups ran in one process. The cause was not Metal state but a data race in mlxcel's own C++ bridge: the NemotronH decode-handle registry was a global `std::unordered_map` with a non-atomic counter and no lock. This PR puts the registry behind a mutex and leaks it so no static destructor runs at exit.

---

## 1. Problem Statement

Every `NemotronHModel` registers a handle in `mlx_cxx_nemotron.cpp` when it is built and erases it in `Drop`. Several `server::batch` test modules (`prompt_count_tests`, `request_memory_tests`, `scheduler_tests`) build tiny NemotronH models on parallel libtest threads, so inserts, erases and rehashes raced. The crash reports on the M3 Ultra show three symptoms of the same heap corruption:

- SIGSEGV in `nemotron_free_model` during a hash-table walk.
- SIGTRAP from a malloc freelist check in an unrelated allocation.
- SIGABRT "pointer being freed was not allocated" from the map's static destructor during `exit`, after `test result: ok` had printed. This is the "lockf exit 70" case in the issue.

Production is exposed in the same way when the server loads or drops NemotronH models concurrently.

Before the fix, `server::batch::` with default threads crashed 2 times in 41 runs. The same filter with `--test-threads=1`, and the model-owned cache modules with either thread setting, never crashed (90 runs).

---

## 2. Technical Decisions

### 2.1 One mutex, model destroyed outside it

Register and free take the lock. Free moves the `unique_ptr` out under the lock and destroys the model after releasing it, so MLX array teardown never runs while other threads wait.

### 2.2 Lock only for the lookup in decode

`nemotron_decode_step` copies the model pointer under the lock and runs the step without it. This is safe because `unordered_map` never moves its elements on insert or rehash, and the handle cannot be freed mid-step: the Rust model that owns it is borrowed for the whole call. Holding the lock for the step would serialize decode across models for no benefit.

### 2.3 Leaked registry

The registry is a function-local `*new Registry()`, the pattern the other bridge caches already use. A static object's destructor at `exit` would run while detached threads might still hold it, which was the third crash symptom.

---

## 3. Change Summary

| Item | Value |
|---|---|
| Files changed | 3 |
| Tests added | 1 |

- `mlx_cxx_nemotron.cpp`: `Registry` struct (mutex, map, counter), locked register/free/lookup.
- `nemotron_registry_tests.rs`: 8 threads × 4000 register/free calls on layerless models, asserting every handle is distinct.

### Validation (M3 Ultra)

- New stress test: before the fix, 4 of 4 runs crashed in `nemotron_free_model` or `nemotron_register_model` and a fifth livelocked; after the fix, 10 of 10 pass.
- `server::batch::` with default threads: 20 of 20 pass after the fix. The `nemotron` filter on the root lib tests passes (202 tests).
- fmt, clippy (`--lib --tests -D warnings`, both crates) and the contract checks pass.

---

## 4. Follow-up Actions

- Older crash reports on this host show SIGSEGVs in `mlx::core::fast::GeneratedSourceCache::get`, reached through `metal_gated_delta_forward_impl`, from test binaries built against a different MLX revision. They are not explained by this fix and are not reproduced on the current pin.
