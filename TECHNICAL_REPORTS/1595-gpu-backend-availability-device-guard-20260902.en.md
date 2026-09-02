# Technical Report: PR #1595 - fix(core): separate GPU backend availability from the default device

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (open PR; CI green on all three commits, unit and integration coverage passing on this host; CUDA-specific paths reasoned from the pinned MLX source, see Appendix)
**Languages**: Rust, C++
**Risk Level**: Medium (touches the process-global MLX default device that every generation and test path dispatches through, but the change is additive — a rename with a deprecated shim, a new query function, and RAII guards — and is validated with hundreds of parallel test runs)

---

## Executive Summary

`mlxcel_core::is_gpu_available()` answered "is the MLX default device currently the GPU", not "does a GPU backend exist", so it turned `false` the instant anything called `set_default_device(false)` on a machine that has a GPU. That conflation misled `initialize_runtime()`, which read the flag after applying an operator's `MLXCEL_DEVICE=cpu` override and so echoed the request back as if it were a hardware finding, and it hid a real defect: two test modules moved the default device to the CPU inside a `std::sync::Once` and never moved it back, so under `--test-threads=1` every real-checkpoint gate sorting after them measured the CPU backend instead of the GPU. PR #1420 had already papered over the second symptom with an unconditional GPU pin in the shared test guard. PR #1595 separates the two questions at the bridge (`default_device_is_gpu()` for the device, a new `gpu_backend_available()` for the hardware), replaces the leaking `Once`s with an RAII guard, and — after two rounds of review found the guard alone was not enough — adds one process-wide lock so that restoring the device and excluding concurrent movers of it are no longer two different guarantees.

---

## 1. Problem Statement

### 1.1 `is_gpu_available()` conflated two different questions

The function's only implementation was `mlx::core::default_device() == mlx::core::Device::gpu`. That is a true statement about where dispatch currently lands, and every caller in the tree that used it to mean "is there a GPU" got the wrong answer whenever the default device happened to be pointed at the CPU for any reason — including reasons that had nothing to do with hardware, such as a prior caller having asked for the CPU on purpose.

### 1.2 `initialize_runtime()` echoed the request instead of reporting availability

`initialize_runtime()` used to read `is_gpu_available()` *after* applying the `MLXCEL_DEVICE=cpu` override, which by then had already moved the default device to the CPU. The resulting `RuntimeSetup.device` was therefore not a resolved fact about the host; it was a restatement of what the operator had just asked for, dressed up as if the runtime had checked. An operator or a downstream test asking "did this host actually have a GPU it declined to use" could not get an answer out of the struct.

### 1.3 The `Once` leak: PR #1420's real target

Two test modules — `multimodal::host_preprocessor_tests` and, with the identical pattern, `src/vision/merge_tests.rs` — moved MLX's process-wide default device to the CPU inside a `std::sync::Once`, precisely because a `Once` runs its initializer exactly once and never after that. The device was never moved back. MLX's default device is one process-global value that every op dispatched without an explicit stream lands on, and libtest runs a binary's test modules in name order under `--test-threads=1`. So every real-checkpoint gate — the reranker scores in `rerank::real_checkpoint_tests` among them — that happened to sort after one of the leaking modules silently measured the CPU backend instead of the GPU one. This is exactly the reproduction PR #1420 chased: `rerank::real_checkpoint_tests` scored a 4-bit Qwen3 reranker at 0.35 instead of 0.99 and produced NaN image scores from a bf16 Qwen3-VL reranker, while the same tests passed cleanly in isolation.

### 1.4 PR #1420's fix hid the leak instead of removing it

PR #1420 added an unconditional `set_default_device(true)` pin at the top of the shared `mlx_test_guard()` test helper. That repairs the symptom every time the guard runs, so the reranker gates stopped failing, but it also means the guard can never observe or report the leak: a real regression that moved the default device would be silently repaired on every subsequent call, with no assertion, no log line, and no signal to a future contributor that anything had gone wrong. The underlying `Once`s were untouched.

### 1.5 Acceptance criteria (issue #1421)

- `gpu_backend_available()` true on GB10 CUDA and on Apple Silicon, false on CPU-only; unchanged after `set_default_device(false)`.
- `default_device_is_gpu()` replaces every in-tree `is_gpu_available()` call; the deprecated shim compiles with a warning.
- `initialize_runtime()` under `MLXCEL_DEVICE=cpu` reports `device == Cpu` and the override flag; `cargo run -- generate` still runs on the CPU.
- A test in `host_preprocessor_tests.rs` (or a sibling sorting after it) asserts `default_device_is_gpu()` equals the value recorded before the module ran; passes under `--test-threads=1` with the pin reduced.
- `cargo test -p mlxcel --lib -- --test-threads=1 multimodal:: rerank::real_checkpoint_tests` passes with the pin reduced.
- `RuntimeSetup` consumers and the startup warning use the new field.

### 1.6 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A future caller re-introduces the availability/default-device conflation under a name that reads as a hardware query | High — repeats this defect | Reduced: the deprecated shim forces a compiler warning at every remaining call site, and the doc comments on both functions cross-reference each other |
| `DefaultDeviceGuard::gpu()` throws on a CPU-only build the way the old unconditional pin did | High if it happened | Eliminated: `gpu()` checks `gpu_backend_available()` before calling `set_default_device(Device::gpu)` and is a no-op otherwise |
| The RAII guard restores correctly per-test but two test modules still interleave on the one process-global device under a parallel `cargo test --lib` | High — this is what actually happened during review (see 2) | Fixed with the third commit's process-wide lock |
| A CUDA build on a driverless host reports `Cpu` but still dispatches to a GPU that MLX itself thinks exists | High — this is what actually happened during review (see 2) | Fixed by applying every CPU resolution rather than only the operator's override |

---

## 2. Technical Review

The review cycle ran two passes over the initial implementation (`7bd6e18b`) and found defects in both, each fixed in its own commit rather than folded back into the first.

### 2.1 Implementation review (`d1f2a38c`)

| Finding | Severity | Status |
|---|---|---|
| `initialize_runtime` computed `device` from `gpu_backend_available()` but applied only the operator's override to MLX, so a CPU resolution that came from the backend answer (no GPU backend, or a CUDA host without a driver) was reported without ever being installed | HIGH | Fixed |
| `vision::merge_tests` guards were not serialized by any lock, so two of its tests could interleave into the exact leak the guard exists to prevent | HIGH | Fixed |
| `host_preprocessor_tests::cpu_device` returned `(MutexGuard, DefaultDeviceGuard)`; tuple fields drop in declaration order, so the module lock released before the device was restored | HIGH | Fixed |
| `mlx_test_guard`'s assertion could not distinguish a real leak from a guard a concurrent test was legitimately holding | HIGH | Fixed with `default_device_guards_held()` |
| `DefaultDeviceGuard::drop` restores via `set_default_device(previous_is_gpu)` without the `gpu()`-style availability check | LOW | Reported only — unreachable, but worth a comment (addressed in this finalization, see §4.4) |
| `RuntimeSetup` is `pub` without `#[non_exhaustive]`, so adding `cpu_override` is a breaking change for any out-of-tree struct literal | LOW | Reported only — pre-existing style, not changed by this PR |
| `the_default_device_is_restored_after_the_export_tests` is a tautology when the test runs alone (the `OnceLock` records whatever the current value is at that moment) | LOW | Reported only — its cross-module assertion still does real verification work |

Measured effect of this commit: a representative parallel `cargo test --lib` filter failed 31 of 40 runs before the fix and 0 of 40 after; a wider 805-test parallel filter ran clean across 12 runs.

### 2.2 Security and performance review (`3741c8fd`)

| Finding | Severity | Status |
|---|---|---|
| Per-module locks do not serialize a process-global device: `multimodal::host_preprocessor`, `vision::merge`, and `mlxcel_core::streams::tests` each had their own mutex, so the modules still interleaved on the one MLX default device | HIGH | Fixed with `lock_default_device()` |
| Consequence 1: `the_default_device_is_restored_after_the_export_tests` could read the CPU default a concurrent `vision::merge` guard was legitimately holding and report it as a leak (4 of 300 parallel two-module runs failed) | HIGH | Fixed (0 of 300 after) |
| Consequence 2: any gate holding `mlx_test_guard`, including `rerank::real_checkpoint_tests` — the gate issue #1421 exists to protect — could measure a real checkpoint on the CPU backend while a concurrent guard held the device there, and the `default_device_guards_held()` exemption would stand the leak assertion down at exactly that moment | HIGH | Fixed |
| `gpu_backend_available()` throwing on any backend | — | Checked, no defect: `no_gpu` returns 0, Metal wraps in try/catch, CUDA calls noexcept `cudaGetDeviceCount` |
| `DefaultDeviceGuard::drop` hitting a throwing `set_default_device(true)` | — | Checked, no defect |
| Atomic ordering of `LIVE_DEVICE_GUARDS` | — | Checked, sound (AcqRel on the write side, Acquire on the read side) |
| Startup cost of `gpu_backend_available()` | — | Checked, negligible — see the follow-up comment on the "cheap" wording in §4.4 |
| New log lines from the startup-line change | — | Checked, emit only a boolean and a fixed string, no new trust boundary |
| MLX keeps `default_device_` in a plain non-atomic global; the one unlocked mover left after this PR is `initialize_runtime()`'s startup-only `set_default_device(false)` | MEDIUM | Reported only — not worth serializing given when it runs, but flagged so no future mover skips the lock without the same reasoning (addressed with a comment in this finalization, see §4.4) |
| `models::bert_tests::mlx_test_guard` / `models::modernbert_tests::mlx_guard` are pure delegations to the shared helper | MEDIUM | Reported only — no action needed, informational |
| The `gpu_backend_available` bridge comment says the query is "safe before runtime init", true, but reads as implying the call is cheap; on Metal the first call constructs the Metal device singleton (a metallib load) | LOW | Reported only (addressed in this finalization, see §4.4) |
| `tests/sampling_*_kill_switch.rs` moved from a default-device check to a backend check; nothing moves the default device in those binaries, so the distinction is currently unreachable there | LOW | Reported only, informational |

Measured effect: 300 parallel runs of the two-module filter (0 failures after the fix), 200 parallel runs of the mlxcel-core `streams::` filter (0), and 60 parallel runs of a 2,432-test filter spanning `models::`, `rerank::`, `embeddings::`, `vision::`, and `multimodal::` (0 failures, no hang).

---

## 3. Technical Decisions

### 3.1 Rename plus a new function, not a semantic change under the old name

**Context.** `is_gpu_available()` had callers on both sides of the availability/default-device distinction: `streams.rs` genuinely wants "is the default device the GPU right now" for its stream-selection helpers, while `initialize_runtime()`, the sampling kill-switch tests, and the sampling microbenchmarks want "does a GPU backend exist at all".

**Alternatives considered.**

| Option | Pros | Cons |
|---|---|---|
| Change `is_gpu_available()`'s meaning in place to answer hardware availability | No new function name to learn | Silently breaks every caller that relied on the old, correct-for-its-purpose default-device meaning (`streams.rs`), with no compiler signal |
| **Chosen: rename to `default_device_is_gpu()`, add `gpu_backend_available()`, keep `is_gpu_available()` as a deprecated shim** | Every caller keeps compiling; the deprecation warning forces a conscious choice of which of the two new names is correct at each site; the shim's body makes the old default-device meaning explicit in its own doc comment | Two function names to maintain instead of one, and a one-release deprecation window before the shim can be removed |

**Rationale.** The two meanings are both legitimate and both needed somewhere in the tree; the defect was never that one of them was wrong, it was that one name was standing in for both. A silent semantic change would have fixed `initialize_runtime()` and broken `streams.rs` in the same commit, with nothing in the diff calling that out. The deprecated shim converts every remaining ambiguous call site into a compile-time warning that a human has to resolve one way or the other, which is a stronger guarantee than a comment or a changelog entry.

**Trade-off.** The shim stays in the tree for one release before removal (tracked as a follow-up, §8).

### 3.2 Unclamped `device_count(Device::gpu)`, not `cfg!(feature = "cuda")`

**Context.** `gpu_backend_available()` needs to answer `false` on two different kinds of host: a CPU-only build (no GPU code compiled in at all) and a CUDA build running on a host without a driver (GPU code compiled in, but nothing to drive). A build-time `cfg!(feature = "cuda")` check can only ever see the first case.

**Rationale.** `mlx::core::device_count(Device::gpu)` is already portable across every MLX backend without an `#ifdef`: the pinned MLX's `no_gpu::device_count()` returns 0, Metal returns 1, and the CUDA backend returns whatever `cudaGetDeviceCount` reports — 0 when it fails, without throwing. `gpu_device_count()` (the existing, older bridge function) already calls the same underlying primitive unconditionally for exactly this reason; `gpu_backend_available()` reuses it but does not clamp the result to `>= 1` the way `gpu_device_count()` does, because clamping exists there so that `Device::gpu` is always a selectable index, which is precisely the wrong behavior for an availability query. A `cfg!(feature = "cuda")` check would report `true` on a CUDA build sitting on a driverless GB10 host, which is the exact failure mode `initialize_runtime()`'s CPU-resolution bug (§2.1, §3.4) depends on this function not having.

### 3.3 `assert!`, not `debug_assert!`, in the test-only leak check

**Context.** The issue's own text suggested `debug_assert!` for the `mlx_test_guard` leak assertion, on the reasoning that it is test-only code and a `debug_assert!` costs nothing in a release binary.

**Rationale.** Every gate this repository runs — `make verify-test`, `make verify-test-cuda`, `make test-fast`, and the `--profile test-fast` this finalization itself uses — builds with `--profile test-fast`, which inherits `release` and therefore compiles `debug_assert!` out entirely. A `debug_assert!` here would be `cfg`'d away everywhere the check is supposed to run, and would only ever fire in a debug build nobody's CI or local workflow produces. `mlx_test_guard` is itself `#[cfg(test)]` code, so a plain `assert!` costs nothing in a production binary regardless — the debug/release distinction that motivates `debug_assert!` elsewhere in the tree does not apply to code that is compiled out of every non-test build to begin with.

### 3.4 Applying every CPU resolution, not only the operator's override

**Context.** `initialize_runtime()` resolves `RuntimeSetup.device` from `gpu_backend_available()` and the operator's `MLXCEL_DEVICE` request through a pure `resolve_device()` function, then has to decide which resolutions to actually apply to MLX's default device. The first cut (`7bd6e18b`) applied only the operator's `MLXCEL_DEVICE=cpu` override.

**The CUDA `gpu::is_available()` quirk.** MLX seeds its own default device from `mlx::core::gpu::is_available()`, and the CUDA backend answers that call `true` unconditionally — it does not consult the driver the way `device_count(Device::gpu)` does. So a CUDA build running on a host without a working driver starts up with MLX's default device already pointed at a GPU that `gpu_backend_available()` (which does check `cudaGetDeviceCount`) correctly calls unusable. If `initialize_runtime()` only ever applied the operator's explicit override, that host would resolve `RuntimeSetup.device == Cpu` (because `resolve_device()` sees `gpu_backend_available() == false`) while MLX itself kept dispatching every uncovered op to the GPU it had already defaulted to — the struct would be reporting a fact that was false about where the runtime actually ran.

**The fix, and why the GPU direction needs no counterpart.** `initialize_runtime()` now applies `set_default_device(false)` whenever `device.uses_gpu()` is false, regardless of whether that resolution came from the operator's override or from the backend answering "no GPU here". The GPU direction is asymmetric on purpose: `device == Gpu` can only be reached when `gpu_backend_available()` is true, which requires `device_count(Device::gpu) > 0`, which on every backend implies `gpu::is_available()` is also true — so MLX's own default is already the GPU in that branch, and there is nothing to apply. Not pinning the GPU direction has a second benefit: it keeps this call out of the reach of the leak assertion in `mlx_test_guard`, which only ever expects the GPU to already be the default, not something that gets pinned there on every runtime initialization.

### 3.5 RAII guard, one process-wide lock, and the live-guard backstop — three layers, not one

**Context.** The obvious fix for a leaking `Once` is an RAII guard whose `Drop` restores the device. That is necessary, and it is what the first commit shipped. Review then found, twice, that it was not sufficient.

**Why a guard alone is not enough.** `DefaultDeviceGuard` restores the device it moved, but restoring is not the same as excluding: nothing stops a second thread from moving the same process-global device in the middle of the span an existing guard is holding. Under `cargo test --lib` without `--test-threads=1` — which `scripts/run_quality_gate.sh` runs and `docs/adding-models.md` recommends to contributors — two test modules holding two *different* guards can still interleave freely on the one MLX default device.

**Why per-module locks were insufficient (and this was caught in review, not designed in from the start).** The first attempt at excluding concurrent movers gave each of `multimodal::host_preprocessor_tests`, `vision::merge_tests`, and `mlxcel_core::streams::tests` its own private mutex. A private lock only excludes that module's own tests from each other; it does nothing to stop `multimodal::host_preprocessor` from reading the device while `vision::merge` is mid-span holding its own, private lock. That is exactly what the security review's parallel runs caught: `the_default_device_is_restored_after_the_export_tests` read a CPU default that a concurrent `vision::merge` guard was legitimately holding and reported it as a leak — a false positive from the multi-lock design, not a real regression.

**The chosen design.** `mlxcel_core::streams::lock_default_device()` is the single process-wide lock for the one process-wide device. Every mover of the default device — the two leaking-`Once`-turned-guard test modules, the `mlxcel-core::streams` tests, and `mlx_test_guard` for the whole span of the tests it serializes — takes this one lock before touching the device, so a measurement and a concurrent move can no longer overlap anywhere in the tree except the one deliberately-excepted startup call (§4.4). `DefaultDeviceGuard` itself deliberately does **not** take the lock, so guards continue to nest freely inside a span that already holds it — `mlx_test_guard` holds the lock and then constructs guards inside that span without deadlocking against itself. Lock ordering is fixed and documented: `lock_default_device()` is always taken after each module's own pre-existing serialization mutex (where one exists) and never in the other order, and no code that takes the device lock also takes that serialization mutex from the other direction, so the pair cannot deadlock.

**The live-guard backstop stays.** `default_device_guards_held()` (added in the second commit, kept in the third) remains as the exemption for a guard created without the lock — the lock now covers every in-tree caller, so in practice the backstop is defense in depth rather than the primary mechanism, but it costs nothing under the `--test-threads=1` every real gate uses, where no guard can be alive while an unrelated test runs and a genuine leak still fails loudly.

---

## 4. Implementation Details

### 4.1 The bridge: one rename, one new function

`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` / `.h`:

```cpp
// Before (mlx_cxx_bridge.cpp)
// Check whether the current default device is GPU
bool is_gpu_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// After
// Check whether the current default device is GPU
bool default_device_is_gpu() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// New
bool gpu_backend_available() {
    return mlx::core::device_count(mlx::core::Device::gpu) > 0;
}
```

The body of `default_device_is_gpu` is byte-identical to the old `is_gpu_available` — this is a pure rename at the bridge layer, with the semantic split expressed entirely by adding the second function rather than by changing the first one's behavior. `gpu_backend_available` is portable across backends without a preprocessor `#ifdef`, since `device_count` already dispatches per-backend inside MLX itself.

### 4.2 The Rust shim and the deprecation

`src/lib/mlxcel-core/src/lib.rs`:

```rust
#[deprecated(note = "use default_device_is_gpu or gpu_backend_available")]
pub fn is_gpu_available() -> bool {
    ffi::default_device_is_gpu()
}
```

Every remaining call in `is_gpu_available`'s old default-device sense now goes through `default_device_is_gpu()` directly; the shim exists purely so an out-of-tree consumer's build keeps compiling, with a warning, for one release.

### 4.3 `initialize_runtime`: resolve, then apply, in that order

`src/execution/runtime.rs`:

```rust
// Availability is a backend fact and the override is an operator request;
// resolve them separately so the setup can say which one put the runtime
// on the CPU (issue #1421).
let (device, cpu_override) =
    resolve_device(requested_device, mlxcel_core::gpu_backend_available());
// Apply every CPU resolution, not just the operator's override. ...
if !device.uses_gpu() {
    mlxcel_core::set_default_device(false);
}
```

`resolve_device` (§3.4) is a pure function taking the requested device and the backend answer and returning `(device, cpu_override)`; all four input combinations are covered by `resolve_device_separates_backend_availability_from_the_cpu_override` in `runtime_tests.rs` with no environment variable and no real device involved. The new `RuntimeSetup.cpu_override: bool` field lets the server startup log (`src/server/startup.rs`) and the CLI runtime printouts (`src/commands/generate.rs`, `src/commands/chat.rs`) report *why* the device is `Cpu` — requested, or the only option — instead of a bare `CPU` that reads the same in both cases.

### 4.4 `DefaultDeviceGuard`, `lock_default_device`, and the live-guard counter

`src/lib/mlxcel-core/src/streams.rs`:

```rust
impl DefaultDeviceGuard {
    pub fn capture() -> Self { /* records without moving */ }
    pub fn cpu() -> Self { /* captures, then moves to the CPU */ }
    pub fn gpu() -> Self {
        let guard = Self::capture();
        if ffi::gpu_backend_available() {
            ffi::set_default_device(true);
        }
        guard
    }
}

impl Drop for DefaultDeviceGuard {
    fn drop(&mut self) {
        ffi::set_default_device(self.previous_is_gpu);
        LIVE_DEVICE_GUARDS.fetch_sub(1, Ordering::AcqRel);
    }
}
```

`gpu()` checks `gpu_backend_available()` before moving, because the pinned MLX throws from `set_default_device(Device::gpu)` on a backend with no GPU — the exact way PR #1420's unconditional pin would have terminated a CPU-only test binary had it ever run there. `Drop`, by contrast, needs no such check: `previous_is_gpu` can only be `true` if a GPU backend was present when it was captured, which is the same condition under which `set_default_device(Device::gpu)` does not throw, so the restore path can never hit the throwing case. This finalization adds that reasoning as an explicit comment on the `Drop` impl (review finding, §2.1, reported as LOW and left uncommented at review time).

`lock_default_device()` (added in the third commit) is the single process-wide serialization point described in §3.5:

```rust
pub fn lock_default_device() -> DefaultDeviceLock {
    DefaultDeviceLock {
        _guard: DEFAULT_DEVICE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}
```

A poisoned lock is recovered rather than propagated, so one panicking test fails alone instead of cascading a lock-poisoning failure into every test that runs after it.

### 4.5 Replacing the leaking `Once`s

`src/multimodal/host_preprocessor_tests.rs` (the identical pattern is applied to `src/vision/merge_tests.rs`):

```rust
// Before
static CPU_DEVICE: Once = Once::new();
fn ensure_cpu_device() {
    CPU_DEVICE.call_once(|| {
        mlxcel_core::set_default_device(false);
    });
}

// After
fn cpu_device() -> (DefaultDeviceGuard, DefaultDeviceLock) {
    let lock = lock_default_device();
    DEFAULT_DEVICE_BEFORE.get_or_init(mlxcel_core::default_device_is_gpu);
    let device = DefaultDeviceGuard::cpu();
    (device, lock)
}
```

The tuple order is load-bearing, and was wrong in the first cut: tuple fields drop in declaration order, so `(DefaultDeviceLock, DefaultDeviceGuard)` would release the lock before the device guard restores, letting a waiting sibling take the lock and record the still-moved CPU device as its own baseline. The corrected order, `(DefaultDeviceGuard, DefaultDeviceLock)`, restores the device before releasing the lock. A new test, `the_default_device_is_restored_after_the_export_tests`, sorts after every test in the module that moves the device and asserts `default_device_is_gpu()` equals the value `DEFAULT_DEVICE_BEFORE` recorded — the same invariant `mlx_test_guard` checks, exercised locally to this module as well.

### 4.6 `mlx_test_guard`: from an unconditional pin to a real assertion

`src/models/embedding_test_support.rs`:

```rust
// Before (PR #1420)
mlxcel_core::set_default_device(true);

// After
let device = mlxcel_core::streams::lock_default_device();
assert!(
    mlxcel_core::default_device_is_gpu()
        || !mlxcel_core::gpu_backend_available()
        || crate::execution::runtime::cpu_override_requested()
        || mlxcel_core::streams::default_device_guards_held() > 0,
    "MLX's default device is the CPU although a GPU backend is available, \
     MLXCEL_DEVICE=cpu is not set, and no DefaultDeviceGuard is held: an \
     earlier test moved the default device and never restored it. ..."
);
```

The assertion is a disjunction over the four legitimate reasons the default device can be the CPU right now (no GPU backend, an explicit operator override, a concurrently held guard) plus the actual failure mode (a leak). It holds `lock_default_device()` for the whole span of the tests it serializes, so no concurrent mover can interleave with the checkpoint it is about to measure.

### 4.7 Comment fixes added in this finalization

Three review findings were left as reported-only because they change no behavior; this finalization closes them as documentation:

- `DefaultDeviceGuard::drop` (§4.4 above) — why the restore needs no `gpu_backend_available()` guard.
- `gpu_backend_available()`'s bridge comment (`mlx_cxx_bridge.cpp`) — the existing text said the call is "safe before runtime initialization finishes", which is true but reads as implying the call is cheap. On Metal the first call constructs the `metal::Device` singleton (a metallib load); the comment now says so explicitly, while noting that every array allocation ends up constructing that singleton anyway, so the query only moves the cost earlier rather than adding a new one, and that later calls (on Metal, and the CUDA backend's cached `cudaGetDeviceCount`) are a magic-static check.
- `initialize_runtime`'s `set_default_device(false)` call (§4.3) — the security review flagged (MEDIUM, reported only) that this is the one in-tree default-device mover left unlocked after the third commit. It is not worth serializing given it runs once at startup and is inert on a GPU host under test, but the comment now says so explicitly and states the rule for any future mover: take `lock_default_device()`.

---

## 5. Change Summary

### Statistics (`git diff --stat origin/main...HEAD`, three commits)

| Item | Value |
|---|---|
| Files changed | 21 |
| Lines added | 686 |
| Lines removed | 78 |
| Commits | 3 |

### Files by size of change

| File | +/- |
|---|---|
| `src/lib/mlxcel-core/src/streams.rs` | +306 |
| `src/execution/runtime_tests.rs` | +61/- |
| `src/execution/runtime.rs` | +69/- |
| `src/models/embedding_test_support.rs` | +89/- |
| `src/multimodal/host_preprocessor_tests.rs` | +80/- |
| `src/vision/merge_tests.rs` | +37/- |
| `src/lib/mlxcel-core/src/lib.rs` | +26/- |
| `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` | +17/- |
| `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h` | +14/- |
| `src/lib/mlxcel-core/src/ffi_tests.rs` | +8/- |
| `src/server/startup.rs` | +8 |
| `src/commands/generate.rs` | +8 |
| `src/commands/chat.rs` | +6 |
| `examples/gumbel_sampling_microbench.rs` | +6/- |
| `src/models/bert_tests.rs`, `src/models/modernbert_tests.rs` | +9/- each |
| `examples/rejection_sampling_microbench.rs` | +4/- |
| `tests/sampling_gumbel_kill_switch.rs`, `tests/sampling_rejection_kill_switch.rs` | +2/-1 each |
| `CHANGELOG.md`, `docs/environment-variables.md` | +1, +1/-1 |

### Related commits

| Hash | Type | Message |
|---|---|---|
| `7bd6e18b` | fix(core) | separate GPU backend availability from the default device |
| `d1f2a38c` | fix(core) | apply the resolved device and serialize the device guards |
| `3741c8fd` | fix(core) | serialize every default-device mover on one process lock |

### Changes by category

| Category | Summary |
|---|---|
| Bridge (C++) | Rename `is_gpu_available` → `default_device_is_gpu`; add `gpu_backend_available` |
| Core (Rust) | Deprecated shim in `mlxcel-core`; `DefaultDeviceGuard`, `lock_default_device`, `default_device_guards_held` in `streams.rs` |
| Runtime | `initialize_runtime` resolves device from backend availability, applies every CPU resolution, adds `RuntimeSetup.cpu_override` |
| Tests | Replace two leaking `Once`s with the RAII guard + lock; reduce `mlx_test_guard`'s unconditional pin to an assertion; new restore-invariant tests |
| CLI / server | Startup and runtime printouts report the override next to the backend answer |
| Docs | `CHANGELOG.md`, `docs/environment-variables.md` describe the rename, the new field, and the startup line |

---

## 6. Follow-up Actions

### Required (tracked, not blocking this PR)

- Remove the deprecated `is_gpu_available()` shim after one release, once out-of-tree callers have had a chance to migrate to `default_device_is_gpu()` or `gpu_backend_available()`.

### Reported in review, left open by design

- `RuntimeSetup` is `pub` without `#[non_exhaustive]`; adding `cpu_override` is already a breaking change for any out-of-tree struct literal that constructs it directly. This is pre-existing style in the crate and was not changed by this PR; a future pass could add `#[non_exhaustive]` to `RuntimeSetup` (and audit other public structs the crate exports) as a separate, deliberately scoped change.
- `the_default_device_is_restored_after_the_export_tests` is a tautology when run in isolation (the `OnceLock` records whatever value is current at that moment); its value comes from the cross-module case, which is exercised by the parallel-run verification in §2.2 rather than by the test itself.
- `models::bert_tests::mlx_test_guard` / `models::modernbert_tests::mlx_guard` are pure delegations to the shared `embedding_test_support::mlx_test_guard`; no action needed, noted for anyone auditing call sites.
- `tests/sampling_*_kill_switch.rs` moved from a default-device check to a backend-availability check; nothing in those binaries currently moves the default device, so the distinction is correct but currently unreachable there.

### Not verifiable on this host

- The CUDA build of the bridge (`--features cuda`): `gpu_backend_available()` uses only `mlx::core::device_count`, which `gpu_device_count()` already calls unconditionally on every backend without an `#ifdef`, and the `no_cuda` stub path is untouched, but no CUDA toolchain is available on this Apple Silicon host.
- `gpu_backend_available()` returning `true` on the GB10 CUDA build with a driver present, and `false` on a CUDA build without a driver — reasoned from the pinned MLX's `mlx/device.cpp` and `mlx/backend/cuda/device_info.cpp`, not executed.
- The reranker gate numbers PR #1420 was chasing (0.9883 for the Beijing document, finite Qwen3-VL image scores): the checkpoints are not present on this host, so `rerank::real_checkpoint_tests` soft-skips rather than scoring.

---

## Appendix

### A. Test results (this host, Apple Silicon, `--profile test-fast --features metal,accelerate`)

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --profile test-fast --features metal,accelerate --lib --tests -- -D warnings` | clean |
| `cargo test -p mlxcel-core --lib streams::` | 17 passed |
| `cargo test --lib execution::runtime` (with and without `MLXCEL_DEVICE=cpu`) | 19 passed each |
| `cargo test --lib multimodal::host_preprocessor` | 24 passed, 1 ignored |
| `cargo test --lib vision::merge` | 4 passed |
| `cargo test --lib models::gemma3_embedding` | 11 passed |
| `cargo test --lib models::bert` | 28 passed |
| `cargo test --lib models::modernbert` | 19 passed |
| `cargo test --lib -- --test-threads=1 multimodal:: rerank::real_checkpoint_tests` (PR #1420 reproduction) | 262 passed, 26 ignored (reranker gates soft-skip: checkpoints absent) |
| `cargo test --test sampling_gumbel_kill_switch` / `sampling_rejection_kill_switch` | 1 passed each |
| Parallel-run verification (review) | 300× two-module filter, 200× `streams::`, 60× 2,432-test filter across `models::`, `rerank::`, `embeddings::`, `vision::`, `multimodal::` — all 0 failures, no hang |
| CI (GitHub) | Green on `7bd6e18b` and `d1f2a38c`: cargo-fmt, cargo-clippy, cargo-deny, OpenXLA feature compile, crate versions, cross-repo refs, kernel dtype keys, llama-compat manifest, CLA; CUDA sm_70 compile and MLX pin extraction path-skipped (no CUDA runner on this repo's CI for this check) |

### B. `MLXCEL_DEVICE=cpu` generate smoke test

```
target/test-fast/mlxcel generate -m mlx-community/Phi-3.5-mini-instruct-4bit -p "Say hello." -n 3
```

Without the override: `Runtime device: Apple GPU (Metal)`, 13.27 tok/s.

With `MLXCEL_DEVICE=cpu`: `Runtime device: CPU`, plus `Running on the CPU because MLXCEL_DEVICE=cpu asked for it (GPU backend available: true).`, 0.44 tok/s.

Both lines are produced by the new `RuntimeSetup.cpu_override` field: the second line is only reachable when the override is `true` *and* a GPU backend was in fact available, which is exactly the distinction §1.2 and §3.4 exist to make reportable.

### C. What was not run

`--features cuda` cannot be compiled or exercised on this host (Apple Silicon only); see "Not verifiable on this host" above. `mlx-community/Phi-3.5-mini-instruct-4bit` and `mlx-community/Qwen3-4B-4bit` are the only checkpoints available locally, which is why the reranker real-checkpoint gates soft-skip rather than score.
